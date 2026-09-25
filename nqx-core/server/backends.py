"""Two backends: CPU (NQX emulator) and GPU (PyTorch / optional Triton)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from nqx.constants import NQXConfig, GOLDEN_ANGLE
from nqx.cpu import NQXCore


class Backend(Protocol):
    name: str
    device: str
    config: NQXConfig

    def encode(self, x: np.ndarray, bits: Optional[int] = None) -> dict: ...
    def decode(
        self, q: np.ndarray, sign: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bits: int
    ) -> np.ndarray: ...
    def info(self) -> dict: ...


class CPUBackend:
    name = "cpu-nqx"

    def __init__(self, dim: int = 128, bits: int = 3):
        self._reconfigure(dim, bits)
        self.device = f"cpu (numpy {np.__version__})"

    def _reconfigure(self, dim: int, bits: int) -> None:
        self.config = NQXConfig(dim=dim, bits=bits)
        self.core = NQXCore(self.config)

    def encode(self, x: np.ndarray, bits: Optional[int] = None) -> dict:
        if bits is not None and bits != self.config.bits:
            self._reconfigure(self.config.dim, bits)
        if x.shape[-1] != self.config.dim:
            self._reconfigure(x.shape[-1], self.config.bits)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        t0 = time.perf_counter()
        enc = self.core.encode(x.astype(np.float32))
        dt = (time.perf_counter() - t0) * 1000.0

        polar_rmse = float(
            np.sqrt(
                (
                    (
                        enc.polar
                        - self._dequantize(
                            enc.quantized_indices, enc.mins, enc.maxs, self.config.bits
                        )
                    )
                    ** 2
                ).mean()
            )
        )

        return {
            "q": enc.quantized_indices,
            "sign": enc.sign_bits,
            "mins": enc.mins,
            "maxs": enc.maxs,
            "packed": enc.packed_bytes,
            "encode_ms": dt,
            "cycles": enc.cycles,
            "energy_nj": enc.energy_nj,
            "polar_rmse": polar_rmse,
            "n": x.shape[0],
            "dim": x.shape[1],
            "bits": self.config.bits,
        }

    def decode(
        self, q: np.ndarray, sign: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bits: int
    ) -> dict:
        if bits != self.config.bits:
            self._reconfigure(self.config.dim, bits)
        from nqx.cpu import EncodeResult

        polar = self._dequantize(q, mins, maxs, bits).astype(np.float32)
        enc_shim = EncodeResult(
            quantized_indices=q,
            sign_bits=sign,
            mins=mins.astype(np.float32),
            maxs=maxs.astype(np.float32),
            packed_bytes=b"",
            polar=polar,
            cycles=0,
            energy_nj=0.0,
        )
        t0 = time.perf_counter()
        dec = self.core.decode(enc_shim)
        dt = (time.perf_counter() - t0) * 1000.0
        return {"x": dec.reconstructed, "decode_ms": dt}

    def info(self) -> dict:
        return {
            "dim": self.config.dim,
            "bits": self.config.bits,
            "phi": self.config.phi,
            "golden_angle_deg": math.degrees(GOLDEN_ANGLE),
            "n_pairs_l1": len(self.core.lut.layers["L1"]),
            "n_pairs_l2": len(self.core.lut.layers["L2"]),
            "n_pairs_l3": len(self.core.lut.layers["L3"]),
            "rom_bytes": self.core.lut.rom_bytes(),
        }

    def rotation_matrix(self) -> np.ndarray:
        return self.core.rotation_matrix()

    def forward_rotation(self, x: np.ndarray) -> np.ndarray:
        return self.core.forward_rotation(x)

    def inverse_rotation(self, x: np.ndarray) -> np.ndarray:
        return self.core.inverse_rotation(x)

    @staticmethod
    def _dequantize(q: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bits: int) -> np.ndarray:
        levels = 2**bits
        ranges = np.maximum(maxs - mins, 1e-8)
        return (q.astype(np.float32) / (levels - 1)) * ranges + mins


@dataclass
class _GPUImpl:
    torch: object
    ref: object
    device: str


class GPUBackend:
    name = "gpu-torch"

    def __init__(self, dim: int = 128, bits: int = 3):
        import torch

        try:
            from nautilus_triton import NautilusConfig, NautilusQuantPyTorch
        except ImportError as e:
            raise RuntimeError(
                "GPUBackend requires nautilus_triton.py (reference/ in the NautilusQuant repo) "
                "on PYTHONPATH. Set PYTHONPATH or git clone it into the image."
            ) from e

        self.torch = torch
        self.NautilusConfig = NautilusConfig
        self.NautilusQuantPyTorch = NautilusQuantPyTorch
        self.dim = dim
        self.bits = bits
        self.config = NQXConfig(dim=dim, bits=bits)
        self._reconfigure(dim, bits)

        self.cuda = torch.cuda.is_available()
        self.device = (
            f"cuda ({torch.cuda.get_device_name(0)})"
            if self.cuda
            else f"cpu (torch {torch.__version__})"
        )
        self._device = "cuda" if self.cuda else "cpu"

    def _reconfigure(self, dim: int, bits: int) -> None:
        self.dim = dim
        self.bits = bits
        self.config = NQXConfig(dim=dim, bits=bits)
        cfg = self.NautilusConfig(dim=dim, bits=bits)
        self.ref = self.NautilusQuantPyTorch(cfg)

    def encode(self, x: np.ndarray, bits: Optional[int] = None) -> dict:
        if bits is not None and bits != self.bits:
            self._reconfigure(self.dim, bits)
        if x.shape[-1] != self.dim:
            self._reconfigure(x.shape[-1], self.bits)
        torch = self.torch

        x_t = torch.from_numpy(x.astype(np.float32))
        if self.cuda:
            x_t = x_t.cuda()

        t0 = time.perf_counter()
        if self.cuda:
            torch.cuda.synchronize()
        enc = self.ref.encode(x_t)
        if self.cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0

        rotated = self.ref.forward(x_t)
        polar = self.ref._to_polar(rotated)
        polar_np = polar.detach().cpu().numpy() if self.cuda else polar.detach().numpy()
        corrected_np = (
            enc["corrected"].detach().cpu().numpy()
            if self.cuda
            else enc["corrected"].detach().numpy()
        )
        scales_np = (
            enc["scales"].detach().cpu().numpy() if self.cuda else enc["scales"].detach().numpy()
        )
        zeros_np = (
            enc["zeros"].detach().cpu().numpy() if self.cuda else enc["zeros"].detach().numpy()
        )

        mins = zeros_np.astype(np.float32).reshape(-1)
        maxs = (zeros_np + scales_np).astype(np.float32).reshape(-1)

        levels = 2**self.bits
        ranges = np.maximum(maxs - mins, 1e-8)
        q = (
            np.round(((corrected_np - mins) / ranges) * (levels - 1))
            .clip(0, levels - 1)
            .astype(np.uint8)
        )
        sign = (polar_np >= corrected_np).astype(np.uint8)

        from nqx.functional_units import PackUnit

        packed, _ = PackUnit(self.config).pack3plus1(q, sign)
        polar_rmse = float(np.sqrt(((polar_np - corrected_np) ** 2).mean()))

        return {
            "q": q,
            "sign": sign,
            "mins": mins,
            "maxs": maxs,
            "packed": packed,
            "encode_ms": dt,
            "cycles": 0,
            "energy_nj": 0.0,
            "polar_rmse": polar_rmse,
            "n": x.shape[0],
            "dim": x.shape[1],
            "bits": self.bits,
        }

    def decode(
        self, q: np.ndarray, sign: np.ndarray, mins: np.ndarray, maxs: np.ndarray, bits: int
    ) -> dict:
        if bits != self.bits:
            self._reconfigure(self.dim, bits)
        torch = self.torch

        levels = 2**bits
        ranges = np.maximum(maxs - mins, 1e-8)
        polar_np = (q.astype(np.float32) / (levels - 1)) * ranges + mins
        polar_t = torch.from_numpy(polar_np)
        if self.cuda:
            polar_t = polar_t.cuda()

        t0 = time.perf_counter()
        if self.cuda:
            torch.cuda.synchronize()
        x = self.ref.decode(polar_t)
        if self.cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0

        x_np = x.detach().cpu().numpy() if self.cuda else x.detach().numpy()
        return {"x": x_np, "decode_ms": dt}

    def info(self) -> dict:
        return {
            "dim": self.dim,
            "bits": self.bits,
            "phi": self.config.phi,
            "golden_angle_deg": math.degrees(GOLDEN_ANGLE),
            "n_pairs_l1": len(self.ref.layer1_pairs),
            "n_pairs_l2": len(self.ref.layer2_pairs),
            "n_pairs_l3": len(self.ref.layer3_pairs),
            "rom_bytes": (
                len(self.ref.layer1_pairs) + len(self.ref.layer2_pairs) + len(self.ref.layer3_pairs)
            )
            * 10,
        }


def auto_select(dim: int = 128, bits: int = 3, prefer: str = "auto") -> Backend:
    if prefer == "cpu":
        return CPUBackend(dim, bits)
    if prefer == "gpu":
        return GPUBackend(dim, bits)
    try:
        return GPUBackend(dim, bits)
    except Exception:
        return CPUBackend(dim, bits)
