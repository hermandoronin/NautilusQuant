#!/usr/bin/env python3
"""Numerics report for the NQX-S1 specification (spec/04_numerics_results.md).

Measures, on the bit-accurate model:
  1. CORDIC accuracy against float64.
  2. Rotation round-trip error (GVNS x3 then GVNS_INV x3) at int16 precision.
  3. Reconstruction error of the complete encode/decode path, compared with
     the nqx-core reference pipeline and with other rotations.

Usage: python tools/bench_numerics.py [--out spec/04_numerics_results.md]
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT / "verif" / "cocotb")]

from nqx_s1 import S1Core, S1Params, host  # noqa: E402
from nqx_s1 import cordic  # noqa: E402
from nqx_s1 import reference as ref  # noqa: E402
from stimulus import full_scale_vector, kv_vector  # noqa: E402

P = S1Params()
NQX_CORE = ROOT.parent / "nqx-core"


def cordic_accuracy(n: int = 20000) -> dict[str, float]:
    rng = random.Random(11)
    rot, vec_r, vec_t = [], [], []
    for _ in range(n):
        a = rng.randint(-(2**22), 2**22)
        b = rng.randint(-(2**22), 2**22)
        mag = math.hypot(a, b)
        if mag < 2**12:
            continue
        z = rng.randrange(1 << P.zw)
        ang = z / (1 << P.zw) * 2 * math.pi
        x, y, _ = cordic.rotate(P, a, b, z)
        ex, ey = a * math.cos(ang) - b * math.sin(ang), a * math.sin(ang) + b * math.cos(ang)
        rot.append(math.hypot(x - ex, y - ey) / mag)
        r, t, _ = cordic.vector(P, a, b)
        vec_r.append(abs(r - mag) / mag)
        err = (t / (1 << P.zw) * 2 * math.pi - math.atan2(b, a) + math.pi) % (2 * math.pi) - math.pi
        vec_t.append(abs(err))
    return {
        "rot_rel_rms": float(np.sqrt(np.mean(np.square(rot)))),
        "rot_rel_max": float(np.max(rot)),
        "vec_r_rel_max": float(np.max(vec_r)),
        "vec_t_rad_max": float(np.max(vec_t)),
    }


def roundtrip(vectors: list[list[int]]) -> tuple[int, float]:
    worst, exact = 0, 0
    for v in vectors:
        prog = host.ldv(v) + b"".join(host.gvns(layer) for layer in (0, 1, 2))
        prog += b"".join(host.gvns(layer, True) for layer in (2, 1, 0)) + host.stv()
        out = host.unpack16(S1Core().run(prog))
        e = max(abs(a - b) for a, b in zip(out, v))
        worst = max(worst, e)
        exact += e == 0
    return worst, exact / len(vectors)


def rel_rmse(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - x) ** 2)) / np.sqrt(np.mean(x.astype(float) ** 2)))


def s1_fixed(x: np.ndarray, refine: bool = True) -> np.ndarray:
    out = []
    for v in x:
        pkt = S1Core().run(host.enc(list(v)))
        c = S1Core(ctrl=1 if refine else 0)
        out.append(host.unpack16(c.run(host.dec(pkt))))
    return np.array(out, dtype=float)


def nqx_core_reference(x: np.ndarray) -> np.ndarray | None:
    if not NQX_CORE.exists():
        return None
    sys.path.insert(0, str(NQX_CORE))
    from nqx.constants import NQXConfig
    from nqx.cpu import NQXCore

    core = NQXCore(NQXConfig(dim=P.dim))
    enc = core.encode(x.astype(np.float32))
    dq, _ = core.qu.dequantize(enc.quantized_indices, enc.mins, enc.maxs, 3)
    y, _ = core.pu.from_polar(dq)
    for layer in (2, 1, 0):
        y, _ = core.gu.apply_layer(y, layer, inverse=True)
    return y


def s1_float_with_rotation(x: np.ndarray, q: np.ndarray | None) -> np.ndarray:
    """S1 quantization rules (float) around an arbitrary orthogonal rotation q."""
    xr = x.astype(float) if q is None else x.astype(float) @ q.T
    r, t = np.hypot(xr[:, 0::2], xr[:, 1::2]), np.arctan2(xr[:, 1::2], xr[:, 0::2])
    rmin = r.min(axis=1, keepdims=True)
    rng = np.maximum(r.max(axis=1, keepdims=True) - rmin, 1e-12)
    u = (r - rmin) / rng * 7
    qr = np.clip(np.floor(u + 0.5), 0, 7)
    rh = rmin + (qr + np.where(u >= qr, 0.25, -0.25)) * rng / 7
    tt = t / (2 * math.pi) * 8
    qt = np.floor(tt + 0.5)
    th = (qt + np.where(tt >= qt, 0.25, -0.25)) / 8 * 2 * math.pi
    y = np.empty_like(xr)
    y[:, 0::2], y[:, 1::2] = rh * np.cos(th), rh * np.sin(th)
    return y if q is None else y @ q


def quality(n: int) -> list[tuple[str, dict[str, float | None]]]:
    rng = random.Random(21)
    sets = {
        "KV-like (outlier channels)": np.array([kv_vector(rng, P.dim) for _ in range(n)]),
        "Gaussian, full int16 scale": np.array([full_scale_vector(rng, P.dim) for _ in range(n)]),
    }
    q_random, _ = np.linalg.qr(np.random.default_rng(5).standard_normal((P.dim, P.dim)))
    q_golden = ref.rotate(P, np.eye(P.dim)).T
    rows = []
    for name, x in sets.items():
        core = nqx_core_reference(x)
        rows.append(
            (
                name,
                {
                    "nqx-core reference (per-feature batch min/max, sign bit unused)": (
                        rel_rmse(core, x) if core is not None else None
                    ),
                    "S1 rules, float, no rotation": rel_rmse(s1_float_with_rotation(x, None), x),
                    "S1 rules, float, random orthogonal rotation": rel_rmse(
                        s1_float_with_rotation(x, q_random), x
                    ),
                    "S1 rules, float, golden-angle rotation": rel_rmse(
                        s1_float_with_rotation(x, q_golden), x
                    ),
                    "S1 rules, float, golden-angle, no residual refinement": rel_rmse(
                        ref.encode(P, x.astype(float), refine=False), x
                    ),
                    "NQX-S1 silicon (bit-accurate model)": rel_rmse(s1_fixed(x), x),
                },
            )
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "spec" / "04_numerics_results.md")
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()

    rng = random.Random(31)
    c = cordic_accuracy()
    rt_kv = roundtrip([kv_vector(rng, P.dim) for _ in range(200)])
    rt_fs = roundtrip([full_scale_vector(rng, P.dim) for _ in range(200)])
    q = quality(args.n)

    lines = [
        "# NQX-S1 numerics results",
        "",
        "Generated by `tools/bench_numerics.py` from the bit-accurate model "
        f"(DIM={P.dim}, DW={P.dw}, ZW={P.zw}, NITER={P.n_iter}). Do not edit by hand.",
        "",
        "## CORDIC accuracy against float64",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Rotation, relative error, RMS | {c['rot_rel_rms']:.2e} |",
        f"| Rotation, relative error, max | {c['rot_rel_max']:.2e} |",
        f"| Vectoring, radius relative error, max | {c['vec_r_rel_max']:.2e} |",
        f"| Vectoring, angle error, max | {c['vec_t_rad_max']:.2e} rad |",
        "",
        "## Rotation round trip (3 layers forward + 3 inverse), int16 output",
        "",
        "| Data (200 vectors each) | max abs error [LSB] | vectors returned bit-exact |",
        "|---|---|---|",
        f"| KV-like | {rt_kv[0]} | {rt_kv[1] * 100:.1f} % |",
        f"| Uniform full-scale int16 | {rt_fs[0]} | {rt_fs[1] * 100:.1f} % |",
        "",
        f"## Encode/decode reconstruction error ({args.n} vectors per set)",
        "",
        "Relative RMSE = RMS(decoded - input) / RMS(input). Lower is better. "
        "4 bits per value in every row except the nqx-core reference, whose "
        "stored sign bit is not used by its decoder.",
        "",
    ]
    methods = list(q[0][1].keys())
    lines.append("| Method | " + " | ".join(name for name, _ in q) + " |")
    lines.append("|---|" + "---|" * len(q))
    for m in methods:
        cells = []
        for _, res in q:
            v = res[m]
            cells.append("n/a" if v is None else f"{v:.4f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
