"""Architecture parameters of NQX-S1 and every constant derived from them.

This module is the single source of truth for numeric constants. The RTL
include file ``rtl/nqx_s1_params.vh`` is generated from it by
``tools/gen_params.py``; never edit the generated file by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

PHI = (1.0 + math.sqrt(5.0)) / 2.0

PHASE_W = 32
TURN = 1 << PHASE_W

# Phase increment of layer L, in units of 2^-32 turn. The reference angle of
# pair k in layer L is (k + 1) * golden_angle * phi^L with golden_angle =
# 2*pi/phi^2, i.e. (k + 1) * phi^(L-2) turns. Only the fractional part matters.
#   L1: 1/phi^2 turn -> 0x61C88647 (= 2^32 - 0x9E3779B9)
#   L2: 1/phi   turn -> 0x9E3779B9 (Knuth's Fibonacci-hashing constant)
#   L3: 1       turn -> 0          (layer 3 of the reference is an identity)
INC_DEFAULT = (
    round(TURN / PHI**2) % TURN,
    round(TURN / PHI) % TURN,
    round(TURN * 1.0) % TURN,
)

CHIP_ID = 0x4E515831  # "NQX1"
VERSION = 0x00010000  # 1.0.0


def _ceil_log2(n: int) -> int:
    return max(0, (n - 1).bit_length())


@dataclass(frozen=True)
class S1Params:
    dim: int = 128
    in_w: int = 16
    fw: int = 4
    gb: int = 2
    n_iter: int = 18
    zw: int = 20
    kf: int = 18
    step_f: int = 8

    def __post_init__(self) -> None:
        if self.dim < 8 or self.dim & (self.dim - 1):
            raise ValueError("dim must be a power of two >= 8")
        if self.fw < 1 or self.gb < 1 or self.step_f < 1:
            raise ValueError("fw, gb and step_f must be at least 1")
        if self.zw > self.dw:
            raise ValueError("zw must not exceed the register width")
        if self.n_iter >= self.cw:
            raise ValueError("n_iter must be smaller than the CORDIC width")

    @property
    def log2_dim(self) -> int:
        return self.dim.bit_length() - 1

    @property
    def iw(self) -> int:
        return self.in_w + (_ceil_log2(self.dim) + 1) // 2

    @property
    def dw(self) -> int:
        return self.iw + self.fw

    @property
    def cw(self) -> int:
        return self.dw + self.gb + 2

    @property
    def hb(self) -> int:
        return (self.dw + 7) // 8

    @property
    def n_pairs(self) -> int:
        return self.dim // 2

    @property
    def l3_stride(self) -> int:
        return max(2, self.dim // 4)

    @property
    def packet_bytes(self) -> int:
        return 2 * self.hb + self.n_pairs

    @cached_property
    def atan_table(self) -> tuple[int, ...]:
        return tuple(
            round(math.atan(2.0**-i) / (2.0 * math.pi) * (1 << self.zw)) for i in range(self.n_iter)
        )

    @cached_property
    def cordic_gain(self) -> float:
        k = 1.0
        for i in range(self.n_iter):
            k *= math.sqrt(1.0 + 2.0 ** (-2 * i))
        return k

    @cached_property
    def kinv(self) -> int:
        return round((1 << self.kf) / self.cordic_gain)

    @property
    def cordic_latency(self) -> int:
        return self.n_iter + 2

    def params_word(self) -> int:
        return self.log2_dim | (self.dw << 8) | (self.n_iter << 16) | (self.zw << 24)

    def layer_pairs(self, layer: int) -> list[tuple[int, int, int]]:
        """(i, j, k) triples in issue order; the phase of the pair is (k+1)*INC."""
        n = self.dim
        if layer == 0:
            return [(2 * k, 2 * k + 1, k) for k in range(n // 2)]
        if layer == 1:
            return [(2 * k + 1, 2 * k + 2, k) for k in range((n - 1) // 2)]
        if layer == 2:
            s = self.l3_stride
            return [(k, k + s, k) for k in range(n) if not (k & s)]
        raise ValueError(f"layer must be 0, 1 or 2, got {layer}")


DEFAULT = S1Params()
