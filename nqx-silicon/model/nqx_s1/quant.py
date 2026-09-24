"""Bit-accurate model of the NQX-S1 quantizer / dequantizer (rtl/nqx_s1_quant.sv).

One packed byte per polar pair k:
    bits [2:0] q_r   3-bit radius code
    bit  [3]   s_r   residual sign of the radius
    bits [6:4] q_t   3-bit angle code
    bit  [7]   s_t   residual sign of the angle
This is the 3+1-bit little-endian layout of nqx-core's PackUnit.
"""

from __future__ import annotations

from nqx_s1.cordic import wrap
from nqx_s1.params import S1Params


def radius_code(r: int, rmin: int, rmax: int) -> tuple[int, int]:
    """3-bit code and residual sign of radius r within [rmin, rmax]."""
    rng = rmax - rmin
    d = r - rmin
    q = sum(1 for m in range(1, 8) if 14 * d > (2 * m - 1) * rng)
    s = 1 if 7 * d >= q * rng else 0
    return q, s


def angle_code(p: S1Params, theta: int) -> tuple[int, int]:
    """3-bit circular code and residual sign of a ZW-bit binary angle."""
    t = theta & ((1 << p.zw) - 1)
    q = ((t + (1 << (p.zw - 4))) >> (p.zw - 3)) & 7
    e = wrap(t - (q << (p.zw - 3)), p.zw)
    return q, 1 if e >= 0 else 0


def encode_pair(p: S1Params, r: int, theta: int, rmin: int, rmax: int) -> int:
    qr, sr = radius_code(r, rmin, rmax)
    qt, st = angle_code(p, theta)
    return qr | (sr << 3) | (qt << 4) | (st << 7)


def radius_step(p: S1Params, rmin: int, rmax: int) -> int:
    """floor((rmax - rmin) * 2^STEP_F / 28): 1/4 of a quantization step."""
    rng = max(0, rmax - rmin)
    return (rng << p.step_f) // 28


def decode_pair(
    p: S1Params, byte: int, rmin: int, step: int, refine: bool
) -> tuple[int, int, bool]:
    """Byte -> (r_hat, theta_hat, saturated) using a precomputed radius_step."""
    qr, sr = byte & 7, (byte >> 3) & 1
    qt, st = (byte >> 4) & 7, (byte >> 7) & 1
    num = 4 * qr + ((2 * sr - 1) if refine else 0)
    r = rmin + ((num * step + (1 << (p.step_f - 1))) >> p.step_f)
    rmax_repr = (1 << (p.dw - 1)) - 1
    sat = r > rmax_repr
    r = min(r, rmax_repr)
    t = qt << (p.zw - 3)
    if refine:
        t += (1 << (p.zw - 5)) if st else -(1 << (p.zw - 5))
    return r, wrap(t, p.zw), sat
