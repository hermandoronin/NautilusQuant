"""Bit-accurate model of the NQX-S1 pipelined CORDIC (rtl/nqx_s1_cordic.sv).

All values are Python ints holding two's-complement quantities. ``>>`` on a
negative int floors, which is what an arithmetic right shift does in RTL.
"""

from __future__ import annotations

from nqx_s1.params import S1Params


def wrap(v: int, bits: int) -> int:
    """Interpret the low ``bits`` bits of ``v`` as a signed number."""
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def saturate(v: int, bits: int) -> tuple[int, bool]:
    hi = (1 << (bits - 1)) - 1
    lo = -(1 << (bits - 1))
    if v > hi:
        return hi, True
    if v < lo:
        return lo, True
    return v, False


def _iterate(p: S1Params, x: int, y: int, z: int, vectoring: bool) -> tuple[int, int, int]:
    for i, a in enumerate(p.atan_table):
        d = (y < 0) if vectoring else (z >= 0)
        if d:
            x, y, z = x - (y >> i), y + (x >> i), z - a
        else:
            x, y, z = x + (y >> i), y - (x >> i), z + a
        x = wrap(x, p.cw)
        y = wrap(y, p.cw)
        z = wrap(z, p.zw)
    return x, y, z


def _compensate(p: S1Params, v: int) -> int:
    shift = p.kf + p.gb
    return (v * p.kinv + (1 << (shift - 1))) >> shift


def rotate(p: S1Params, a: int, b: int, z: int) -> tuple[int, int, bool]:
    """Rotate (a, b) by angle z (ZW-bit binary angle, full turn = 2^ZW).

    Returns the two outputs saturated to DW bits and a saturation flag.
    """
    x = a << p.gb
    y = b << p.gb
    z = wrap(z, p.zw)
    top = (z >> (p.zw - 2)) & 0b11
    if top in (0b01, 0b10):
        x, y = -x, -y
        z = wrap(z ^ (1 << (p.zw - 1)), p.zw)
    x, y, _ = _iterate(p, x, y, z, vectoring=False)
    xo, sx = saturate(_compensate(p, x), p.dw)
    yo, sy = saturate(_compensate(p, y), p.dw)
    return xo, yo, sx or sy


def vector(p: S1Params, a: int, b: int) -> tuple[int, int, bool]:
    """Cartesian (a, b) -> (r, theta). theta is a signed ZW-bit binary angle."""
    x = a << p.gb
    y = b << p.gb
    if x < 0:
        x, y, z = -x, -y, -(1 << (p.zw - 1))
    else:
        z = 0
    x, _, z = _iterate(p, x, y, z, vectoring=True)
    r, sat = saturate(_compensate(p, x), p.dw)
    return r, z, sat
