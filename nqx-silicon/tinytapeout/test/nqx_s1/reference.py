"""Floating-point reference of the NautilusQuant pipeline, as NQX-S1 executes it.

``rotate``/``unrotate`` follow nqx-core's GoldenAngleLUT exactly (float64).
``encode``/``decode`` apply the S1 quantization rules (per-vector radius range,
circular angle code, residual sign refinement) in floating point, so the
difference between this module and the bit-accurate model is fixed-point
error only.
"""

from __future__ import annotations

import math

import numpy as np

from nqx_s1.params import PHI, S1Params

GOLDEN_ANGLE = 2.0 * math.pi / PHI**2


def layer_angles(p: S1Params, layer: int) -> list[tuple[int, int, float]]:
    scale = PHI**layer
    return [(i, j, GOLDEN_ANGLE * (k + 1) * scale) for i, j, k in p.layer_pairs(layer)]


def _apply(p: S1Params, x: np.ndarray, layer: int, inverse: bool) -> np.ndarray:
    out = x.astype(np.float64).copy()
    for i, j, a in layer_angles(p, layer):
        c, s = math.cos(a), math.sin(a)
        if inverse:
            s = -s
        xi, xj = out[..., i].copy(), out[..., j].copy()
        out[..., i] = xi * c - xj * s
        out[..., j] = xi * s + xj * c
    return out


def rotate(p: S1Params, x: np.ndarray) -> np.ndarray:
    for layer in (0, 1, 2):
        x = _apply(p, x, layer, inverse=False)
    return x


def unrotate(p: S1Params, x: np.ndarray) -> np.ndarray:
    for layer in (2, 1, 0):
        x = _apply(p, x, layer, inverse=True)
    return x


def to_polar(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = x[..., 0::2], x[..., 1::2]
    return np.hypot(a, b), np.arctan2(b, a)


def encode(p: S1Params, x: np.ndarray, refine: bool = True) -> np.ndarray:
    """Float encode -> decode round trip under S1 quantization rules."""
    r, t = to_polar(rotate(p, x))
    rmin = r.min(axis=-1, keepdims=True)
    rng = r.max(axis=-1, keepdims=True) - rmin
    safe = np.where(rng > 0, rng, 1.0)
    u = (r - rmin) / safe * 7.0
    qr = np.clip(np.floor(u + 0.5), 0, 7)
    sr = u >= qr
    rhat = rmin + (qr + (np.where(sr, 0.25, -0.25) if refine else 0.0)) * safe / 7.0
    rhat = np.where(rng > 0, rhat, rmin)
    turns = t / (2 * math.pi)
    qt = np.floor(turns * 8 + 0.5)
    et = turns * 8 - qt
    that = (qt + (np.where(et >= 0, 0.25, -0.25) if refine else 0.0)) / 8 * 2 * math.pi
    y = np.empty_like(x, dtype=np.float64)
    y[..., 0::2] = rhat * np.cos(that)
    y[..., 1::2] = rhat * np.sin(that)
    return unrotate(p, y)
