#!/usr/bin/env python3
"""Reverse study: where does the NautilusQuant / NQX-S1 design win, and why?

    python research/reverse_study.py [--out research/reverse_study_results.md]

Instead of asking "is the golden angle better on average" (it is not; see
spec/11_algorithm_findings.md), this study starts from what open KV-cache
compressors and accelerators do and looks for the operating points where
the properties of NQX-S1 (no rotation state, multiplier-free CORDIC, pairs
of coordinates in polar form, integer determinism, any even dimension)
decide the outcome. Every scenario is a measurement on an emulator; nothing
is assumed to win.

KV emulator (no real model weights are reachable from the build machine, so
the statistics follow published observations and are stated here):

- RoPE with NeoX pairing (i, i + d/2), frequencies base^(-2i/d).
- Pre-RoPE keys: per-channel means that are zero except on a few
  low-frequency RoPE pairs, where one of the two dimensions carries a large
  fixed value (KIVI: fixed outlier channels in K; "Massive values", ICML
  2025: massive values in Q and K sit in low-frequency RoPE dimensions;
  PolarQuant: the outlier usually sits in one of the two paired dimensions).
  On top: per-channel Gaussian noise with log-normal scales and a
  log-normal per-token scale.
- Queries: the same low-frequency pairs carry large values (same sources).
- Values: no channel outliers, per-token scale (KIVI).

Metrics: relative RMSE of K and V, and the relative error of the attention
output softmax(q.k / sqrt(d)) V for the last queries of each sequence, which
is what the model actually consumes. Bits per value include every scale and
header.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model")]

PHI = (1 + 5 ** 0.5) / 2
GOLDEN = 2 * math.pi / PHI**2


# ---------------------------------------------------------------- rotations
def givens_matrix(d: int, layers: list[list[tuple[int, int, float]]]) -> np.ndarray:
    """Orthogonal matrix Q (rows = transform) for a list of Givens layers."""
    q = np.eye(d)
    for layer in layers:
        g = np.eye(d)
        for i, j, a in layer:
            c, s = math.cos(a), math.sin(a)
            g[i, i], g[i, j], g[j, i], g[j, j] = c, -s, s, c
        q = g @ q
    return q


def golden_nqx(d: int) -> np.ndarray:
    """NautilusQuant layers L1 (2k,2k+1), L2 (2k+1,2k+2), L3 (k,k+d/4)."""
    l1 = [(2 * k, 2 * k + 1) for k in range(d // 2)]
    l2 = [(2 * k + 1, 2 * k + 2) for k in range(d // 2 - 1)]
    q4 = d // 4
    l3 = [(k, k + q4) for k in list(range(q4)) + list(range(2 * q4, 3 * q4))]
    return givens_matrix(
        d, [[(i, j, GOLDEN * (k + 1) * PHI**L) for k, (i, j) in enumerate(p)]
            for L, p in enumerate((l1, l2, l3))]
    )


def butterfly_pairs(d: int) -> list[list[tuple[int, int]]]:
    layers, s = [], 1
    while s < d:
        layers.append([(i, i + s) for i in range(d) if not (i & s) and i + s < d])
        s <<= 1
    return layers


def golden_butterfly(d: int, incs: list[float] | None = None) -> np.ndarray:
    """Butterfly topology; pair k of layer L turns by (k+1)*inc_L mod 2*pi."""
    layers = butterfly_pairs(d)
    incs = incs if incs is not None else [GOLDEN * PHI**L for L in range(len(layers))]
    return givens_matrix(d, [[(i, j, (k + 1) * incs[L]) for k, (i, j) in enumerate(p)]
                             for L, p in enumerate(layers)])


def hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def randomized_hadamard(d: int, seed: int = 3) -> np.ndarray:
    """Sylvester Hadamard with random signs; block-diagonal if d is not 2^k."""
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], d)
    q, pos = np.zeros((d, d)), 0
    for b in [1 << k for k in range(12, -1, -1)]:
        while d - pos >= b:
            q[pos:pos + b, pos:pos + b] = hadamard(b)
            pos += b
    return q * signs


def random_orthogonal(d: int, seed: int = 5) -> np.ndarray:
    return np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))[0]


# ------------------------------------------------------------- quantizers
@dataclass
class Result:
    xhat: np.ndarray
    bits_per_value: float


def s1_polar(x: np.ndarray, q: np.ndarray | None, pairs: str = "interleaved") -> Result:
    """NQX-S1 rules: per-vector radius range (2 x 16-bit header), 3-bit radius
    code, 3-bit circular angle code, one residual bit per value."""
    d = x.shape[1]
    xr = x if q is None else x @ q.T
    a, b = (xr[:, 0::2], xr[:, 1::2]) if pairs == "interleaved" else (xr[:, : d // 2], xr[:, d // 2:])
    r, t = np.hypot(a, b), np.arctan2(b, a)
    rmin = r.min(axis=1, keepdims=True)
    rng = np.maximum(r.max(axis=1, keepdims=True) - rmin, 1e-12)
    u = (r - rmin) / rng * 7
    qr = np.clip(np.floor(u + 0.5), 0, 7)
    rh = rmin + (qr + np.where(u >= qr, 0.25, -0.25)) * rng / 7
    tt = t / (2 * math.pi) * 8
    qt = np.floor(tt + 0.5)
    th = (qt + np.where(tt >= qt, 0.25, -0.25)) / 8 * 2 * math.pi
    y = np.empty_like(xr)
    if pairs == "interleaved":
        y[:, 0::2], y[:, 1::2] = rh * np.cos(th), rh * np.sin(th)
    else:
        y[:, : d // 2], y[:, d // 2:] = rh * np.cos(th), rh * np.sin(th)
    return Result(y if q is None else y @ q, 4 + 32 / d)


def uniform_asym(x: np.ndarray, axis: int, bits: int = 4) -> np.ndarray:
    lo, hi = x.min(axis=axis, keepdims=True), x.max(axis=axis, keepdims=True)
    step = np.maximum(hi - lo, 1e-12) / (2**bits - 1)
    return lo + np.clip(np.round((x - lo) / step), 0, 2**bits - 1) * step


def int4_per_token(x: np.ndarray) -> Result:
    return Result(uniform_asym(x, axis=1), 4 + 32 / x.shape[1])


def kivi4_keys(x: np.ndarray, group: int = 128) -> Result:
    """KIVI: keys per channel over a group of tokens (fp16 min and scale)."""
    y = np.empty_like(x)
    for s in range(0, x.shape[0], group):
        y[s:s + group] = uniform_asym(x[s:s + group], axis=0)
    return Result(y, 4 + 32 / group)


def q4_0_blocks(x: np.ndarray, block: int = 32) -> Result:
    """llama.cpp q4_0: symmetric 4-bit per 32 values, one fp16 scale."""
    t, d = x.shape
    xb = x.reshape(t, d // block, block)
    amax = np.abs(xb).max(axis=2, keepdims=True)
    scale = np.maximum(amax, 1e-12) / 8
    q = np.clip(np.round(xb / scale), -8, 7)
    return Result((q * scale).reshape(t, d), 4 + 16 / block)


def rope_polar_keys(k_pre: np.ndarray, group: int = 128) -> Result:
    """Pre-RoPE keys in polar form per RoPE pair (i, i+d/2); per pair and
    per group of tokens: radius range and angle arc (4 x 16 bit), 3-bit codes
    and one residual bit each. RoPE is applied afterwards as an angle
    addition, which is exact on the dequantized angle."""
    t, d = k_pre.shape
    h = d // 2
    a, b = k_pre[:, :h], k_pre[:, h:]
    r, ang = np.hypot(a, b), np.arctan2(b, a)
    rh, th = np.empty_like(r), np.empty_like(ang)
    for s in range(0, t, group):
        sl = slice(s, s + group)
        rr, aa = r[sl], ang[sl]
        rmin = rr.min(axis=0, keepdims=True)
        rng = np.maximum(rr.max(axis=0, keepdims=True) - rmin, 1e-12)
        u = (rr - rmin) / rng * 7
        qr = np.clip(np.floor(u + 0.5), 0, 7)
        rh[sl] = rmin + (qr + np.where(u >= qr, 0.25, -0.25)) * rng / 7
        # smallest arc containing all angles of the pair in this group
        c = np.angle(np.exp(1j * aa).mean(axis=0, keepdims=True))
        dev = np.angle(np.exp(1j * (aa - c)))
        w = np.maximum(np.abs(dev).max(axis=0, keepdims=True), 1e-9)
        v = (dev + w) / (2 * w) * 7
        qa = np.clip(np.floor(v + 0.5), 0, 7)
        th[sl] = c - w + (qa + np.where(v >= qa, 0.25, -0.25)) * 2 * w / 7
    y = np.concatenate([rh * np.cos(th), rh * np.sin(th)], axis=1)
    return Result(y, 4 + 64 / (2 * group))


def calibrate_channels(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x.min(axis=0), x.max(axis=0)


def static_per_channel(x: np.ndarray, cal: tuple[np.ndarray, np.ndarray]) -> Result:
    """Per-channel 4-bit with ranges calibrated offline (no header, streaming)."""
    lo, hi = cal
    step = np.maximum(hi - lo, 1e-12) / 15
    return Result(lo + np.clip(np.round((x - lo) / step), 0, 15) * step, 4.0)


def calibrate_rope_polar(k_pre: np.ndarray) -> dict:
    h = k_pre.shape[1] // 2
    r = np.hypot(k_pre[:, :h], k_pre[:, h:])
    ang = np.arctan2(k_pre[:, h:], k_pre[:, :h])
    c = np.angle(np.exp(1j * ang).mean(axis=0))
    dev = np.angle(np.exp(1j * (ang - c)))
    return {"rmin": r.min(axis=0), "rmax": r.max(axis=0), "c": c,
            "w": np.maximum(np.abs(dev).max(axis=0), 1e-9)}


def rope_polar_static(k_pre: np.ndarray, cal: dict) -> Result:
    """NQX-RN with per-pair constants from calibration: streaming, no header.
    Radius 3 bit + residual over [rmin, rmax]; angle 3 bit + residual over
    the calibrated arc c +- w (values outside are clipped)."""
    h = k_pre.shape[1] // 2
    r = np.hypot(k_pre[:, :h], k_pre[:, h:])
    ang = np.arctan2(k_pre[:, h:], k_pre[:, :h])
    rng_ = np.maximum(cal["rmax"] - cal["rmin"], 1e-12)
    u = np.clip((r - cal["rmin"]) / rng_ * 7, 0, 7)
    qr = np.clip(np.floor(u + 0.5), 0, 7)
    rh = cal["rmin"] + (qr + np.where(u >= qr, 0.25, -0.25)) * rng_ / 7
    dev = np.clip(np.angle(np.exp(1j * (ang - cal["c"]))), -cal["w"], cal["w"])
    v = (dev + cal["w"]) / (2 * cal["w"]) * 7
    qa = np.clip(np.floor(v + 0.5), 0, 7)
    th = cal["c"] - cal["w"] + (qa + np.where(v >= qa, 0.25, -0.25)) * 2 * cal["w"] / 7
    return Result(np.concatenate([rh * np.cos(th), rh * np.sin(th)], axis=1), 4.0)


# ------------------------------------------------------------ KV emulator
def rope(x: np.ndarray, pos: np.ndarray, base: float) -> np.ndarray:
    d = x.shape[1]
    h = d // 2
    th = pos[:, None] * base ** (-np.arange(h) * 2 / d)[None, :]
    c, s = np.cos(th), np.sin(th)
    a, b = x[:, :h], x[:, h:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=1)


@dataclass
class Head:
    k_pre: np.ndarray
    q_pre: np.ndarray
    v: np.ndarray
    pos: np.ndarray
    base: float


def make_head(rng: np.random.Generator, d: int, t: int, base: float, n_out: int = 4,
              out_mag: float = 12.0, structure_seed: int | None = None) -> Head:
    srng = np.random.default_rng(structure_seed) if structure_seed is not None else rng
    h = d // 2
    mu_k, mu_q = np.zeros(d), np.zeros(d)
    low = np.arange(h - max(8, n_out * 2), h)                  # lowest-frequency pairs
    for i in srng.choice(low, n_out, replace=False):
        dim = i if srng.random() < 0.5 else i + h              # one of the two paired dims
        mu_k[dim] = srng.choice([-1, 1]) * out_mag * srng.uniform(0.6, 1.4)
        mu_q[dim] = srng.choice([-1, 1]) * out_mag * srng.uniform(0.3, 0.8)
    sig_k = np.exp(srng.normal(0, 0.35, d))
    sig_q = np.exp(srng.normal(0, 0.35, d))
    sig_v = np.exp(srng.normal(0, 0.25, d))
    tok = np.exp(rng.normal(0, 0.25, (t, 1)))
    k_pre = tok * (mu_k + sig_k * rng.standard_normal((t, d)))
    q_pre = np.exp(rng.normal(0, 0.25, (t, 1))) * (mu_q + sig_q * rng.standard_normal((t, d)))
    v = np.exp(rng.normal(0, 0.35, (t, 1))) * sig_v * rng.standard_normal((t, d))
    return Head(k_pre, q_pre, v, np.arange(t, dtype=float), base)


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, n_q: int) -> np.ndarray:
    t, d = k.shape
    out = []
    for p in range(t - n_q, t):
        s = k[: p + 1] @ q[p] / math.sqrt(d)
        w = np.exp(s - s.max())
        out.append((w / w.sum()) @ v[: p + 1])
    return np.array(out)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ----------------------------------------------------------------- methods
def key_methods(d: int, tuned_incs: list[float] | None) -> dict:
    """name -> (kind, function). kind: 'post' works on post-RoPE keys,
    'pre' on pre-RoPE keys (RoPE applied after dequantization, as an angle
    addition in polar form), '*-static' also receives calibration data."""
    qg, qb = golden_nqx(d), golden_butterfly(d)
    qr, qh = random_orthogonal(d), randomized_hadamard(d)
    m = {
        "INT4 per token (asym)":                   ("post", int4_per_token, True),
        "llama.cpp q4_0 (blocks of 32)":           ("post", q4_0_blocks, True),
        "KIVI-4 (per channel, groups of 128 tok)": ("post", kivi4_keys, False),
        "KIVI, calibrated ranges":                 ("post-static", static_per_channel, True),
        "KVQuant-like: pre-RoPE, calibrated":      ("pre-static", static_per_channel, True),
        "S1 polar, no rotation":                   ("post", lambda x: s1_polar(x, None), True),
        "NQX-S1: golden L1+L2+L3 + S1":            ("post", lambda x: s1_polar(x, qg), True),
        "Golden butterfly + S1":                   ("post", lambda x: s1_polar(x, qb), True),
        "TurboQuant-like: random + S1":            ("post", lambda x: s1_polar(x, qr), True),
        "QuaRot-like: rand. Hadamard + S1":        ("post", lambda x: s1_polar(x, qh), True),
        "PolarQuant-like: S1 on RoPE pairs":       ("post", lambda x: s1_polar(x, None, "neox"), True),
        "NQX-RN: pre-RoPE polar, groups of 128":   ("pre", rope_polar_keys, False),
        "NQX-RN static: pre-RoPE polar, calibrated": ("pre-static", rope_polar_static, True),
    }
    if tuned_incs is not None:
        qt = golden_butterfly(d, tuned_incs)
        m["Tuned golden butterfly + S1 (7 registers)"] = ("post", lambda x: s1_polar(x, qt), True)
    return m


def evaluate_keys(pairs: list[tuple[Head, Head]], d: int, n_q: int,
                  tuned_incs: list[float] | None = None) -> dict[str, dict]:
    """Keys quantized by each method; values by one common quantizer (INT4
    per token) so that the attention error isolates the key method."""
    res = {}
    for name, (kind, fk, streaming) in key_methods(d, tuned_incs).items():
        ek, eo, bits = [], [], []
        for cal, h in pairs:
            k = rope(h.k_pre, h.pos, h.base)
            q = rope(h.q_pre, h.pos, h.base)
            if kind == "post":
                rk = fk(k)
                khat = rk.xhat
            elif kind == "post-static":
                rk = fk(k, calibrate_channels(rope(cal.k_pre, cal.pos, cal.base)))
                khat = rk.xhat
            elif kind == "pre":
                rk = fk(h.k_pre)
                khat = rope(rk.xhat, h.pos, h.base)
            else:   # pre-static
                c = (calibrate_rope_polar(cal.k_pre) if fk is rope_polar_static
                     else calibrate_channels(cal.k_pre))
                rk = fk(h.k_pre, c)
                khat = rope(rk.xhat, h.pos, h.base)
            vhat = int4_per_token(h.v).xhat
            ek.append(rel(khat, k))
            eo.append(rel(attention(q, khat, vhat, n_q), attention(q, k, h.v, n_q)))
            bits.append(rk.bits_per_value)
        res[name] = {"K": float(np.mean(ek)), "attn": float(np.mean(eo)),
                     "bits": float(np.mean(bits)), "streaming": streaming}
    return res


def value_methods(vs: list[np.ndarray]) -> dict[str, dict]:
    d = vs[0].shape[1]
    qg, qb, qr, qh = golden_nqx(d), golden_butterfly(d), random_orthogonal(d), randomized_hadamard(d)
    fs = {"INT4 per token (asym)": int4_per_token, "llama.cpp q4_0": q4_0_blocks,
          "S1 polar, no rotation": lambda x: s1_polar(x, None),
          "NQX-S1 golden + S1": lambda x: s1_polar(x, qg),
          "Golden butterfly + S1": lambda x: s1_polar(x, qb),
          "Random + S1": lambda x: s1_polar(x, qr),
          "Rand. Hadamard + S1": lambda x: s1_polar(x, qh)}
    return {n: {"V": float(np.mean([rel(f(v).xhat, v) for v in vs])),
                "bits": f(vs[0]).bits_per_value} for n, f in fs.items()}


def head_pairs(n: int, d: int, t: int, seed: int, base: float = 500000.0):
    """Calibration and test sequence per head: same head statistics, new tokens."""
    return [(make_head(np.random.default_rng(seed + 2 * i), d, t, base, structure_seed=seed * 7 + i),
             make_head(np.random.default_rng(seed + 2 * i + 1), d, t, base, structure_seed=seed * 7 + i))
            for i in range(n)]


# --------------------------------------------------------------- scenarios
def scenario_main(tuned_incs, d=128, t=1024, n_heads=16, n_q=64, seed=11):
    pairs = head_pairs(n_heads, d, t, seed)
    return evaluate_keys(pairs, d, n_q, tuned_incs), value_methods([h.v for _, h in pairs])


def scenario_dims(t=512, n_heads=8, n_q=32, seed=2):
    out = {}
    for d in (64, 80, 96, 128):
        rng = np.random.default_rng(seed + d)
        heads = [make_head(rng, d, t, 500000.0) for _ in range(n_heads)]
        sub = {}
        for name, q in (("no rotation", None), ("golden L1+L2+L3", golden_nqx(d)),
                        ("golden butterfly", golden_butterfly(d)),
                        ("random orthogonal (d^2 state)", random_orthogonal(d)),
                        ("Hadamard (block-diagonal if d != 2^k)", randomized_hadamard(d))):
            e = []
            for h in heads:
                k = rope(h.k_pre, h.pos, h.base)
                e.append(rel(s1_polar(k, q).xhat, k))
            sub[name] = float(np.mean(e))
        # zero-padding to the next power of two for a plain Hadamard
        p2 = 1 << (d - 1).bit_length()
        if p2 != d:
            qh = randomized_hadamard(p2)
            e = []
            for h in heads:
                k = rope(h.k_pre, h.pos, h.base)
                kp = np.concatenate([k, np.zeros((k.shape[0], p2 - d))], axis=1)
                e.append(rel(s1_polar(kp, qh).xhat[:, :d], k))
            sub[f"Hadamard, zero-padded to {p2} (+{100 * (p2 - d) / d:.0f} % bits)"] = float(np.mean(e))
        out[d] = sub
    return out


def scenario_determinism(d=128, n=4000, seed=4):
    """Random rotation in float32 with two summation orders: how many 4-bit
    codes differ? Integer CORDIC (NQX-S1) and integer Hadamard are exact."""
    rng = np.random.default_rng(seed)
    q = random_orthogonal(d).astype(np.float32)
    x = (rng.standard_normal((n, d)) * np.exp(rng.normal(0, 0.3, d))).astype(np.float32)
    y1 = x @ q.T
    y2 = np.zeros_like(y1)
    for j in reversed(range(d)):                      # other accumulation order
        y2 += np.outer(x[:, j], q[:, j]).astype(np.float32)

    def codes(y):
        r, t = np.hypot(y[:, 0::2], y[:, 1::2]), np.arctan2(y[:, 1::2], y[:, 0::2])
        rmin, rmax = r.min(1, keepdims=True), r.max(1, keepdims=True)
        u = (r - rmin) / np.maximum(rmax - rmin, 1e-12) * 7
        tt = t / (2 * np.pi) * 8
        return np.floor(u + 0.5), (np.floor(u + 0.5) <= u), np.floor(tt + 0.5) % 8, (np.floor(tt + 0.5) <= tt)
    def bf16(a):
        b = np.asarray(a, np.float32).view(np.uint32)
        return ((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000).view(np.float32)

    y3 = bf16(bf16(x) @ bf16(q).T)                   # bf16 inputs and output, fp32 accumulate

    def diff(ya, yb):
        ca, cb = codes(ya), codes(yb)
        dv = np.zeros(ca[0].shape, bool)
        for a, b in zip(ca, cb):
            dv |= a != b
        return dv
    d12, d13 = diff(y1, y2), diff(y1, y3)
    return {"fp32, two accumulation orders: max |difference|": float(np.abs(y1 - y2).max()),
            "fp32, two accumulation orders: pairs with a different code": float(d12.mean()),
            "fp32 vs bf16 platform: pairs with a different code": float(d13.mean()),
            "fp32 vs bf16 platform: vectors with at least one different code": float(d13.any(axis=1).mean())}


def scenario_tuning(d=128, t=768, n_heads=12, n_q=48, trials=400, seed=6):
    """Template chip: tune the butterfly phase increments (one 32-bit register
    per layer) on calibration sequences of a fixed 'task', report held-out
    sequences of the same heads."""
    struct = list(range(100, 100 + n_heads))
    cal = [make_head(np.random.default_rng(seed + i), d, t, 500000.0, structure_seed=s)
           for i, s in enumerate(struct)]
    test = [make_head(np.random.default_rng(seed + 1000 + i), d, t, 500000.0, structure_seed=s)
            for i, s in enumerate(struct)]
    n_layers = len(butterfly_pairs(d))

    def score(q, heads):
        e = []
        for h in heads:
            k = rope(h.k_pre, h.pos, h.base)
            e.append(rel(s1_polar(k, q).xhat, k))
        return float(np.mean(e))

    rng = np.random.default_rng(seed)
    best_inc = [GOLDEN * PHI**L for L in range(n_layers)]
    best = score(golden_butterfly(d, best_inc), cal)
    start = best
    for it in range(trials):                          # coordinate-wise random search
        cand = list(best_inc)
        L = it % n_layers
        cand[L] = (cand[L] + rng.normal(0, 0.6 if it < trials // 2 else 0.15)) % (2 * math.pi)
        s = score(golden_butterfly(d, cand), cal)
        if s < best:
            best, best_inc = s, cand
    out = {
        "golden butterfly (default increments), calibration": start,
        "tuned butterfly, calibration": best,
        "golden butterfly (default), held-out": score(golden_butterfly(d), test),
        "tuned butterfly, held-out": score(golden_butterfly(d, best_inc), test),
        "random orthogonal, held-out": score(random_orthogonal(d), test),
        "randomized Hadamard, held-out": score(randomized_hadamard(d), test),
        "NQX-S1 golden L1+L2+L3, held-out": score(golden_nqx(d), test),
        "tuned increments (hex, 32-bit phase)": [f"0x{int(round(c / (2 * math.pi) * 2**32)) % 2**32:08X}" for c in best_inc],
    }
    return out


def scenario_significance(tuned_incs, d=128, seeds=(11, 23, 37), n_heads=16, t=1024, n_q=64):
    """Paired per-head differences of the attention error, three independent
    head populations: is any mixing rotation really better than another?"""
    rots = {"golden butterfly": golden_butterfly(d), "tuned butterfly": golden_butterfly(d, tuned_incs),
            "randomized Hadamard": randomized_hadamard(d), "random orthogonal": random_orthogonal(d),
            "NQX-S1 golden L1+L2+L3": golden_nqx(d)}
    rows = []
    for seed in seeds:
        err = {n: [] for n in rots}
        for _, h in head_pairs(n_heads, d, t, seed):
            k, q = rope(h.k_pre, h.pos, h.base), rope(h.q_pre, h.pos, h.base)
            vh, ref = int4_per_token(h.v).xhat, None
            ref = attention(q, k, h.v, n_q)
            for n, m in rots.items():
                err[n].append(rel(attention(q, s1_polar(k, m).xhat, vh, n_q), ref))
        e = {n: np.array(v) for n, v in err.items()}
        for a, b in (("golden butterfly", "randomized Hadamard"), ("golden butterfly", "random orthogonal"),
                     ("tuned butterfly", "randomized Hadamard"), ("NQX-S1 golden L1+L2+L3", "randomized Hadamard")):
            dl = e[a] - e[b]
            rows.append((seed, f"{a} - {b}", float(dl.mean()), float(dl.std(ddof=1) / math.sqrt(len(dl))),
                         int((dl < 0).sum()), len(dl)))
    return rows


def hardware_table(d=128):
    lg = int(math.log2(d))
    pairs_nqx = d // 2 + d // 2 - 1          # L1 + L2 (L3 is skipped in silicon)
    return [
        ("NQX-S1 golden (CORDIC)", "12 B (3 x 32-bit)", "any even d", 0,
         f"{pairs_nqx * 18 * 3} add/sub (18 CORDIC steps x 3 per pair)", "yes (integer)"),
        ("Golden butterfly (CORDIC, programmable)", f"{4 * lg} B ({lg} x 32-bit increments)", "any d", 0,
         f"{lg * d // 2 * 18 * 3} add/sub", "yes (integer)"),
        ("Golden butterfly (angles fixed in the mask)", "0 (constants)", "any d", 0,
         f"about {lg * d // 2 * 12} add/sub (shift-add cos/sin, ~12 per pair)", "yes (integer)"),
        ("Randomized Hadamard (FWHT)", f"{d // 8} B sign vector", "2^k (block-diagonal otherwise)", 0,
         f"{d * lg} add/sub", "yes if done in integers"),
        ("Random orthogonal (TurboQuant)", f"{d * d * 2 // 1024} KB (fp16)", "any d", d * d,
         f"{d * d} MAC", "no (float reduction order)"),
        ("Learned (SpinQuant)", f"{d * d * 2 // 1024} KB per layer (fp16)", "any d", d * d,
         f"{d * d} MAC", "no (float)"),
        ("RoPE as angle addition (NQX-RN)", "phase accumulator per pair", "any even d", 0,
         f"{d // 2} add (vs {2 * d} mul + {d} add for RoPE on dequantized keys)", "yes"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "reverse_study_results.md")
    args = ap.parse_args()
    t0 = time.time()

    # the generic golden layers equal the silicon reference (power-of-two d)
    from nqx_s1 import S1Params, reference as ref
    for d in (64, 128):
        assert np.allclose(golden_nqx(d), ref.rotate(S1Params(dim=d), np.eye(d)).T, atol=1e-12)

    tun = scenario_tuning()
    tuned = [int(h, 16) / 2**32 * 2 * math.pi for h in tun["tuned increments (hex, 32-bit phase)"]]
    keys, vals = scenario_main(tuned)
    dims = scenario_dims()
    det = scenario_determinism()
    sig = scenario_significance(tuned)
    hw = hardware_table()

    L = ["# Reverse study results (generated)", "",
         f"Generated by `research/reverse_study.py` in {time.time() - t0:.0f} s. Emulated KV data;"
         " see the script docstring for the statistics and their sources.", "",
         "## 1. Keys (d = 128, 16 heads, 1024 tokens, last 64 queries; values INT4 per token for all)", "",
         "Calibrated methods take their constants from a separate sequence of the same head.", "",
         "| Key method | bits/value | streaming (no token buffer) | K rel. RMSE | attention output rel. error |",
         "|---|---|---|---|---|"]
    for name, r in sorted(keys.items(), key=lambda kv: kv[1]["attn"]):
        L.append(f"| {name} | {r['bits']:.2f} | {'yes' if r['streaming'] else 'no'} | {r['K']:.4f} | {r['attn']:.4f} |")
    L += ["", "## 1b. Values (same heads)", "", "| Value method | bits/value | V rel. RMSE |", "|---|---|---|"]
    for name, r in sorted(vals.items(), key=lambda kv: kv[1]["V"]):
        L.append(f"| {name} | {r['bits']:.2f} | {r['V']:.4f} |")
    L += ["", "## 2. Head dimension (keys, S1 quantizer, relative RMSE)", ""]
    names = list(next(iter(dims.values())).keys())
    extra = sorted({n for sub in dims.values() for n in sub} - set(names))
    L.append("| d | " + " | ".join(names + ["Hadamard, zero-padded"]) + " |")
    L.append("|---" * (len(names) + 2) + "|")
    for d, sub in dims.items():
        pad = next((f"{v:.4f} ({k.split('(')[1].rstrip(')')})" for k, v in sub.items() if "zero-padded" in k), "n/a")
        L.append(f"| {d} | " + " | ".join(f"{sub[n]:.4f}" for n in names) + f" | {pad} |")
    L += ["", "## 3. Determinism of a float random rotation (4 000 vectors)", "",
          "| Quantity | Value |", "|---|---|"]
    for k, v in det.items():
        L.append(f"| {k} | {v:.3g} |")
    L.append("| NQX-S1 (integer CORDIC), integer Hadamard | 0 by construction |")
    L += ["", "## 4. Template chip: tuned butterfly increments (calibration vs held-out sequences)", "",
          "| Configuration | keys rel. RMSE |", "|---|---|"]
    for k, v in tun.items():
        L.append(f"| {k} | {v if isinstance(v, list) else f'{v:.4f}'} |")
    L += ["", "## 5. Is one mixing rotation better than another? (attention error, paired per head)", "",
          "| head population (seed) | difference | mean | standard error | first wins |", "|---|---|---|---|---|"]
    for seed, name, m, se, w, n in sig:
        L.append(f"| {seed} | {name} | {m:+.4f} | {se:.4f} | {w}/{n} |")
    L += ["", "## 6. Hardware cost of the rotation step (d = 128)", "",
          "| Rotation | state | dimensions | multipliers | operations per vector | bit-exact |",
          "|---|---|---|---|---|---|"]
    for row in hw:
        L.append("| " + " | ".join(str(c) for c in row) + " |")
    args.out.write_text("\n".join(L) + "\n")
    args.out.with_suffix(".json").write_text(json.dumps(
        {"keys": keys, "values": vals, "dims": {str(k): v for k, v in dims.items()}, "determinism": det,
         "tuning": tun, "significance": sig}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
