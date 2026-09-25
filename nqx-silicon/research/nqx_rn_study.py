#!/usr/bin/env python3
"""NQX-RN: pre-RoPE keys in polar form, RoPE as an integer phase addition.

    python research/nqx_rn_study.py [--quick]

Part II of the reverse study (reverse_study.py) found one operating point
where the structure of NQX (coordinate pairs, polar form, phase
accumulators, CORDIC) matches the data instead of approximating a random
rotation: store each key before RoPE, per RoPE pair (i, i + d/2), as a
radius and an angle. RoPE then adds the angle m * w_i, and a score can be
computed from the codes as r_q r_k cos(phi_q - theta_k + (n - m) w_i).

This script develops that point into a codec and tests it:

1. Codec. Per key token one 16-bit scale (the largest pair radius, the same
   header NQX-S1 writes). Per pair a radius code over a calibrated range
   and an angle code over a calibrated arc; the radius/angle bit widths of
   every pair come from a greedy allocation that weights each pair by the
   query energy in that pair. Pair energy is invariant under RoPE, so the
   weight is exact at every relative position (a per-channel weight is
   not). All constants come from one calibration sequence of the head.
2. Main comparison at 2, 3 and 4 bits per value (plus header) against
   rotation codecs (random and Hadamard rotation with a Lloyd-Max scalar
   quantizer, TurboQuant-MSE style; the S1 rows of part II), KIVI with a
   128-token buffer, and pre-RoPE Cartesian codecs with the same
   calibration (KVQuant-style per-channel ranges, uniform or with a
   Block-GTQ-style per-pair allocation). Three populations of 16 heads,
   paired differences.
3. Sensitivity: the emulator puts massive values in a few low-frequency
   RoPE pairs (KVQuant, "Massive values", PolarQuant report this for real
   models). Each assumption is broken in turn, plus calibration drift,
   RoPE base 10 000 and d = 64 / 96.
4. Position arithmetic: integer phase words (m * W_i mod 2^B) against fp32
   and bf16 RoPE at absolute positions up to 2^20.
5. Scoring in the angle domain: cos table size needed.
6. Operation count per key and pair.

The emulator (reverse_study.make_head) is not a language model; see the
docstring of reverse_study.py for the statistics and their sources. The
real-model check is still open.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reverse_study as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HDR = 16                                   # bits of the per-token scale


# ------------------------------------------------------------ primitives
def wrap(a: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * a))


def uq(x, lo, hi, bits):
    """Uniform mid-rise quantizer on [lo, hi]; bits may differ per column,
    bits == 0 reconstructs the midpoint."""
    n = 2.0 ** np.asarray(bits)
    step = np.maximum(hi - lo, 1e-12) / n
    idx = np.clip(np.floor((x - lo) / step), 0, n - 1)
    return lo + (idx + 0.5) * step


def aq(dev, w, bits):
    """Angle quantizer: deviation from the arc centre, arc half-width w;
    w >= pi means the full circle (circular code, no clipping)."""
    bits = np.broadcast_to(np.asarray(bits), np.shape(w))
    n = 2.0 ** bits
    step_f = 2 * math.pi / n
    rec_f = -math.pi + (np.mod(np.floor((dev + math.pi) / step_f), n) + 0.5) * step_f
    rec_f = np.where(bits == 0, 0.0, rec_f)
    wc = np.minimum(w, math.pi)
    return np.where(w >= math.pi * 0.999, rec_f, uq(np.clip(dev, -wc, wc), -wc, wc, bits))


def polar(k: np.ndarray):
    h = k.shape[1] // 2
    return np.hypot(k[:, :h], k[:, h:]), np.arctan2(k[:, h:], k[:, :h])


def from_polar(r, t):
    return np.concatenate([r * np.cos(t), r * np.sin(t)], axis=1)


# ------------------------------------------------------------ the codec
class NQXRN:
    """Pre-RoPE polar key codec with calibrated per-pair constants."""

    def __init__(self, k_cal: np.ndarray, q_cal: np.ndarray, pct: float = 0.1, arc_pct: float = 99.5):
        r, a = polar(k_cal)
        s = self.scale(r)
        rn = r / s
        self.lo = np.percentile(rn, pct, axis=0)
        self.hi = np.percentile(rn, 100 - pct, axis=0)
        self.c = np.angle(np.exp(1j * a).mean(axis=0))                 # arc centre
        w = np.percentile(np.abs(wrap(a - self.c)), arc_pct, axis=0)
        self.w = np.where(w > 0.8 * math.pi, math.pi, w)                # wide arcs: full circle
        rq, _ = polar(q_cal)
        self.wq = (rq ** 2).mean(axis=0)                                # RoPE-invariant weight
        self._cal = (r, a, s)

    @staticmethod
    def scale(r: np.ndarray) -> np.ndarray:
        return r.max(axis=1, keepdims=True)                             # comparator only

    def encode(self, k: np.ndarray, br, ba) -> np.ndarray:
        """Returns the dequantized pre-RoPE keys (for the float reference path)."""
        r, a = polar(k)
        s = self.scale(r)
        return from_polar(uq(r / s, self.lo, self.hi, br) * s, self.c + aq(wrap(a - self.c), self.w, ba))

    def codes(self, k: np.ndarray, br, ba):
        """Dequantized radius and angle per pair (what the scorer reads)."""
        r, a = polar(k)
        s = self.scale(r)
        return uq(r / s, self.lo, self.hi, br) * s, self.c + aq(wrap(a - self.c), self.w, ba)

    def allocate(self, bits_per_pair: float, bmax: int = 7, weighted: bool = True):
        r, a, s = self._cal
        h = r.shape[1]
        E = np.zeros((h, bmax + 1, bmax + 1))
        for br in range(bmax + 1):
            rh = uq(r / s, self.lo, self.hi, br) * s
            for ba in range(bmax + 1):
                th = self.c + aq(wrap(a - self.c), self.w, ba)
                E[:, br, ba] = (r ** 2 + rh ** 2 - 2 * r * rh * np.cos(a - th)).mean(axis=0)
        w = self.wq if weighted else np.ones(h)
        br, ba = np.zeros(h, int), np.zeros(h, int)
        ar = np.arange(h)
        for _ in range(int(round(bits_per_pair * h))):
            cur = w * E[ar, br, ba]
            g_r = np.where(br < bmax, cur - w * E[ar, np.minimum(br + 1, bmax), ba], -np.inf)
            g_a = np.where(ba < bmax, cur - w * E[ar, br, np.minimum(ba + 1, bmax)], -np.inf)
            if g_r.max() >= g_a.max():
                br[np.argmax(g_r)] += 1
            else:
                ba[np.argmax(g_a)] += 1
        return br, ba


class CartesianPreRoPE:
    """KVQuant-style: pre-RoPE per-channel ranges from calibration, per-token
    scale header, uniform bits or a Block-GTQ-style per-pair allocation."""

    def __init__(self, k_cal: np.ndarray, q_cal: np.ndarray, scale: str = "max", pct: float = 0.1):
        self.kind = scale
        s = self.scale(k_cal)
        x = k_cal / s
        self.lo = np.percentile(x, pct, axis=0)
        self.hi = np.percentile(x, 100 - pct, axis=0)
        rq, _ = polar(q_cal)
        self.wp = (rq ** 2).mean(axis=0)
        self._cal = (x, s)

    def scale(self, k):
        if self.kind == "max":
            return np.abs(k).max(axis=1, keepdims=True)
        r, _ = polar(k)
        return np.sqrt((r ** 2).mean(axis=1, keepdims=True))

    def encode(self, k, bits):
        s = self.scale(k)
        return uq(k / s, self.lo, self.hi, bits) * s

    def allocate(self, bits_avg: float, bmax: int = 7):
        x, s = self._cal
        d = x.shape[1]
        h = d // 2
        E = np.stack([(((uq(x, self.lo, self.hi, b) - x) * s) ** 2).mean(axis=0) for b in range(bmax + 1)], 1)
        Ep = E[:h] + E[h:]
        b = np.zeros(h, int)
        ar = np.arange(h)
        for _ in range(int(round(bits_avg * h))):
            g = np.where(b < bmax, self.wp * (Ep[ar, b] - Ep[ar, np.minimum(b + 1, bmax)]), -np.inf)
            b[np.argmax(g)] += 1
        return np.concatenate([b, b])


_LLOYD: dict[int, np.ndarray] = {}


def lloyd_gauss(bits: int) -> np.ndarray:
    if bits not in _LLOYD:
        z = np.sort(np.random.default_rng(0).standard_normal(400_000))
        n = 2 ** bits
        lv = np.quantile(z, (np.arange(n) + 0.5) / n)
        for _ in range(60):
            idx = np.searchsorted((lv[1:] + lv[:-1]) / 2, z)
            lv = np.array([z[idx == j].mean() for j in range(n)])
        _LLOYD[bits] = lv
    return _LLOYD[bits]


def rotate_lloyd(x: np.ndarray, q: np.ndarray, bits: int) -> np.ndarray:
    """TurboQuant-MSE style: rotation, norm header, Lloyd-Max per coordinate."""
    d = x.shape[1]
    y = x @ q.T
    nrm = np.linalg.norm(y, axis=1, keepdims=True)
    lv = lloyd_gauss(bits)
    u = lv[np.searchsorted((lv[1:] + lv[:-1]) / 2, y / nrm * math.sqrt(d))]
    return (u * nrm / math.sqrt(d)) @ q


# ------------------------------------------------------------ emulator variants
def make_head2(rng, d, t, base, n_out=4, out_mag=12.0, mode="one", structure_seed=0, drift=0.0) -> R.Head:
    """reverse_study.make_head with switches for the sensitivity study.
    mode: 'one' massive value in one dim of a low-frequency pair (default),
    'split' fixed direction across both dims, 'ring' fixed magnitude with a
    random direction per token. drift: the test sequence's channel means and
    scales differ from the calibration sequence by this relative amount."""
    srng = np.random.default_rng(structure_seed)
    h = d // 2
    mu_k, mu_q = np.zeros(d), np.zeros(d)
    low = np.arange(h - max(8, n_out * 2), h)
    ring = np.zeros(d, bool)
    for i in srng.choice(low, n_out, replace=False):
        mk = srng.choice([-1, 1]) * out_mag * srng.uniform(0.6, 1.4)
        mq = srng.choice([-1, 1]) * out_mag * srng.uniform(0.3, 0.8)
        if mode == "one":
            dim = i if srng.random() < 0.5 else i + h
            mu_k[dim], mu_q[dim] = mk, mq
        elif mode == "split":
            a = srng.uniform(0, 2 * math.pi)
            mu_k[i], mu_k[i + h] = mk * math.cos(a), mk * math.sin(a)
            mu_q[i], mu_q[i + h] = mq * math.cos(a), mq * math.sin(a)
        else:
            mu_k[i], mu_q[i] = abs(mk), abs(mq)
            ring[i] = True
    sig_k = np.exp(srng.normal(0, 0.35, d))
    sig_q = np.exp(srng.normal(0, 0.35, d))
    sig_v = np.exp(srng.normal(0, 0.25, d))
    if drift:
        drng = np.random.default_rng(structure_seed + 999)
        mu_k = mu_k * drng.uniform(1 - drift, 1 + drift, d)
        sig_k = sig_k * np.exp(drng.normal(0, drift, d))
    tok = np.exp(rng.normal(0, 0.25, (t, 1)))
    bk, bq = np.tile(mu_k, (t, 1)), np.tile(mu_q, (t, 1))
    for i in np.where(ring)[0]:
        ang = rng.uniform(0, 2 * math.pi, t)
        bk[:, i], bk[:, i + h] = mu_k[i] * np.cos(ang), mu_k[i] * np.sin(ang)
        bq[:, i], bq[:, i + h] = mu_q[i] * np.cos(ang), mu_q[i] * np.sin(ang)
    k_pre = tok * (bk + sig_k * rng.standard_normal((t, d)))
    q_pre = np.exp(rng.normal(0, 0.25, (t, 1))) * (bq + sig_q * rng.standard_normal((t, d)))
    v = np.exp(rng.normal(0, 0.35, (t, 1))) * sig_v * rng.standard_normal((t, d))
    return R.Head(k_pre, q_pre, v, np.arange(t, dtype=float), base)


# ------------------------------------------------------------ evaluation
def eval_head(c: R.Head, h: R.Head, n_q: int, bits=(2, 3, 4), full: bool = True) -> dict:
    d = h.k_pre.shape[1]
    kpost = R.rope(h.k_pre, h.pos, h.base)
    q = R.rope(h.q_pre, h.pos, h.base)
    vq = R.int4_per_token(h.v).xhat
    ref = R.attention(q, kpost, h.v, n_q)

    def post(kh):
        return R.rel(R.attention(q, kh, vq, n_q), ref)

    def pre(kh_pre):
        return post(R.rope(kh_pre, h.pos, h.base))

    qr, qh = R.random_orthogonal(d), R.randomized_hadamard(d)
    rn = NQXRN(c.k_pre, c.q_pre)
    cm = CartesianPreRoPE(c.k_pre, c.q_pre, "max")
    cr = CartesianPreRoPE(c.k_pre, c.q_pre, "rms")
    hb = HDR / d
    out = {"KIVI-4, 128-token buffer|4.250": post(R.kivi4_keys(kpost).xhat),
           "Random + S1 (TurboQuant-like, part II)|4.250": post(R.s1_polar(kpost, qr).xhat)}
    if full:
        out["Hadamard + S1 (QuaRot-like, part II)|4.250"] = post(R.s1_polar(kpost, qh).xhat)
        out["Golden butterfly + S1 (part II)|4.250"] = post(R.s1_polar(kpost, R.golden_butterfly(d)).xhat)
        out["NQX-RN of part II, 128-token groups|4.250"] = pre(R.rope_polar_keys(h.k_pre).xhat)
    for b in bits:
        tag = f"{b + hb:.3f}"
        out[f"Random rotation + Lloyd-Max {b} bit|{tag}"] = post(rotate_lloyd(kpost, qr, b))
        out[f"Hadamard + Lloyd-Max {b} bit|{tag}"] = post(rotate_lloyd(kpost, qh, b))
        cart = {"max": pre(cm.encode(h.k_pre, b)), "rms": pre(cr.encode(h.k_pre, b)),
                "pair": pre(cm.encode(h.k_pre, cm.allocate(b)))}
        out[f"Cartesian pre-RoPE {b} bit, max scale|{tag}"] = cart["max"]
        out[f"Cartesian pre-RoPE {b} bit, RMS scale|{tag}"] = cart["rms"]
        out[f"Cartesian pre-RoPE, per-pair allocation {b} bit|{tag}"] = cart["pair"]
        out[f"NQX-RN {b}+{b} per pair, no allocation|{tag}"] = pre(rn.encode(h.k_pre, b, b))
        out[f"NQX-RN, unweighted allocation {b} bit|{tag}"] = pre(rn.encode(h.k_pre, *rn.allocate(2 * b, weighted=False)))
        out[f"NQX-RN {b} bit|{tag}"] = pre(rn.encode(h.k_pre, *rn.allocate(2 * b)))
    return out


def paired(a, b):
    dd = np.asarray(a) - np.asarray(b)
    return float(dd.mean()), float(dd.std(ddof=1) / math.sqrt(len(dd))), int((dd < 0).sum()), len(dd)


def scenario_main(seeds=(11, 23, 37), n_heads=16, d=128, t=1024, n_q=64):
    res: dict[str, list[float]] = {}
    for seed in seeds:
        for c, h in R.head_pairs(n_heads, d, t, seed):
            for k, v in eval_head(c, h, n_q).items():
                res.setdefault(k, []).append(v)
    cmp = {}
    for b in (2, 3, 4):
        rn = res[next(k for k in res if k.startswith(f"NQX-RN {b} bit|"))]
        carts = [k for k in res if k.startswith("Cartesian") and f" {b} bit" in k]
        best = min(carts, key=lambda k: np.mean(res[k]))
        cmp[f"NQX-RN {b} bit - best Cartesian pre-RoPE ({best.split('|')[0]})"] = paired(rn, res[best])
        rot = [k for k in res if "Lloyd-Max" in k and f" {b} bit" in k]
        bestr = min(rot, key=lambda k: np.mean(res[k]))
        cmp[f"NQX-RN {b} bit - best rotation at {b} bit ({bestr.split('|')[0]})"] = paired(rn, res[bestr])
    rn3 = res[next(k for k in res if k.startswith("NQX-RN 3 bit|"))]
    for k in res:
        if k.startswith(("Hadamard + Lloyd-Max 4", "Random rotation + Lloyd-Max 4", "Random + S1", "KIVI")):
            cmp[f"NQX-RN 3 bit - {k.split('|')[0]}"] = paired(rn3, res[k])
    return res, cmp


SENS = {
    "base (as the main table)": {},
    "no massive values": {"n_out": 0},
    "massive value across both dims": {"mode": "split"},
    "massive magnitude, direction per token": {"mode": "ring"},
    "calibration drift 20 %": {"drift": 0.2},
    "calibration drift 40 %": {"drift": 0.4},
    "RoPE base 10 000": {"base": 10000.0},
    "d = 96": {"d": 96},
    "d = 64": {"d": 64},
}


def scenario_sensitivity(n_heads=16, t=1024, n_q=64, seed=5, names=None):
    out = {}
    for name in names or list(SENS):
        kw = dict(SENS[name])
        d, base, drift = kw.pop("d", 128), kw.pop("base", 500000.0), kw.pop("drift", 0.0)
        res: dict[str, list[float]] = {}
        for i in range(n_heads):
            ss = seed * 100 + i
            c = make_head2(np.random.default_rng(2 * ss), d, t, base, structure_seed=ss, **kw)
            h = make_head2(np.random.default_rng(2 * ss + 1), d, t, base, structure_seed=ss, drift=drift, **kw)
            for k, v in eval_head(c, h, n_q, bits=(3, 4), full=False).items():
                res.setdefault(k.split("|")[0], []).append(v)
        keep = ["Random + S1 (TurboQuant-like, part II)", "KIVI-4, 128-token buffer",
                "Hadamard + Lloyd-Max 4 bit", "Cartesian pre-RoPE 4 bit, max scale", "NQX-RN 4 bit",
                "Hadamard + Lloyd-Max 3 bit", "Cartesian pre-RoPE 3 bit, max scale", "NQX-RN 3 bit"]
        out[name] = {"means": {k: float(np.mean(res[k])) for k in keep},
                     "rn3_vs_cart3": paired(res["NQX-RN 3 bit"], res["Cartesian pre-RoPE 3 bit, max scale"]),
                     "rn3_vs_had4": paired(res["NQX-RN 3 bit"], res["Hadamard + Lloyd-Max 4 bit"]),
                     "rn3_vs_had3": paired(res["NQX-RN 3 bit"], res["Hadamard + Lloyd-Max 3 bit"])}
        print(f"  sensitivity: {name}: " + ", ".join(f"{k} {v:.3f}" for k, v in out[name]["means"].items()), flush=True)
    return out


# ------------------------------------------------------------ position arithmetic
def bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def rope_impl(x: np.ndarray, pos: np.ndarray, base: float, impl: str) -> np.ndarray:
    d = x.shape[1]
    h = d // 2
    if impl == "exact":
        th = np.outer(pos.astype(np.float64), base ** (-np.arange(h) * 2.0 / d))
        c, s = np.cos(th), np.sin(th)
    elif impl in ("fp32", "fp32 angle, bf16 cos/sin", "bf16 angle"):
        inv = (1.0 / (np.float32(base) ** (np.arange(0, d, 2, dtype=np.float32) / np.float32(d)))).astype(np.float32)
        if impl == "bf16 angle":
            th = bf16(np.outer(bf16(pos.astype(np.float32)), bf16(inv)))
        else:
            th = np.outer(pos.astype(np.float32), inv).astype(np.float32)   # as in HF transformers
        c = np.cos(th.astype(np.float64)).astype(np.float32)
        s = np.sin(th.astype(np.float64)).astype(np.float32)
        if impl != "fp32":
            c, s = bf16(c), bf16(s)
    else:                                                    # "int B": B-bit phase words
        b = int(impl.split()[1])
        w = np.round(base ** (-np.arange(h) * 2.0 / d) / (2 * math.pi) * 2.0 ** b).astype(np.int64) % (1 << b)
        th = ((pos.astype(np.int64)[:, None] * w[None, :]) % (1 << b)) * (2 * math.pi / 2.0 ** b)
        c = np.round(np.cos(th) * 32768) / 32768            # 16-bit CORDIC output
        s = np.round(np.sin(th) * 32768) / 32768
    a, b_ = x[:, :h], x[:, h:]
    return np.concatenate([a * c - b_ * s, a * s + b_ * c], axis=1)


PHASE_IMPLS = ["fp32", "fp32 angle, bf16 cos/sin", "bf16 angle", "int 16", "int 20", "int 24", "int 32"]


def scenario_phase(bases=(500000.0, 10000.0), offsets=(0, 2**15, 2**17, 2**20), n_heads=6, t=512, n_q=32):
    out = {}
    for base in bases:
        for off in offsets:
            errs = {k: [] for k in PHASE_IMPLS}
            for i in range(n_heads):
                hd = R.make_head(np.random.default_rng(700 + i), 128, t, base, structure_seed=900 + i)
                pos = np.arange(t) + off
                ref = R.attention(rope_impl(hd.q_pre, pos, base, "exact"), rope_impl(hd.k_pre, pos, base, "exact"), hd.v, n_q)
                for im in PHASE_IMPLS:
                    errs[im].append(R.rel(R.attention(rope_impl(hd.q_pre, pos, base, im),
                                                      rope_impl(hd.k_pre, pos, base, im), hd.v, n_q), ref))
            out[f"{int(base)}|{off}"] = {k: float(np.mean(v)) for k, v in errs.items()}
    return out


def scenario_lut(n_heads=8, t=1024, n_q=32, bits_list=(6, 8, 10, 12), b=3):
    """Scores from the codes: r_q r_k cos(table[index]); index = top bits of the
    32-bit phase phi_q + (n - m) W_i - theta_k (one add per key and pair)."""
    out = {str(x): [] for x in bits_list}
    out["float"] = []
    for i in range(n_heads):
        c, h = R.head_pairs(1, 128, t, 300 + i)[0]
        d = 128
        rn = NQXRN(c.k_pre, c.q_pre)
        br, ba = rn.allocate(2 * b)
        kq = rn.encode(h.k_pre, br, ba)
        q = R.rope(h.q_pre, h.pos, h.base)
        ref = R.attention(q, R.rope(h.k_pre, h.pos, h.base), h.v, n_q)
        out["float"].append(R.rel(R.attention(q, R.rope(kq, h.pos, h.base), h.v, n_q), ref))
        rk, tk = rn.codes(h.k_pre, br, ba)
        rq, tq = polar(h.q_pre)
        w = np.round(h.base ** (-np.arange(d // 2) * 2.0 / d) / (2 * math.pi) * 2.0 ** 32).astype(np.int64)
        tk_i = np.round(tk / (2 * math.pi) * 2.0 ** 32).astype(np.int64)
        tq_i = np.round(tq / (2 * math.pi) * 2.0 ** 32).astype(np.int64)
        for bits in bits_list:
            n = 1 << bits
            lut = np.round(np.cos(2 * math.pi * (np.arange(n) + 0.5) / n) * 2047) / 2047   # 12-bit entries
            att = []
            for p in range(t - n_q, t):
                m = np.arange(p + 1)
                ph = (tq_i[p][None, :] + (p - m)[:, None] * w[None, :] - tk_i[m]) % (1 << 32)
                s = (rq[p][None, :] * rk[m] * lut[ph >> (32 - bits)]).sum(axis=1) / math.sqrt(d)
                e = np.exp(s - s.max())
                att.append((e / e.sum()) @ h.v[: p + 1])
            out[str(bits)].append(R.rel(np.array(att), ref))
    return {k: float(np.mean(v)) for k, v in out.items()}


def op_table(d=128, b=3):
    h = d // 2
    return [
        ("Dequantize + RoPE + dot product (KVQuant-style pre-RoPE keys)",
         "2 mul + 2 add (dequantize), 4 mul + 2 add (RoPE), 2 MAC", 8, "cos/sin per position: table or fp32 compute"),
        ("Post-RoPE Cartesian keys (KIVI, rotations)", "2 mul + 2 add (dequantize), 2 MAC", 4,
         "positions only at encode time; rotations add d mul (random) or log2 d add (Hadamard) per coordinate"),
        ("PolarQuant, post-RoPE polar, table per query", "1 table read + 1 add",
         0, f"table of 2^(bits) entries per pair built per query ({h} x 2^{2 * b} = {h * 2 ** (2 * b)} mul at {b}+{b} bits)"),
        ("NQX-RN, pre-RoPE polar, angle domain", "2 add (phase step, angle difference), 2 table reads (cos, radius), 1 mul, 1 add",
         1, f"per query: CORDIC vectoring of the query, {h} radius tables of 2^b_r entries; phase words W_i are constants"),
    ]


# ------------------------------------------------------------ report
def fmt_pm(p):
    m, se, wins, n = p
    return f"{m:+.4f} ± {se:.4f} (better in {wins}/{n})"


def write_report(path: Path, main, cmp, sens, phase, lut, secs):
    L = ["# NQX-RN study results (generated)", "",
         f"Generated by `research/nqx_rn_study.py` in {secs:.0f} s. Emulated KV data (reverse_study.make_head); "
         "attention output relative error, keys quantized, values INT4 per token for every method. "
         "Bits per value include the per-token header; calibrated methods take their constants from a "
         "separate sequence of the same head.", "",
         "## 1. Main comparison (d = 128, 3 populations x 16 heads, 1024 tokens, last 64 queries)", "",
         "| Key method | bits/value | streaming | calibrated | attention error |", "|---|---|---|---|---|"]
    rows = sorted(((k.split("|")[0], float(k.split("|")[1]), float(np.mean(v))) for k, v in main.items()), key=lambda r: r[2])
    for name, bpv, err in rows:
        streaming = "no" if "buffer" in name or "groups" in name else "yes"
        cal = "yes" if name.startswith(("Cartesian", "NQX-RN")) and "groups" not in name else "no"
        L.append(f"| {name} | {bpv:.3f} | {streaming} | {cal} | {err:.4f} |")
    L += ["", "## 2. Paired differences (negative: NQX-RN better)", "", "| Comparison | mean ± s.e. |", "|---|---|"]
    L += [f"| {k} | {fmt_pm(v)} |" for k, v in cmp.items()]
    L += ["", "## 3. Sensitivity (16 heads each, attention error)", ""]
    names = list(next(iter(sens.values()))["means"].keys())
    L += ["| Scenario | " + " | ".join(names) + " | NQX-RN 3 - Cartesian 3 | NQX-RN 3 - Hadamard 4 | NQX-RN 3 - Hadamard 3 |",
          "|" + "---|" * (len(names) + 4)]
    for s, v in sens.items():
        L.append(f"| {s} | " + " | ".join(f"{v['means'][n]:.3f}" for n in names) +
                 f" | {fmt_pm(v['rn3_vs_cart3'])} | {fmt_pm(v['rn3_vs_had4'])} | {fmt_pm(v['rn3_vs_had3'])} |")
    L += ["", "## 4. RoPE position arithmetic at large absolute positions (attention error vs float64 RoPE, unquantized keys)", "",
          "| RoPE base | position offset | " + " | ".join(PHASE_IMPLS) + " |", "|" + "---|" * (len(PHASE_IMPLS) + 2)]
    for k, v in phase.items():
        base, off = k.split("|")
        L.append(f"| {base} | {off} | " + " | ".join(f"{v[i]:.1e}" for i in PHASE_IMPLS) + " |")
    L += ["", "`int B`: phase word (m * W_i) mod 2^B, exact integer arithmetic, 16-bit cos/sin.", "",
          "## 5. Scoring in the angle domain (3-bit NQX-RN keys, 8 heads)", "",
          "| cos table index bits | attention error |", "|---|---|"]
    L += [f"| {k} | {v:.4f} |" for k, v in lut.items()]
    L += ["", "## 6. Work per cached key and RoPE pair in the attention scan", "",
          "| Key format | operations | multiplications | note |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in op_table()]
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "nqx_rn_results.md")
    ap.add_argument("--quick", action="store_true", help="one population of 8 heads, fewer scenarios")
    a = ap.parse_args()
    t0 = time.time()
    if a.quick:
        main_res, cmp = scenario_main(seeds=(11,), n_heads=8)
        sens = scenario_sensitivity(n_heads=4, names=list(SENS)[:2])
        phase = scenario_phase(bases=(500000.0,), offsets=(0, 2**20), n_heads=2)
        lut = scenario_lut(n_heads=2)
    else:
        main_res, cmp = scenario_main()
        print(f"main done {time.time() - t0:.0f} s", flush=True)
        sens = scenario_sensitivity()
        print(f"sensitivity done {time.time() - t0:.0f} s", flush=True)
        phase = scenario_phase()
        lut = scenario_lut()
    write_report(a.out, main_res, cmp, sens, phase, lut, time.time() - t0)
    a.out.with_suffix(".json").write_text(json.dumps(
        {"main": main_res, "paired": cmp, "sensitivity": sens, "phase": phase, "lut": lut}, indent=1))
    print(a.out.read_text())


if __name__ == "__main__":
    main()
