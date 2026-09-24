#!/usr/bin/env python3
"""Datapath width sweep used to choose FW, NITER, ZW and KF (spec/04_numerics.md §3).

For each parameter set: max rotation error vs float64 and max round-trip error
(3 layers forward + 3 inverse), both in int16 LSB before output rounding.

Usage: python tools/sweep_widths.py [--n 40]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT / "verif" / "cocotb")]

from nqx_s1 import S1Core, S1Params, host  # noqa: E402
from nqx_s1 import reference as ref  # noqa: E402
from stimulus import kv_vector  # noqa: E402

SETS = [
    {},
    {"fw": 3},
    {"fw": 2},
    {"n_iter": 16, "zw": 18},
    {"n_iter": 14, "zw": 16},
    {"gb": 1},
    {"gb": 3},
    {"kf": 14},
]


def measure(p: S1Params, vectors: list[list[int]]) -> tuple[float, float]:
    rot, rt = 0.0, 0.0
    for v in vectors:
        c = S1Core(p=p)
        c.run(host.ldv(v) + host.gvns(0) + host.gvns(1) + host.gvns(2))
        fixed = np.array(c.rf) / 2**p.fw
        rot = max(rot, float(np.max(np.abs(fixed - ref.rotate(p, np.array(v, float))))))
        c.run(host.gvns(2, True) + host.gvns(1, True) + host.gvns(0, True))
        rt = max(rt, float(np.max(np.abs(np.array(c.rf) / 2**p.fw - np.array(v)))))
    return rot, rt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    rng = random.Random(1)
    vectors = [kv_vector(rng, 128) for _ in range(args.n)]
    print("| Parameters | DW | CW | max rotation error [LSB] | max round-trip error [LSB] |")
    print("|---|---|---|---|---|")
    for kw in SETS:
        p = S1Params(**kw)
        rot, rt = measure(p, vectors)
        name = ", ".join(f"{k}={v}" for k, v in kw.items()) or "default"
        print(f"| {name} | {p.dw} | {p.cw} | {rot:.3f} | {rt:.3f} |")


if __name__ == "__main__":
    main()
