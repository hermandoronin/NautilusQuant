#!/usr/bin/env python3
"""Per-head tuning of the golden butterfly (companion to reverse_study.py).

    python research/reverse_study_perhead.py

The template-chip idea tunes the rotation registers to the data of one
task. The harshest version of that: tune the 7 phase increments of the
butterfly separately for each head, on the attention-output error of one
sequence (calibration), then measure new tokens of the same head (test).
Hill climbing, 140 steps per head, the same search as the tuned row of
reverse_study.py. Output goes to stdout; the article quotes the means and
the paired differences (paper/nautilusquant_v1_v2.ru.md, table 8).

Result on the emulator: calibration error drops by about a third, the test
error does not move; the attention error of 32 queries is a noisy
objective and 7 registers fit its noise.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reverse_study as R  # noqa: E402

D, T, N_HEADS, N_Q, STEPS, SEED = 128, 768, 12, 32, 140, 211


def prepare(head: R.Head):
    k = R.rope(head.k_pre, head.pos, head.base)
    q = R.rope(head.q_pre, head.pos, head.base)
    v = R.int4_per_token(head.v).xhat
    return k, q, v, R.attention(q, k, head.v, N_Q)


def score(rot: np.ndarray, data) -> float:
    k, q, v, ref = data
    return R.rel(R.attention(q, R.s1_polar(k, rot).xhat, v, N_Q), ref)


def main() -> None:
    n_layers = len(R.butterfly_pairs(D))
    q_had, q_rand, q_gold = R.randomized_hadamard(D), R.random_orthogonal(D), R.golden_butterfly(D)
    rows = []
    for hi, (cal_head, test_head) in enumerate(R.head_pairs(N_HEADS, D, T, SEED)):
        cal, test = prepare(cal_head), prepare(test_head)
        rng = np.random.default_rng(hi)
        incs = [R.GOLDEN * R.PHI**layer for layer in range(n_layers)]
        best = score(R.golden_butterfly(D, incs), cal)
        for step in range(STEPS):
            cand = list(incs)
            layer = step % n_layers
            sigma = 0.6 if step < STEPS // 2 else 0.15
            cand[layer] = (cand[layer] + rng.normal(0, sigma)) % (2 * math.pi)
            s = score(R.golden_butterfly(D, cand), cal)
            if s < best:
                best, incs = s, cand
        rows.append([score(q_gold, cal), best, score(q_gold, test),
                     score(R.golden_butterfly(D, incs), test), score(q_had, test), score(q_rand, test)])
        r = rows[-1]
        print(f"head {hi}: cal {r[0]:.3f}->{r[1]:.3f} | test golden {r[2]:.3f} tuned {r[3]:.3f} "
              f"hadamard {r[4]:.3f} random {r[5]:.3f}", flush=True)
    a = np.array(rows)
    print("means: cal default %.4f tuned %.4f | test golden %.4f tuned %.4f hadamard %.4f random %.4f"
          % tuple(a.mean(0)))
    for j, name in ((2, "golden"), (4, "hadamard"), (5, "random")):
        diff = a[:, 3] - a[:, j]
        print(f"  per-head tuned - {name}: {diff.mean():+.4f} +- {diff.std(ddof=1) / math.sqrt(len(diff)):.4f}, "
              f"wins {(diff < 0).sum()}/{len(diff)}")


if __name__ == "__main__":
    main()
