# NQX-S1 — Numerics

**Document status:** preliminary. Revision 1.0, 2026-09-24.
Measured numbers: [`04_numerics_results.md`](04_numerics_results.md), which
`tools/bench_numerics.py` regenerates.

## 1. Three levels of model

| Level | File | Role |
|---|---|---|
| nqx-core emulator | `nqx-core/nqx/` | The original algorithm in FP32, with batch quantization |
| Float reference | `model/nqx_s1/reference.py` | The S1 algorithm in float64: same rotation, S1 quantization rules |
| Bit-accurate model | `model/nqx_s1/{cordic,quant,core}.py` | Integer arithmetic, identical to the RTL bit for bit |

The difference between rows 2 and 3 is pure fixed-point error. The
difference between rows 1 and 2 is the effect of the S1 algorithm
decisions ([`02_architecture.md`](02_architecture.md) §8).

## 2. Fixed-point error budget

| Source | Size | Mechanism |
|---|---|---|
| Input scaling | 0 | int16 is exact; ×2⁴ into the VR |
| Phase increment rounding | < 5·10⁻⁸ rad | INC is round(2³²·frac(φ^(L−2))); worst case at k = 128 |
| Phase → 20-bit angle | ≤ 3·10⁻⁶ rad | Round to nearest |
| CORDIC residual angle and atan table rounding | 3·10⁻⁵ relative, max | 18 iterations, 20-bit table |
| CORDIC shift truncation | a few LSB of the 28-bit datapath | 2 guard bits below the VR LSB |
| Gain compensation | 2⁻¹⁸ relative | KINV = 159188 / 2¹⁸ |
| Output rounding to VR | ½ LSB of Q.4 = 1/32 int16 LSB | round-half-up |
| Radius quantization | R/14 (½ step) | 3 bits over [RMIN, RMAX] |
| Angle quantization | 22.5° (½ sector) | 3 bits, circular |
| Residual refinement | halves both of the above | 1 bit each |

The quantization terms are about 10⁴ times larger than every fixed-point
term. The measurement agrees: silicon relative RMSE is 0.1464 against 0.1471
for float64.

## 3. Why these widths

`tools/sweep_widths.py` (40 KV-like vectors with full-scale outliers,
errors in int16 LSB before the final rounding to int16):

| Parameters | DW | CW | max rotation error | max round-trip error |
|---|---|---|---|---|
| **default (FW=4, GB=2, NITER=18, ZW=20, KF=18)** | 24 | 28 | **0.578** | **0.562** |
| FW=3 | 23 | 27 | 0.646 | 0.750 |
| FW=2 | 22 | 26 | 0.896 | 1.000 |
| NITER=16, ZW=18 | 24 | 28 | 1.809 | 2.750 |
| NITER=14, ZW=16 | 24 | 28 | 7.086 | 14.688 |
| GB=1 | 24 | 27 | 0.646 | 0.688 |
| GB=3 | 24 | 29 | 0.583 | 0.625 |
| KF=14 | 24 | 28 | 1.577 | 3.250 |

- **DW = 24 (Q20.4).** 20 integer bits because ‖x‖ ≤ √128·2¹⁵ < 2¹⁹, so no
  element can overflow under any rotation. The fraction bits are chosen
  from the table above: going from 4 to fewer bits roughly doubles the
  round-trip error.
- **CW = 28.** DW + 2 guard bits + 2 headroom bits. The pair norm is at
  most 2²³ in VR units, K·√2 < 2.33, so 25 magnitude bits + sign suffice for
  any VR content. The CORDIC never wraps. A third guard bit gives no
  measurable improvement.
- **ZW = 20, NITER = 18.** Each 2 iterations fewer multiplies the error by
  about 3–5. 18 is the smallest setting that keeps the round trip, after
  rounding, at ≤ 1 LSB for full-scale data.
- **KF = 18.** Gain compensation precision; with KF = 14 the gain error
  dominates.

## 4. What the results say about the algorithm

From [`04_numerics_results.md`](04_numerics_results.md), relative RMSE at
4 bits/value on KV-like data:

| Change | Effect |
|---|---|
| nqx-core quantizer → S1 quantizer (same golden rotation) | 0.316 → 0.147, **−53 %** |
| S1 without residual refinement → with | 0.259 → 0.147, **−43 %** |
| No rotation → golden-angle rotation | 0.161 → 0.147, **−9 %** |
| Golden-angle rotation → random orthogonal rotation | 0.147 → 0.122, **−17 %** |

On isotropic Gaussian data all three rotations are equivalent, as expected.

The honest reading:

- Most of the gain over nqx-core comes from the quantizer, not from the
  golden ratio.
- The golden-angle rotation does help relative to no rotation on
  outlier-heavy data, but a random rotation helps more. The same
  observation is in `nqx-core/bench/phi_vs_random.md`.
- What the golden angle buys in hardware is **zero rotation state**: 12
  bytes of increment registers instead of a 32 KB matrix. Whether −17 %
  error is worth 32 KB per head dimension is a system-level decision.
  Because the INC registers are programmable, other deterministic
  sequences can be evaluated on the same silicon.
