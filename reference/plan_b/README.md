# Plan B — experimental ideas (untested)

Six ideas from the early phase of the project, each a possible replacement
for one stage of the version 1 pipeline. None has been validated; the
"hypothesis" column states what each was meant to test, not a result.

## Modules

| # | Module | What it replaces | Hypothesis |
|---|--------|-----------------|----------------|
| 1 | `quasicrystal.py` | Stage 4 (Lloyd-Max) | 2-bit quality at 3-bit budget |
| 2 | `golden_jl.py` | Stage 5 (QJL) | Drop the QJL stage: 3 instead of 3 + 1 bits (5.3× instead of 4×) |
| 3 | `phinary.py` | Stage 4 (scalar quant) | Better outlier coverage via φ-base |
| 4 | `fractal_hash.py` | Angle encoding | Sub-1-bit per angle |
| 5 | `groq_dataflow.py` | Runtime scheduler | Static LUT for Groq/Cerebras/TPU |
| 6 | `multimodal_spiral.py` | Fixed φ constant | Adaptive φ/silver ratio per modality |

## Quick test (from `reference/`, numpy only)

```bash
python plan_b/quasicrystal.py    # 8D quasi-crystal quantizer
python plan_b/golden_jl.py       # Can we kill QJL?
python plan_b/phinary.py         # φ-base quantization
python plan_b/fractal_hash.py    # Sub-1-bit fractal encoding
```

## Idea 1: Quasi-crystal codebook

Replace scalar Lloyd-Max with 8D quasi-crystalline vector quantization.
Codebook built on Fibonacci proportions — fills space more densely than
periodic lattices (E8). Like Penrose tiling in 8 dimensions.

## Idea 2: Golden JL transform

**Hypothesis:** NautilusRotate IS a Johnson-Lindenstrauss projector.
If golden angles preserve dot products well enough after quantization,
QJL correction (1 bit/value) becomes unnecessary.
Result: 3 bits instead of 3+1 → compression jumps from 4x to **5.3x**.

## Idea 3: Phinary Quantization (base φ)

Quantize in base φ instead of base 2. The φ-exponential scale
expands slower than binary → naturally covers both tiny values (±0.5)
and huge outliers (±60) without wasting bits.

Includes `ZeckendorfEncoder` (Fibonacci number system) and
`PhinaryFloat` (φ-exponent floating point format).

## Idea 4: Fractal Sub-1-bit Hashing

Encode angles as step indices on the golden orbit (137.5° × k).
With delta encoding, angles compress to **< 2 bits** on correlated data.
Fibonacci-fractal segmentation achieves **1.44x more information per bit**
than binary partitioning.
