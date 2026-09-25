# Landscape: related chips and prior art

**Document status:** Revision 1.0, 2026-09-24. Compiled from public sources
(company pages, Hot Chips/ISSCC reports, arXiv, press). Specifications of
commercial chips change quickly, and some of them are marketing figures.
Anything marked *unverified* could not be confirmed from a primary source.

## 1. Inference accelerators

| Chip | Process | What it accelerates | KV-cache compression | Source |
|---|---|---|---|---|
| Groq LPU (v1); Groq 3 LPX after NVIDIA's Dec-2025 licensing deal | GF 14 nm / Samsung SF4X | Deterministic, compiler-scheduled SRAM-based inference | Not documented | tomshardware.com (Groq/NVIDIA deal) |
| Tenstorrent Wormhole / Blackhole | 12 nm / TSMC 6 nm | Tensix cores + RISC-V, up to 745 FP8 TFLOPS | Not documented | docs.tenstorrent.com |
| Etched Sohu | TSMC N4P | Transformer-only ASIC, 144 GB HBM3E | Not documented | techtimes.com, 2026-06 |
| Cerebras WSE-3 | TSMC 5 nm | Wafer-scale, 44 GB on-chip SRAM | Not documented | aiwiki.ai |
| d-Matrix Corsair | TSMC 6 nm | Digital in-memory compute, block floating point (MX) | MX block formats; KV-specific: *unverified* | servethehome.com (Hot Chips 2025) |
| SambaNova SN40L | TSMC 5 nm | Dataflow, three memory tiers | Tiering rather than compression | IEEE Xplore 10904578 |
| Positron Asimov | TSMC N3P (tape-out end 2026) | LPDDR5X instead of HBM | Not documented | prnewswire.com |
| MatX One | *unverified* | Splittable systolic array | Not documented | matx.com |
| Taalas HC1 (acquired by AMD, Aug 2026) | TSMC N6, 815 mm² | Llama-3.1-8B hard-wired, 3/6-bit weights | Not documented | cnx-software.com; amd.com |
| EnCharge EN100 | 16 nm | Analog in-memory compute, edge | Not documented | siliconangle.com |
| NVIDIA Blackwell | TSMC 4NP | GPU | **NVFP4 KV cache**: E2M1 in blocks of 16 with an FP8 scale; about 50 % smaller than FP8. No rotation | developer.nvidia.com |
| AMD MI355X | TSMC N3P | GPU | MXFP4/6/8 formats; no dedicated KV compression | rocm.docs.amd.com |

## 2. Academic silicon and simulated designs

| Work | Type | Relation to NQX-S1 |
|---|---|---|
| ISSCC 2026 paper 31.1 (55 nm, ReRAM on logic) | Silicon | Has a "local rotation unit for outlier-free low-bit quantization", applied to weights and activations. The closest silicon precedent for rotation before quantization. |
| Titanus (GLSVLSI 2025) | Simulated | On-the-fly KV pruning and quantization; not in silicon |
| CORDIC-based QR/Givens systolic arrays | Silicon, classic | Same arithmetic (CORDIC Givens rotations), used for linear algebra, not quantization |
| "CORDIC Is All You Need", CARMEN, CORDIC LSTM accelerators | Papers | CORDIC for ML activation functions, not for KV rotation |

## 3. Algorithmic prior art (software)

| Method | Rotation | Quantizer | Residual | Relation |
|---|---|---|---|---|
| **TurboQuant** (Google, ICLR 2026) | Random orthogonal | PolarQuant + scalar | 1-bit QJL | Structurally the same pipeline; NautilusQuant replaces the random rotation with a deterministic one |
| PolarQuant | Random preconditioning | Polar coordinates | — | Origin of the polar step |
| QuaRot, SpinQuant | Hadamard / learned | Integer | — | Rotations for outlier removal (weights, activations, KV) |
| RotorQuant (2026) | Block-diagonal 2-D Givens / quaternions, in llama.cpp | Integer | — | Closest software relative: Givens blocks on the KV cache |
| ButterflyQuant | Learnable Givens butterfly | Integer | — | Learned angles instead of golden angles |
| FibQuant | — | Fibonacci-sphere codebook (golden angle) | — | Uses the golden angle for codebook placement, not rotation |
| KIVI | None | Per-channel K, per-token V | — | Baseline KV quantizer |

## 4. Position of NQX-S1

- **No chip found uses deterministic golden-angle, quasi-random or
  phase-accumulator rotations for quantization.** None of the commercial
  parts above documents a rotation stage for the KV cache at all. NVIDIA and
  AMD use block-scaled FP4/MX without rotation.
- The algorithmic idea, rotation + polar + 3-bit + 1-bit residual, is
  TurboQuant's. NautilusQuant's contribution is making the rotation a
  constant.
- NQX-S1's specific hardware contribution is **angle generation by phase
  accumulation + CORDIC**. The rotation needs no memory and no multipliers,
  and it is reprogrammable through three registers.
- The measured quality cost of the deterministic rotation (about 17 % higher
  error than random on outlier-heavy synthetic data) must be disclosed
  alongside that benefit ([`11_algorithm_findings.md`](11_algorithm_findings.md)).

## Sources

- Groq / NVIDIA: https://www.tomshardware.com/tech-industry/semiconductors/nvidias-20-billion-groq-deal-produces-its-first-chip
- Tenstorrent Blackhole: https://docs.tenstorrent.com/aibs/blackhole/specifications.html
- Etched Sohu: https://www.techtimes.com/articles/319393/20260630/transformer-chip-startup-etched-exits-stealth-800m-raised-1b-contracts.htm
- Cerebras WSE-3: https://aiwiki.ai/wiki/cerebras_wse_3
- d-Matrix Corsair: https://www.servethehome.com/d-matrix-corsair-in-memory-computing-for-ai-inference-at-hot-chips-2025/
- SambaNova SN40L: https://ieeexplore.ieee.org/document/10904578/
- Positron: https://www.prnewswire.com/news-releases/positron-ai-raises-875-million-at-a-5-billion-valuation-to-bring-its-next-generation-inference-silicon-to-market-302874601.html
- MatX: https://matx.com/research/series_b
- Taalas HC1: https://www.cnx-software.com/2026/02/22/taalas-hc1-hardwired-llama-3-1-8b-ai-accelerator-delivers-up-to-17000-tokens-s/ and https://newsroom.amd.com/news/amd-acquires-taalas-ai-inference/
- EnCharge EN100: https://siliconangle.com/2025/05/29/encharges-en100-accelerator-chip-sets-stage-powerful-device-ai-inference/
- NVIDIA NVFP4 KV cache: https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/
- AMD MI350: https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi350.html
- ISSCC 2026 31.1 (rotation unit): https://arxiv.org/abs/2605.09375
- Titanus: https://arxiv.org/abs/2505.17787
- TurboQuant: https://arxiv.org/abs/2504.19874
- PolarQuant: https://www.researchgate.net/publication/388754617_PolarQuant_Quantizing_KV_Caches_with_Polar_Transformation
- RotorQuant: https://github.com/scrya-com/rotorquant
- ButterflyQuant: https://arxiv.org/pdf/2509.09679
- FibQuant: https://arxiv.org/html/2605.11478
- CORDIC QR systolic array: https://ieeexplore.ieee.org/document/7082764/
- CORDIC Is All You Need: https://arxiv.org/pdf/2503.11685
- CARMEN: https://arxiv.org/pdf/2605.06878
