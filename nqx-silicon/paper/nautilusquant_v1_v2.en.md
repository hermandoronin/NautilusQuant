# From the golden-angle hypothesis to silicon: deterministic KV-cache compression in two versions

**Technical preprint, revision 2 · 24 September 2026 · Herman Doronin**
Repository: <https://github.com/hermandoronin/NautilusQuant>

*Translated from the Russian original ([nautilusquant_v1_v2.ru.md](nautilusquant_v1_v2.ru.md)).*

> **Status.** Chip logic and verification are complete (model, RTL, pin-level
> simulation, formal proofs, 91 golden transactions).
> The final IHP sign-off run is clean: DRC against the full IHP rule deck —
> 0 violations, metal density and antennas — 0, LVS — circuits match,
> timing closes at all three corners (slack +1.08 ns at the slow corner). Foundry
> package: `nqx-silicon/tapeout/ihp-sg13g2/`. The final post-route netlist
> with IHP cell models passes all 91 golden transactions bit for
> bit.

## Abstract

The key-value cache (KV cache) determines memory consumption during inference
of large language models. The best methods for compressing it rotate the
vectors with a random orthogonal matrix and then quantize them. NautilusQuant
replaces the random matrix with layers of Givens rotations whose angles are
multiples of the golden angle 2π/φ², and obtains a fully deterministic
transform that needs almost no memory. Version 1 tested this idea in software
and in an emulator: the rotation state shrank to a 1,910-byte table, but in
reconstruction quality the rotation lost to a random one by 7.9%.

Version 2 moves the algorithm to silicon. The conversion to fixed point
revealed that the third rotation layer is the identity and that the
second-layer angles equal the first-layer angles with the opposite sign, so
the whole rotation state fits in two 32-bit registers. A new quantizer lowers
the reconstruction error from 0.316 to 0.146 at the same 4 bits per value. The
NQX-S1 test chip matches the model bit for bit, its interface protocol is
formally proven, and the full 2×2 mm die is designed for the IHP SG13G2
130 nm process and has passed sign-off: DRC, LVS, density, antennas and
timing at three corners. A mixing analysis explains why the golden angle falls
behind: rotation over adjacent pairs spreads each channel over only 4 of 128.
The work ends with a proposal for a third version, a task-specific chip
template whose parameters are fitted to the task data with separate
validation.

Part II runs a reverse study: open KV compression methods (KIVI,
KVQuant, QuaRot, TurboQuant, PolarQuant, llama.cpp) and NQX are compared on a
KV-cache emulator with RoPE, channel outliers and attention output.
The butterfly rotation with golden angles is statistically indistinguishable
from Hadamard and random rotation, with zero multipliers and with a
bit-exactness that float rotation lacks: on the switch fp32 → bf16, 73% of
vectors change at least one code. A polar code of pre-RoPE keys over RoPE
pairs, with a token buffer, reaches the accuracy of the best streaming
rotation methods without any rotation at all.
Tuning the angles on attention error does not transfer to new data.

Part III develops the point found there into the NQX-RN codec. The key is
stored before RoPE, each RoPE pair is written as a radius and an angle, bits
are split between pairs by query energy, which does not change under RoPE, and
the position is an integer 32-bit phase word. In emulation, NQX-RN at
3.125 bits per value is more accurate than Hadamard rotation and random
rotation at 4.125 bits, that is, it saves about one bit per value. Positions
are exact at any context length.
The gain disappears without massive values in the keys and when the
statistics drift 40% from calibration; both conditions remain to be checked
on real models.

**Keywords:** KV cache, quantization, Givens rotations, golden angle,
CORDIC, RoPE, polar quantization, ASICs, open PDKs,
IHP SG13G2.

## 1. Introduction

During language model inference, each new token reads the keys and values of
all previous tokens. For a 7-billion-parameter model with a 128K-token
context, the KV cache takes about 64 GB in FP16 and becomes the main consumer
of HBM memory, so compressing it directly increases throughput.

Methods such as TurboQuant [1] and QuaRot [2] first rotate the vector with an
orthogonal matrix. The rotation preserves lengths and dot products, and so
the attention scores, but it spreads outliers over all coordinates, after
which the vector quantizes well to 3–4 bits. The random matrix has to be
stored or generated: 32 KB in FP16 at dimension 128 and 2 MB at
dimension 1024. It also maps poorly onto deterministic hardware without a
random number generator.

The NautilusQuant hypothesis: replace the random matrix with a product of
Givens rotations with angles

    θ_k = (2π / φ²) · (k + 1) ≈ 137.508° · (k + 1),   φ = (1 + √5) / 2      (1)

The golden-angle sequence is uniformly distributed on the circle with the
smallest possible discrepancy, of order O(1/N) (Weyl's theorem [6]).
The question tested was whether this uniformity of angles turns into lower
quantization error.

## 2. Version 1: software prototype (March — July 2026)

### 2.1 Algorithm

A vector of dimension d passes through five stages: three layers of Givens
rotations, conversion to polar coordinates per pair, 3-bit quantization, a
1-bit residual correction following the QJL scheme, and packing into 4 bits
per value. The layers act on the pairs L1: (2k, 2k+1), L2: (2k+1, 2k+2),
L3: (k, k+d/4); the angle of pair k in layer ℓ equals θ_k·φ^ℓ. An early
formulation with centripetal scaling φ^(−i/d) broke orthogonality; it was
removed before version 1.

### 2.2 Implementation

- Reference in PyTorch and Triton, and an angle table (ROM with cos/sin):
  1,910 bytes at d = 128, 15,350 bytes at d = 1024.
- NQX-Core: a cycle-accurate emulator in NumPy, an instruction set of 24
  opcodes, an assembler, a FastAPI server, 247 tests.
- A SystemVerilog RTL skeleton with stubs in place of arithmetic
  (`polar_unit.sv` computed `x ^ y` where a CORDIC should be).

### 2.3 Results and their revision

Orthogonality was confirmed (TᵀT − I error of order 1.6·10⁻⁷), and the output
was repeatable bit for bit. A direct comparison with a random orthogonal
matrix on synthetic data with outliers showed that the golden angle is
**7.9% worse** in RMSE (0.1429 vs 0.1325). The speed figure (32 times fewer
cycles) and the energy figure (9.7 times less) came from the project's
analytical model; they were not measured. In July 2026 all published figures
were checked against the code, measured figures were separated from modelled
ones, and the negative result was moved to the top of the README.

**Outcome of version 1:** the compactness of the rotation state and the
determinism are defensible; superior quality was not confirmed; there is no
hardware implementation.

## 3. Version 2: the NQX-S1 test chip (September 2026)

The goal is a manufacturable chip that performs compression in hardware and
matches the reference model bit for bit, with a complete document package for
ordering at a foundry. The work follows the industrial order: bit-accurate
model → RTL → simulation and formal verification → synthesis, placement,
routing → DRC, LVS and timing analysis → foundry package.

### 3.1 What the conversion to fixed point revealed

    θ_3(k) = (2π/φ²)(k+1)·φ² = 2π(k+1)                                    (2)

| No. | Finding | Consequence for the chip |
|---|---|---|
| F1 | The third-layer angle is a whole number of turns (2): L3 is the identity and mixes nothing | A layer with zero increment is skipped; encoding costs two layers instead of three, with the same result |
| F2 | Since 1/φ + 1/φ² = 1, the L2 angles equal the L1 angles with the opposite sign | The phase increments `0x9E3779B9` and `0x61C88647` are Knuth's Fibonacci hashing constants [7] |
| F3 | The angles form a linear phase ramp (k+1)·c mod 2π | The 1.9 KB table is not needed: one 32-bit phase accumulator per layer; the whole state is 12 bytes of registers |
| F4 | The quantizer, not the rotation, has the strongest effect on the error | New quantization rules lower RMSE from 0.316 to 0.146 at the same 4 bits (§3.3) |

*Table 1.* Details: [`spec/11_algorithm_findings.md`](../spec/11_algorithm_findings.md).

### 3.2 Architecture

A vector of 128 int16 numbers is held in a 128×24-bit register file
(2 read ports, 2 write ports). Rotations are performed by a pipelined CORDIC [8]
using shifts and additions: 18 iterations, a 28-bit datapath, angles in 20 bits
per full turn, 20-cycle latency, one pair per cycle. An iterative variant gives
the same result bit for bit with 25% less area. The interface is an 8-bit
bus with an asynchronous four-phase handshake.

| Micro-operation | LDV | ROT L1 | ROT L2 | ROT L3 | POLAR | QUANT | **ENC** | **DEC** |
|---|---|---|---|---|---|---|---|---|
| Cycles | 258 | 88 | 87 | 3 (skip) | 87 | 200 | **720** | **690** |

*Table 2.* Cycles of the ENC instruction (256 bytes → 70-byte packet,
3.66× compression), measured in RTL simulation. At 50 MHz this is about
69,000 vectors per second.

### 3.3 The S1 quantizer

1. **Per-vector radius range.** The minimum and maximum are stored in the
   6-byte packet header; the 3-bit code is computed by an exact integer
   comparison q = #{14d > (2m−1)R}.
2. **Circular angle code.** 8 sectors around the circle with no discontinuity
   at ±π.
3. **Using the residual in decoding.** The QJL bit shifts the reconstructed
   value by a quarter of the quantization step.

### 3.4 Verification

| Level | What was checked | Result |
|---|---|---|
| Model | 23 tests: constants, pair tables against the reference (d from 8 to 256), CORDIC accuracy, rounding | pass |
| CORDIC block | 4,254 operations, edge-case and random inputs | bit for bit |
| Core | random programs, illegal opcodes, bus delays of 10–30% | bit for bit |
| Chip pins | 91 golden transactions (36,062 bytes) through the asynchronous interface | bit for bit |
| Formal | handshake and streaming protocol, k-induction (SymbiYosys) | proven |
| With IHP pads | the same set through sg13g2_io cell models | bit for bit |
| Lint | Verilator -Wall, Yosys 0.33 and 0.69, Icarus | 0 warnings |

*Table 3.* Details: [`spec/05_verification.md`](../spec/05_verification.md).

### 3.5 Physical implementation

IHP SG13G2 (130 nm), open LibreLane flow [13]: 2.0×2.0 mm die, 31 bond
pads, seal ring, metal fill, 1.2 V / 3.3 V supply, 50 MHz clock. A variant
for GF180MCU (wafer.space) and a reduced 32-value version for Tiny Tapeout
were built in parallel.

| Problem | Cause | Fix | Before → after |
|---|---|---|---|
| Timing at the slow corner (1.08 V, 125 °C) | weak buffers on high fanouts; repair based on post-placement parasitic capacitance estimates; only the typical corner checked | repair after global routing, all corners checked, max transition 1.5 ns during placement | −2.5 ns → +1.08 ns |
| Excess antenna diodes | a heuristic inserted a diode on almost every net | heuristic disabled, antennas fixed after routing | 47,343 → 0 |
| Hold buffers | the 0.25 ns clock uncertainty was applied to hold | a separate hold margin of 0.10 ns | 7,254 → 731 |
| Out of memory during fill | the IHP script flattens the die into flat geometry | the same rules in KLayout hierarchical mode | >12 GB → 2 GB |
| Metal2 density (≥ 25% in an 800 µm window) | the core is entirely filled with cells | core 1.04 mm instead of 1.2 mm, fill step per layer | 16–22% → rule met in all windows |

*Table 4.* Cell area dropped from 1.05 to 0.68 mm², core utilization
from 73 to 63%.

| Sign-off check | Tool | Result |
|---|---|---|
| DRC of the layout with fill, full IHP rule deck (174 categories) | KLayout | 0 violations |
| Metal and active-area density | KLayout, IHP rules | 0 violations |
| Antennas | OpenROAD and KLayout | 0 / 0 |
| LVS | Magic + Netgen | circuits match uniquely |
| Timing: setup / hold, slow corner | OpenSTA | +1.08 / +0.38 ns |
| Timing: hold, fast corner | OpenSTA | +0.08 ns |
| Slew and capacitance, all corners | OpenSTA | 0 violations |
| Power, typical corner | OpenROAD | 4.6 mW |

*Table 4a.* Result of the final run. In full:
[`spec/06_physical_design.md`](../spec/06_physical_design.md) and
`tapeout/ihp-sg13g2/SIGNOFF.md`.

The layout-versus-schematic comparison
(LVS) at first failed on 13 nets: the bond pads were placed edge to edge
with the pad pin, and Magic, extracting from abstracts, did not see the
contact. The bond pads were shifted by 1 µm to overlap the pad metal. IHP's
reference KLayout rule deck (`sg13g2.lvs`) does not accept the I/O cell
schematics from the same PDK, so LVS sign-off is done by Magic and Netgen:
the I/O cells enter as abstracts, and every connection to them is checked,
but not their contents (foundry IP).

## 4. Results: version 1 vs version 2

| Characteristic | Version 1 | Version 2 (NQX-S1) |
|---|---|---|
| Rotation state, d = 128 | 1,910 bytes of ROM | 12 bytes (3 registers) |
| Rotation arithmetic | FP32 multipliers (in the model) | CORDIC with shifts and additions |
| Layer L3 | computed for nothing | skipped in hardware |
| RMSE, KV-like data, 4 bits | 0.316 | 0.146 |
| Rotation accuracy | float32 | ≤ 3·10⁻⁵ rel., round trip ≤ 1 LSB |
| Compression | 4.00× (no header) | 3.66× (70 bytes with header) |
| Core cell area | — | 0.60 mm² (synthesis), 0.68 mm² after placement |
| Flip-flops | — | 5,634, all with reset |
| How the figures were confirmed | analytical model | RTL simulation, STA, DRC |

| Variant (S1 quantizer, except the first) | Relative RMSE |
|---|---|
| Version 1, NQX-Core pipeline | 0.316 |
| No rotation | 0.161 |
| Golden angle, NQX-S1 | 0.146 |
| Golden angle, butterfly topology (7 layers) | 0.133 |
| Random orthogonal matrix | 0.122 |
| Randomized Hadamard | 0.120 |

*Table 5.* Synthetic KV-cache-like data (channels with outliers),
4 bits per value. The first three rows are from
[`spec/04_numerics_results.md`](../spec/04_numerics_results.md) (256
vectors), the rest are from the mixing analysis (400 vectors, §5).

## 5. Why the golden angle falls behind random rotation

Rotation helps quantization only if the energy of an outlier spreads over
many coordinates. The measure is how many output channels receive a
noticeable share of the energy of one input channel.

| Rotation, d = 128 | Output channels per input channel (median) |
|---|---|
| Golden angle, adjacent pairs (versions 1 and 2) | 4 |
| Golden angle, butterfly (pairs with stride 1, 2, 4 … 64) | 58 |
| Random orthogonal matrix | 127 |
| Randomized Hadamard | 128 |

If the golden angles are kept and only the pairing scheme is changed to a
butterfly, the error drops from 0.146 to 0.133: the gap to random rotation
shrinks by more than half. The weakness of versions 1 and 2 is mainly
topological. This also limits the main argument for the golden angle: zero
memory and determinism are also provided by the randomized Hadamard
transform, which according to Table 5 is more accurate and simpler in hardware
(only additions and subtractions). A variant with trainable butterfly angles
is described as ButterflyQuant [5].

## 6. Discussion: a task-specific chip

NQX-S1 is not meant as a general-purpose accelerator. The project goal is a
template: the compression algorithm adapts to a specific task (for example,
data from a robot's vision system), and a chip for exactly that task is
produced from the template. This approach already exists in pieces: Taalas
HC1 hardwires one language model into silicon [9]; Robomorphic Computing
builds an accelerator from a parameterized template based on the robot's
structure [10]; ECON-T at CERN compresses detector data with a neural network
whose weights are tuned per detector region [11]; SpinQuant learns rotations
for a given model [3].

The evaluation criterion changes accordingly: what matters is the gain on the
data of the specific task, not superiority over random rotation on average
data. Tuning is acceptable if the task data are split into a part for fitting
and a held-out part for validation, and parameters go into silicon only after
validation on held-out data. The typical reason a result degrades once tuning
is switched off is fitting and validating on the same data.

## 7. Limitations

- All quality estimates come from synthetic data; there have been no runs on
  real KV caches and no perplexity measurements yet.
- The chip has not been fabricated; frequency, power and timing slack are
  results of static analysis with the PDK libraries.
- Speed is limited by the 8-bit bus: this is a test chip for checking the idea.
- There are no scan chains; the fabricated chip is tested functionally, with
  golden vectors through the pins.
- LVS checks the IHP I/O cells as abstracts: every connection to them, but
  not their contents (foundry IP).

## 8. Version 3: plan

1. **Programmable rotation topology:** a pair stride per layer and a
   constant angle, so that one die works both as a golden butterfly and
   as a Hadamard-like transform.
2. **Parameter tuner:** fitting the topology, angles and quantization levels
   to task data with mandatory held-out validation; the output is a
   configuration for the chip generator.
3. **Validation on real data** under a protocol fixed in advance.
4. **FPGA prototype** from the same RTL.
5. **Fabrication** of the first iteration through Tiny Tapeout (GF180), then
   wafer.space or IHP.

## 9. Conclusion

Version 1 showed that golden angles give a compact and deterministic
rotation, but not a more accurate one. Version 2 turned the algorithm into a
verified test chip and produced three results that matter more than the
original hypothesis: the third layer of the algorithm is the identity; the
whole rotation state reduces to two constants; the error is halved by the
quantizer, not by the choice of angles. The mixing analysis showed that the
weak point is the pairing scheme, not the angle, and pointed the way for the
third version: a programmable template that adapts to the task using data,
with separate validation.

---

# Part II. Reverse study: where NQX wins and why

## 10. The reverse framing

Part I asked "is the golden angle better on average" and got a negative
answer. Here the question is reversed: what properties do the open projects
that solve the same problem have, and at which point of the requirement space
do the properties of NQX decide the outcome? Every scenario is a measurement,
not an assumption. Code: [`research/reverse_study.py`](../research/reverse_study.py),
results: [`research/reverse_study_results.md`](../research/reverse_study_results.md).

### 10.1 What the open projects do

| Project | What it does with KV | Rotation | Quantizer | Hardware cost |
|---|---|---|---|---|
| KIVI [4] | K per channel, V per token, 2–4 bits | none | uniform, asymmetric, token groups | buffer of recent tokens in full precision |
| KVQuant [14] | K per channel **before RoPE**, non-uniform grid | none | NUQ, calibrated scales | RoPE after dequantization (multiplications) |
| PolarQuant [15] | key as RoPE pairs in polar form | none | radius and angle of the pair | attention via a q·k table over codes |
| TurboQuant [1] | random rotation + polar code + QJL | random orthogonal | scalar + 1 bit | d² state, d² multiplications |
| QuaRot [2] | randomized Hadamard | Walsh–Hadamard | int4 | d log d additions, 2ᵏ only |
| SpinQuant [3] | learned rotation | learned matrix | int4 | d² per layer, training |
| ButterflyQuant [5], HARP [16] | trainable butterfly of Givens rotations | structured | int2–4 | n log n / 2 parameters; HARP supports non-2ᵏ |
| llama.cpp q4_0/q8_0 | blocks of 32 values | none | symmetric + fp16 scale | 4.5 bits, cheap dequantization |
| NVFP4 (Blackwell) | blocks of 16, E2M1 + FP8 scale | none | floating-point format | hardware support in GPUs |

There is no open silicon for KV-cache compression. The closest works are a
"local rotation" block in ISSCC 2026 31.1 and simulated accelerators
(see `spec/09_landscape.md`). The survey [17] (200 papers, 43 transform
methods) states a general principle: quantization with a shared scale
per group benefits from "flattening" the energy within the group, while coding
with flexible bit allocation benefits from concentrating it.

### 10.2 How NQX differs from all of them

1. **Rotation state** — 12 bytes (phase increments) vs 32 KB for a
   random matrix and 16 bytes of signs for randomized Hadamard.
2. **Arithmetic** — CORDIC with shifts and additions, no multipliers.
3. **Dimension** — any even value, no 2ᵏ requirement.
4. **Determinism** — an integer datapath; the result is the same on any
   device.
5. **Pairs in polar form** — the same data form in which RoPE rotates the
   key pairs.
6. **Programmability** — phase increments in registers: one transform is
   replaced by another without a die respin.

### 10.3 KV-cache emulator

Weights of open models cannot be downloaded from the build machine (Hugging
Face is blocked by the environment's network policy), so the data are
emulated from published properties: RoPE with pairs (i, i + d/2) and base
500,000; in keys and queries, large values sit in a few low-frequency
RoPE pairs, in one of the two dimensions of the pair [4, 15, 18]; values have
no channel outliers [4]. Metrics: relative RMSE of keys and values, and the
relative attention output error softmax(q·k/√d)·V for the last
queries, which is what the model actually receives. Methods with calibration
take their constants from another sequence of the same head. Bits per value
include all scales and headers.

## 11. Results

### 11.1 Keys: all methods at ~4.25 bits

| Method (keys; values INT4 per token for all) | bits/value | no token buffer | key RMSE | attention error |
|---|---|---|---|---|
| KIVI-4, groups of 128 tokens | 4.25 | no | 0.060 | **0.233** |
| **NQX-RN**: pre-RoPE keys, polar over RoPE pairs, RoPE as angle addition | 4.25 | no | 0.073 | 0.336 |
| TurboQuant-like: random + S1 | 4.25 | yes | 0.122 | 0.338 |
| Tuned golden butterfly + S1 (7 registers) | 4.25 | yes | 0.128 | 0.347 |
| Golden butterfly + S1 | 4.25 | yes | 0.131 | 0.356 |
| QuaRot-like: Hadamard + S1 | 4.25 | yes | 0.120 | 0.372 |
| PolarQuant-like: S1 on RoPE pairs | 4.25 | yes | 0.164 | 0.386 |
| KVQuant-like, calibrated | 4.00 | yes | 0.082 | 0.409 |
| KIVI with calibrated ranges | 4.00 | yes | 0.084 | 0.421 |
| NQX-RN with calibrated constants | 4.00 | yes | 0.081 | 0.434 |
| NQX-S1 as built (adjacent pairs) | 4.25 | yes | 0.148 | 0.463 |
| S1 without rotation | 4.25 | yes | 0.161 | 0.479 |
| llama.cpp q4_0 | 4.50 | yes | 0.154 | 0.496 |
| INT4 per token | 4.25 | yes | 0.200 | 0.502 |

*Table 6.* d = 128, 16 heads, 1024 tokens, 64 last queries.

### 11.2 Significance test

The differences in attention error between the golden butterfly, its tuned
variant, Hadamard and random rotation are smaller than one to one and a half
standard errors, and they change sign from one head population to another
(three independent populations of 16 heads). **These four rotations are
statistically indistinguishable.** NQX-S1 in its as-built topology is
consistently worse than Hadamard: by 0.09–0.20, that is, 2.1 to 2.8 standard
errors in each of the three populations
(`research/reverse_study_results.md`, section 5).

### 11.3 Head dimension

| d | no rotation | golden, as in NQX-S1 | golden butterfly | random | Hadamard (block-wise at d ≠ 2ᵏ) | Hadamard with zero padding |
|---|---|---|---|---|---|---|
| 64 | 0.156 | 0.135 | 0.126 | 0.120 | 0.119 | — |
| 80 | 0.157 | 0.142 | 0.137 | 0.121 | 0.120 | 0.095 (+60% bits) |
| 96 | 0.156 | 0.139 | 0.129 | 0.121 | 0.123 | 0.104 (+33% bits) |
| 128 | 0.163 | 0.150 | 0.131 | 0.121 | 0.120 | — |

*Table 7.* Key RMSE. Block-wise Hadamard (64 + 16, 64 + 32) handles
non-2ᵏ dimensions as well as random rotation does. NQX has no quality
advantage here.

### 11.4 Cross-platform determinism

A random rotation in fp32 with a different summation order gives a difference
of 3·10⁻⁶ and does not change a single 4-bit code. But if one platform
computes in fp32 and another with inputs and output in bf16 (the usual
accelerator mode), **2.1% of pairs get a different code, and 73% of vectors
get at least one differing code.** The integer CORDIC of NQX-S1 (and an
integer Hadamard) give 0 mismatches by construction.

### 11.5 Tuning the template to the task

| Tuning of the 7 butterfly registers | on tuning data | on held-out data |
|---|---|---|
| on key RMSE; held out: new tokens of the same heads | 0.1358 → 0.1253 (−7.7%) | 0.1358 → 0.1253 (−7.7%) |
| on attention error; held out: **other** heads | 0.425 → 0.361 (−15%) | 0.350 → 0.354 (+1%, within noise) |
| on attention error, **separate tuning for each head**; held out: new tokens | 0.333 → 0.212 (−36%) | 0.391 → 0.395 (+1%; Hadamard 0.384, random 0.361) |

*Table 8.* The second and third rows reproduce in emulation what was
observed in the project's earlier experiments: a tuned result looks strong
on the tuning data and disappears on other data, even on new tokens of the
same head (12 heads: beats the original butterfly in 6 of
12, difference +0.004 ± 0.026). Attention error on 32 queries is a noisy
objective: in 140 steps, 7 registers manage to fit the noise of the
particular sample. Only tuning on a stable statistic transfers
(key RMSE over the whole distribution, first row).

### 11.6 Hardware cost of rotation (d = 128)

| Rotation | state | dimension | multipliers | operations per vector | bit-exact |
|---|---|---|---|---|---|
| NQX-S1 (CORDIC, 2 layers) | 12 B | any even | 0 | 6,858 additions | yes |
| Golden butterfly, programmable | 28 B (7 registers) | any | 0 | 24,192 additions | yes |
| Golden butterfly, angles in the mask | 0 | any | 0 | ~5,400 additions | yes |
| Randomized Hadamard (FWHT) | 16 B | 2ᵏ or blocks | 0 | 896 additions | yes, in integers |
| Random (TurboQuant) | 32 KB | any | 16,384 | 16,384 MAC | no |
| Learned (SpinQuant) | 32 KB per layer | any | 16,384 | 16,384 MAC | no |
| RoPE as angle addition (NQX-RN) | phase accumulator per pair | any even | 0 | 64 additions instead of 256 multiplications + 128 additions | yes |

## 12. Where NQX wins and where it does not

**Scenario A. Deterministic KV compression without multipliers at the network edge**
(a robot, an embedded or certifiable system). Required: no token buffer, no
multipliers, no matrix storage, the same result on any device. The golden
butterfly gives the same attention error as the best streaming rotations
(Table 6, §11.2). Compared with TurboQuant and SpinQuant it needs a thousand
times less state, uses no multipliers, and is bit-exact where float rotation
changes codes in 73% of vectors on the switch fp32 → bf16. **Here NQX beats
float rotations on every axis at equal quality.** Against Hadamard it is a
draw in quality and determinism; Hadamard is cheaper in number of additions.
What remains is programmability (phase registers) and a shared CORDIC for
rotation and polar coding.

**Scenario B. Keys in the form "native" to RoPE.** If keys are stored before
RoPE in polar form over RoPE pairs, a mixing rotation is not needed at
all, and positional encoding becomes angle addition (64 additions instead of
256 multiplications and 128 additions per key). Table-based attention
computation (as in PolarQuant [15]) then maps onto the same NQX datapath. In
emulation this variant, without any rotation, reached the attention error of
the best rotation methods (0.336 vs 0.338 for random rotation) with half the
key RMSE. Caveat: this variant encoded in groups of 128 tokens, that is, with
a buffer, while the rotations are streaming. A fair comparison and a
streaming codec are in Part III. This is the only scenario where the design
of NQX (pairs, polar form, phase accumulators) is not a compromise but an
exact match to the structure of the data.

**Where NQX loses.** If a buffer of 128 tokens in full precision is
affordable, element-wise per-channel quantization of keys (KIVI) is the most
accurate of all (0.233). In number of additions, Hadamard is cheaper than any
CORDIC butterfly. The as-built NQX-S1 topology (adjacent pairs) loses in
every scenario. Tuning the template on the end metric (attention error)
transfers neither to other heads nor to new tokens of the same head.

## 13. Conclusions for version 3

1. Replace the adjacent-pair topology with a butterfly: quality matches
   Hadamard and random rotation with the same 0 multipliers.
2. Add a pair mode (i, i + d/2) and a pre-RoPE key path: polar code over
   RoPE pairs, RoPE by phase addition, per-pair constants from calibration.
3. Tune the template parameters on stable data statistics (key RMSE,
   per-channel distribution) over a large sample, not on a noisy end
   metric. Accept a tuning only if it wins on held-out data with
   statistical significance; otherwise keep the original angles.
4. The main property for the market is not quality but bit-exactness
   without multipliers and without memory for the rotation: robotics,
   embedded and certifiable systems.
5. Check everything on real KV caches (Llama, Qwen, Phi-3 with d = 96) as
   soon as access to the weights is available.

**Limitation of Part II.** All numbers come from the emulator, not from
real models. The emulator reproduces known KV properties but does not
replace validation on real data and perplexity measurement.

# Part III. NQX-RN: pre-RoPE keys in polar form

## 14. The idea and its place among known work

Part II found one operating point where the design of NQX does not
approximate random rotation but matches the structure of the data. The key
is stored **before RoPE**, and each RoPE pair (i, i + d/2) is written as a
radius and an angle. RoPE rotates a pair by the angle m·ωᵢ, so in polar form
positional encoding is angle addition, and the attention score is computed
directly from the codes:

  q·k = Σᵢ ρᵢ · rᵢ · cos(φᵢ − θᵢ + (n − m)·ωᵢ),

where (ρᵢ, φᵢ) is the pre-RoPE query pair, (rᵢ, θᵢ) is the key pair, and n and
m are the query and key positions. This form rests on four properties:

1. **Pre-RoPE key statistics are stationary per channel** [14]: each
   channel has a stable scale and mean. The quantizer constants can be
   calibrated once per head.
2. **Massive values sit in low-frequency RoPE pairs**, usually in one
   of the two coordinates [15, 18]. In polar form such a pair is an almost
   constant radius and a narrow arc of angle; it is cheap to encode.
3. **The query pair energy |qᵢ|² does not change under RoPE.** So the pair
   weight in bit allocation is exact at any relative offset (n − m). The
   weight of a single coordinate lacks this property: RoPE mixes the two
   coordinates of a pair.
4. **The position becomes an integer.** The phase m·ωᵢ is stored as a 32-bit
   word m·Wᵢ mod 2³², Wᵢ = round(ωᵢ·2³²/2π). This is exact modular
   arithmetic: there is no cos or sin of a large argument in float.

| Work | Where the key is quantized | Form | Positions | Bit allocation |
|---|---|---|---|---|
| KVQuant [14] | before RoPE | Cartesian, per channel | RoPE recomputed with multiplications after dequantization | none |
| PolarQuant, Wu et al. [15] | after RoPE | polar per pair | already inside the key | none |
| PolarQuant, Han et al. [19] | after RoPE, after a random rotation | recursive polar | already inside the key | none |
| Block-GTQ [20] | after RoPE | Cartesian, TurboQuant-MSE over RoPE blocks | already inside the key | per RoPE block, by Q/K energy |
| RoPE-aligned rotations [21] | the rotation commutes with RoPE | Cartesian, rotation within each pair | unchanged | none |
| **NQX-RN** | **before RoPE** | **polar over RoPE pairs** | **integer phase, angle addition** | **radius and angle of each pair, by query energy** |

*Table 9.* The authors of PolarQuant [15] explicitly contrast their method
with pre-RoPE quantization: with KVQuant, positions have to be recomputed at
every decoding step. In polar form this recomputation reduces to a single
integer phase addition, so both advantages can be had at once: the
stationary pre-RoPE statistics and no recomputation. The works we found do
not contain this combination. This is a statement about the literature we
reviewed, not a proof of novelty. Work [21] showed that rotations within
RoPE pairs alone lose to a full Hadamard; Table 12 below agrees with this:
the pair-local form by itself does not win.

## 15. Codec

- **Token scale:** the largest radius among the key's pairs, 16 bits. This is
  the same rmax header that NQX-S1 writes. A single comparator finds it. The
  mean and RMS of the radii are worse in emulation (separate run, 16 heads,
  3 bits: 0.346 and 0.301 vs 0.282).
- **Pair radius:** rᵢ/s is quantized uniformly on a calibrated interval
  (0.1 and 99.9% percentiles).
- **Pair angle:** the deviation from the calibrated arc center cᵢ is
  quantized on the arc ±wᵢ (99.5% percentile). If the arc is wider than 0.8π,
  the code is circular.
- **Pair bits:** a greedy algorithm distributes the total budget (2B bits per
  pair at B bits per value) between the radius and the angle of each pair.
  The pair weight is the mean query energy E|qᵢ|²; the error is the mean
  squared error of the pair on the calibration sequence.
- **Calibration:** one sequence of the same head, separate from the
  validation sequence. Per-pair constants: radius bounds, arc center and arc
  width at 16 bits each, plus two bit widths of 3 bits each; 70 bits per
  pair, 560 bytes per head at d = 128.

Key storage at 3 bits per value: 64 pairs × 6 bits + 16 bits of scale
= 50 bytes instead of 256 bytes in fp16.

**Correction to Part II.** The NQX-RN variant of Part II (0.336) quantized
in groups of 128 tokens. It needs a token buffer, like KIVI, so the
comparison with streaming rotations was unequal. Its streaming variant with
static calibration lost (0.434). The codec in this section is streaming.

## 16. Results

The emulator and the metric are the same as in Part II: d = 128, 1024 tokens,
attention output error for the 64 last queries, values INT4 per token for all
methods. Three independent populations of 16 heads, paired differences. The
strong rotation competitor: random rotation or Hadamard rotation, then a
per-coordinate Lloyd–Max scalar quantizer and a 16-bit norm (the
TurboQuant-MSE scheme [1]).

| Key method | bits per value | streaming | calibration | attention error |
|---|---|---|---|---|
| NQX-RN, 4 bits | 4.125 | yes | yes | **0.198** |
| Cartesian pre-RoPE (like KVQuant), 4 bits | 4.125 | yes | yes | 0.212 |
| KIVI-4, 128-token buffer | 4.25 | no | no | 0.251 |
| **NQX-RN, 3 bits** | **3.125** | yes | yes | **0.263** |
| Hadamard + Lloyd–Max, 4 bits | 4.125 | yes | no | 0.297 |
| Random rotation + Lloyd–Max, 4 bits | 4.125 | yes | no | 0.316 |
| Cartesian pre-RoPE, 3 bits | 3.125 | yes | yes | 0.334 |
| Random rotation + S1 (Part II) | 4.25 | yes | no | 0.357 |
| Golden butterfly + S1 (Part II) | 4.25 | yes | no | 0.361 |
| Part II NQX-RN, groups of 128 tokens | 4.25 | no | no | 0.386 |
| **NQX-RN, 2 bits** | **2.125** | yes | yes | **0.404** |
| Hadamard + Lloyd–Max, 3 bits | 3.125 | yes | no | 0.478 |
| Cartesian pre-RoPE, 2 bits, per-pair bits | 2.125 | yes | yes | 0.489 |
| Hadamard + Lloyd–Max, 2 bits | 2.125 | yes | no | 0.824 |

*Table 10.* 48 heads. In full: `research/nqx_rn_results.md`.

| Paired difference (below zero means NQX-RN is better) | mean ± std. error | NQX-RN better |
|---|---|---|
| NQX-RN 3 bits − Hadamard + Lloyd–Max 4 bits | −0.033 ± 0.014 | 33 of 48 |
| NQX-RN 3 bits − random rotation + Lloyd–Max 4 bits | −0.052 ± 0.016 | 36 of 48 |
| NQX-RN 3 bits − random rotation + S1, 4.25 bits | −0.093 ± 0.019 | 38 of 48 |
| NQX-RN 3 bits − KIVI-4 with buffer | +0.012 ± 0.019 | 17 of 48 (draw) |
| NQX-RN 3 bits − best Cartesian pre-RoPE, 3 bits | −0.070 ± 0.014 | 40 of 48 |
| NQX-RN 2 bits − best Cartesian pre-RoPE, 2 bits | −0.085 ± 0.015 | 43 of 48 |
| NQX-RN 4 bits − best Cartesian pre-RoPE, 4 bits | −0.014 ± 0.009 | 36 of 48 (borderline) |

*Table 11.*

**Main emulation result: NQX-RN spends about one bit per value less than
the best rotation, at the same or lower error.** At
3.125 bits it is more accurate than Hadamard and random rotation at
4.125 bits. At 2.125 bits it is more accurate than Hadamard at 3.125. Against
KIVI, 3 bits vs 4.25 is a draw, but KIVI needs a 128-token buffer in full
precision.

**Where the gain comes from.** Breakdown by component at 3 bits:

| Variant | attention error |
|---|---|
| Cartesian pre-RoPE, equal bits | 0.334 |
| Cartesian pre-RoPE, per-pair bits weighted by query energy (like Block-GTQ) | 0.369 |
| polar, 3 + 3 bits for each pair | 0.347 |
| polar, bits by key error without query weight | 0.354 |
| polar, bits weighted by query energy (NQX-RN) | **0.263** |

*Table 12.* The polar form by itself is no better than the Cartesian one
(0.347 vs 0.334). The gain comes from combining the polar form with bit
allocation by query energy. In Cartesian form the same allocation did not
help, even per pair (0.369). Our explanation: the polar form separates
components that matter differently, namely the radius, which is almost
constant in massive pairs, and the angle, which is uniform in noise pairs.
The weight is also exact at any position offset. This is an explanation, not
a proven mechanism.

## 17. Where NQX-RN stops winning

The emulator builds in properties that the literature describes for real
models: stationary pre-RoPE channels and massive values in low-frequency
pairs. The check breaks each assumption in turn (16 heads
per scenario):

| Scenario | Hadamard, 4 bits | Cartesian, 3 bits | NQX-RN, 3 bits | Hadamard, 3 bits | outcome for NQX-RN at 3 bits |
|---|---|---|---|---|---|
| baseline | 0.307 | 0.318 | **0.260** | 0.504 | best of all |
| no massive values | **0.203** | 0.472 | 0.389 | 0.334 | loses to rotation |
| massive value in both coordinates of the pair | 0.299 | 0.333 | **0.268** | 0.514 | best of all |
| massive magnitude, direction changes from token to token | 0.214 | 0.450 | **0.177** | 0.353 | best of all; Cartesian breaks down |
| statistics shifted from calibration by 20% | 0.300 | 0.452 | 0.353 | 0.472 | draw with Hadamard at 4 bits |
| statistics shifted by 40% | **0.285** | 0.595 | 0.539 | 0.501 | loses to Hadamard at 4 bits, draw at 3 |
| RoPE, base 10,000 | 0.330 | 0.348 | **0.266** | 0.497 | best of all |
| d = 96 | 0.377 | 0.329 | **0.254** | 0.579 | best of all |
| d = 64 | 0.349 | 0.368 | **0.251** | 0.739 | best of all |

*Table 13.* Conclusions:

- **Structure in the data is required.** Without massive values the keys
  are close to Gaussian, calibration gives nothing, and rotation wins. The
  whole gain of NQX-RN rests on a property that has to be confirmed on real
  models.
- **The polar form is robust to outliers rotating within a pair.** If the
  massive magnitude changes direction from token to token, the pre-RoPE
  Cartesian code breaks down (0.450), while the polar code works best (0.177):
  the radius is constant. This is the observation of PolarQuant [15], carried
  over to pre-RoPE keys.
- **Calibration is the main risk.** A 20% shift in the statistics removes
  the gain over Hadamard at 4 bits; at 40%, NQX-RN loses to it. This is the
  same trap as with angle tuning in Part II. What is needed is on-the-fly
  recalibration (sliding percentiles in the encoding datapath) or a fallback
  to Hadamard rotation if the encoding error grows.

## 18. Hardware

**Positions.** Attention output error with exact keys, with RoPE computed in
different ways at positions 0 … 2²⁰:

| RoPE base; position | fp32 | fp32 angle, cos/sin in bf16 | angle in bf16 | integer phase, 16 bits | 24 bits | 32 bits |
|---|---|---|---|---|---|---|
| 500,000; 0 | 1.6·10⁻⁶ | 1.1·10⁻³ | 0.14 | 5.2·10⁻³ | 4.2·10⁻⁵ | 2.5·10⁻⁵ |
| 500,000; 131,072 | 4.0·10⁻⁴ | 7.0·10⁻³ | 0.52 | 5.2·10⁻³ | 8.3·10⁻⁵ | 7.4·10⁻⁵ |
| 500,000; 1,048,576 | 3.8·10⁻³ | 8.8·10⁻³ | 0.52 | 5.2·10⁻³ | 6.7·10⁻⁵ | 6.8·10⁻⁵ |
| 10,000; 1,048,576 | 5.8·10⁻³ | 1.4·10⁻² | 0.64 | 7.9·10⁻³ | 1.2·10⁻⁴ | 7.8·10⁻⁵ |

*Table 14.* The float-RoPE error grows with position because the absolute
precision of the angle m·ω falls. An angle in bf16 breaks RoPE completely;
this is a known problem [22]. The usual bf16 model mode (angle in fp32, cos
and sin in bf16) gives 1–14·10⁻³. **The integer phase does not depend on
position by construction.** 24 bits are enough for an error of about 10⁻⁴;
the limit is set by the 16-bit cos and sin.

**Attention score directly from the codes.** The cosine table index is the
upper bits of the 32-bit phase φᵢ + (n − m)·Wᵢ − θᵢ. Moving to the next key
changes the phase by one addition. A table with an 8-bit index (256 values of
12 bits, 384 bytes) adds 0.001 to the error (0.1505 vs 0.1494 when
computed in float). A 6-bit table adds 0.009.

| Key format | work per key and pair during the attention scan | multiplications |
|---|---|---|
| pre-RoPE, Cartesian (KVQuant) | dequantization, RoPE, dot product | 8 |
| post-RoPE, Cartesian (KIVI, rotations) | dequantization, dot product | 4 |
| post-RoPE, polar, per-query table (PolarQuant) | 1 table lookup, 1 addition | 0 (but 64 × 2⁶ multiplications to build the table for each query) |
| **pre-RoPE, polar, by angle (NQX-RN)** | 2 additions, 2 table lookups, 1 multiplication, 1 addition | **1** |

*Table 15.* In operations per key, PolarQuant is cheaper than NQX-RN.
The advantage of NQX-RN is not in operations but in statistics (about one bit
per value) and in integer positions.

**What NQX-S1 already has:**
- CORDIC in vectoring mode (pair → radius and angle) and in rotation mode;
- a polar quantizer with 3-bit radius and angle codes;
- an rmax header per vector, which is exactly the NQX-RN scale;
- a 32-bit phase accumulator with programmable increments.

**What it lacks:**
- pairs (i, i + d/2): the third layer has stride 32, but the host can permute
  the coordinates;
- per-pair calibration constants (560 bytes);
- different code widths for different pairs;
- 64 phase words Wᵢ instead of three registers;
- a scoring block: cosine table, per-query radius tables,
  a multiplier and an adder.

The fabricated chip can serve as a partial test bench (encoding the key into
polar form), but it does not implement NQX-RN in full.

## 19. How to check on real models

An experiment plan that closes the main limitation:

1. **Models:** Llama-3.2-1B (d = 64, base 500,000), Qwen2.5-1.5B (d = 128),
   Phi-3-mini (d = 96, base 10,000), Llama-3.1-8B.
2. **Pre-RoPE data:** outputs of `k_proj` and `q_proj`. In the Hugging Face
   implementations RoPE is applied after the projection, so capturing the
   layer outputs is enough.
3. **Calibration** on 512 tokens of one corpus (C4), validation on another
   (WikiText-2), then a domain shift (calibration on text, validation on
   code): a direct test of the risk from Section 17.
4. **Metrics:** perplexity with quantized keys (values INT4 for all methods),
   attention output error per layer, a selection of LongBench tasks for long
   context.
5. **Competitors:** KIVI, a KVQuant-like pre-RoPE code, PolarQuant,
   Hadamard and random rotation with Lloyd–Max, and Block-GTQ if its code is
   available.
6. **Success criteria:**
   - NQX-RN at 3.125 bits is no worse in perplexity than Hadamard at 4.125;
   - under domain shift the loss is no larger than for the KVQuant-like code;
   - massive values do sit in low-frequency pairs, as the emulator
     assumes.

This takes a few hours on one GPU. Model weights are not accessible from the
build environment, so the experiment has not been run.

## 20. Conclusions of Part III

NQX-RN is neither a rotation nor a copy of a known method. It combines
pre-RoPE quantization (like KVQuant), a polar form over RoPE pairs (like
PolarQuant, but before RoPE), an integer phase instead of float positions,
and bit allocation by query energy, which does not change under RoPE. In
emulation it saves about one bit per value compared with the best rotations
and makes positions exact at any context length. Its gain rests on two
conditions: the keys contain massive values in low-frequency pairs, and the
statistics do not drift far from calibration. Both conditions can be checked
on real models in a few hours; until then it is too early to put NQX-RN into
the version 3 silicon.


## References

1. Zandieh A. et al. TurboQuant: online vector quantization with near-optimal distortion rate. ICLR 2026. arXiv:2504.19874.
2. Ashkboos S. et al. QuaRot: outlier-free 4-bit inference in rotated LLMs. 2024. arXiv:2404.00456.
3. Liu Z. et al. SpinQuant: LLM quantization with learned rotations. 2024. arXiv:2405.16406.
4. Liu Z. et al. KIVI: a tuning-free asymmetric 2bit quantization for KV cache. 2024. arXiv:2402.02750.
5. ButterflyQuant: trainable butterflies of Givens rotations for LLM quantization (overview in [`spec/09_landscape.md`](../spec/09_landscape.md)).
6. Weyl H. Über die Gleichverteilung von Zahlen mod. Eins. Mathematische Annalen 77, 1916.
7. Knuth D. E. The Art of Computer Programming, vol. 3, §6.4.
8. Volder J. E. The CORDIC trigonometric computing technique. IRE Trans. Electronic Computers, 1959.
9. Taalas HC1: hardwired Llama-3.1 8B accelerator. CNX Software, 2026.
10. Neuman S. M. et al. Robomorphic computing. ASPLOS 2021. doi:10.1145/3445814.3446746.
11. Di Guglielmo G. et al. A reconfigurable neural network ASIC for detector front-end data compression at the HL-LHC. 2021. arXiv:2105.01683.
12. IHP Open PDK documentation: filler generation using KLayout.
13. LibreLane: <https://github.com/librelane/librelane>.
14. Hooper C. et al. KVQuant: towards 10 million context length LLM inference with KV cache quantization. NeurIPS 2024.
15. Wu S. et al. PolarQuant: leveraging polar transformation for key cache quantization and decoding acceleration. NeurIPS 2025. arXiv:2502.00527.
16. HARP: Hadamard-preconditioned adaptive rotation processor for extreme LLM quantization. 2026. arXiv:2605.29843.
17. Transforms for LLM quantization: the Great Inversion and format co-design. 2026. arXiv:2608.25188.
18. Jin M. et al. Massive values in self-attention modules are the key to contextual knowledge understanding. ICML 2025. arXiv:2502.01563.
19. Han I. et al. PolarQuant: quantizing KV caches with polar transformation. 2025. arXiv:2502.02617.
20. Liang F., Zhang Y., Jia J. RoPE-aware bit allocation for KV-cache quantization (Block-GTQ). 2026. arXiv:2606.24033.
21. Wang S., Luo Y., Xu N., Cheung C. W. When local variance optimality is not enough: RoPE-aligned Q/K rotations for dynamic 4-bit quantisation. 2026. arXiv:2608.13365.
22. Wang H. et al. When precision meets position: BFloat16 breaks down RoPE in long-context training. 2024. arXiv:2411.13476.

---

The hardware implementation, verification and physical design of version 2
were done with the help of Claude Code (Anthropic).
