<div align="center">

# 🐚 NautilusQuant

### Deterministic Orthogonal KV-Cache Quantization with a 1.9 KB Rotation ROM

[![Status](https://img.shields.io/badge/status-research%20prototype-orange?style=for-the-badge)](nqx-core/docs/PRD.md)
[![Tests](https://img.shields.io/badge/tests-246_passing%2C_1_skipped-brightgreen?style=for-the-badge)](nqx-core/tests)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Triton](https://img.shields.io/badge/triton-GPU%20kernel-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://triton-lang.org)

**[TL;DR](#tldr)** ·
**[How it works](#how-it-works)** ·
**[NQX-Core](#nqx-core--pre-silicon-emulator-and-chip-development-kit)** ·
**[NQX-S1 silicon](#nqx-s1--silicon-test-chip)** ·
**[Results](#results)** ·
**[What is not true yet](#what-is-not-true-yet)** ·
**[Maritime](#industrial-applications--shipboard-edge-ai)** ·
**[Roadmap](#roadmap)**

</div>

---

## TL;DR

Rotation-based KV-cache quantization ([TurboQuant](https://arxiv.org/abs/2504.19874), Google ICLR 2026) needs a **random orthogonal matrix** — `dim × dim` FP16 of persistent state per configuration: 32 KB at dim=128, 2 MB at dim=1024. NautilusQuant replaces the random matrix with three layers of **golden-angle Givens rotations**, so the entire rotation collapses into a **1 910-byte ROM at dim=128** (15 KB at dim=1024) and produces **bit-identical output on every run** — no PRNG, no seed, no per-layer matrix in HBM.

**That size and determinism claim is the defensible core of this project.** It is reproducible from [`nqx-core/bench/lut_budget.md`](nqx-core/bench/lut_budget.md) and [`nqx-core/bench/determinism.md`](nqx-core/bench/determinism.md).

**What is *not* established:** that the golden angle gives *better reconstruction quality* than a random matrix. The repo's own head-to-head says it does not — φ is **7.9 % worse in RMSE** ([`bench/phi_vs_random.md`](nqx-core/bench/phi_vs_random.md)). See [Results](#results).

**v0.1.0** ships an upstream-faithful reference implementation (this repo) plus **NQX-Core** — a pre-silicon emulator and chip development kit at [`nqx-core/`](nqx-core/): 24-opcode ISA, cycle-accurate NumPy emulator, SystemVerilog RTL *skeleton* (placeholder datapath, see [E2](#roadmap)), Yosys + OpenLane configuration, ASIC floorplan and timing notes, FastAPI server, demo runner with side-by-side TurboQuant comparison, **247 tests (246 pass, 1 skip)**.

**NQX-S1** at [`nqx-silicon/`](nqx-silicon/) turns the algorithm into a manufacturable test chip. It has a bit-accurate model, verified fixed-point RTL, formal proofs, a full-chip layout for IHP SG13G2 with pad ring and seal ring, a port to wafer.space GF180MCU, golden test vectors, and a 12-part specification package for ordering. The rotation needs **no ROM at all**: two 32-bit phase increments and a CORDIC regenerate every golden angle.

---

## The Problem

LLM inference is **memory-bound**, not compute-bound. The KV-cache for a 7B model at 128K context is **64 GB in FP16**, the dominant HBM consumer. Compression directly buys throughput.

State of the art ([TurboQuant](https://arxiv.org/abs/2504.19874)): random orthogonal rotation, then polar-quantize to 3 bits. It works — but the rotation matrix is PRNG-derived, must be stored or regenerated, has only statistical (O(1/√N)) angular uniformity, and does not map cleanly onto deterministic-dataflow hardware (Groq, Cerebras, shipboard PLCs) that has no PRNG block and no room for a multi-MB matrix.

**The question this repo asks:** how much of the rotation can be replaced by a closed-form constant, and what does that cost in quality?

---

## Core Hypothesis

The rotation matrix is a product of **non-overlapping Givens pairs** with golden-angle θ:

```
θ_k = (2π / φ²) × (k + 1) ≈ 137.5077640500° × (k + 1)
```

Hermann Weyl (1916) proved that the golden angle has the slowest-converging continued fraction `[1; 1, 1, 1, …]` of any number, giving **angular discrepancy O(1/N)** instead of the O(1/√N) of an i.i.d. random sequence. That is a statement about *angle coverage*, not about end-to-end quantization error — the two turn out not to be the same thing (see [Results](#results)).

| Property                        | Random Rotation (TurboQuant) | **Golden Rotation (NautilusQuant)** |
|---------------------------------|------------------------------|-------------------------------------|
| Deterministic                   | No (seed-dependent)          | **Yes** (φ and π are constants)     |
| Angular uniformity              | O(1/√N) statistical          | **O(1/N)** (Weyl)                   |
| Reproducibility                 | Depends on PRNG state        | **100 %** bit-identical every run   |
| Rotation state (dim=128)        | 32 KB FP16 matrix            | **1 910 B ROM**                     |
| Rotation state (dim=1024)       | 2 MB FP16 matrix             | **15 KB ROM** (137× smaller)        |
| Runtime state                   | seed + rotation matrix       | **0** (precomputed angles)          |
| Maps onto static dataflow?      | No (random matmul)           | **Yes** (Givens pipeline 1:1)       |
| Reconstruction RMSE             | **better by 7.9 %**          | worse — see [Results](#results)     |

The matrix must be **orthogonal** so attention scores survive: `‖Tv‖ = ‖v‖`, `⟨Tq, Tk⟩ = ⟨q, k⟩`. v1 of this design included `φ^(-i/d)` centripetal scaling that broke orthogonality — fixed in v2 with pure Givens.

---

## How It Works

```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
│ 1. Input │──▶│ 2. Rotate │──▶│ 3. Polar │──▶│ 4. Quant  │──▶│ 5. QJL   │
│   FP16   │   │  Golden φ │   │  (r, θ)  │   │ Lloyd-Max │   │  ±1 bit  │
│  16 bit  │   │  T^T·T=I  │   │          │   │   3 bit   │   │  1 bit   │
└──────────┘   └───────────┘   └──────────┘   └───────────┘   └──────────┘
     HBM ──────────── SRAM (fused, single pass) ────────────▶ HBM
```

Three layers of non-overlapping Givens rotations, all orthogonal by construction:

```python
# Layer 1: adjacent pairs
for k in range(dim // 2):
    givens(v, 2*k, 2*k+1, GOLDEN_ANGLE * (k + 1))

# Layer 2: shifted pairs (offset by 1)
for k in range((dim - 1) // 2):
    givens(v, 2*k+1, 2*k+2, GOLDEN_ANGLE * (k + 1) * φ)

# Layer 3: butterfly with stride dim/4 (non-overlapping pairs only)
for k in range(dim):
    if not_overlapping(k):
        givens(v, k, (k + dim//4) % dim, GOLDEN_ANGLE * (k + 1) * φ²)
```

Decode is the same in reverse with negated angles: `T⁻¹ = L₁ᵀ·L₂ᵀ·L₃ᵀ`.

The ROM stores one `(pair_i, pair_j, cos, sin)` record per pair — 10 bytes each, 191 pairs at dim=128 → **1 910 bytes**, independent of model, layer and device.

---

## NQX-Core — pre-silicon emulator and chip development kit

> **The deterministic dataflow processor that NautilusQuant maps to 1:1.**
> Lives at [`nqx-core/`](nqx-core/). **Software-only.** No FPGA bring-up, no silicon.

```
        ┌────────── HBM (off-chip, FP16) ──────────┐
        v                                          ^
   ┌────────┐                                ┌──────────┐
   │  DMA   │--> SRAM_in (24KB) ─────> ... ──│   PACK   │
   └────────┘                                │   3+1bit │
                                             └──────────┘
                                                  ^
   SRAM_in ──> [ VRF FP32, 16 × 128 elem ]        │
                       │                          │
        ┌──────────────┴────────────────┐    ┌────┴──────┐
        v                               v    │   QJL     │
  ┌──────────┐  ┌──────────┐  ┌──────────┐   │ sign+corr │
  │  GU-L1   │─▶│  GU-L2   │─▶│  GU-L3   │   └─────▲─────┘
  │ 64 lanes │  │ 63 lanes │  │ ~32 lns  │         │
  │ adj pair │  │ shifted  │  │ butterfly│         │
  └──────────┘  └──────────┘  └──────────┘         │
                       │                           │
                       v                           │
                 ┌──────────┐    ┌──────────┐      │
                 │  POLAR   │───▶│  QUANT   │──────┘
                 │ √+atan2  │    │ Lloyd-Max│
                 │ 64 lanes │    │   3-bit  │
                 └──────────┘    └──────────┘
                       ^
                 ┌─────┴────┐
                 │ ROM LUT  │  golden cos/sin, 1 910 B
                 │ 191 pair │
                 └──────────┘
```

**Pipeline depth: 18 cycles. Steady-state throughput: 1 vec/cycle** — as modelled by the emulator's cycle counter, not measured on hardware.

| Layer                     | Artifact                                                          |
|---------------------------|-------------------------------------------------------------------|
| ISA + assembler           | [`nqx-core/nqx/`](nqx-core/nqx/) — 24 opcodes (`LDV`, `GVNS`, `POLAR`, `QUANT`, `QJL`, `PACK3`, `MXPACK`, `SUBBIT_ENC`, `ATTN_DOT`, `LDV_ASYNC`, …) |
| Cycle-accurate emulator   | [`nqx-core/nqx/cpu.py`](nqx-core/nqx/cpu.py) — pure NumPy, no torch dep |
| RTL **skeleton**          | [`nqx-core/rtl/`](nqx-core/rtl/) — 5 SystemVerilog design modules + Verilator testbench. `polar_unit.sv` and `quant_unit.sv` still carry **placeholder datapaths** (XOR/ADD instead of CORDIC, truncation instead of Lloyd-Max). Structure and interfaces only. |
| Synthesis                 | [`nqx-core/rtl/synth/`](nqx-core/rtl/synth/) — Yosys flow with sky130 target |
| Open-source tape-out      | [`nqx-core/rtl/openlane/`](nqx-core/rtl/openlane/) — OpenLane2 config (Skywater MPW path) |
| Formal verification       | [`nqx-core/rtl/formal/`](nqx-core/rtl/formal/) — SymbiYosys harness for orthogonality |
| ASIC floorplan + timing   | [`nqx-core/asic/`](nqx-core/asic/) — paper study: 50 mm² TSMC 7 nm target, 1 GHz, tape-out checklist |
| HTTP service              | [`nqx-core/server/`](nqx-core/server/) — FastAPI, monitoring, chaos tests |
| Side-by-side vs TurboQuant| [`nqx-core/demos/side_by_side.md`](nqx-core/demos/side_by_side.md) |
| φ vs random head-to-head  | [`nqx-core/bench/phi_vs_random.md`](nqx-core/bench/phi_vs_random.md) |
| Pre-silicon SDK           | [`nqx-core/sdk/`](nqx-core/sdk/) — libnqx C ABI, install.sh, errata, programming guide |
| Linux driver skeleton     | [`nqx-core/firmware/driver/`](nqx-core/firmware/driver/) |
| Roadmap (E1-E6)           | [`nqx-core/docs/PRD.md`](nqx-core/docs/PRD.md)                     |

```bash
git clone https://github.com/hermandoronin/NautilusQuant && cd NautilusQuant/nqx-core
pip install -r requirements.txt
python -m pytest -q                        # 246 passed, 1 skipped (247 collected)
python run.py verify --dim 128             # acceptance criteria
python run.py bench --vectors 4096         # cycles + throughput + energy (model)
python demos/run_demo.py                   # TurboQuant vs NQX side-by-side
```

---

## NQX-S1 — silicon test chip

> Lives at [`nqx-silicon/`](nqx-silicon/). Russian overview: [`nqx-silicon/README.ru.md`](nqx-silicon/README.ru.md).

| | |
|---|---|
| Function | 128 × int16 → 70-byte packet (golden-angle Givens → polar → 3+1 bit), and back |
| Rotation state | 3 × 32-bit phase-increment registers (`0x61C88647`, `0x9E3779B9`, `0`); no angle ROM |
| Datapath | Pipelined 18-stage CORDIC (shift-and-add), 128 × 24-bit vector register, exact integer quantizer |
| Interface | 8-bit in / 8-bit out, asynchronous 4-phase handshake, any MCU |
| Verification | Bit-exact vs the Python model at core, pin and gate level; SymbiYosys proofs; golden vectors for silicon |
| Processes | IHP SG13G2 130 nm (full chip, signed off: DRC, LVS, density, antenna, STA at 3 corners); wafer.space GF180MCU (template port, CI build + precheck) |
| Findings | Layer 3 of the reference rotation is an identity; the S1 quantizer halves reconstruction error ([details](nqx-silicon/spec/11_algorithm_findings.md)) |

Specification package: [`nqx-silicon/spec/`](nqx-silicon/spec/). How to order: [`08_tapeout_package.md`](nqx-silicon/spec/08_tapeout_package.md).

---

## Results

Two kinds of number live in this table and they are labelled as such:

- **measured** — produced by running code in this repo (pytest, emulator, benchmarks);
- **model** — produced by the project's own analytical cycle/energy model in [`nqx-core/nqx/energy.py`](nqx-core/nqx/energy.py) and [`nqx-core/nqx/pipeline.py`](nqx-core/nqx/pipeline.py). **No silicon, no FPGA and no GPU wall-clock measurement backs these.**

| Metric                                             | Value                | Kind      | Source                                 |
|----------------------------------------------------|----------------------|-----------|----------------------------------------|
| Orthogonality `T^T·T = I` (dim=128)                | err **1.6 × 10⁻⁷**   | measured  | `nqx-core/tests/test_orthogonality.py` |
| Roundtrip without quantization                     | RMSE **9.6 × 10⁻⁸**  | measured  | same                                   |
| Matches the reference implementation `nautilus_triton.py` | within **10⁻⁴** max abs diff | measured | `nqx-core/tests/test_vs_reference.py` |
| Compression ratio                                  | exactly **4.00×**    | measured  | `nqx-core/tests/test_roundtrip.py`     |
| ROM-LUT size (dim=128)                             | **1 910 bytes**      | measured  | `nqx-core/nqx/lut.py`, `bench/lut_budget.md` |
| ROM-LUT size (dim=64 / 1024)                       | **950 B / 15 350 B** | measured  | `nqx-core/bench/lut_budget.md`         |
| Determinism — 100 runs, same input                 | **100 %** identical  | measured  | `nqx-core/bench/determinism.md`        |
| **Reconstruction quality vs random rotation**      | **φ is 7.9 % WORSE** (RMSE 0.1429 vs 0.1325) | measured | [`nqx-core/bench/phi_vs_random.md`](nqx-core/bench/phi_vs_random.md) |
| Reconstruction quality vs TurboQuant pipeline      | φ is 1.5 % worse (RMSE 0.2860 vs 0.2818) | measured | [`nqx-core/demos/side_by_side.md`](nqx-core/demos/side_by_side.md) |
| Pipeline depth                                     | **18** cycles        | model     | emulator cycle counter                 |
| Throughput (steady state)                          | **1 vec / cycle**    | model     | emulator cycle counter                 |
| Cycles per vector vs TurboQuant emulation          | **32× fewer**        | model     | `nqx-core/demos/side_by_side.md`       |
| Energy per vector vs TurboQuant emulation          | **9.7× lower**       | model     | `nqx-core/nqx/energy.py`               |
| Energy / encode-vec, TSMC 7 nm assumptions         | **≈ 5.1 nJ**         | model     | `nqx-core/nqx/energy.py`               |
| NQ-ISA opcode count                                | **24**               | measured  | `nqx-core/nqx/isa.py`                  |
| Unit tests                                         | **246 passing, 1 skipped** (247 collected) in < 20 s | measured | `pytest -q` in `nqx-core/` |

### The negative result

The head-to-head benchmark in [`nqx-core/bench/phi_vs_random.md`](nqx-core/bench/phi_vs_random.md) compares φ-Givens against a fresh QR-orthonormal random matrix on identical synthetic KV-like inputs (Gaussian + 1/64 outliers ≈6σ), across dim ∈ {64, 128} and 3/4-bit quantization:

| dim | bits | φ RMSE | random RMSE | φ worse by |
|----:|-----:|-------:|------------:|-----------:|
| 64  | 3    | 0.2009 | 0.1832      | +9.7 %     |
| 64  | 4    | 0.0940 | 0.0854      | +10.1 %    |
| 128 | 3    | 0.1887 | 0.1781      | +6.0 %     |
| 128 | 4    | 0.0879 | 0.0831      | +5.8 %     |
| **avg** | | **0.1429** | **0.1325** | **+7.9 %** |

**The golden angle does not beat random rotation on reconstruction quality.** Weyl-optimal angular *coverage* did not translate into lower quantization error here. The three-layer Givens topology is far from a dense random orthogonal matrix, and that structural deficit outweighs the coverage advantage.

What survives the negative result, and what the project is actually about:

- rotation state shrinks from 32 KB (dim=128) / 2 MB (dim=1024) to **1 910 B / 15 KB**;
- output is **bit-identical** across runs — no seed to record, no PRNG to reproduce;
- the schedule is fully static, so it maps onto hardware with no PRNG and no scheduler;
- 88.9 % fewer modelled compute cycles than a dense random matmul.

Whether ~8 % extra RMSE is an acceptable price for those properties is an application decision, not a claim this repo can make for you.

---

## What is not true yet

Explicitly, so nobody has to reverse-engineer it from the code:

- **No silicon, no FPGA bitstream, no tape-out.** Everything hardware-related is a paper design plus a Python emulator.
- **The RTL is a skeleton.** `polar_unit.sv` computes `x ^ y` and `x + y` where CORDIC belongs; `quant_unit.sv` truncates instead of applying Lloyd-Max. Interfaces, pipelining and the module hierarchy are real; the arithmetic is not.
- **Cycle and energy numbers come from this project's own model**, not from measurement.
- **No end-to-end LLM quality evaluation** (perplexity, long-context accuracy) has been run on a real model with these kernels.
- **The φ hypothesis is not confirmed.** On quality it loses to random rotation; it wins on state size and determinism.

---

## Compression Comparison

| Config                                | Bits/value | Memory reduction (reported) | Rotation state (dim=128) | Determinism |
|---------------------------------------|------------|------------------------------|--------------------------|-------------|
| FP16 baseline                         | 16         | 1.0×                         | —                        | n/a         |
| KIVI (no rotation)                    | 2          | 2.6× (end-to-end, per paper) | per-channel scales       | yes         |
| TurboQuant (random)                   | 3 + 1      | 4.0× (KV-cache)              | 32 KB → 2 MB at dim=1024 | seed-dependent |
| **NautilusQuant (φ)**                 | **3 + 1**  | **4.0×** (KV-cache)          | **1.9 KB → 15 KB**       | **bit-identical** |

The "memory reduction" column mixes sources: KIVI's 2.6× is the end-to-end system figure reported in [its paper](https://arxiv.org/abs/2402.02750), while the 4.0× rows are the KV-cache tensor ratio measured in `tests/test_roundtrip.py`. They are not directly comparable — hence the column name.

`scale + zero-point` overhead currently matches TurboQuant (32 bit / group). Whether golden-angle rotation produces a tight enough output distribution to drop them entirely is still an **open empirical question**.

---

## Hardware fit

The pipeline is a **static dataflow** — fixed schedule, zero data-dependent branches, no PRNG, LUT in constant memory. This is the execution model of inference accelerators without a hardware scheduler. The "NQX status" column below describes *design fit*, not ported and benchmarked code:

| Platform                         | Why it fits              | NQX status                   |
|----------------------------------|--------------------------|------------------------------|
| **Groq LPU** (Tensor Streaming)  | Fully static schedule, no HBM, 230 MB on-chip SRAM | architectural fit on paper |
| **Cerebras WSE-3**               | Large on-chip SRAM, dataflow scheduling | architectural fit on paper |
| **Google TPU v5/v6**             | Systolic MXU, XLA static schedule | XLA path not implemented |
| **AWS Trainium**                 | MXFP4 native + dataflow  | MX fallback exists in the emulator |
| **NVIDIA Blackwell / RTX 5090**  | MXFP4 / NVFP4 tensor cores | Triton kernel exists (`nautilus_triton.py`), not benchmarked on Blackwell |
| **AMD MI355X** (CDNA4)           | FP4 / FP6 native         | ROCm path not implemented |
| **Skywater 130 nm**              | Open PDK, free MPW slots | OpenLane2 config present, never run to GDS |
| **NVIDIA Jetson / Movidius**     | Edge GPU + 4–8 GB RAM    | target platform, not yet deployed |
| **PLCs / FPGAs with ≥ 1 KB ROM** | Constrained controllers  | 950 B ROM at dim=64 fits a small register file |

Random rotation does not map onto these targets cleanly — it needs a PRNG block and a persistent matrix that grows with `dim²`.

---

## Industrial Applications — shipboard edge AI

NautilusQuant did not start as an academic curiosity. It started in the engine room.

A modern ship power plant generates thousands of sensor readings per second. Satellite uplink between vessel and shore is **64–512 kbps** (VSAT or Iridium Certus), shared with crew comms, ECDIS updates and IMO mandatory reporting. Pushing raw telemetry plus an LLM-based decision-support model up that pipe needs aggressive and *auditable* compression.

| Constraint                          | Property of this design that addresses it                                   |
|-------------------------------------|-----------------------------------------------------------------------------|
| **VSAT / Iridium uplink 64–512 kbps** | 4× KV-cache compression, measured on synthetic tensors                     |
| **IMO / SOLAS auditability**        | No PRNG seed → bit-identical results, reproducible from the code alone      |
| **Resource-constrained controllers**| 950 B rotation ROM at dim=64; no per-model rotation state at all            |
| **Real-time determinism**           | Fixed schedule, no data-dependent branches, no cache-miss jitter            |

These are the properties the design *has*. Running a real condition-monitoring LLM on a real vessel is future work — see [Roadmap](#roadmap) E4.

---

## Roadmap

| Stage | What | Status | Notes |
|---|---|---|---|
| **E1** | Software emulator + 24-opcode ISA + assembler | ✅ shipped | `nqx-core/nqx/`, 247 tests |
| **E2** | RTL skeleton (Verilator + Yosys + OpenLane2 + SymbiYosys) | ✅ superseded by NQX-S1 | The `nqx-core/rtl/` skeleton stays for history; the implemented, verified RTL is [`nqx-silicon/rtl/`](nqx-silicon/rtl/) |
| **E3** | FPGA bring-up (Alveo U280 / V80 / AWS F1) | ⏳ not started | needs E2 datapath first |
| **E4** | LLM stack integration (HF Cache / vLLM / Triton kernel) | ⏳ not started | needs a rented GPU |
| **E5** | MPW tape-out of a test chip | 🚧 GDSII signed off, not ordered | NQX-S1: IHP SG13G2 full chip; wafer.space GF180MCU port. Efabless closed in 2025; options and prices in [`08_tapeout_package.md`](nqx-silicon/spec/08_tapeout_package.md) |
| **E6** | Commercial ASIC TSMC 12 / 7 nm | 🔮 future | $1.5–5 M depending on node |

---

## Risks

Three things can break the central thesis:

| Risk | What breaks | Mitigation |
|---|---|---|
| **Structural resonance** | Golden angles align with outlier dims → MSE explodes | Fixed permutation layer before rotation |
| **0-overhead failure**  | Angle distribution not predictable enough → still need scale/zero-point | MX-format fallback (0.25 bit/value overhead) |
| **FP16 drift**          | Roundtrip errors accumulate over 100K-token contexts | Kahan summation / periodic renormalization |

The first risk is partly realised already: on synthetic outlier-heavy data φ is 7.9 % worse than random rotation. What is left is the state-size and determinism argument, which does not depend on the quality result.

Experimental drop-in alternatives live in [`plan_b/`](plan_b/) — `quasicrystal.py`, `golden_jl.py`, `phinary.py`, `fractal_hash.py`, `groq_dataflow.py`, `multimodal_spiral.py`. Untested, marked experimental.

---

## Quick Start (this repo)

```bash
git clone https://github.com/hermandoronin/NautilusQuant && cd NautilusQuant
pip install -r requirements.txt

# Browse interactively (no install needed)
xdg-open index.html        # 3D pipeline visualization

# Synthetic validation with realistic outliers
python validate_real_kv.py --sweep --dim 128 --count 500

# Real KV-cache from Gemma 3
pip install transformers accelerate
python validate_real_kv.py --model google/gemma-3-4b-it --sweep

# GPU kernel (Triton)
pip install triton
python nautilus_triton.py --dim 128 --n 10000

# Hardware co-design concepts (Concept 1-4)
python nautilus_hardware.py

# Needle-in-a-Haystack on 104K tokens
python benchmark_needle.py --model google/gemma-3-4b-it --method both

# Pure-numpy GloVe vector-search benchmark
python benchmark_glove.py --profile
```

For the **chip development kit**, jump to [`nqx-core/README.md`](nqx-core/README.md).

> Some design documents (`RISKS.md`, `nqx-core/docs/architecture.md`,
> `nqx-core/docs/FINAL_REPORT.md`, `nqx-core/audits/`) are written in Russian.
> The English documentation set is this file, [`nqx-core/README.md`](nqx-core/README.md),
> [`nqx-core/docs/PRD.md`](nqx-core/docs/PRD.md) and [`nqx-core/docs/paper/`](nqx-core/docs/paper/).

---

## Related Work

| Method            | Year | Approach                                          | Bits   | Paper                                            |
|-------------------|------|---------------------------------------------------|--------|--------------------------------------------------|
| GPTQ              | 2022 | Layer-wise Hessian quantization                   | 4      | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| AWQ               | 2023 | Activation-aware weight protection                | 4      | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| QuIP#             | 2023 | Hadamard rotation + E8 lattice codebooks          | 2      | [arXiv:2402.04396](https://arxiv.org/abs/2402.04396) |
| SqueezeLLM        | 2023 | Dense-and-sparse quantization                     | 3–4    | [arXiv:2306.07629](https://arxiv.org/abs/2306.07629) |
| KIVI              | 2024 | Per-channel KV-cache quantization                 | 2      | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) |
| BitNet b1.58      | 2024 | Ternary weights from training                     | 1.58   | [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) |
| **TurboQuant**    | 2026 | **Random** rotation + PolarQuant + QJL            | 3 + 1  | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) |
| **NautilusQuant** | 2026 | **Golden ratio** rotation + PolarQuant + QJL      | 3 + 1  | this repo (paper draft: [`nqx-core/docs/paper/`](nqx-core/docs/paper/)) |

---

## Citation

```bibtex
@software{nautilusquant2026,
  author = {Doronin, Herman},
  title  = {NautilusQuant: Deterministic Orthogonal KV-Cache Quantization
            via Golden Ratio Geometry},
  year   = {2026},
  url    = {https://github.com/hermandoronin/NautilusQuant},
  note   = {Includes NQX-Core pre-silicon emulator (nqx-core/, MIT)}
}
```

Machine-readable: [`nqx-core/CITATION.cff`](nqx-core/CITATION.cff).

License: MIT — see [`LICENSE`](LICENSE).

---

<div align="center">

**φ = 1.618 033 988 749 894 848 …**

*A 1.9 KB constant instead of a 32 KB random matrix —*<br>
*at the price of 8 % reconstruction error.*

</div>
