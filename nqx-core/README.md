<div align="center">

# 🐚 NQX-Core

**Pre-silicon emulator and chip development kit for the NautilusQuant
golden-ratio KV-cache accelerator.**

[![Tests](https://img.shields.io/badge/tests-246_passing%2C_1_skipped-brightgreen?style=for-the-badge)](tests)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](../LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/numpy-2.x-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Status](https://img.shields.io/badge/status-software%20only-orange?style=for-the-badge)](docs/PRD.md)

**[Quick start](#quick-start)** ·
**[What is real](#what-is-real-and-what-is-not)** ·
**[Numbers](#numbers)** ·
**[Architecture](#architecture-one-picture)** ·
**[Paper draft](docs/paper/)** ·
**[Roadmap](docs/PRD.md#3-roadmap-e1e6)**

</div>

---

## TL;DR

NQX-Core is a software emulator of a special-purpose processor for
[NautilusQuant](https://github.com/hermandoronin/NautilusQuant) — 4× deterministic
KV-cache compression built on golden-angle (φ) Givens rotations. The point of the
design is that the rotation collapses into a **1 910-byte ROM at dim=128**
instead of the 32 KB random matrix a TurboQuant-style rotation needs (2 MB at
dim=1024), and that the output is bit-identical on every run.

The package contains: ISA + assembler + disassembler, functional-unit models,
a **SystemVerilog RTL skeleton**, Verilator / Yosys / OpenLane build files
(a path towards a Skywater MPW shuttle), ASIC floorplan and timing notes,
a FastAPI HTTP server with monitoring and chaos tests, a demo runner, and
7 benchmark documents with numbers against a TurboQuant emulation.

## What is real, and what is not

| | Status |
|---|---|
| Cycle-accurate NumPy emulator, ISA, assembler/disassembler | **Real, tested** (246 passing tests) |
| Functional-unit math (Givens, polar, Lloyd-Max, QJL, pack, MX, sub-bit) | **Real, tested** against the upstream reference |
| FastAPI service, metrics, chaos tests, Docker images | **Real** |
| Benchmarks in `bench/`, demos in `demos/` | **Real**, all on synthetic KV-like data |
| SystemVerilog RTL | **Skeleton.** `polar_unit.sv` computes `x^y` / `x+y` where CORDIC belongs; `quant_unit.sv` truncates instead of applying Lloyd-Max. Hierarchy, interfaces and pipelining are real; the arithmetic is a placeholder. The implemented and verified fixed-point RTL, with full-chip layout, is **NQX-S1** in [`../nqx-silicon/`](../nqx-silicon/). |
| Yosys / OpenLane / SymbiYosys flows | Configuration files exist; never run to GDS or to a formal proof |
| ASIC floorplan, timing, tape-out checklist in `asic/` | Paper study, no PDK run behind it |
| Cycle counts and energy figures | **Analytical model** in `nqx/pipeline.py` + `nqx/energy.py`, not measurement |
| FPGA bitstream, silicon | **Do not exist** |

## Why this design

| | TurboQuant (Google ICLR 2026) | **NQX / NautilusQuant** |
|---|---|---|
| **Approach** | random rotation matrix + PRNG | **deterministic Givens × φ** |
| **Rotation state (dim=128)** | 32 KB FP16 matrix | **1 910 B ROM** |
| **Rotation state (dim=1024)** | 2 MB FP16 matrix | **15 350 B ROM** (137× smaller) |
| **Determinism** | seed-dependent | **bit-identical always** |
| **PRNG cost** | proportional to `dim²` per matrix | **0** (precomputed constants) |
| **Hardware fit** | dense random matmul | **static dataflow**, 1:1 with the pipeline |
| **Compression ratio** | 4× | **4×** (equal) |
| **Reconstruction RMSE** | **better by 7.9 %** | worse — see [`bench/phi_vs_random.md`](bench/phi_vs_random.md) |

The last row is the honest headline: on quality, the golden angle **loses** to a
random orthogonal matrix on synthetic outlier-heavy data. What it buys instead is
a 17–137× smaller rotation state, zero runtime state and bit-exact
reproducibility. Whether that trade is worth ~8 % extra RMSE depends on the
deployment, and this repo does not pretend to answer that for you.

## Quick start

```bash
git clone https://github.com/hermandoronin/NautilusQuant
cd NautilusQuant/nqx-core
pip install -r requirements.txt

# Run the whole suite (247 collected: 246 pass, 1 skip)
# CI runs `pytest tests -q`, which collects 241 (sdk/ and firmware/ excluded)
python -m pytest -q

# Verify acceptance criteria (orthogonality, roundtrip, match vs reference)
python run.py verify --dim 128

# Benchmark cycles + throughput + energy (emulator model)
python run.py bench --vectors 4096

# Run the encode pipeline as an NQ-ASM program
python run.py run programs/encode_dim128.nqasm --vectors 1000

# Show the demo (TurboQuant vs NautilusQuant side-by-side)
python demos/run_demo.py
```

### Run as an HTTP API

```bash
# Local CPU
docker compose --profile cpu up
curl http://localhost:8000/health

# Rented GPU host
bash deploy/quickstart-vastai.sh
```

See [`deploy/vastai.md`](deploy/vastai.md) for the full deployment guide.

## Architecture (one picture)

```
        ┌────────── HBM (off-chip, FP16) ──────────┐
        │                                          │
        v                                          ^
   ┌────────┐                                ┌──────────┐
   │  DMA   │--> SRAM_in (24KB) ------> ... -│   PACK   │
   └────────┘                                │   3+1bit │
                                             └──────────┘
                                                  ^
   SRAM_in --> [ VRF FP32, 16 × 128 elem ]        │
                       │                          │
        ┌──────────────┴────────────────┐    ┌────┴──────┐
        v                               v    │   QJL     │
  ┌──────────┐  ┌──────────┐  ┌──────────┐   │ sign+corr │
  │ GU-L1    │->│ GU-L2    │->│ GU-L3    │   └─────▲─────┘
  │ 64 lanes │  │ 63 lanes │  │ ~32 lanes│         │
  │ adj pair │  │ shifted  │  │ butterfly│         │
  └──────────┘  └──────────┘  └──────────┘         │
                       │                           │
                       v                           │
                 ┌──────────┐    ┌──────────┐      │
                 │ POLAR    │--->│  QUANT   │------┘
                 │ sqrt+at2 │    │ Lloyd-Max│
                 │ 64 lanes │    │ 3-bit    │
                 └──────────┘    └──────────┘
                       ^
                 ┌──────┴───┐
                 │ ROM LUT  │  golden angle cos/sin, 1 910 B
                 │ 191 pair │
                 └──────────┘
```

Pipeline: **18 cycles** depth, **1 vec / cycle** steady-state throughput — both
from the emulator's cycle model. Detailed spec: [`docs/architecture.md`](docs/architecture.md)
(written in Russian).

## Numbers

`measured` = produced by running code in this repo. `model` = produced by the
analytical cycle/energy model in `nqx/pipeline.py` and `nqx/energy.py`; no
silicon, FPGA or GPU wall-clock measurement backs those rows.

| Metric | Value | Kind | Source |
|---|---|---|---|
| Orthogonality `T^T·T = I` (dim=128) | err ≤ **1.6e-7** | measured | `tests/test_orthogonality.py` |
| Roundtrip without quantization (RMSE) | **9.6e-8** | measured | `tests/test_orthogonality.py` |
| Matches upstream NautilusQuant math | within **1e-4** max abs diff | measured | `tests/test_vs_reference.py` |
| Compression ratio | exactly **4.00×** | measured | `tests/test_roundtrip.py` |
| ROM-LUT size (dim=64 / 128 / 1024) | **950 / 1 910 / 15 350 bytes** | measured | `nqx/lut.py`, [`bench/lut_budget.md`](bench/lut_budget.md) |
| Determinism (100 runs, same input) | **100 % bit-identical** | measured | [`bench/determinism.md`](bench/determinism.md) |
| φ vs random rotation, reconstruction | **φ 7.9 % worse** (0.1429 vs 0.1325 RMSE) | measured | [`bench/phi_vs_random.md`](bench/phi_vs_random.md) |
| φ vs random rotation, compute cycles | **88.9 % fewer** | model | [`bench/phi_vs_random.md`](bench/phi_vs_random.md) |
| Number of NQ-ISA opcodes | **24** | measured | `nqx/isa.py` |
| Tests | **246 passed, 1 skipped** (247 collected) | measured | `pytest -q` |
| Throughput steady-state | **1 vec / cycle** | model | emulator cycle counter |
| Pipeline depth | **18 cycles** | model | `docs/architecture.md` |
| Energy per encode-vec (TSMC 7 nm assumptions) | ≈ **5.1 nJ** | model | `nqx/energy.py` |

## Roadmap

```
E1 ✅ Software emulator (NQX-Core)                    ← DONE
E2 🚧 RTL skeleton — placeholder datapath in
      polar_unit.sv / quant_unit.sv                   ← IN PROGRESS
E3 ⏳ FPGA bring-up (Alveo U280 / AWS F1)             ← blocked on E2
E4 ⏳ vLLM / HF / llama.cpp integration               ← not started
E5 ⏳ Skywater 130nm MPW via Efabless                 ← planned
E6 ⏳ TSMC 12nm / 7nm tape-out                        ← future
```

Details: [`docs/PRD.md`](docs/PRD.md), [`asic/floorplan.md`](asic/floorplan.md),
[`asic/timing.md`](asic/timing.md), [`asic/tapeout_checklist.md`](asic/tapeout_checklist.md).

## Project layout

```
nqx/                  Software emulator core (16 Python modules)
  constants.py          φ, golden angle, NQXConfig
  lut.py                ROM-LUT generator
  memory.py             HBM, SRAM, register files
  functional_units.py   GivensUnit, PolarUnit, QuantUnit, QJLUnit, PackUnit, AttentionUnit
  mx_unit.py            OCP MX-format quantization
  subbit_unit.py        Sub-1-bit quantization experiments
  pipeline.py           Cycle/energy accounting model
  isa.py                24 opcodes: definitions, encoding, decoding
  assembler.py          NQ-ASM → bytecode
  disassembler.py       bytecode → NQ-ASM
  cpu.py                NQXCore: emulator orchestrator
  energy.py             Energy model + random-rotation baseline
  coverage.py           Opcode/state coverage tracking
  counters.py           Performance counters (MMIO-mapped)
  jtag.py               IEEE 1149.1 TAP controller model

rtl/                  SystemVerilog skeleton (5 design modules + testbench)
  givens_lane.sv        One Givens rotation lane (4 mul + 2 add)
  golden_rom.sv         ROM with cos/sin LUT
  polar_unit.sv         CORDIC sqrt + atan2 — PLACEHOLDER datapath
  quant_unit.sv         Lloyd-Max + min/max reduce tree — PLACEHOLDER mapping
  nqx_top.sv            Top-level wrapper
  tb_nqx.sv             Verilator testbench
  formal/               SymbiYosys harness (never run to a proof)
  synth/                Yosys synthesis flow
  openlane/             OpenLane config for Skywater MPW

asic/                 Paper design docs (floorplan, timing, tape-out checklist)
programs/             NQ-ASM example programs
tests/                46 test files, 241 tests (+6 in sdk/ and firmware/)
server/               FastAPI HTTP service
deploy/               Docker, GPU host scripts, smoke tests
demos/                TurboQuant comparison, pitch deck, scaling demo, notebooks
bench/                7 benchmark documents (angular_uniformity, phi_vs_random, …)
docs/                 PRD, architecture, paper draft
sdk/                  libnqx C ABI, install scripts
firmware/             Boot ROM, Linux driver skeleton
integrations/         vLLM/HF/llama.cpp adapter notes (not implemented)
tools/                gen_rom.py, rig.py, dump_for_rtl.py, debug/
tools/cli/            20 shell launchers
audits/               AI-agent prompts used while building this (Russian)
```

## Compare against TurboQuant

```bash
python demos/turboquant_emul.py             # TurboQuant random-rotation emulation
python demos/run_demo.py                    # Full comparison: cycles, energy, RMSE, determinism
cat demos/side_by_side.md                   # Main comparison table
cat bench/phi_vs_random.md                  # φ vs random head-to-head (φ loses on RMSE)
```

## How to cite

```bibtex
@misc{nqxcore2026,
  title  = {NQX-Core: Pre-silicon emulator for golden-ratio KV-cache quantization},
  author = {Doronin, Herman},
  year   = {2026},
  url    = {https://github.com/hermandoronin/NautilusQuant/tree/main/nqx-core}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable form.

## Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md). The AI-agent task lists
used to build this project live in [`audits/prompts/`](audits/prompts/) (Russian)
for transparency.

## License

MIT — see [`LICENSE`](../LICENSE).

## Acknowledgments

- Theory: Hermann Weyl's [equidistribution theorem](https://en.wikipedia.org/wiki/Equidistribution_theorem) (1916).
- [TurboQuant](https://arxiv.org/abs/2504.19874) (Google ICLR 2026) established that
  rotation before polar quantization works; this project asks what happens when the
  rotation is a constant.
- Built with **Claude** and **DeepSeek** as coding agents; the development log is in `audits/`.

---

<div align="center">

**Status**: software only. RTL skeleton with placeholder arithmetic.
No FPGA bring-up, no silicon.

[Open an issue](../../issues) · [Read the pitch](demos/pitch.md)

</div>
