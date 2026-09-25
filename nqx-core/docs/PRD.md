# NQX-Core — Product Requirements Document

## 1. Vision in one line

A special-purpose processor (NPU/ASIC class) for the five-stage NautilusQuant
pipeline: deterministic KV-cache quantization via the golden ratio φ.
NQX-Core = software emulator → SystemVerilog RTL → FPGA prototype → ASIC tape-out.

## 2. Current state

| Artifact | State |
|---|---|
| Software emulator `nqx/` (16 modules, pure NumPy) | ✅ works, 246 tests pass, 1 skipped |
| NQ-ISA v2 (24 opcodes, assembler, disassembler) | ✅ |
| NQ-ASM encode/decode programs | ✅ |
| Acceptance: orthogonality 1.6e-7, roundtrip 9.6e-8 | ✅ |
| Matches upstream NautilusQuant math within 1e-4 | ✅ (`tests/test_vs_reference.py`) |
| HTTP service (FastAPI, CPU + GPU backends) | ✅ |
| Docker (CPU + CUDA images) + rented-GPU instructions | ✅ |
| CLI launchers (`nqx-claude`, `nqx-deepseek`, `nqx-audit`, …) | ✅ |
| MXFP4 backend (Concept 3 from upstream) | ✅ emulated in `nqx/mx_unit.py` |
| Sub-bit ISA extension (Concept 4) | ✅ emulated in `nqx/subbit_unit.py` |
| CI (GitHub Actions) | ✅ `.github/workflows/nqx-core-ci.yml` |
| **SystemVerilog RTL** (`rtl/`) | 🚧 **skeleton only** — hierarchy and interfaces exist; `polar_unit.sv` and `quant_unit.sv` contain placeholder datapaths (`x^y` / `x+y` instead of CORDIC; truncation instead of Lloyd-Max) |
| Yosys / OpenLane / SymbiYosys flows | 🚧 configuration files only, never run to GDS or to a formal proof |
| **vLLM / HuggingFace integration** | ⏳ TODO |
| **Triton kernel inside `server/`** | ⏳ TODO |
| FPGA bitstream, silicon | ❌ do not exist |

## 3. Roadmap (E1–E6)

| ID | Stage | Artifact | Goal | Status |
|---|---|---|---|---|
| E1 | Software emulator | `nqx/`, `server/` | Cycle-accurate emulation, bit-exact against the reference | ✅ done |
| E2 | RTL bit-exact | `rtl/*.sv` + Verilator testbench | RTL produces the same output as Python for batch=1024, dim=128 | 🚧 skeleton — placeholder datapath, not bit-exact yet |
| E3 | FPGA bring-up | `rtl/build/` + Vivado project | Synthesizable on Alveo U280, ≥100 MHz, throughput ≥10 K vec/s | ⏳ blocked on E2 |
| E4 | LLM stack integration | `integrations/vllm_kvquant.py`, `integrations/hf_kv_hook.py` | Llama-3.1-8B inference with NQX KV-quant without quality loss | ⏳ not started |
| E5 | ASIC floor-plan | `asic/floorplan.md`, `asic/timing.md` | TSMC 7 nm, 50 mm², 1 GHz, ready for tape-out | 🚧 paper study only |
| E6 | PCIe board bring-up | `firmware/`, kernel driver | A real device in a Linux host via `/dev/nqx0` | 🔮 future |

**E2 is explicitly not finished.** The RTL directory contains a synthesizable
module hierarchy with correct interfaces and pipelining, and the build flows to
drive it, but the arithmetic inside `polar_unit.sv` and `quant_unit.sv` is a
placeholder. Any statement that RTL is "shipped" or bit-exact is wrong until
those two datapaths are implemented and `tb_nqx.sv` checks them against the
emulator.

## 4. Architecture parameters (fixed)

| Parameter | Value | Changeable? |
|---|---|---|
| dim (KV vector) | 128 default; supports {16, 32, 64, 128, 256, 512} | yes, via `NQXConfig` |
| Quantization bits | 3 (Lloyd-Max) + 1 (QJL sign) | yes, ISA supports up to 8 |
| φ (golden ratio) | (1+√5)/2 = 1.618… | NO, hard-coded |
| L1 pairs | adjacent, 64 for dim=128 | NO |
| L2 pairs | shifted by 1, 63 for dim=128 | NO |
| L3 pairs | butterfly, stride dim/4 | NO |
| SIMD lanes | 64 (for 64 parallel pairs) | yes, dim=256 ⇒ 128 lanes |
| VRF | 16 × dim × FP32 | yes |
| Pipeline depth | 18 (steady state 1 vec/cycle, modelled) | yes, with optimization |
| ROM-LUT | 10 B per Givens pair → 950 B (dim=64), 1 910 B (dim=128), 15 350 B (dim=1024) | derived from dim |

## 5. Acceptance criteria (checked by CI)

| Check | File | Criterion |
|---|---|---|
| `T^T·T = I` | `tests/test_orthogonality.py` | err < 1e-5 |
| Roundtrip without quantization | `tests/test_orthogonality.py` | RMSE < 1e-5 |
| Rotation vs reference implementation | `tests/test_vs_reference.py` | max abs diff < 1e-4 |
| ISA encode/decode | `tests/test_isa.py` | bit-exact roundtrip of all opcodes |
| Pipeline cycle counter | `tests/test_pipeline.py` | predicted == measured |
| Pack/unpack 3+1 bit | `tests/test_roundtrip.py` | exact inverse |
| Compression ratio | `tests/test_roundtrip.py` | == 4.00× exactly |
| `tests/` suite (run by CI) | `pytest tests -q` | 241 collected, 0 failures, 1 skip |
| Whole repo (adds `sdk/`, `firmware/`) | `pytest -q` from `nqx-core/` | 247 collected, 0 failures, 1 skip |

**After any PR or edit the whole test suite must pass.** No xfail.

## 6. Out of scope

- ❌ GUI (the web index in the parent repo is enough)
- ❌ Model training / fine-tuning — inference KV-cache only
- ❌ Quantization below 4 effective bits (radius + angle)
- ❌ Non-NVIDIA / non-AMD GPU support in `server/backends.py`
- ❌ Distributed inference (multi-GPU sharding)
- ❌ Visualization / 3D (that lives in the parent repo's `labs/quantsim3d.html`)

## 7. Working rules (for AI agents and humans)

### Language
- Code, comments, docstrings, branch names, commit messages: **English**
- The English documentation set is `README.md`, `docs/PRD.md`, `docs/paper/`
  and the parent repository's `README.md`. Some older documents
  (`docs/architecture.md`, `docs/FINAL_REPORT.md`, `audits/`) are still Russian.

### Code style
- Python ≥ 3.11, type hints required for public API
- NumPy 2.x; PyTorch and Triton are optional dependencies (only in
  `server/backends.py` and the upstream `nautilus_triton.py`)
- **Do not add docstrings/comments unless the task explicitly asks**
- No speculative error handling
- No refactors outside the scope of the task
- No unrequested features

### Tests
- Each new functional unit gets its own `test_<unit>.py`
- If you break an acceptance test, fix it before anything else
- Benchmarks go into `python run.py bench`, not into pytest

### Git
- Branches: `feat/short-name`, `fix/...`, `chore/...`, `rtl/...`
- One logical change = one commit
- No amends, no force-push to main

## 8. Stack

| Layer | Technology |
|---|---|
| Core emulator | Python 3.11+, NumPy 2.x |
| HTTP service | FastAPI, Uvicorn, Pydantic 2.x |
| GPU backend | PyTorch ≥ 2.2, Triton ≥ 2.2 (optional) |
| RTL | SystemVerilog, Verilator (sim), Yosys / OpenLane (synth) |
| Tests | pytest 8.x |
| Containers | Docker, docker-compose |

## 9. Repository layout

```
nqx/                emulator (constants/lut/memory/FU/pipeline/cpu/isa/assembler/energy)
programs/           NQ-ASM programs
tests/              pytest — 46 files, 241 tests (6 more live in sdk/ and firmware/)
server/             HTTP API
deploy/             Docker, GPU host instructions
docs/               architecture.md, PRD.md (this file), paper/
tools/cli/          20 shell launchers
audits/             AI-agent prompts and results (Russian)
bench/              7 benchmark documents
rtl/                SystemVerilog skeleton (E2)          ← placeholder datapath
integrations/       vLLM / HF / llama.cpp notes (E4)     ← TODO
asic/               floor-plan, timing study (E5)        ← paper only
firmware/           boot ROM, driver skeleton (E6)       ← skeleton
```

## 10. Success metrics

| Metric | Goal | Today |
|---|---|---|
| Throughput (RTX 5090) | > 1 M vec/s | not measured |
| Throughput (FPGA Alveo U280) | > 100 K vec/s | not measured |
| Compression | 4.00× | ✅ measured |
| Match with the reference implementation | within 1e-4 | ✅ measured |
| Latency p99 | < 1 ms per 1024-vector batch | not measured |
| Energy per vector on 7 nm ASIC | < 5 nJ | 5.1 nJ from the analytical model in `nqx/energy.py` — no silicon |
| Reconstruction quality vs random rotation | ≤ random | ❌ φ is 7.9 % worse — see `bench/phi_vs_random.md` |

## 11. Links

- Upstream NautilusQuant: https://github.com/hermandoronin/NautilusQuant
- OCP MX standard: https://www.opencompute.org/documents/ocp-mx-spec
- TurboQuant: https://arxiv.org/abs/2504.19874
- KIVI: https://arxiv.org/abs/2402.02750
- QuIP#: https://arxiv.org/abs/2402.04396
