# NQX-S1 — NautilusQuant silicon

**A test chip that runs NautilusQuant KV-cache compression in hardware.**
It covers everything from bit-accurate model to sign-off GDSII, using only
open-source tools and open PDKs.

`nqx-core/` is the software emulator of an idealised accelerator. This
directory designs a chip that can be manufactured. It has a model you can
trust bit for bit, synthesizable RTL verified against that model, a full
chip layout for a real foundry process, and the documents and test vectors
a fab and a bring-up engineer need.

## What the chip does

```
int16[128] ──► Givens L1 ─► Givens L2 ─► (L3) ─► polar ─► 3-bit r, 3-bit θ + residual bits ──► 70-byte packet
                 ▲ angles from a 32-bit phase accumulator (0x61C88647, 0x9E3779B9); no ROM
```

- **Rotation without memory.** The golden-angle sequence (k+1)·2π/φ² is a
  phase ramp. A 32-bit accumulator regenerates it, and a shift-and-add
  CORDIC applies it. nqx-core needs a 1.9 KB cos/sin ROM and a random
  rotation needs a 32 KB matrix; this chip needs three 32-bit registers.
- **One pipelined CORDIC.** It serves both the rotations and the polar
  transform. There are no multipliers except one constant gain correction.
- **Bit-exact** with the Python model on every command, from the core down
  to the pins, and on the gate-level netlist.
- **Asynchronous 8-bit handshake bus.** Any microcontroller can drive it.

See [`spec/00_datasheet.md`](spec/00_datasheet.md).

## Status

| Item | State | Evidence |
|---|---|---|
| Bit-accurate model | done | `model/`, 23 unit tests |
| RTL (SystemVerilog, 10 modules) | done, lint-clean (Verilator -Wall, Yosys, Icarus) | `rtl/`, `make lint yosys-check` |
| RTL verification | done: CORDIC unit, core random, pin level, golden vectors | `verif/cocotb`, `make test-rtl` |
| Formal verification | done: unbounded proofs of the handshake bridge and the stream protocol | `verif/formal`, `make formal` |
| Full chip, IHP SG13G2 (130 nm) | GDSII with pad ring, seal ring and fill; sign-off results in `spec/06` | `flow/ihp-sg13g2`, `tapeout/ihp-sg13g2` |
| Full chip, GF180MCU (wafer.space) | ready to build (template port, CI job) | `flow/wafer-space-gf180` |
| Silicon | not yet ordered | see [`spec/08_tapeout_package.md`](spec/08_tapeout_package.md) |

## Specification package

| # | Document |
|---|---|
| 00 | [Datasheet](spec/00_datasheet.md) |
| 01 | [Requirements](spec/01_requirements.md) |
| 02 | [Architecture and micro-architecture](spec/02_architecture.md) |
| 03 | [Programming model: commands, registers, packet format](spec/03_programming_model.md) |
| 04 | [Numerics](spec/04_numerics.md) and [measured results](spec/04_numerics_results.md) |
| 05 | [Verification plan and results](spec/05_verification.md) |
| 06 | [Physical design and sign-off](spec/06_physical_design.md) |
| 07 | [Design for test, production test](spec/07_dft_and_test.md) |
| 08 | [Tape-out package: what goes to the fab, and how to order](spec/08_tapeout_package.md) |
| 09 | [Landscape: Groq, Tenstorrent, Etched, Blackwell NVFP4, TurboQuant …](spec/09_landscape.md) |
| 10 | [Bring-up guide](spec/10_bringup.md) |
| 11 | [Algorithm findings from the hardware mapping](spec/11_algorithm_findings.md) |

A Russian overview for the project owner is in
[`README.ru.md`](README.ru.md).

## Quick start

```bash
cd nqx-silicon
pip install numpy pytest cocotb==2.0.1     # plus iverilog, verilator, yosys
make test-model                            # bit-accurate model
make lint yosys-check                      # RTL static checks
make test-rtl                              # cocotb: unit, core, pins, golden vectors
make formal                                # SymbiYosys (needs sby + yices)
python tools/bench_numerics.py             # regenerate spec/04_numerics_results.md
python tools/gen_vectors.py                # regenerate vectors/
make gds-ihp                               # full chip, IHP SG13G2 (LibreLane 3.0.14)
make gds-gf180                             # full chip, wafer.space GF180MCU
```

GitHub Actions (`.github/workflows/nqx-silicon.yml`) runs the model, lint,
formal, RTL and golden-vector checks on every push. It builds both GDSII
files when started manually (`workflow_dispatch`).

## Layout

```
model/nqx_s1/     params.py (single source of constants), cordic.py, quant.py,
                  core.py (transaction model), host.py (command builders),
                  reference.py (float64 reference)
rtl/              nqx_s1_top, _hsio, _core, _cordic, _rf, _rcode, _rsign,
                  _tcode, _qdec, _div28; generated nqx_s1_params.vh / _atan.vh
verif/cocotb/     test_cordic.py, test_core.py, test_top.py, stimulus.py
verif/formal/     hsio.sby, core.sby + property harnesses
flow/ihp-sg13g2/  chip_top.sv (pads), config.yaml, SDC, PDN, bond-pad macro
flow/wafer-space-gf180/   wafer.space template with the NQX-S1 core
tapeout/          sign-off GDS, netlists, reports, checksums (per process)
vectors/          golden test vectors for silicon (W/R byte files)
host/             MicroPython driver for a Pico, vector runner
tools/            gen_params, gen_vectors, bench_numerics, sweep_widths
spec/             the documents listed above
```

## Relation to the rest of the repository

- `nqx-core/nqx/lut.py` defines the pair order and angles; the model tests
  check that they match exactly.
- `nqx-core/rtl/` is the earlier placeholder RTL skeleton (FP32 lanes,
  unimplemented CORDIC/Lloyd-Max). NQX-S1 replaces it with an implemented,
  verified fixed-point design. The skeleton is kept for history.
- The NQX-S1 opcodes follow NQ-ISA (`nqx-core/nqx/isa.py`) where the
  operation exists there.
