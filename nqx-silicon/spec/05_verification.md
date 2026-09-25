# NQX-S1 — Verification plan and results

**Document status:** Revision 1.0, 2026-09-24. Results below were produced
by the commands in the right-hand column, on the committed sources.

## 1. Strategy

```
 nqx-core FP32 emulator ──(pairs, angles, pack layout)──► float reference ──(fixed-point error)──► bit-accurate model
                                                                                                     │ bit-exact
                                                                     RTL ◄───────── cocotb ──────────┘
                                                                      │  bit-exact, pin level
                                           gate-level netlist ◄───────┤
                                                                      │  same files
                                                      golden vectors ─┴──► silicon bring-up / production test
```

- **Single source of truth.** `model/nqx_s1/` defines every result bit. The
  RTL is compared against it byte for byte, never against a tolerance.
- **Link to the algorithm.** The model is tied to the original algorithm
  through `model/tests/test_model.py` (identical pair order and angles to
  `nqx-core/nqx/lut.py`) and through the float reference (fixed-point error
  quantified in [`04_numerics_results.md`](04_numerics_results.md)).
- **Silicon test.** The pin-level test replays the same golden-vector files
  that the silicon test uses, so bring-up starts from vectors already proven
  on RTL and on the gate-level netlist.

## 2. Test inventory

| Level | Test | What it checks | Command |
|---|---|---|---|
| Model | `model/tests/test_model.py` (23 tests) | Constants, widths, pair tables vs nqx-core for DIM 8…256, CORDIC accuracy bounds, quantizer rounding, circular angle code, encode/decode vs float, round trip, status flags | `make test-model` |
| Unit | `verif/cocotb/test_cordic.py` | 4 254 CORDIC operations: every corner of x, y ∈ {0, ±1, ±max, …}, quadrant-boundary angles, 4 000 random operations with random input gaps; bit-exact vs `cordic.py` | `make test-rtl` |
| Core | `verif/cocotb/test_core.py::csr_and_sync` | CSR map, reset values, scratch, unmapped addresses, SYNC | 〃 |
| Core | `::load_store` | LDV/STV/STVR on edge and random vectors | 〃 |
| Core | `::single_ops` | Every micro-op on its own with full VR readback after each; non-zero INC2 | 〃 |
| Core | `::encode_decode` | 7 edge + 18 KV/random vectors, ENC then DEC, REFINE on and off, counters | 〃 |
| Core | `::errors_and_saturation` | Illegal opcode, illegal layer, W1C, decode and store saturation | 〃 |
| Core | `::cycle_counters` | CYCLES/BUSY_CYCLES monotonic and cover an encode | 〃 |
| Core | `::random_programs` | 8 constrained-random programs × 25 commands: random opcodes including illegal ones, random INC values, random packets, random CTRL/STATUS writes; 10 % input gaps, 30 % output stalls | 〃 |
| Core | `::perf_cycles` | Measures the cycle table in [`02_architecture.md`](02_architecture.md) §5.1 | 〃 |
| Pins | `verif/cocotb/test_top.py::pins_csr`, `::pins_encode_decode` | Full top with the handshake bridge; host model toggles pins with random delays unrelated to the clock | 〃 |
| Pins | `::golden_vectors` | Replays all 9 files in `vectors/` (91 transactions, 36 062 bytes written) from reset through the pins | 〃 |
| Formal | `verif/formal/hsio.sby` | 4-phase handshake: protocol obligations, every byte delivered exactly once and unchanged in both directions, bounded response. **Unbounded proof** (k-induction) | `make formal` |
| Formal | `verif/formal/core.sby` | Core stream valid/ready compliance; idle ⇒ ready. **Unbounded proof**, DIM=128, for both CORDIC builds; reachability covers | 〃 |
| Lint | Verilator `-Wall` | Zero warnings (the waivers are listed in §4) | `make lint` |
| Lint | Yosys 0.33 and 0.69 `check -assert`; Icarus `-g2012` | Elaboration, no multiple drivers or combinational loops | `make yosys-check` |
| Config | DIM = 16 and 32 | Complete core + pin regression with regenerated parameters | `NQX_DIM=16 NQX_RTL_DIR=… pytest verif/test_rtl.py` |
| Config | ITER_CORDIC = 1 (iterative CORDIC build) | Complete core + pin regression including all golden vectors: bit-identical to the pipelined build | `NQX_ITER=1 pytest verif/test_rtl.py -k "core or top"` |
| Chip, IHP | `test_chip_ihp` | `chip_top` with the IHP `sg13g2_io` pad models + RTL core: pin tests and golden vectors through the pad cells | `NQX_IHP_PDK=… pytest verif/test_rtl.py -k chip_ihp` |
| Gate level | `test_chip_ihp_gate_level`: pin tests + golden vectors on the post-route IHP netlist with the foundry cell models | See [`06_physical_design.md`](06_physical_design.md) §6 | `NQX_IHP_PDK=… pytest verif/test_rtl.py -k gate_level` |
| Chip, GF180 | `flow/wafer-space-gf180/cocotb/chip_top_tb.py` | Pin tests + golden vectors through the wafer.space pad ring and `chip_core` mapping | `make sim` in that directory |

## 3. Test quality checks

- **Mutation check.** Changing the rounding constant of the angle code
  (`1 << (ZW-4)` → `1 << (ZW-5)`) made `single_ops`, `encode_decode` and
  `random_programs` fail at once. The test suite detects single-constant
  arithmetic changes.
- **Cross-simulator.** Model tests run on CPython. RTL tests run on
  Icarus 12 (4-state, so X-propagation is visible). The CORDIC and core
  regressions also pass on Verilator 5.052 (`NQX_SIM=verilator`).
- **Reset.** Every flip-flop resets, so no test depends on initial X values.

## 4. Lint waivers

| File | Waiver | Reason |
|---|---|---|
| `nqx_s1_cordic.sv` | UNUSEDSIGNAL `px`, `py` low bits | Rounded off by design |
| `nqx_s1_core.sv` | UNUSEDSIGNAL `wshift[7:0]`, `st_shr` high bits | Shift-register byte that is overwritten; integer part beyond int16 is checked for saturation |
| `nqx_s1_qdec.sv` | UNUSEDSIGNAL `prod` low bits | Rounded off by design |
| `nqx_s1_div28.sv` | UNUSEDSIGNAL `quo` top bits | Always zero because rng < 2²³ |
| `nqx_s1_top.sv` | SYNCASYNCNET `rst_sync` | Standard reset synchronizer |

## 5. Not verified (open items)

| Item | Why | Plan |
|---|---|---|
| Timing-annotated (SDF) gate-level simulation | Static timing is the sign-off method; SDF simulation of the asynchronous interface adds little | Optional, with OpenSTA-written SDF |
| Metastability behaviour in silicon | Cannot be simulated; covered by design (two-flop synchronizers, MTBF analysis in [`06_physical_design.md`](06_physical_design.md)) | Stress test with an asynchronous host during bring-up |
| Fault coverage (stuck-at) | No scan; functional test only | Estimate in [`07_dft_and_test.md`](07_dft_and_test.md) |
| Real LLM KV-cache data | Synthetic data only | Run `tools/bench_numerics.py` style evaluation on dumped KV tensors before an S2 tape-out |
| Power | Static estimate from OpenROAD only | Measure on silicon |
