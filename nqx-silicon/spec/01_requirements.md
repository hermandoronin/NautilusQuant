# NQX-S1 — Product requirements

**Document status:** preliminary. Revision 1.0, 2026-09-24.

## 1. Purpose

NQX-S1 is the first silicon implementation of the NautilusQuant KV-cache
compression pipeline. It takes a 128-element int16 vector and produces a
70-byte packet: golden-angle Givens rotation → polar transform → 3-bit
quantization → 1-bit residual → packing. It also runs the inverse path.

It is a test chip. Its jobs are to:

1. prove that the golden-angle rotation can be generated in hardware from
   phase increments alone, with no angle ROM;
2. provide bit-exact silicon that can be measured against the software model;
3. validate the complete open-source flow (RTL → verified GDSII → fab)
   before a larger, bandwidth-oriented design (NQX-S2) is attempted.

NQX-S1 is **not** an LLM accelerator. Throughput is limited by its 8-bit
host interface. A production part would need a wide memory interface
(HBM/LPDDR/PCIe) and many parallel lanes; see
[`02_architecture.md`](02_architecture.md) §9.

## 2. Functional requirements

| ID | Requirement | Verified by |
|---|---|---|
| F1 | Load a DIM=128 vector of signed 16-bit integers | cocotb `load_store` |
| F2 | Apply Givens layers L1, L2 and L3 of the NautilusQuant reference, forward and inverse, with the reference pair order and angles | model tests vs `nqx-core/nqx/lut.py`; cocotb `single_ops` |
| F3 | Generate the rotation angles from 32-bit phase increments; after reset the increments are the golden-ratio values | `test_phase_increments_are_golden_ratio_constants` |
| F4 | Let software change the phase increments without new silicon | CSR INC0..2; cocotb `random_programs` |
| F5 | Pairwise polar transform (r, θ) and its inverse | cocotb `single_ops` |
| F6 | 3-bit radius code over the per-vector [r_min, r_max] range; 3-bit circular angle code; 1 residual sign bit per value | model `test_radius_code_is_rounding`, `test_angle_code_is_circular` |
| F7 | Pack to 4 bits per value in the nqx-core `PackUnit` layout, plus a 6-byte header | model `test_encode_decode_matches_float_reference` |
| F8 | Decode a packet back to int16. The residual-sign refinement can be switched off (CTRL.REFINE) to reproduce the nqx-core decoder | cocotb `encode_decode` |
| F9 | Fused ENC and DEC macro commands | cocotb `encode_decode`, `pins_encode_decode` |
| F10 | Identification, status and performance counters readable by the host | cocotb `csr_and_sync`, `cycle_counters` |
| F11 | Detect illegal commands and arithmetic saturation, reported in sticky status bits | cocotb `errors_and_saturation` |

## 3. Performance requirements

| ID | Requirement | Target | Status |
|---|---|---|---|
| P1 | Core clock | 50 MHz at the slow corner (1.08 V, 125 °C, IHP) | see [`06_physical_design.md`](06_physical_design.md) |
| P2 | Encode latency, core level | ≤ 1000 cycles per vector with a 1 byte/cycle host | 720 cycles, measured in RTL simulation |
| P3 | Decode latency, core level | ≤ 1000 cycles | 690 cycles, measured |
| P4 | Rotation accuracy vs float64 | relative error ≤ 1e-4 | 3.0e-5 max ([`04_numerics_results.md`](04_numerics_results.md)) |
| P5 | Rotation round trip at int16 precision | ≤ 1 LSB | 1 LSB max, measured |
| P6 | Reconstruction error vs the float version of the same algorithm | within 1 % | 0.1464 vs 0.1471 relative RMSE |

## 4. Interface requirements

| ID | Requirement |
|---|---|
| I1 | 8-bit input bus and 8-bit output bus, each with an asynchronous 4-phase request/acknowledge handshake, so that a microcontroller can bit-bang it at any speed |
| I2 | No timing relationship between the host and the chip clock; synchronizers are required on every handshake input |
| I3 | Single clock input, active-low asynchronous reset with synchronous de-assertion |
| I4 | Few enough pads to fit an MPW die (23 signal pads) |

## 5. Implementation requirements

| ID | Requirement |
|---|---|
| M1 | Fully synthesizable Verilog-2005/SystemVerilog subset, readable by Yosys, Verilator and Icarus without vendor tools |
| M2 | Clean under `verilator --lint-only -Wall`; no latches; one clock domain |
| M3 | Implemented with the open-source flow (LibreLane / OpenROAD / Yosys / Magic / KLayout / Netgen) on an open PDK |
| M4 | Primary process: IHP SG13G2 130 nm. Secondary: GlobalFoundries GF180MCU through wafer.space |
| M5 | Sign-off: DRC clean with the foundry KLayout deck, LVS clean, antenna clean, density rules met, seal ring present, no setup or hold violations at any corner |
| M6 | All flip-flops reset to a known value (deterministic power-up for test) |
| M7 | RTL bit-exact with the Python model on every command |

## 6. Non-goals

- A general-purpose CPU or instruction fetch. The chip executes host
  commands; the command opcodes follow NQ-ISA where they overlap.
- On-chip memory for more than one vector, DMA, HBM or PCIe.
- Scan-chain DFT. Production test is functional, through the command
  interface ([`07_dft_and_test.md`](07_dft_and_test.md)).
- Analog or mixed-signal blocks, PLL, or on-chip clock generation.
- Quantum computing. "Quantization" means reducing the number of bits per
  value.

## 7. Deviations from the nqx-core emulator

These are deliberate. Each is justified in
[`02_architecture.md`](02_architecture.md) §8.

1. Arithmetic is fixed point instead of FP32. The rounding is specified
   bit-exactly in `model/nqx_s1/`.
2. Input is int16. The host supplies a power-of-two scale per vector (block
   floating point); the chip never sees it.
3. The radius range is per vector rather than per feature over a batch.
   Single-token decode has no batch to take a range over.
4. The angle code is circular: 8 sectors of 45°. The reference maps
   [−π, π] linearly, which wastes one of its 8 levels.
5. The residual sign bit is used on decode (±¼ step). The reference stores
   it but ignores it; CTRL.REFINE = 0 restores the reference behaviour.
6. Layer 3 is bypassed when its phase increment is zero, which is the
   reference value (see [`11_algorithm_findings.md`](11_algorithm_findings.md)).
