# NQX-S1 — Architecture and micro-architecture

**Document status:** preliminary. Revision 1.0, 2026-09-24.
RTL: `rtl/`. Golden model: `model/nqx_s1/`. If this document and the model
ever disagree, the model is the specification and the document has a bug.

## 1. Block diagram

```
                 pads                                   nqx_s1_top
   in_bus[7:0] ────────┐
   in_req  ──[sync]──┐ │   ┌──────────────┐  byte   ┌─────────────────────────────────────────┐
   in_ack  ◄─────────┴─┴──►│  nqx_s1_hsio │ ──────► │ nqx_s1_core                              │
                           │  4-phase <-> │         │                                          │
   out_bus[7:0] ◄──────────│  valid/ready │ ◄────── │  command decoder + micro-op sequencer    │
   out_req ◄───────────────│              │  byte   │  CSRs, perf counters                     │
   out_ack ──[sync]───────►└──────────────┘         │                                          │
                                                    │   ┌───────────┐ 2R2W  ┌─────────────────┐ │
   clk ────────────────────────────────────────────►│   │ VR        │──────►│ CORDIC          │ │
   rst_n ──[async assert / sync de-assert]─────────►│   │ 128 x 24b │◄──────│ 18 stages       │ │
   busy ◄───────────────────────────────────────────│   │ flip-flops│       │ rotate / vector │ │
                                                    │   └───────────┘       └────────▲────────┘ │
                                                    │        │  ▲                     │ angle   │
                                                    │        ▼  │            ┌────────┴──────┐  │
                                                    │   ┌──────────────┐     │ phase accum.  │  │
                                                    │   │ quantizer /  │     │ 32 b, INC0..2 │  │
                                                    │   │ dequantizer  │     └───────────────┘  │
                                                    │   │ ÷28 serial   │                         │
                                                    │   └──────────────┘                         │
                                                    └─────────────────────────────────────────┘
```

| Block | RTL | Function |
|---|---|---|
| Handshake I/O | `nqx_s1_hsio.sv` | 2-flop synchronizers; converts the asynchronous 4-phase pin protocol to a synchronous byte stream |
| Core | `nqx_s1_core.sv` | Command decoder, micro-op sequencer, CSRs, write-back muxing |
| Vector register (VR) | `nqx_s1_rf.sv` | 128 × 24-bit flip-flops, 2 read + 2 write ports |
| CORDIC | `nqx_s1_cordic.sv` | Pipelined CORDIC, 18 iterations, rotation and vectoring modes, gain compensation |
| Radius code | `nqx_s1_rcode.sv`, `nqx_s1_rsign.sv` | 3-bit radius code by exact integer comparison, plus residual sign |
| Angle code | `nqx_s1_tcode.sv` | 3-bit circular angle code plus residual sign |
| Dequantizer | `nqx_s1_qdec.sv`, `nqx_s1_div28.sv` | Byte → (r̂, θ̂); serial divider for the radius step |
| Top | `nqx_s1_top.sv` | Reset synchronizer, I/O + core |

## 2. Data formats

| Quantity | Format | Width | Notes |
|---|---|---|---|
| Input element | signed integer | 16 | Host applies a per-vector power-of-two scale (block floating point) |
| VR word | signed fixed point, Q20.4 | DW = 24 | 20 integer bits hold ‖x‖ ≤ √128 · 2¹⁵ < 2¹⁹ with margin |
| CORDIC datapath | signed | CW = 28 | DW + 2 guard fraction bits + 2 headroom bits; never overflows for any VR content |
| Angle | binary angle, full turn = 2²⁰ | ZW = 20 | Modulo-2π wrap is plain integer overflow |
| Phase accumulator | binary angle, full turn = 2³² | 32 | Rounded to 20 bits before the CORDIC |
| Radius after POLAR | unsigned in a VR word | 24 | Always ≥ 0 |
| θ after POLAR | 20-bit angle, sign-extended into a VR word | 24 | |
| Packed value | {s, q[2:0]} | 4 | Two values per byte, radius in the low nibble |

All widths come from `model/nqx_s1/params.py`. `tools/gen_params.py`
emits them as `rtl/nqx_s1_params.vh`, and CI fails if the generated file is
out of date.

## 3. Algorithm mapping

### 3.1 Givens layers

The pair order and angles are exactly those of `nqx-core/nqx/lut.py`
(asserted by `model/tests/test_model.py`):

| Layer | Pairs (i, j) | Pair count (DIM=128) | Angle of pair with loop index k |
|---|---|---|---|
| L1 | (2k, 2k+1), k = 0..63 | 64 | (k+1) · 2π/φ² |
| L2 | (2k+1, 2k+2), k = 0..62 | 63 | (k+1) · 2π/φ |
| L3 | (k, k+32), k ∈ [0,32) ∪ [64,96) | 64 | (k+1) · 2π ≡ 0 |

In binary angles each layer is an arithmetic progression, so a single
accumulator generates it. The accumulator starts at INC and adds INC once
per loop index k:

| Layer | INC (reset value) | Meaning |
|---|---|---|
| L1 | `0x61C88647` | round(2³²/φ²) |
| L2 | `0x9E3779B9` | round(2³²/φ): Knuth's Fibonacci-hashing constant |
| L3 | `0x00000000` | 2π·(k+1) is a whole number of turns |

**No angle ROM exists.** The rotation state of the whole transform is these
three 32-bit registers (12 bytes). nqx-core's 1 910-byte LUT, and the 32 KB a
random rotation would need, are both replaced by the accumulator. The angle
error from rounding INC is below 5·10⁻⁸ rad for every pair.

When the selected INC is zero the sequencer skips the layer entirely: a
rotation by 0 is the identity, and skipping it avoids 64 needless CORDIC
passes and their rounding.

Inverse layers use z = −round(phase). The CORDIC then takes exactly the
opposite micro-rotation decisions, so a forward/inverse pair cancels to
within datapath rounding. That is why the round trip is ≤ 1 LSB even though
the absolute rotation error is 3·10⁻⁵.

### 3.2 Polar transform

POLAR runs the CORDIC in vectoring mode on the pairs (2k, 2k+1).
r = K⁻¹·x_N goes back to VR[2k] and θ = z_N to VR[2k+1]. While the results
are written back, the running minimum and maximum of r are captured in the
RMIN/RMAX registers. The quantizer needs them, and so no second pass over
the data is required.

IPOLAR runs the CORDIC in rotation mode on (r, 0) with z = θ.

### 3.3 Quantization (QUANT)

For each pair k, with R = RMAX − RMIN and d = r − RMIN:

- radius code q_r = #{m ∈ 1..7 : 14·d > (2m−1)·R}, which is round(7d/R)
  computed with exact integer comparisons: no divider, no reciprocal;
- radius residual s_r = (7·d ≥ q_r·R);
- angle code q_θ = round(θ · 8 / turn) mod 8, the top 3 bits of θ after
  adding half a sector;
- angle residual s_θ = sign of (θ − q_θ · 45°), taken as a wrapped 20-bit
  angle.

The output packet is RMIN (3 bytes), RMAX (3 bytes), then 64 bytes of
`{s_θ, q_θ, s_r, q_r}`. Both header fields are little-endian.

### 3.4 Dequantization (DEQUANT)

The header gives RMIN and RMAX, each clamped to 2²³−1. A serial divider
computes step = ⌊R · 2⁸ / 28⌋, a quarter of a radius step, in 32 cycles.
Each payload byte then decodes to:

- r̂ = RMIN + round(n · step / 2⁸), with n = 4q_r ± 1 (refinement on) or
  4q_r (off);
- θ̂ = q_θ · 45° ± 11.25° (refinement on) or q_θ · 45° (off).

## 4. CORDIC

Two implementations exist, selected by the `ITER_CORDIC` parameter of
`nqx_s1_top`. Their results are bit-identical; the sequencer waits for the
CORDIC's `in_ready`:

| | `nqx_s1_cordic.sv` (ITER_CORDIC=0) | `nqx_s1_cordic_iter.sv` (ITER_CORDIC=1) |
|---|---|---|
| Structure | 18 pipeline stages | 1 register set, 1 micro-rotation per clock, barrel shifter |
| Throughput | 1 pair / clock | 1 pair / 20 clocks |
| Latency | 20 clocks | 20 clocks |
| Cell area, whole chip core | IHP 0.60 mm², GF180 1.02 mm² | IHP 0.45 mm², GF180 0.77 mm² |

Common to both:

- 18 iterations, one pipeline stage each. There is one input register stage
  (with quadrant pre-rotation) and one output stage (gain compensation and
  saturation). Latency is 20 cycles, and it accepts one pair per cycle.
- **Pre-rotation.** In rotation mode, if the angle lies outside
  [−90°, 90°) the vector is negated and 180° is added to the angle, so the
  18 micro-rotations (±99.9° range) always converge. In vectoring mode,
  x < 0 is negated and the angle accumulator starts at 180°.
- **Micro-rotation i** uses only shifts and adds:
  x ← x ∓ (y >>> i), y ← y ± (x >>> i), z ← z ∓ atan(2⁻ⁱ).
  The atan table has 18 constants of 20 bits.
- **Gain compensation.** One constant multiplication by
  K⁻¹ = 159188 / 2¹⁸ with round-half-up, then saturation to 24 bits. If
  saturation occurs it sets STATUS.SAT; it can happen only for decoded
  packets with out-of-range headers.
- The pipeline carries a 14-bit tag (the destination pair i, j) so that
  write-back needs no separate address queue.

## 5. Sequencer

Each host command is decoded into one micro-op, or into a fixed list of
micro-ops for the macro commands:

| Command | Micro-ops |
|---|---|
| ENC (0x60) | LDV → ROT L1 → ROT L2 → ROT L3 → POLAR → QUANT |
| DEC (0x61) | DEQUANT → IPOLAR → ROT⁻¹ L3 → ROT⁻¹ L2 → ROT⁻¹ L1 → STV |

CORDIC micro-ops (ROT, POLAR, IPOLAR) issue one pair per clock from a pair
counter k:

1. **Issue.** The addresses (i, j) are generated from k and the layer. For
   L3 the bit log2(32) of k gates validity; the phase accumulator still
   steps on invalid k, matching the reference's (k+1) indexing.
2. **Drain.** The sequencer waits until the in-flight counter returns to
   zero. Pairs within one layer are disjoint, so no hazard exists inside a
   layer. The drain handles the dependency between layers.
3. **Write-back.** At the CORDIC output, both VR ports write (i, j). Tags
   travel with the data.

QUANT reads one pair and spends 3 cycles per output byte (read →
radius-code compare → sign/pack). DEQUANT spends 2 cycles per input byte.
Neither is on the critical throughput path; the handshake interface is.

### 5.1 Measured cycle counts (RTL simulation, 1 byte/cycle host)

| Command | Busy cycles | Bytes in | Bytes out |
|---|---|---|---|
| LDV | 258 | 257 | 0 |
| GVNS 0 / GVNS 1 | 88 / 87 | 2 | 0 |
| GVNS 2 (INC2 = 0, bypassed) | 3 | 2 | 0 |
| POLAR / IPOLAR | 87 / 87 | 1 | 0 |
| QUANT | 200 | 1 | 70 |
| DEQUANT | 170 | 71 | 0 |
| STV | 258 | 1 | 256 |
| **ENC** | **720** | 257 | 70 |
| **DEC** | **690** | 71 | 256 |

Source: `verif/cocotb/test_core.py::perf_cycles`. At 50 MHz the core
could encode about 69 000 vectors/s if the host delivered one byte per
clock. Through the 4-phase pins the host-side handshake dominates: about
8 clock cycles per byte plus the host's own latency.

## 6. Handshake interface

```
 host → chip                          chip → host
 in_bus  ==X=====data=====X====        out_bus ==X====data======X=====
 in_req  __/‾‾‾‾‾‾‾‾‾‾‾‾\______        out_req ____/‾‾‾‾‾‾‾‾‾‾\______
 in_ack  ______/‾‾‾‾‾‾‾‾‾‾‾\___        out_ack _______/‾‾‾‾‾‾‾‾‾‾\___
          (1)  (2)      (3)(4)                 (1) (2)       (3)(4)
```

- **Inbound.**
  1. The host sets in_bus, then raises in_req.
  2. The chip samples in_bus two clocks after it sees in_req (through the
     synchronizer) and raises in_ack.
  3. The host drops in_req and may change in_bus.
  4. The chip drops in_ack.
- **Outbound.** The chip drives out_bus one clock before it raises out_req,
  so the data pins are already stable when the host sees the request. The
  host reads out_bus, raises out_ack, and waits for out_req to fall before
  dropping out_ack.

The protocol is delay-insensitive on the host side: any MCU, at any speed,
works. Details and command formats are in
[`03_programming_model.md`](03_programming_model.md).

## 7. Clock, reset, power

- **Clock.** One clock domain, `clk`, with a target of 50 MHz. The only
  asynchronous inputs are in_req, out_ack and rst_n; each goes through two
  flip-flops.
- **Reset.** Asynchronous assertion, synchronous de-assertion (a 2-flop
  reset synchronizer in `nqx_s1_top`). All 5 633 flip-flops have an
  asynchronous reset: the VR clears to 0, the INC registers load the golden
  constants and CTRL.REFINE = 1.
- **Power.**
  - IHP SG13G2: core VDD = 1.2 V, pads IOVDD = 3.3 V.
  - GF180MCU (wafer.space): 5 V cells and pads, as set by the template.

## 8. Design decisions

| # | Decision | Alternatives considered | Reason |
|---|---|---|---|
| D1 | CORDIC with shifts and adds only (one constant multiply) | FP32 multipliers as in nqx-core's `GivensUnit` | A 4-multiplier FP32 Givens lane is about 30× the area of one CORDIC stage at 130 nm. The rotation angles form a phase ramp, which is exactly what CORDIC consumes. |
| D2 | Phase accumulator instead of an angle ROM | 1.9 KB cos/sin ROM (nqx-core) | The golden angle sequence is (k+1)·INC mod 2³². The ROM carries no information beyond one constant per layer. |
| D3 | Programmable INC registers | Hard-wired constants | Costs 96 flip-flops. Allows a different irrational increment, or a fix to layer 3, to be evaluated on real silicon. |
| D4 | One pipelined CORDIC shared by all layers and by POLAR/IPOLAR | 64 parallel lanes (nqx-core `architecture.md`) | IO-bound at 8 bits; parallel lanes would idle 99 % of the time. |
| D5 | Flip-flop vector register | SRAM macro | 2R2W access on arbitrary pairs; fully testable through LDV/STVR; no macro integration risk. |
| D6 | Per-vector radius range | Per-feature range over a batch (nqx-core) | Works for single-token decode, needs no batch buffer, and halves the reconstruction error on KV-like data ([`04_numerics_results.md`](04_numerics_results.md)). |
| D7 | Circular angle quantizer | Linear min/max over [−π, π] | θ is periodic. The linear scheme spends two of 8 levels on the same angle (±π). |
| D8 | Residual sign used on decode | Stored but ignored (nqx-core) | Same 4 bits per value, 1.8× lower error. CTRL.REFINE can switch it off. |
| D9 | Exact integer threshold compare for the radius code | Reciprocal + multiplier | No division at encode time; bit-exact and cheap. |
| D10 | 4-phase asynchronous byte interface | SPI, synchronous parallel bus | Robust for any host MCU and with no clock relationship; standard synchronizer design. |
| D11 | All flip-flops reset | Reset only the control state | The IHP cell library has no flip-flop without reset (yosys tied 4 871 reset pins high), so resetting everything costs no area and makes power-up state deterministic. |
| D12 | Build option `ITER_CORDIC`: one micro-rotation per clock on a single register set | Pipelined only | Bit-identical results. Cell area is 25 % smaller (IHP: 0.60 → 0.45 mm², GF180: 1.02 → 0.77 mm²), and CORDIC throughput is 1/20 of the pipelined version. Used for the smallest wafer.space slot. FEATURES[0] tells the host which build it is talking to. |

## 9. Scaling beyond S1

S1 spends about 60 % of its area on the vector register and 30 % on the
CORDIC. A bandwidth-oriented part (NQX-S2) would:

- replace the byte interface with a 64–128-bit AXI-stream or LPDDR/PCIe
  front end;
- instantiate N CORDIC pipelines (N = 8…64) fed from banked SRAM.
  Layer L1 pairs are adjacent and L2 pairs are shifted by one, so a
  two-bank even/odd organization serves L1 and L2 without conflicts;
- pipeline QUANT to 1 byte/cycle and fuse the radius min/max into POLAR,
  which S1 already does;
- keep the phase-accumulator angle generation, which is independent of N.

The same RTL parameters (DIM, widths) apply. DIM is a generator parameter
(`tools/gen_params.py --dim`), and the model and RTL tests pass for any
power of two ≥ 8.
