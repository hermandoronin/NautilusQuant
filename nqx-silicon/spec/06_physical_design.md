# NQX-S1 — Physical design and sign-off

**Document status:** Revision 1.1, 2026-09-24 (final run `full5`). The sign-off numbers in §5
are generated into `tapeout/ihp-sg13g2/SIGNOFF.md` by
`tools/collect_tapeout.py`; this document explains them.

## 1. Targets

| | IHP SG13G2 (primary) | wafer.space GF180MCU |
|---|---|---|
| Process | 130 nm SiGe BiCMOS, CMOS part only; 5 thin metals + TopMetal1/2 | 180 nm, 5 metals, 5 V cells (`gf180mcu_fd_sc_mcu7t5v0`) |
| PDK | IHP-Open-PDK `c4b8b4e5` (LibreLane 3.0.14's qualified revision) | gf180mcuD via ciel `f6eeac7d` (wafer.space template pin) |
| Flow | LibreLane 3.0.14, `Chip` flow (`flow/ihp-sg13g2/config.yaml`) | LibreLane (template flake), `Chip` flow (`flow/wafer-space-gf180/librelane/`) |
| Die | 2.000 × 2.000 mm | slot 0p5x0p5: 1.936 × 2.531 mm; slot 0p5x1: 1.936 × 5.122 mm |
| Core area | 1.04 × 1.04 mm (480 µm from the die edge) | 1.052 × 1.647 mm (0p5x0p5) |
| Clock | 50 MHz (20 ns) | 25 MHz (40 ns, template default) |
| CORDIC build | pipelined | iterative in 0p5x0p5, pipelined in 0p5x1 |
| Supplies | VDD 1.2 V core, IOVDD 3.3 V I/O | per template (5 V core cells) |

## 2. Floorplan (IHP)

```
 ┌──────────────────────────── 2000 µm ────────────────────────────┐
 │ seal ring                                                        │
 │   bond pads (70×70 µm, 1 µm overlap on each I/O cell's pad pin)  │
 │   ┌──────── I/O ring: 180 µm deep sg13g2 pad cells ───────────┐ │
 │   │ N: IOVSS IOVDD out_ack out_req in_ack in_req VSS VDD       │ │
 │ W │   ┌───── core ring (TopMetal1/2, 10 µm) ────────────────┐   │ E │
 │   │   │                                                     │   │ out_bus
 │in_│   │     core 1040 × 1040 µm, ≈ 63 % standard cells      │   │ [0..7]
 │bus│   │     PDN straps TopMetal1 (V) / TopMetal2 (H),       │   │   │
 │[7:│   │     75.6 µm pitch; Metal1 rails                     │   │   │
 │ 0]│   └─────────────────────────────────────────────────────┘   │   │
 │   │ S: VDD VSS clk rst_n busy IOVDD IOVSS                     │   │
 │   └───────────────────────────────────────────────────────────┘   │
 └──────────────────────────────────────────────────────────────────┘
```

- **Pads.** 31 foundry I/O cells (`sg13g2_io`): 12 × `IOPadIn`,
  11 × `IOPadOut4mA`, and 2 each of `IOPadVdd`, `IOPadVss`, `IOPadIOVdd`,
  `IOPadIOVss`. Corner cells and I/O fillers close the ring. The I/O supply
  rails connect by abutment.
- **Bond pads.** `bondpad_70x70` is generated from IHP's `SG13_dev`
  bondpad PCell (`flow/ihp-sg13g2/macros/`: GDS plus a hand-written LEF,
  class COVER, the whole Metal3..TopMetal2 stack declared as pin `pad`).
  The LibreLane pad-ring step places it at offset (5, −69) µm, overlapping
  the I/O cell's pad pin by 1 µm. At the PDK default (5, −70) the two only
  abut, and Magic's extraction from the LEF abstracts left bond pad and
  pad cell on different nets (13 unmatched nets in LVS). The passivation
  opening stays outside the I/O cell.
- **Core size.** The core is deliberately smaller than the space inside the
  pad ring. IHP's density rule MxFil.h needs ≥ 25 % Metal2..5 (drawn plus
  fill) in every 800 × 800 µm window; with a 1.2 mm core the central
  windows held only routed cells and Metal2 reached 16–22 %. With 1.04 mm
  every window also contains fillable space.
- **Power.** The core ring (10 µm) connects to the VDD/VSS pad pins.
  `pdn_cfg.tcl` creates the IOVDD/IOVSS nets for the pad pins but keeps
  them out of the core voltage domain, so no core ring is built for the
  I/O supply.
- **Seal ring and fill.** The seal ring is produced by IHP's `sealring.py`
  PCell script at 140 µm edge spacing (PDK default). Metal and active fill
  come from IHP's `filler.py`, in LibreLane's `KLayout.SealRing` and
  `KLayout.Filler` steps.

## 3. Timing constraints

`flow/ihp-sg13g2/chip_top.sdc`:

- One clock, created on `clk_pad/p2c` with a 20 ns period, propagated.
  Setup uncertainty 0.25 ns, hold uncertainty 0.10 ns, 0.15 ns clock
  transition, ±5 % derate. The hold margin is separate: with 0.25 ns on
  hold the resizer inserted 7 254 hold buffers, with 0.10 ns 731.
- Placement and routing use `chip_top_pnr.sdc`, which adds
  `set_max_transition 1.5` so the resizer keeps transitions short. The
  sign-off SDC keeps the library limits; max slew and max capacitance are
  checked against them at every corner.
- **Asynchronous interface.** in_req, out_ack and rst_n pass through
  2-flop synchronizers. in_bus is sampled ≥ 2 cycles after in_req
  (bundled data). out_bus changes one cycle before out_req. None of these
  ports is timed against clk. `set_max_delay 0.6 × T` bounds every
  pad-to-flop and flop-to-pad path so the bundling assumptions hold with a
  large margin.
- **Reset.** Asynchronous assertion. De-assertion comes from a flop, so
  recovery/removal of all 5 634 flip-flops is checked by STA against clk.
- **Corners** (IHP liberty): `nom_slow_1p08V_125C`, `nom_typ_1p20V_25C`,
  `nom_fast_1p32V_m40C`. Setup, hold, max slew and max capacitance are
  checked at all three (`TIMING_VIOLATION_CORNERS`, `MAX_SLEW_…`,
  `MAX_CAP_VIOLATION_CORNERS` = `*`). Design and timing repair also run
  after global routing (`RUN_POST_GRT_*`): the placement-based estimate
  missed the slow corner by 2.5 ns in an earlier run.
- `set_max_fanout 10` is a guideline for the resizer. 452 nets exceed it;
  none of them violates max slew or max capacitance, which are the
  electrical checks, at any corner.

### 3.1 Synchronizer MTBF

MTBF = e^(t_r/τ) / (T₀ · f_clk · f_data). The following values are
typical of a 130 nm flip-flop and are **not** IHP data: τ ≈ 30 ps,
T₀ ≈ 100 ps. With t_r ≈ 18 ns (one period minus clock-to-Q and setup),
f_clk = 50 MHz and f_data ≤ 5 MHz, MTBF ≈ e⁶⁰⁰ / (10⁻¹⁰·5·10⁷·5·10⁶), far
beyond the age of the universe. Two flip-flops are more than enough.

## 4. Implementation results

See `tapeout/ihp-sg13g2/SIGNOFF.md` for the complete generated table,
including every corner. Summary (IHP SG13G2, final run):

| Quantity | Value |
|---|---|
| Die / core | 2.000 × 2.000 mm / 1.040 × 1.040 mm (1.081 mm²) |
| Standard cells | 44 120 (93 630 instances with taps, fill, decaps, pads) |
| Flip-flops | 5 634 (`sg13g2_dfrbpq_1`) |
| Standard-cell area / core utilization | 0.679 mm² / 62.8 % |
| Routed wire length | 2.09 m |
| Setup slack, slow / typ / fast | +1.083 / +7.946 / +9.620 ns (period 20 ns) |
| Hold slack, slow / typ / fast | +0.383 / +0.194 / +0.084 ns |
| Setup / hold violations, all corners | 0 / 0 |
| Max slew / max capacitance violations, all corners | 0 / 0 |
| Clock skew (setup) | 0.40 ns |
| Power, typical corner (vectorless) / leakage | 4.6 mW / 29.7 µW |
| Worst IR drop | 5.0 mV |

The slow-corner slack corresponds to about 53 MHz at 1.08 V and 125 °C.

## 5. Sign-off checks

| Check | Tool / deck | Result |
|---|---|---|
| Routing DRC | OpenROAD TritonRoute | 0 |
| Layout DRC, full IHP rule deck (174 rule categories) on the filled GDS | KLayout, `ihp-sg13g2.drc` | 0 |
| Metal and active density (global and 800 µm windows) | KLayout, IHP density deck | 0 |
| Antenna | OpenROAD `check_antennas` after routing; KLayout IHP antenna deck | 0 / 0 |
| LVS: layout netlist vs powered netlist | Magic extraction (DEF, I/O cells as abstracts) + Netgen | circuits match uniquely; 0 unmatched devices, nets, pins |
| GDS consistency | Magic vs KLayout stream-out XOR | 0 differences |
| Disconnected pins; power-grid connectivity (VDD, VSS, IOVDD, IOVSS) | OpenROAD checks | 0 / 0 |
| Static timing, 3 corners | OpenSTA | clean (§4) |
| Magic illegal overlaps | Magic | 19 458, informational: all between obstruction and pin shapes of neighbouring `sg13g2_io` abstracts in the pad ring, none in the core; the real geometry is checked by the KLayout DRC deck |

Why this split of tools:

- **DRC.** IHP's own KLayout deck is the reference. Magic's IHP rules are
  a community port, and a full-chip Magic DRC with fill ran for more than
  80 minutes on one core without finishing, so `RUN_MAGIC_DRC` is off.
- **LVS.** IHP's KLayout LVS deck (`sg13g2.lvs`) stops on the PDK's own
  I/O cell netlists, so sign-off LVS uses Magic and Netgen. The I/O cells
  enter as LEF abstracts: every connection to them is checked, their
  contents (foundry IP) are not.
- **Fill.** The PDK's fill script flattens the die and was killed above
  12 GB. `scripts/filler_deep.py` runs the same rules in KLayout's
  hierarchical (deep) mode in about 2 GB and 25 minutes. The fill
  distance per layer (Metal2..5: 0.42, 1.0, 1.0, 2.0 µm) meets MxFil.h in
  every window while Metal4/5 stay below the 60 % maximum.

Reproduce: `make gds-ihp`, then
`python tools/collect_tapeout.py flow/ihp-sg13g2/runs/<tag> --process ihp-sg13g2`.

## 6. Gate-level simulation

`test_chip_ihp_gate_level` simulates the post-route netlist of run `full5`
(`tapeout/ihp-sg13g2/chip_top.nl.v`, 5 634 flip-flops) with the IHP
standard-cell models (converted to functional Verilog by
`tools/functional_cells.py`), the `sg13g2_io` pad models and a metal-only
model of the bond pad (`flow/ihp-sg13g2/sim/bondpad_70x70.v`). It drives
the same pin-level tests and the 91 golden transactions as the RTL chip
test, through the asynchronous interface, and compares every output byte.

Result on the final netlist: **3 of 3 cocotb tests pass** (`pins_csr`,
`pins_encode_decode`, `golden_vectors`: all 91 transactions bit-exact),
16.2 ms of simulated time, 1 h 35 min wall time in Icarus Verilog.

## 7. Tool versions used for the committed IHP result

| Tool | Version |
|---|---|
| LibreLane | 3.0.14 (pip) |
| OpenROAD / OpenSTA | 26Q2 / 3.1.0 (nixpkgs) |
| Yosys | 0.69 |
| Magic | 8.3.681 |
| Netgen | 1.5.323 |
| KLayout | 0.30.12 |
| IHP-Open-PDK | c4b8b4e5e7a05f375cca3815d51b3a37721fbf5c |

The committed run used tools from nixpkgs rather than LibreLane's own Nix
environment, because this build machine can't reach the FOSSi binary
cache. One compatibility shim was needed: LibreLane's
`scripts/openroad/common/io.tcl` calls `sta::set_scene`, which this
OpenSTA build does not bind, so the shim uses `sta::set_cmd_scene`. The CI
job `gds-ihp` rebuilds the chip in LibreLane's reference environment
(`nix-shell`) without the shim. Its result is the one to submit if the two
differ.
