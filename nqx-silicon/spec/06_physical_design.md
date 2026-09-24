# NQX-S1 — Physical design and sign-off

**Document status:** Revision 1.0, 2026-09-24. The sign-off numbers in §5
are generated into `tapeout/ihp-sg13g2/SIGNOFF.md` by
`tools/collect_tapeout.py`; this document explains them.

## 1. Targets

| | IHP SG13G2 (primary) | wafer.space GF180MCU |
|---|---|---|
| Process | 130 nm SiGe BiCMOS, CMOS part only; 5 thin metals + TopMetal1/2 | 180 nm, 5 metals, 5 V cells (`gf180mcu_fd_sc_mcu7t5v0`) |
| PDK | IHP-Open-PDK `c4b8b4e5` (LibreLane 3.0.14's qualified revision) | gf180mcuD via ciel `f6eeac7d` (wafer.space template pin) |
| Flow | LibreLane 3.0.14, `Chip` flow (`flow/ihp-sg13g2/config.yaml`) | LibreLane (template flake), `Chip` flow (`flow/wafer-space-gf180/librelane/`) |
| Die | 2.000 × 2.000 mm | slot 0p5x0p5: 1.936 × 2.531 mm; slot 0p5x1: 1.936 × 5.122 mm |
| Core area | 1.20 × 1.20 mm (400 µm from the die edge) | 1.052 × 1.647 mm (0p5x0p5) |
| Clock | 50 MHz (20 ns) | 25 MHz (40 ns, template default) |
| CORDIC build | pipelined | iterative in 0p5x0p5, pipelined in 0p5x1 |
| Supplies | VDD 1.2 V core, IOVDD 3.3 V I/O | per template (5 V core cells) |

## 2. Floorplan (IHP)

```
 ┌──────────────────────────── 2000 µm ────────────────────────────┐
 │ seal ring                                                        │
 │   bond pads (70×70 µm, 70 µm outside each I/O cell)              │
 │   ┌──────── I/O ring: 180 µm deep sg13g2 pad cells ───────────┐ │
 │   │ N: IOVSS IOVDD out_ack out_req in_ack in_req VSS VDD       │ │
 │ W │   ┌───── core ring (TopMetal1/2, 10 µm) ────────────────┐   │ E │
 │   │   │                                                     │   │ out_bus
 │in_│   │     core 1200 × 1200 µm, ≈ 45 % standard cells      │   │ [0..7]
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
  class COVER). It is placed 70 µm outside each I/O cell by the LibreLane
  pad-ring step.
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
  0.25 ns uncertainty, 0.15 ns transition, ±5 % derate (PDK defaults).
- **Asynchronous interface.** in_req, out_ack and rst_n pass through
  2-flop synchronizers. in_bus is sampled ≥ 2 cycles after in_req
  (bundled data). out_bus changes one cycle before out_req. None of these
  ports is timed against clk. `set_max_delay 0.6 × T` bounds every
  pad-to-flop and flop-to-pad path so the bundling assumptions hold with a
  large margin.
- **Reset.** Asynchronous assertion. De-assertion comes from a flop, so
  recovery/removal of all 5 633 flip-flops is checked by STA against clk.
- **Corners** (IHP liberty): `nom_slow_1p08V_125C`, `nom_typ_1p20V_25C`,
  `nom_fast_1p32V_m40C`. Setup is signed off at slow, hold at fast; all
  three are reported.

### 3.1 Synchronizer MTBF

MTBF = e^(t_r/τ) / (T₀ · f_clk · f_data). The following values are
typical of a 130 nm flip-flop and are **not** IHP data: τ ≈ 30 ps,
T₀ ≈ 100 ps. With t_r ≈ 18 ns (one period minus clock-to-Q and setup),
f_clk = 50 MHz and f_data ≤ 5 MHz, MTBF ≈ e⁶⁰⁰ / (10⁻¹⁰·5·10⁷·5·10⁶), far
beyond the age of the universe. Two flip-flops are more than enough.

## 4. Implementation results

See `tapeout/ihp-sg13g2/SIGNOFF.md` for the complete generated table,
including every corner. Summary (IHP SG13G2, final run):

RESULTS_PLACEHOLDER

## 5. Sign-off checks

| Check | Tool / deck | Result |
|---|---|---|
SIGNOFF_PLACEHOLDER

Reproduce: `make gds-ihp`, then
`python tools/collect_tapeout.py flow/ihp-sg13g2/runs/<tag> --process ihp-sg13g2`.

## 6. Gate-level simulation

GL_PLACEHOLDER

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
