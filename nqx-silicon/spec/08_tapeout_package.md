# NQX-S1 — Tape-out package

**Document status:** Revision 1.0, 2026-09-24.

This document answers two questions: *what exactly goes to the fab*, and
*how an individual orders NQX-S1 in 2026*. Prices and dates come from the
services' own published data (sources in §6). They change often, so check
them before paying.

## 1. What a fab or MPW service receives

Nobody ships "specifications" to a foundry. The foundry receives a
**layout database (GDSII/OASIS)** that already passes its design rules. The
specifications in `spec/` are the engineering record behind that file. For
an MPW (multi-project wafer) service the package is:

| # | Deliverable | NQX-S1 file | Required by |
|---|---|---|---|
| 1 | Final layout, one top cell, origin (0,0), dbu 1 nm, seal ring, metal fill | `tapeout/ihp-sg13g2/chip_top.gds.gz`; wafer.space: `final/gds/chip_top.gds` from CI | all |
| 2 | Clean DRC with the foundry rule deck (KLayout), plus Magic DRC | `tapeout/ihp-sg13g2/reports/` | all |
| 3 | Clean LVS: extracted layout vs netlist | `reports/` + `chip_top.pnl.v` / `.spice` | IHP, ChipFoundry |
| 4 | Antenna check, density check | `reports/` | all |
| 5 | Timing sign-off (setup/hold at slow/typ/fast corners) | `reports/` and [`06_physical_design.md`](06_physical_design.md) | good practice |
| 6 | Gate-level netlist (powered and unpowered) | `chip_top.nl.v`, `chip_top.pnl.v` | IHP (LVS), own GL simulation |
| 7 | Pad list and bonding information | [`00_datasheet.md`](00_datasheet.md) pinout, §4 below | IHP (QFN), Europractice |
| 8 | Project metadata | IHP: `doc/info.json` + TRL document (template in §3.2); wafer.space: none beyond the order | IHP Open-MPW |
| 9 | Test vectors and bring-up procedure | `vectors/`, [`10_bringup.md`](10_bringup.md) | you, after fabrication |
| 10 | SHA-256 checksums of all of the above | `tapeout/ihp-sg13g2/SHA256SUMS` | good practice |

## 2. Routes available to an individual (September 2026)

| | **A. wafer.space Run 3** | **B. IHP SG13G2** | **C. Tiny Tapeout** | **D. ChipFoundry chipIgnite** |
|---|---|---|---|---|
| Process | GlobalFoundries 180 nm (GF180MCU), 5 metals | IHP 130 nm BiCMOS, Frankfurt (Oder) | IHP / GF180 / SKY130 | SkyWater 130 nm |
| Who can order | Anyone (Crowd Supply / shop) | Free MPW: non-commercial research and education, eligibility of individuals unverified; paid MPW: on request | Anyone | Anyone |
| Area / price | 0.5×0.5 slot (1.94×2.53 mm die): **$2 000** early bird until 2026-09-30, then $3 000. 0.5×1 slot: $4 000 / $5 000 | Paid about 2 800 €/mm² (unverified); NQX-S1 die 4.0 mm² | 70 € per tile (~0.03–0.06 mm²) | $14 950 per project (100 QFN parts), 10 mm² |
| Parts | **1 000 dies** per slot; chip-on-board +$1 500 | typically 20 bare dies; QFN with a bonding plan | chip on a demo board | 100 QFN |
| Deadlines | purchase by **2026-12-09**, GDS by **2026-12-16**, delivery about 2027-04 | free MPW: next tape-in 2026-09-29, later dates by announcement | TTGF26c (GF180): 2026-12-07; TTSKY26d: 2026-11-30; TTIHP27a: 2027-03 | CI2612 (Dec 2026) |
| NQX-S1 readiness | **Ready**: `flow/wafer-space-gf180`, CI builds GDS and runs the wafer.space precheck | **Ready**: signed-off GDS in `tapeout/ihp-sg13g2` | **Ready** as DIM=32 package in `tinytapeout/` (see §5) | Needs a Caravel wrapper port (not done) |

### Recommendation

1. **wafer.space, 0.5×0.5 slot, iterative-CORDIC build.** It is the
   cheapest way to get 1 000 real chips ($2 000–3 000). The core cell area
   is 0.77 mm² in a 1.73 mm² core area, about 45 % utilization. If the CI
   precheck for 0.5×0.5 fails, use the 0.5×1 slot with the pipelined build
   ($4 000–5 000).
2. **IHP** if you can reach its research MPW through a university or
   institute partner. The IHP GDS in this repository has already passed
   sign-off here, and the fab is in Germany.
3. **Tiny Tapeout TTGF26c** (about 1 120 €) for a cheap first chip on a demo
   board: the DIM=32 package in `tinytapeout/`.

## 3. Step-by-step

### 3.1 wafer.space

1. In GitHub: *Actions → nqx-silicon → Run workflow*, with "gds" checked.
   The job builds `chip_top.gds` for slots 0p5x0p5 and 0p5x1, runs RTL and
   gate-level chip simulations with the PDK pad models, and runs the
   official `gf180mcu-precheck`. Artifacts: `nqx-s1-wafer-space-gf180-<slot>`.
2. Buy the slot on wafer.space / Crowd Supply for Run 3 (before
   2026-12-09). Add chip-on-board if you want mounted dies.
3. Upload `final/gds/chip_top.gds` of the passing slot by 2026-12-16,
   following wafer.space's submission instructions.
4. Keep the CI run ID, commit hash and artifact checksums; they identify
   exactly what was fabricated.

### 3.2 IHP Open-Silicon MPW (research route)

The procedure follows `IHP-GmbH/Open-Silicon-MPW` (`IP-development-steps.md`):

1. Open an issue in IHP-GmbH/Open-Silicon-MPW named `IHP__<subcategory><4 digits>`
   (the naming scheme is in their repository).
2. Provide in `release/<version>/`:
   - `chip_top.gds` (seal ring and fill included: done);
   - the netlist (`chip_top.pnl.v`, or the SPICE from LVS);
   - `doc/info.json` (name, author, area, pad list, supply voltages);
   - the TRL document;
   - DRC reports from IHP's minimal and maximal KLayout decks (the flow
     runs the main deck; run the maximal deck as in §4 of
     [`06_physical_design.md`](06_physical_design.md)).
3. For packaged samples, supply a bonding plan for QFN (24–64 pins): 31 pads
   → QFN-32 or QFN-40.

### 3.3 Paid MPW (IHP directly or via Europractice)

Ask IHP's MPW service (contact through the page linked in §6) for a quote:
a 2.0 × 2.0 mm SG13G2 design with 31 pads, 20 samples, optional QFN
packaging. Europractice pricing needs an
institutional account.

## 4. Pre-submission checklist

| Item | IHP SG13G2 | wafer.space GF180 |
|---|---|---|
| RTL frozen, tagged commit | ☐ tag `nqx-s1-v1.0` after the CI run | ☐ same |
| Model tests, RTL tests, formal, golden vectors: pass | ☑ local and CI | ☑ same RTL |
| Chip-level RTL sim with foundry pad models | ☑ `test_chip_ihp` | ☐ CI job |
| Gate-level sim of the final netlist | see [`06`](06_physical_design.md) §6 | ☐ CI job |
| DRC (foundry KLayout deck) | see [`06`](06_physical_design.md) §5 | ☐ CI + precheck |
| LVS | see [`06`](06_physical_design.md) §5 | ☐ CI |
| Antenna | see [`06`](06_physical_design.md) §5 | ☐ precheck |
| Density / fill | see [`06`](06_physical_design.md) §5 | ☐ precheck |
| Seal ring | ☑ KLayout.SealRing step | ☑ template |
| Setup/hold, all corners | see [`06`](06_physical_design.md) §4 | ☐ CI |
| IO/power pad count and ESD | ☑ foundry pad cells only; 2 pairs of each supply | ☑ template pad ring unchanged |
| Chip ID readable | ☑ CSR ID = "NQX1", VERSION, FEATURES | ☑ same + wafer.space QR/ID cells |
| Bonding plan | ☐ needed for QFN | ☑ default template bonding |
| Checksums recorded | ☑ `tapeout/*/SHA256SUMS` | ☐ from CI artifacts |

## 5. Smaller variants (for Tiny Tapeout or cheaper area)

Cell areas are from Yosys + ABC synthesis against the foundry TT liberty
files, before placement buffering (typically +5–10 %).

| Build | IHP cell area | GF180 cell area | Fits |
|---|---|---|---|
| DIM=128, pipelined CORDIC | 0.60 mm² | 1.02 mm² | IHP die, wafer.space 0.5×1 |
| DIM=128, iterative CORDIC | 0.45 mm² | 0.77 mm² | wafer.space 0.5×0.5 |
| DIM=32, iterative CORDIC | 0.22 mm² | 0.38 mm² | TT: 8×2 IHP tiles (0.50 mm², 44 %) or 4×4 GF180 tiles (0.89 mm², 42 %) |

DIM is a generator parameter (`tools/gen_params.py --dim 32`). Model, RTL
and pin tests pass at DIM = 16 and 32. The 24 host signals map one-to-one
onto Tiny Tapeout's `ui_in` / `uo_out` / `uio` pins.

**Tiny Tapeout package:** `tinytapeout/`, generated by
`tools/export_tinytapeout.py`. It contains `info.yaml` (4×4 tiles, GF180),
the `tt_um_hermandoronin_nqx_s1` wrapper, one flattened RTL file (DIM=32,
iterative CORDIC), `docs/info.md`, and a cocotb test that is bit-exact
against the model; it passes locally with Icarus. Copy the package into a
repository made from `ttgf-verilog-template` and submit it for TTGF26c
(deadline 2026-12-07). The GDS is built by Tiny Tapeout's own GitHub Action.
Cost: 16 tiles × 70 € = 1 120 € plus board/shipping.

## 6. Sources

- wafer.space runs and slots (data files of their web site):
  https://raw.githubusercontent.com/wafer-space/wafer-space.github.io/main/_data/runs.yml,
  https://raw.githubusercontent.com/wafer-space/wafer-space.github.io/main/_data/slots.yml
- wafer.space template and precheck: https://github.com/wafer-space/gf180mcu-project-template,
  https://github.com/wafer-space/gf180mcu-precheck
- IHP Open-Silicon MPW: https://github.com/IHP-GmbH/Open-Silicon-MPW (issue #63 for the
  2026-09-29 tape-in, `IP-development-steps.md`)
- IHP MPW service: https://www.ihp-microelectronics.com/services/research-and-prototyping-service/mpw-prototyping-service
- Tiny Tapeout shuttles and prices: https://github.com/TinyTapeout/tinytapeout_www (content/,
  PR #300 for the 2026–2027 schedule)
- ChipFoundry: https://chipfoundry.io/faqs, https://github.com/chipfoundry/cf-cli
- Europractice: https://europractice-ic.com/schedules-prices-2026/
