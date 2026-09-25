# Provenance

This directory is a copy of the wafer.space GF180MCU project template
(https://github.com/wafer-space/gf180mcu-project-template, commit
`0de7e394337a1f7f5303ac7a3681bf2481b58176`, Apache-2.0, see `LICENSE` and
`AUTHORS.md`), adapted for NQX-S1:

- `src/chip_core.sv` replaced by the NQX-S1 pad mapping and `nqx_s1_top`
  instance (RTL in `../../rtl`).
- SRAM macros removed from `librelane/macros/*.yaml` and
  `librelane/pdn/pdn_cfg.tcl` (NQX-S1 uses no SRAM).
- The 0p5x0p5 slot builds NQX-S1 with the iterative CORDIC
  (`NQX_ITER_CORDIC=1`, bit-identical results, about 25 % less area).
- Default slot changed to `0p5x0p5`; placement density moved into the
  slot files (70 % for 0p5x0p5); RTL file list extended in
  `librelane/config.yaml`.
- `cocotb/` replaced by the NQX-S1 chip-level testbench.

`src/chip_top.sv`, the pad ring, power pads and the ID/logo IP are unchanged,
so the default wafer.space bonding and breakout PCB apply.
