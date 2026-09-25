<div align="center">

# NautilusQuant · NQX-S1

### An open-source KV-cache compression chip, from an algorithm idea to a signed-off 130 nm layout

[![nqx-silicon](https://github.com/hermandoronin/NautilusQuant/actions/workflows/nqx-silicon.yml/badge.svg)](https://github.com/hermandoronin/NautilusQuant/actions/workflows/nqx-silicon.yml)
[![nqx-core](https://github.com/hermandoronin/NautilusQuant/actions/workflows/nqx-core-ci.yml/badge.svg)](https://github.com/hermandoronin/NautilusQuant/actions/workflows/nqx-core-ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![Process](https://img.shields.io/badge/process-IHP%20SG13G2%20130%20nm-555)
![Sign-off](https://img.shields.io/badge/DRC%20%C2%B7%20LVS%20%C2%B7%20STA-clean-2F6B45)

<img src="nqx-silicon/tapeout/ihp-sg13g2/chip_top.png" width="440" alt="NQX-S1 layout: 2 × 2 mm die, 31 pads, IHP SG13G2">

*NQX-S1 on IHP SG13G2: 2.0 × 2.0 mm, 31 pads, 44 k standard cells, 50 MHz.*

**[Chip](#the-chip)** · **[What I built](#what-i-built)** · **[Sign-off problems solved](#sign-off-problems-solved)** · **[Findings](#research-findings)** · **[Repository](#repository-map)** · **[Reproduce](#reproduce)** · **[Русский](README.ru.md)**

</div>

---

Large language models keep a key/value cache for every token, and at long
context it dominates memory. The best compressors rotate each vector with a
random orthogonal matrix and then quantize it. **NautilusQuant** asks whether
the random matrix can be replaced by a closed-form constant: Givens
rotations whose angles step by the golden angle 2π/φ². The rotation becomes
deterministic and needs almost no stored state.

The project took that idea through the whole chain a hardware team would:
a bit-accurate model, synthesizable RTL, simulation and formal
verification, a full-chip layout for a real foundry process, sign-off, a
tape-out package, and a research study that measures honestly where the
idea helps and where it does not. Everything uses open-source tools and
open PDKs.

## The chip

| | |
|---|---|
| Function | Compresses a 128 × int16 KV vector to a 70-byte packet (golden-angle rotation → polar form → 3-bit radius, 3-bit angle, residual bits) and back |
| Rotation state | Two 32-bit phase increments (`0x61C88647`, `0x9E3779B9`) and a phase accumulator; no angle ROM, no matrix |
| Datapath | One pipelined 18-stage shift-and-add CORDIC for both the rotation and the polar transform; no general multipliers |
| Interface | 8-bit in / 8-bit out, asynchronous four-phase handshake, drivable by any microcontroller |
| Process | IHP SG13G2 130 nm: 2.0 × 2.0 mm die, 1.04 mm core, 63 % utilization, 5 634 flip-flops, 50 MHz, 4.6 mW |
| Sign-off | KLayout DRC with IHP's full rule deck (174 rule categories): 0 · density and antenna: 0 · LVS: circuits match · STA at three corners: +1.08 ns setup slack at 1.08 V / 125 °C, no slew or capacitance violations |
| Verification | Bit-exact against the Python model for the CORDIC unit (4 254 operations), the core (random programs, bus stalls), the pins (91 golden transactions) and the post-route netlist; SymbiYosys proofs of the handshake and stream protocol |
| Second process | wafer.space GF180MCU: CI builds two die sizes, runs gate-level simulation and passes the official wafer.space precheck |
| Status | Tape-out package ready ([`nqx-silicon/tapeout/ihp-sg13g2/`](nqx-silicon/tapeout/ihp-sg13g2/)); silicon not ordered |

Datasheet and the 12-document specification package:
[`nqx-silicon/spec/`](nqx-silicon/spec/).

## What I built

| Stage | Result | Where |
|---|---|---|
| Algorithm | Golden-angle Givens rotation, polar quantizer; version 1 in PyTorch and Triton | [`reference/`](reference/) |
| Pre-silicon emulator | Cycle-accurate NumPy emulator of an idealised accelerator, 24-opcode ISA, assembler, SDK, 247 tests | [`nqx-core/`](nqx-core/) |
| Bit-accurate model | Fixed-point model that the RTL must match bit for bit; single source of constants for RTL headers | [`nqx-silicon/model/`](nqx-silicon/model/) |
| RTL | 10 SystemVerilog modules: pipelined CORDIC, radius and angle coders, 28-bit divider, 2R2W register file, asynchronous byte interface; lint-clean in Verilator, Yosys and Icarus | [`nqx-silicon/rtl/`](nqx-silicon/rtl/) |
| Verification | cocotb regressions, SymbiYosys k-induction proofs, golden vectors for silicon bring-up, gate-level simulation with the foundry cell models | [`nqx-silicon/verif/`](nqx-silicon/verif/), [`vectors/`](nqx-silicon/vectors/) |
| Physical design | LibreLane Chip flow: pad ring, bond pads, power grid, placement, clock tree, routing, seal ring, metal fill | [`nqx-silicon/flow/ihp-sg13g2/`](nqx-silicon/flow/ihp-sg13g2/) |
| Sign-off | DRC, density, antenna, LVS, static timing at three corners, IR drop | [`spec/06_physical_design.md`](nqx-silicon/spec/06_physical_design.md) |
| Second process | Port to the wafer.space GF180MCU template, built and prechecked in CI | [`nqx-silicon/flow/wafer-space-gf180/`](nqx-silicon/flow/wafer-space-gf180/) |
| Tiny Tapeout | Reduced 32-value version packaged for a Tiny Tapeout tile | [`nqx-silicon/tinytapeout/`](nqx-silicon/tinytapeout/) |
| Bring-up | MicroPython driver for a Raspberry Pi Pico and a vector runner | [`nqx-silicon/host/`](nqx-silicon/host/), [`spec/10_bringup.md`](nqx-silicon/spec/10_bringup.md) |
| Research | Reverse study against open KV compressors; the NQX-RN codec; a three-part preprint | [`nqx-silicon/research/`](nqx-silicon/research/), [`nqx-silicon/paper/`](nqx-silicon/paper/) |
| CI | GitHub Actions: model, lint, formal, RTL and golden-vector checks on every push; full GDSII builds on demand | [`.github/workflows/`](.github/workflows/) |

## Sign-off problems solved

Getting the IHP layout clean took more than running the flow. Each problem
was traced to its cause and fixed in the flow configuration:

| Problem | Cause | Fix | Before → after |
|---|---|---|---|
| Setup slack at the slow corner (1.08 V, 125 °C) | Resizer used placement-based parasitics and checked the typical corner only | Design and timing repair after global routing; all corners checked | −2.5 ns → +1.08 ns |
| Antenna diodes on almost every net | Heuristic diode insertion | Disabled; antennas repaired after routing and checked with IHP's deck | 47 343 diodes → 0 |
| Hold buffers | Setup clock uncertainty also applied to hold | Separate 0.10 ns hold margin | 7 254 → 731 buffers |
| Metal fill killed by memory | PDK fill script flattens the die | Same rules in KLayout's hierarchical (deep) mode | > 12 GB → 2 GB |
| Metal2 density below 25 % in 800 µm windows | Core fully covered by routed cells | Smaller core, fill spacing set per metal layer | 16–22 % → rule met in every window |
| 13 unmatched nets in LVS | Bond pads only abutted the pad pin, so extraction saw no contact | 1 µm overlap on the pad metal | 13 → 0 |
| IHP's KLayout LVS deck unusable | It rejects the PDK's own I/O cell netlists | Magic extraction + Netgen, I/O cells as abstracts | circuits match |

## Research findings

The study reports negative results with the same weight as positive ones.

1. **The golden angle is not better than a random rotation.** In version 1
   it lost 7.9 % in reconstruction RMSE. What it does give is determinism
   and tiny state: 1.9 KB of angle table instead of a 32 KB matrix, and on
   the chip two 32-bit registers.
2. **Mapping to fixed point exposed the structure.** The third rotation
   layer is the identity, and the second layer's angles are the first
   layer's negated, so the whole rotation state fits in two registers
   ([`spec/11`](nqx-silicon/spec/11_algorithm_findings.md)). A new
   quantizer cut reconstruction error from 0.316 to 0.146 at the same
   4 bits per value.
3. **Why the golden angle loses.** Adjacent-pair rotations spread each input
   channel to only 4 of 128 outputs. A butterfly topology with golden
   angles ties with Hadamard and random rotation on attention error, with
   zero multipliers.
4. **Bit-exactness is the real advantage over float rotations.** Moving a
   float random rotation from fp32 to bf16 changes at least one 4-bit code
   in 73 % of vectors; the integer CORDIC path changes none
   ([`research/reverse_study.py`](nqx-silicon/research/reverse_study.py)).
5. **Tuning angles to the data does not transfer.** Per-head tuning cut the
   calibration error by 36 % and changed the error on new tokens of the
   same head by +1 %.
6. **NQX-RN, a codec that fits the data rather than a rotation.** Store keys
   before RoPE, each RoPE pair as radius and angle; RoPE becomes an integer
   phase addition, and bits go to pairs by the query energy, which RoPE
   leaves unchanged. On an emulated KV cache it reaches, at 3.1 bits per
   value, lower attention error (0.263) than Hadamard and random rotation at
   4.1 bits (0.297, 0.316), and its positions are exact at any context
   length. It stops winning without massive values in the keys or with
   40 % calibration drift, and it has **not yet been checked on a real
   model** ([`research/nqx_rn_study.py`](nqx-silicon/research/nqx_rn_study.py)).

The full write-up is a three-part preprint:
**[English](nqx-silicon/paper/nautilusquant_v1_v2.en.md)** · [Deutsch](nqx-silicon/paper/nautilusquant_v1_v2.de.md) · [中文](nqx-silicon/paper/nautilusquant_v1_v2.zh.md) · [Русский (original)](nqx-silicon/paper/nautilusquant_v1_v2.ru.md).

## Repository map

| Path | Content |
|---|---|
| [`nqx-silicon/`](nqx-silicon/) | **The NQX-S1 chip**: model, RTL, verification, IHP and GF180 flows, tape-out package, specification, research scripts, preprint |
| [`nqx-core/`](nqx-core/) | Version 1: emulator of an idealised accelerator, ISA, SDK, server, 247 tests |
| [`reference/`](reference/) | Version 1 software: PyTorch/Triton reference and benchmark scripts |
| [`labs/`](labs/) | Interactive browser visualizations from the exploration phase |
| [`docs/`](docs/) | Early design notes and the version 1 README |

## Reproduce

```bash
git clone https://github.com/hermandoronin/NautilusQuant
cd NautilusQuant/nqx-silicon
pip install numpy pytest cocotb==2.0.1     # plus iverilog, verilator, yosys, sby
make test-model                            # bit-accurate model
make lint yosys-check                      # RTL static checks
make test-rtl                              # cocotb: CORDIC, core, pins, golden vectors
make formal                                # SymbiYosys proofs
make gds-ihp                               # full chip on IHP SG13G2 (LibreLane 3.0.14)
python tools/collect_tapeout.py flow/ihp-sg13g2/runs/<tag> --process ihp-sg13g2
python research/nqx_rn_study.py            # the NQX-RN study (NumPy only)
```

## Tools

Python and NumPy · SystemVerilog · Verilator · Icarus Verilog · Yosys ·
cocotb · SymbiYosys · LibreLane (OpenROAD, OpenSTA, Magic, Netgen, KLayout) ·
IHP-Open-PDK SG13G2 · GF180MCU · GitHub Actions · Nix

## Next steps

- Check NQX-RN on real KV caches (Llama, Qwen, Phi-3); the protocol is in
  section 19 of the preprint.
- Prototype the version 3 datapath (butterfly topology, RoPE pairs,
  pre-RoPE polar keys) on an FPGA.
- Silicon: an IHP research shuttle or the Tiny Tapeout version.

## Citation

```bibtex
@software{doronin2026nautilusquant,
  author = {Doronin, Herman},
  title  = {NautilusQuant and NQX-S1: deterministic KV-cache compression,
            from algorithm to signed-off silicon layout},
  year   = {2026},
  url    = {https://github.com/hermandoronin/NautilusQuant}
}
```

MIT License, see [`LICENSE`](LICENSE). The hardware implementation,
verification and physical design of version 2 were done with the help of
Claude Code (Anthropic).
