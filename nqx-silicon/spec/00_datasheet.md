# NQX-S1 — Preliminary datasheet

**Golden-angle KV-cache compression engine. Test chip, revision 1.0.**

> **Preliminary.** Electrical values are taken from the foundry PDK
> libraries and from sign-off analysis. They have not been measured on
> silicon.

## Features

- Encodes a 128 × int16 vector into a 70-byte packet (3.66×; 4.0× excluding
  the header). Decodes a packet back to int16.
- NautilusQuant pipeline:
  1. deterministic golden-angle Givens rotation;
  2. polar transform;
  3. 3-bit radius code and 3-bit circular angle code;
  4. 1-bit residual per value.
- The rotation needs **no angle memory**. A phase accumulator per layer
  generates the angles (increments `0x61C88647`, `0x9E3779B9`), and a
  shift-and-add CORDIC applies them.
- Programmable phase increments, so other rotation sequences can be tried
  on the same silicon.
- Bit-exact with the published Python model (`model/nqx_s1`).
- Asynchronous 8-bit in / 8-bit out 4-phase handshake interface. Works with
  any microcontroller at any speed.
- One 50 MHz clock (target), fully static, all flip-flops reset.

## Block diagram

See [`02_architecture.md`](02_architecture.md) §1.

## Pinout: IHP SG13G2 die (31 pads)

| Side | Pads (in placement order) |
|---|---|
| South | VDD, VSS, clk, rst_n, busy, IOVDD, IOVSS |
| East | out_bus[0] … out_bus[7] |
| North | IOVSS, IOVDD, out_ack, out_req, in_ack, in_req, VSS, VDD |
| West | in_bus[7] … in_bus[0] |

| Pin | Type | Description |
|---|---|---|
| clk | input | Core clock |
| rst_n | input | Asynchronous reset, active low (de-assertion synchronized on chip) |
| in_bus[7:0] | input | Host → chip data |
| in_req | input | Host → chip request (4-phase) |
| in_ack | output, 4 mA | Chip → host acknowledge |
| out_bus[7:0] | output, 4 mA | Chip → host data |
| out_req | output, 4 mA | Chip → host request |
| out_ack | input | Host → chip acknowledge |
| busy | output, 4 mA | High while a command executes |
| VDD / VSS | power | Core supply 1.2 V (×2 each) |
| IOVDD / IOVSS | power | I/O supply 3.3 V (×2 each) |

The pinout of the wafer.space GF180MCU die follows the wafer.space template
pad ring; see [`10_bringup.md`](10_bringup.md) §2.

## Operating conditions (IHP SG13G2, from PDK characterization)

| Parameter | Min | Typ | Max | Unit |
|---|---|---|---|---|
| Core supply VDD | 1.08 | 1.20 | 1.32 | V |
| I/O supply IOVDD | 3.0 | 3.3 | 3.6 | V |
| Junction temperature | −40 | 25 | 125 | °C |
| Clock frequency | 0 | — | 50 | MHz |

The frequency limit is signed off at the slow corner (1.08 V, 125 °C).
Actual values per corner are in [`06_physical_design.md`](06_physical_design.md).

## Functional summary

| Command | Opcode | Latency (cycles, 1 B/cycle host) |
|---|---|---|
| ENC (encode vector) | `0x60` | 720 |
| DEC (decode packet) | `0x61` | 690 |
| CSRR / CSRW | `0x72` / `0x71` | 6 / 6 |

Full command set, register map and packet format:
[`03_programming_model.md`](03_programming_model.md).

## Accuracy

| Quantity | Value |
|---|---|
| Rotation round trip, int16 | ≤ 1 LSB |
| Rotation vs float64 | ≤ 3·10⁻⁵ relative |
| Reconstruction error, KV-like data, 4 bits/value | 0.146 relative RMSE |

Details: [`04_numerics_results.md`](04_numerics_results.md).
