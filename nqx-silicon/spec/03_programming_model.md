# NQX-S1 — Programming model

**Document status:** preliminary. Revision 1.0, 2026-09-24.
Reference implementation of the host side: `model/nqx_s1/host.py`.
Reference behaviour: `model/nqx_s1/core.py` (`S1Core.run`).

## 1. Transport

The host writes a byte stream to the chip and reads a byte stream back,
using the 4-phase handshakes of [`02_architecture.md`](02_architecture.md)
§6. The chip processes commands strictly in order. Output bytes appear in
the same order as the commands that produce them. The two directions are
independent, so a host can read while it is still writing (and must, if a
command produces more output than it is willing to buffer).

The `busy` pin is high while a command executes. It is informational; the
handshakes alone are sufficient for correct operation.

## 2. Commands

All multi-byte values are little-endian. DIM = 128 and HB = 3 in this
revision. *In* and *Out* count bytes, excluding the opcode.

| Opcode | Mnemonic | Arguments (in) | Out | Effect |
|---|---|---|---|---|
| `0x00` | NOP | — | — | Nothing |
| `0x01` | LDV | 2·DIM: int16 elements | — | VR[i] ← element i · 2⁴ |
| `0x02` | STV | — | 2·DIM | int16 elements: round(VR[i] / 2⁴), saturated to int16 (sets SAT) |
| `0x03` | STVR | — | HB·DIM | Raw 24-bit VR words, sign-extended to HB bytes (debug/test) |
| `0x10` | GVNS | 1: layer 0..2 | — | Forward Givens layer; skipped if INC[layer] = 0 |
| `0x11` | GVNS_INV | 1: layer 0..2 | — | Inverse Givens layer |
| `0x20` | POLAR | — | — | (VR[2k], VR[2k+1]) ← (r, θ); updates RMIN/RMAX |
| `0x21` | IPOLAR | — | — | (VR[2k], VR[2k+1]) ← (r·cos θ, r·sin θ) |
| `0x30` | QUANT | — | 2·HB + DIM/2 | Emits a packet from the polar VR contents (§3) |
| `0x31` | DEQUANT | 2·HB + DIM/2 | — | Loads a packet into VR in polar form |
| `0x60` | ENC | 2·DIM | 2·HB + DIM/2 | LDV, GVNS 0, GVNS 1, GVNS 2, POLAR, QUANT |
| `0x61` | DEC | 2·HB + DIM/2 | 2·DIM | DEQUANT, IPOLAR, GVNS_INV 2, GVNS_INV 1, GVNS_INV 0, STV |
| `0x70` | SYNC | — | 1 | Emits `0xA5`. All earlier commands have completed |
| `0x71` | CSRW | 1 addr + 4 data | — | Write CSR |
| `0x72` | CSRR | 1 addr | 4 | Read CSR |

Any other opcode, or a layer argument above 2, sets STATUS.ILLEGAL and is
otherwise ignored. An illegal GVNS layer byte is consumed. The opcodes
match NQ-ISA (`nqx-core/nqx/isa.py`) wherever the operation exists there.

## 3. Packet format (QUANT output, DEQUANT/DEC input)

```
byte  0..2   RMIN  (24-bit unsigned, little-endian)
byte  3..5   RMAX  (24-bit unsigned, little-endian)
byte  6+k    pair k, k = 0..63:
             bit 7   s_θ   angle residual sign  (1: θ ≥ reconstruction)
             bit 6:4 q_θ   angle sector, θ ≈ q_θ · 45°
             bit 3   s_r   radius residual sign (1: r ≥ reconstruction)
             bit 2:0 q_r   radius level, r ≈ RMIN + q_r · (RMAX−RMIN)/7
```

70 bytes per 128-element vector (256 bytes in int16): 3.66× compression.
Excluding the 6-byte header, as nqx-core counts it, the ratio is 4.0×. The
nibble layout equals nqx-core's `PackUnit.pack3plus1` (little-endian bit
stream, radius value first).

**Decoding rule (REFINE = 1).**

- r̂ = RMIN + (4·q_r + 2·s_r − 1) · (RMAX − RMIN) / 28
- θ̂ = q_θ · 45° + (2·s_θ − 1) · 11.25°

The bit-exact integer form is `model/nqx_s1/quant.py:decode_pair`.

## 4. Control and status registers

| Addr | Name | Access | Reset | Description |
|---|---|---|---|---|
| `0x00` | ID | RO | `0x4E515831` | ASCII "NQX1" |
| `0x01` | VERSION | RO | `0x00010000` | Major.minor.patch = 1.0.0 |
| `0x02` | PARAMS | RO | `0x14121807` | [7:0] log2(DIM)=7, [15:8] DW=24, [23:16] NITER=18, [31:24] ZW=20 |
| `0x03` | CTRL | RW | `0x1` | bit 0 REFINE: use residual signs when decoding |
| `0x04` | STATUS | RW1C | `0x0` | bit 0 ILLEGAL, bit 1 SAT. Sticky; write 1 to clear |
| `0x05` | INC0 | RW | `0x61C88647` | Layer 1 phase increment (2⁻³² turn) |
| `0x06` | INC1 | RW | `0x9E3779B9` | Layer 2 phase increment |
| `0x07` | INC2 | RW | `0x00000000` | Layer 3 phase increment (0 = layer skipped) |
| `0x08` | RMIN | RO | 0 | Smallest radius of the last POLAR |
| `0x09` | RMAX | RO | 0 | Largest radius of the last POLAR |
| `0x0A` | SCRATCH | RW | 0 | Free register for bring-up |
| `0x0B` | FEATURES | RO | build | bit 0: iterative CORDIC build (results identical, fewer pairs per cycle) |
| `0x10` | CYCLES | RO | 0 | Clock cycles since reset |
| `0x11` | BUSY_CYCLES | RO | 0 | Cycles spent outside the idle state |
| `0x12` | CORDIC_OPS | RO | 0 | Pairs issued to the CORDIC |
| `0x13` | ENC_COUNT | RO | 0 | QUANT operations completed |
| `0x14` | DEC_COUNT | RO | 0 | DEQUANT operations completed |

Unmapped addresses read 0; writes to them and to RO registers are ignored.
Counters wrap at 2³².

## 5. Examples

### 5.1 Identify the chip

```
host → 72 00                 CSRR ID
chip → 31 58 51 4E           0x4E515831 = "NQX1"
```

### 5.2 Encode one vector, decode it again

```python
from nqx_s1 import host
stream = host.enc(vector)          # 0x60 + 256 bytes
packet = chip.transact(stream, 70) # 70-byte packet
restored = host.unpack16(chip.transact(host.dec(packet), 256))
```

### 5.3 Try a different rotation in silicon

```python
chip.write(host.csrw(0x07, 0x2F1BBCDC))   # give layer 3 a non-zero increment
chip.write(host.csrw(0x04, 0x3))          # clear STATUS
packet = chip.transact(host.enc(vector), 70)
```

The bit-accurate model reproduces any INC setting
(`S1Core(inc=[...])`), so experiments on silicon are checked against
exact predictions.

## 6. Host timing

- There is no minimum or maximum time between bytes.
- in_bus must be stable before in_req rises and until in_ack rises
  (bundled-data rule).
- out_bus is stable from one clock before out_req rises until out_ack rises.
- After reset de-asserts, wait 4 clock cycles before the first request.
