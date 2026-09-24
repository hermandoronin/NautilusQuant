# NQX-S1 — Bring-up guide

**Document status:** preliminary. Revision 1.0, 2026-09-24.

## 1. What you need

| Item | Notes |
|---|---|
| NQX-S1 dies on a carrier | wafer.space: chip-on-board option ($1 500) or its breakout PCB. IHP: QFN package or bare dies with a bonding plan |
| Microcontroller with ≥ 23 free GPIO | Raspberry Pi Pico (RP2040) or Pico 2 (RP2350) with MicroPython. The pin map is in `host/nqx_pins.py` |
| Supplies | IHP die: 1.2 V core (VDD) and 3.3 V I/O (IOVDD). GF180 die: as specified by the wafer.space board |
| Clock | Pico PWM output (1–25 MHz) or a crystal oscillator (up to 50 MHz) |
| Bench meter | Supply current |

**Check the I/O voltage before you connect anything.** The IHP I/O pads
run at 3.3 V and match the Pico directly. The GF180MCU pads run at the
DVDD chosen for the wafer.space board (3.3 V or 5 V). At 5 V, use level
shifters between the chip and a 3.3 V microcontroller.

## 2. Wiring (default map)

| Chip pin | Direction | Pico GPIO |
|---|---|---|
| in_bus[7:0] | in | GP0…GP7 |
| in_req | in | GP8 |
| in_ack | out | GP9 |
| out_bus[7:0] | out | GP10…GP17 |
| out_req | out | GP18 |
| out_ack | in | GP19 |
| busy | out | GP20 |
| rst_n | in | GP21 |
| clk | in | GP22 (PWM) |

For the wafer.space chip, the pad numbers are in
`flow/wafer-space-gf180/src/chip_core.sv`: in_bus = bidir[7:0],
out_bus = bidir[15:8], in_ack/out_req/busy = bidir[16..18],
in_req/out_ack = input[0..1].

## 3. Procedure

```bash
pip install mpremote
mpremote cp nqx-silicon/host/nqx_s1_host.py nqx-silicon/host/nqx_pins.py \
            nqx-silicon/host/run_all_vectors.py :
mpremote mkdir :vectors
for f in nqx-silicon/vectors/*.txt; do mpremote cp "$f" :vectors/; done
mpremote exec "import run_all_vectors; run_all_vectors.main()"
```

1. With the clock off and reset low, apply power. Record the supply currents.
2. Start the clock at 1 MHz and release reset. `busy` must be low.
3. Run
   `mpremote exec "import nqx_pins, nqx_s1_host as h; t=nqx_pins.PinTransport(clock_hz=1_000_000); t.reset(); print(hex(h.NQXS1(t).csr_read(0)))"`.
   The expected answer is `0x4e515831` ("NQX1").
4. Run all vectors (above). Expected result: `RESULT: PASS`.
5. Raise `clock_hz` step by step to 50 MHz and repeat step 4.
6. Record the results for every die in a copy of the table in §5.

## 4. If something fails

| Symptom | Likely cause | Check |
|---|---|---|
| No in_ack ever | Clock not running, reset held, IOVDD missing | Scope clk and rst_n pads |
| ID wrong, bytes shifted | Bit order of in_bus/out_bus swapped | Wiring vs §2 |
| `00_identity` passes, `01_vector_register` fails | Vector-register or hold-time defect | Which word and bit differ (STVR readback) |
| `02_rotation` fails, `01` passes | CORDIC pipeline | Compare against `S1Core` on the first failing transaction |
| Fails only at high clock | Setup timing | Find the maximum frequency; compare with the STA slow corner |
| Random, rare failures | Supply noise or a host protocol violation | Slow down the host; check decoupling |

## 5. Bring-up record

| Die | IDD reset (VDD / IOVDD) | 1 MHz | 10 MHz | 50 MHz | Fmax | Notes |
|---|---|---|---|---|---|---|
| 1 | | | | | | |
