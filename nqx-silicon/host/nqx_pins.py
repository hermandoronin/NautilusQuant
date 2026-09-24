"""MicroPython transport: bit-bang the NQX-S1 4-phase handshake from GPIO.

Default wiring (Raspberry Pi Pico / RP2040, 3.3 V logic):

    GP0..GP7   -> in_bus[7:0]      GP8  -> in_req     GP9  <- in_ack
    GP10..GP17 <- out_bus[7:0]     GP18 <- out_req    GP19 -> out_ack
    GP20 <- busy                   GP21 -> rst_n      GP22 -> clk (PWM)

For the IHP SG13G2 chip, IOVDD = 3.3 V matches the Pico directly. For the
GF180MCU (wafer.space) chip, check the I/O voltage of your breakout board
before connecting a 3.3 V MCU.

Usage on the board (copy host/*.py and vectors/*.txt with mpremote):

    import nqx_pins, nqx_s1_host as h
    t = nqx_pins.PinTransport(clock_hz=10_000_000)
    t.reset()
    chip = h.NQXS1(t)
    print(hex(chip.csr_read(0)))
"""

import time

from machine import PWM, Pin

from nqx_s1_host import Transport


class PinTransport(Transport):
    def __init__(self, in_bus=0, in_req=8, in_ack=9, out_bus=10, out_req=18, out_ack=19,
                 busy=20, rst_n=21, clk=22, clock_hz=10_000_000, timeout_ms=1000):
        self.inb = [Pin(in_bus + i, Pin.OUT, value=0) for i in range(8)]
        self.outb = [Pin(out_bus + i, Pin.IN) for i in range(8)]
        self.in_req = Pin(in_req, Pin.OUT, value=0)
        self.in_ack = Pin(in_ack, Pin.IN)
        self.out_req = Pin(out_req, Pin.IN)
        self.out_ack = Pin(out_ack, Pin.OUT, value=0)
        self.busy = Pin(busy, Pin.IN)
        self.rst_n = Pin(rst_n, Pin.OUT, value=0)
        self.timeout_ms = timeout_ms
        self.rx = bytearray()
        if clk is not None:
            self.pwm = PWM(Pin(clk))
            self.pwm.freq(clock_hz)
            self.pwm.duty_u16(32768)

    def reset(self):
        self.rst_n.value(0)
        time.sleep_ms(1)
        self.rst_n.value(1)
        time.sleep_ms(1)
        self.rx = bytearray()

    def _wait(self, pin, level):
        t0 = time.ticks_ms()
        while pin.value() != level:
            if time.ticks_diff(time.ticks_ms(), t0) > self.timeout_ms:
                raise OSError("NQX-S1 handshake timeout")

    def _poll_out(self):
        # Drain one chip -> host byte if one is offered (non-blocking).
        if self.out_req.value():
            v = 0
            for i in range(8):
                v |= self.outb[i].value() << i
            self.out_ack.value(1)
            self._wait(self.out_req, 0)
            self.out_ack.value(0)
            self.rx.append(v)

    def write(self, data):
        for byte in data:
            for i in range(8):
                self.inb[i].value((byte >> i) & 1)
            self.in_req.value(1)
            t0 = time.ticks_ms()
            while not self.in_ack.value():
                self._poll_out()
                if time.ticks_diff(time.ticks_ms(), t0) > self.timeout_ms:
                    raise OSError("NQX-S1 in_ack timeout")
            self.in_req.value(0)
            self._wait(self.in_ack, 0)
            self._poll_out()

    def read(self, n):
        t0 = time.ticks_ms()
        while len(self.rx) < n:
            self._poll_out()
            if time.ticks_diff(time.ticks_ms(), t0) > self.timeout_ms:
                raise OSError("NQX-S1 read timeout (%d of %d bytes)" % (len(self.rx), n))
        out = bytes(self.rx[:n])
        self.rx = self.rx[n:]
        return out
