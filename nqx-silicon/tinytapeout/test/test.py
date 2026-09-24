# SPDX-License-Identifier: Apache-2.0
"""NQX-S1 on Tiny Tapeout: pin-level test, bit-exact against the model."""

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

from nqx_s1 import Csr, S1Core, S1Params, host

P = S1Params(dim=32)
IN_REQ, IN_ACK, OUT_REQ, OUT_ACK = 0, 1, 2, 3


class Pins:
    """Host side of the 4-phase handshakes on the TT pins."""

    def __init__(self, dut):
        self.dut = dut
        self.uio = 0
        dut.uio_in.value = 0

    def _set(self, bit, level):
        self.uio = (self.uio & ~(1 << bit)) | (level << bit)
        self.dut.uio_in.value = self.uio

    def _get(self, bit):
        return (int(self.dut.uio_out.value) >> bit) & 1

    async def _wait(self, bit, level, limit=500000):
        for _ in range(limit):
            if self._get(bit) == level:
                return
            await ClockCycles(self.dut.clk, 1)
        raise TimeoutError(f"uio[{bit}] stuck")

    async def write(self, data):
        for b in data:
            self.dut.ui_in.value = b
            await ClockCycles(self.dut.clk, 1)
            self._set(IN_REQ, 1)
            await self._wait(IN_ACK, 1)
            self._set(IN_REQ, 0)
            await self._wait(IN_ACK, 0)

    async def read(self, n):
        out = bytearray()
        for _ in range(n):
            await self._wait(OUT_REQ, 1)
            out.append(int(self.dut.uo_out.value))
            self._set(OUT_ACK, 1)
            await self._wait(OUT_REQ, 0)
            self._set(OUT_ACK, 0)
        return bytes(out)

    async def transact(self, stream, n):
        reader = cocotb.start_soon(self.read(n))
        await self.write(stream)
        return await reader


async def start(dut):
    cocotb.start_soon(Clock(dut.clk, 40, unit="ns").start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    return Pins(dut)


@cocotb.test()
async def test_identity(dut):
    pins = await start(dut)
    stream = host.csrr(Csr.ID) + host.csrr(Csr.PARAMS) + host.csrr(Csr.FEATURES) + host.sync()
    exp = S1Core(p=P, features=1).run(stream)
    assert await pins.transact(stream, len(exp)) == exp


@cocotb.test()
async def test_encode_decode(dut):
    pins = await start(dut)
    rng = random.Random(1)
    model = S1Core(p=P, features=1)
    vectors = [[0] * P.dim, [32767] * P.dim, [(-1) ** i * 12345 for i in range(P.dim)]]
    vectors += [[rng.randint(-32768, 32767) for _ in range(P.dim)] for _ in range(3)]
    for v in vectors:
        stream = host.enc(v)
        pkt = model.run(stream)
        assert await pins.transact(stream, len(pkt)) == pkt
        stream = host.dec(pkt)
        exp = model.run(stream)
        assert await pins.transact(stream, len(exp)) == exp
