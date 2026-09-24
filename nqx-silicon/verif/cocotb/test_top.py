"""Pin-level test of nqx_s1_top: 4-phase handshake host model, async to clk."""

from __future__ import annotations

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

from nqx_s1 import Csr, S1Core, S1Params, host
from stimulus import csr_smoke, kv_vector

P = S1Params(dim=int(os.environ.get("NQX_DIM", "128")))


class PinHost:
    """Host MCU model: bit-bangs the handshake with its own, unrelated timing."""

    def __init__(self, dut, rng: random.Random):
        self.dut = dut
        self.rng = rng

    async def _delay(self):
        await Timer(self.rng.choice([3, 7, 13, 29, 41]), unit="ns")

    async def _wait(self, sig, level: int, limit_ns: int = 2_000_000):
        waited = 0
        while int(sig.value) != level:
            await Timer(5, unit="ns")
            waited += 5
            if waited > limit_ns:
                raise TimeoutError(f"{sig._name} stuck at {int(sig.value)}")

    async def write(self, data: bytes):
        dut = self.dut
        for byte in data:
            dut.in_bus.value = byte
            await self._delay()
            dut.in_req.value = 1
            await self._wait(dut.in_ack, 1)
            await self._delay()
            dut.in_req.value = 0
            dut.in_bus.value = self.rng.randrange(256)
            await self._wait(dut.in_ack, 0)

    async def read(self, n: int) -> bytes:
        dut = self.dut
        out = bytearray()
        for _ in range(n):
            await self._wait(dut.out_req, 1)
            await self._delay()
            out.append(int(dut.out_bus.value))
            dut.out_ack.value = 1
            await self._wait(dut.out_req, 0)
            await self._delay()
            dut.out_ack.value = 0
        return bytes(out)

    async def transact(self, stream: bytes, n_out: int) -> bytes:
        reader = cocotb.start_soon(self.read(n_out))
        await self.write(stream)
        return await reader


async def bring_up(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())
    dut.rst_n.value = 0
    dut.in_bus.value = 0
    dut.in_req.value = 0
    dut.out_ack.value = 0
    await ClockCycles(dut.clk, 4)
    await Timer(3, unit="ns")
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 4)


@cocotb.test()
async def pins_csr(dut):
    await bring_up(dut)
    h = PinHost(dut, random.Random(7))
    stream = csr_smoke()
    exp = S1Core(p=P).run(stream)
    got = await h.transact(stream, len(exp))
    assert got == exp, f"\n rtl  : {got.hex()}\n model: {exp.hex()}"


@cocotb.test()
async def pins_encode_decode(dut):
    await bring_up(dut)
    rng = random.Random(8)
    h = PinHost(dut, rng)
    model = S1Core(p=P)
    for _ in range(2):
        v = kv_vector(rng, P.dim)
        stream = host.enc(v)
        pkt = model.run(stream)
        got = await h.transact(stream, len(pkt))
        assert got == pkt
        stream = host.dec(pkt)
        exp = model.run(stream)
        got = await h.transact(stream, len(exp))
        assert got == exp
    stream = host.csrr(Csr.ENC_COUNT) + host.csrr(Csr.DEC_COUNT)
    exp = model.run(stream)
    assert await h.transact(stream, len(exp)) == exp
    assert int(dut.busy.value) == 0


def _vector_files():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "vectors")
    pattern = os.environ.get("NQX_VECTORS", "*")
    import fnmatch

    return sorted(
        os.path.join(root, f)
        for f in os.listdir(root)
        if f.endswith(".txt") and f[0].isdigit() and fnmatch.fnmatch(f, pattern + ".txt")
    )


def _transactions(path):
    w, r = bytearray(), bytearray()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line == ".":
                yield bytes(w), bytes(r)
                w, r = bytearray(), bytearray()
            elif line[0] == "W":
                w += bytes.fromhex(line[2:])
            else:
                r += bytes.fromhex(line[2:])


@cocotb.test()
async def golden_vectors(dut):
    """Replay vectors/*.txt (the silicon bring-up vectors) through the pins."""
    await bring_up(dut)
    rng = random.Random(9)
    h = PinHost(dut, rng)
    for path in _vector_files():
        dut.rst_n.value = 0
        await ClockCycles(dut.clk, 3)
        dut.rst_n.value = 1
        await ClockCycles(dut.clk, 4)
        n = 0
        for w, r in _transactions(path):
            got = await h.transact(w, len(r))
            assert got == r, f"{os.path.basename(path)} transaction {n}: mismatch"
            n += 1
        dut._log.info(f"{os.path.basename(path)}: {n} transactions bit-exact")
