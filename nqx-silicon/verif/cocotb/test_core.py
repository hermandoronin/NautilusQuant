"""Core-level tests: nqx_s1_core byte stream vs the transaction-level model.

The driver inserts random idle cycles on the input and random back-pressure on
the output; every test compares the full output byte stream with
S1Core.run() on the same input stream.
"""

from __future__ import annotations

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

from nqx_s1 import Csr, S1Core, S1Params, host
from stimulus import (
    csr_smoke,
    edge_vectors,
    kv_vector,
    random_packet,
    random_program,
    random_vector,
)

P = S1Params(dim=int(os.environ.get("NQX_DIM", "128")))
SEED = int(os.environ.get("NQX_SEED", "1"))
SCALE = float(os.environ.get("NQX_SCALE", "1.0"))
FEATURES = int(os.environ.get("NQX_ITER", "0"))


class Harness:
    def __init__(self, dut, rng: random.Random, in_gap: float = 0.2, out_stall: float = 0.2):
        self.dut = dut
        self.rng = rng
        self.in_gap = in_gap
        self.out_stall = out_stall
        self.out = bytearray()

    async def reset(self):
        dut = self.dut
        cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
        dut.rst_n.value = 0
        dut.in_valid.value = 0
        dut.in_data.value = 0
        dut.out_ready.value = 0
        await ClockCycles(dut.clk, 3)
        dut.rst_n.value = 1
        cocotb.start_soon(self._sink())

    async def _sink(self):
        dut = self.dut
        while True:
            await FallingEdge(dut.clk)
            dut.out_ready.value = 0 if self.rng.random() < self.out_stall else 1
            await RisingEdge(dut.clk)
            if int(dut.out_valid.value) and int(dut.out_ready.value):
                self.out.append(int(dut.out_data.value))

    async def send(self, stream: bytes):
        dut = self.dut
        for byte in stream:
            await FallingEdge(dut.clk)
            while self.rng.random() < self.in_gap:
                dut.in_valid.value = 0
                await FallingEdge(dut.clk)
            dut.in_valid.value = 1
            dut.in_data.value = byte
            while True:
                await RisingEdge(dut.clk)
                if int(dut.in_ready.value):
                    break
        await FallingEdge(dut.clk)
        dut.in_valid.value = 0

    async def wait_idle(self, expected_len: int, limit: int = 200000):
        dut = self.dut
        for _ in range(limit):
            await RisingEdge(dut.clk)
            if len(self.out) >= expected_len and not int(dut.busy.value):
                await ClockCycles(dut.clk, 4)
                return
        raise TimeoutError(f"core still busy, got {len(self.out)}/{expected_len} bytes")

    async def run_and_check(self, stream: bytes, model: S1Core | None = None, label: str = ""):
        model = model or S1Core(p=P)
        exp = model.run(stream)
        start = len(self.out)
        await self.send(stream)
        await self.wait_idle(start + len(exp))
        got = bytes(self.out[start:])
        if got != exp:
            first = next((i for i in range(min(len(got), len(exp))) if got[i] != exp[i]),
                         min(len(got), len(exp)))
            raise AssertionError(
                f"{label}: output mismatch at byte {first} (got {len(got)} B, expected {len(exp)} B)"
                f"\n  rtl  : {got[max(0, first - 8):first + 8].hex()}"
                f"\n  model: {exp[max(0, first - 8):first + 8].hex()}"
            )
        return model


def n(count: int) -> int:
    return max(1, int(count * SCALE))


@cocotb.test()
async def csr_and_sync(dut):
    h = Harness(dut, random.Random(SEED))
    await h.reset()
    await h.run_and_check(csr_smoke(), label="csr")
    await h.run_and_check(host.csrr(Csr.FEATURES), S1Core(p=P, features=FEATURES), "features")


@cocotb.test()
async def load_store(dut):
    h = Harness(dut, random.Random(SEED + 1))
    await h.reset()
    rng = random.Random(SEED + 1)
    model = S1Core(p=P)
    for v in edge_vectors(P.dim) + [random_vector(rng, P.dim) for _ in range(n(4))]:
        await h.run_and_check(host.ldv(v) + host.stv() + host.stvr(), model, "ldv/stv")


@cocotb.test()
async def single_ops(dut):
    """Each micro-op on its own, full register readback after every step."""
    h = Harness(dut, random.Random(SEED + 2))
    await h.reset()
    rng = random.Random(SEED + 2)
    model = S1Core(p=P)
    await h.run_and_check(host.csrw(Csr.INC2, 0x2F1BBCDC), model, "inc2")
    for _ in range(n(3)):
        v = random_vector(rng, P.dim)
        seq = [host.ldv(v)]
        for layer in (0, 1, 2):
            seq += [host.gvns(layer), host.stvr()]
        seq += [host.polar(), host.stvr(), host.csrr(Csr.RMIN), host.csrr(Csr.RMAX)]
        seq += [host.quant(), host.ipolar(), host.stvr()]
        for layer in (2, 1, 0):
            seq += [host.gvns(layer, inverse=True), host.stvr()]
        seq += [host.stv(), host.dequant(random_packet(rng, P)), host.stvr()]
        await h.run_and_check(host.join(seq), model, "single-ops")


@cocotb.test()
async def encode_decode(dut):
    h = Harness(dut, random.Random(SEED + 3), in_gap=0.05, out_stall=0.05)
    await h.reset()
    rng = random.Random(SEED + 3)
    model = S1Core(p=P)
    vecs = edge_vectors(P.dim) + [kv_vector(rng, P.dim) for _ in range(n(12))]
    vecs += [random_vector(rng, P.dim) for _ in range(n(6))]
    for i, v in enumerate(vecs):
        if i == len(vecs) // 2:
            await h.run_and_check(host.csrw(Csr.CTRL, 0), model, "refine-off")
        pkt = S1Core(p=P, ctrl=model.ctrl).run(host.enc(v))
        await h.run_and_check(host.enc(v) + host.dec(pkt), model, f"enc/dec #{i}")
    await h.run_and_check(
        host.join([host.csrr(Csr.ENC_COUNT), host.csrr(Csr.DEC_COUNT), host.csrr(Csr.CORDIC_OPS)]),
        model,
        "counters",
    )


@cocotb.test()
async def errors_and_saturation(dut):
    h = Harness(dut, random.Random(SEED + 4))
    await h.reset()
    model = S1Core(p=P)
    big = ((1 << P.dw) - 1).to_bytes(P.hb, "little")
    pkt = bytes(P.hb) + big + bytes([0x77] * P.n_pairs)
    seq = [
        bytes([0x7F]),
        host.csrr(Csr.STATUS),
        bytes([0x10, 0x03]),
        host.csrr(Csr.STATUS),
        host.csrw(Csr.STATUS, 0x1),
        host.csrr(Csr.STATUS),
        host.dequant(pkt),
        host.csrr(Csr.STATUS),
        host.ipolar(),
        host.stv(),
        host.csrr(Csr.STATUS),
        host.csrw(Csr.STATUS, 0x3),
        host.csrr(Csr.STATUS),
    ]
    await h.run_and_check(host.join(seq), model, "errors")


@cocotb.test()
async def cycle_counters(dut):
    h = Harness(dut, random.Random(SEED + 5))
    await h.reset()
    start = len(h.out)
    await h.send(host.csrr(Csr.CYCLES) + host.csrr(Csr.BUSY_CYCLES))
    await h.wait_idle(start + 8)
    await h.send(host.enc([1000] * P.dim) + host.csrr(Csr.CYCLES) + host.csrr(Csr.BUSY_CYCLES))
    await h.wait_idle(start + 16 + P.packet_bytes)
    words = [int.from_bytes(h.out[start + 4 * i:start + 4 * i + 4], "little") for i in range(2)]
    tail = h.out[start + 8 + P.packet_bytes:]
    after = [int.from_bytes(tail[4 * i:4 * i + 4], "little") for i in range(2)]
    assert after[0] > words[0] and after[1] > words[1], (words, after)
    assert after[1] - words[1] >= 3 * P.n_pairs, "busy counter did not cover the encode"


@cocotb.test()
async def random_programs(dut):
    h = Harness(dut, random.Random(SEED + 6), in_gap=0.1, out_stall=0.3)
    await h.reset()
    rng = random.Random(SEED + 6)
    model = S1Core(p=P)
    for i in range(n(8)):
        await h.run_and_check(random_program(rng, P, 25), model, f"random program #{i}")


@cocotb.test()
async def perf_cycles(dut):
    """Busy cycles per command with an ideal 1 byte/cycle host (logged for the spec)."""
    h = Harness(dut, random.Random(SEED + 7), in_gap=0.0, out_stall=0.0)
    await h.reset()
    model = S1Core(p=P)
    v = kv_vector(random.Random(3), P.dim)
    pkt = S1Core(p=P).run(host.enc(v))
    probes = [
        ("LDV", host.ldv(v)),
        ("GVNS 0", host.gvns(0)),
        ("GVNS 1", host.gvns(1)),
        ("GVNS 2 (bypassed)", host.gvns(2)),
        ("POLAR", host.polar()),
        ("QUANT", host.quant()),
        ("IPOLAR", host.ipolar()),
        ("DEQUANT", host.dequant(pkt)),
        ("STV", host.stv()),
        ("ENC", host.enc(v)),
        ("DEC", host.dec(pkt)),
    ]
    rows = []
    for name, cmd in probes:
        start = len(h.out)
        await h.send(host.csrr(Csr.BUSY_CYCLES))
        await h.wait_idle(start + 4)
        b0 = int.from_bytes(h.out[start:start + 4], "little")
        exp = model.run(cmd)
        await h.send(cmd + host.csrr(Csr.BUSY_CYCLES))
        await h.wait_idle(start + 4 + len(exp) + 4)
        b1 = int.from_bytes(h.out[-4:], "little")
        # BUSY_CYCLES also counts the 6 cycles of the CSRR command itself.
        rows.append((name, b1 - b0 - 6, len(cmd), len(exp)))
    for name, cyc, nin, nout in rows:
        dut._log.info(f"PERF {name:<18} busy={cyc:5d} cycles  in={nin:4d} B  out={nout:4d} B")
