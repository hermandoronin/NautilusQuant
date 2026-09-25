"""CORDIC unit test: RTL vs model/nqx_s1/cordic.py, one operand per cycle."""

from __future__ import annotations

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, RisingEdge

from nqx_s1 import S1Params
from nqx_s1.cordic import rotate, vector

P = S1Params(dim=int(os.environ.get("NQX_DIM", "128")))
N_RANDOM = int(os.environ.get("NQX_CORDIC_N", "4000"))


def operands(rng: random.Random):
    lo, hi = -(1 << (P.dw - 1)), (1 << (P.dw - 1)) - 1
    zmax = (1 << P.zw) - 1
    corner_z = [0, 1, zmax, 1 << (P.zw - 2), (1 << (P.zw - 2)) - 1, 1 << (P.zw - 1),
                (1 << (P.zw - 1)) - 1, 3 << (P.zw - 2), (3 << (P.zw - 2)) - 1]
    corner_xy = [0, 1, -1, lo, hi, lo + 1, hi - 1]
    for x in corner_xy:
        for y in corner_xy:
            yield (1, x, y, 0)
            for z in corner_z[:4]:
                yield (0, x, y, z)
    for z in corner_z:
        yield (0, hi, 0, z)
    for _ in range(N_RANDOM):
        mode = rng.randrange(2)
        mag = rng.choice([P.dw - 1, P.dw - 4, 20, 12, 4])
        x = rng.randint(-(1 << mag), (1 << mag) - 1)
        y = rng.randint(-(1 << mag), (1 << mag) - 1)
        x = max(lo, min(hi, x))
        y = max(lo, min(hi, y))
        yield (mode, x, y, rng.randrange(1 << P.zw))


def expected(op):
    mode, x, y, z = op
    if mode:
        r, t, sat = vector(P, x, y)
        return ("vec", r, t & ((1 << P.zw) - 1), int(sat))
    a, b, sat = rotate(P, x, y, z)
    return ("rot", a, b, int(sat))


@cocotb.test()
async def cordic_matches_model(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_vec.value = 0
    dut.in_x.value = 0
    dut.in_y.value = 0
    dut.in_z.value = 0
    dut.in_tag.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1

    ops = list(operands(random.Random(1234)))
    got: list[tuple] = []

    async def monitor():
        while True:
            await RisingEdge(dut.clk)
            if int(dut.out_valid.value):
                tag = int(dut.out_tag.value)
                if ops[tag][0]:
                    got.append((tag, "vec", dut.out_x.value.to_signed(), int(dut.out_z.value),
                                int(dut.out_sat.value)))
                else:
                    got.append((tag, "rot", dut.out_x.value.to_signed(),
                                dut.out_y.value.to_signed(), int(dut.out_sat.value)))

    cocotb.start_soon(monitor())
    rng = random.Random(99)
    for tag, (mode, x, y, z) in enumerate(ops):
        await FallingEdge(dut.clk)
        while rng.random() < 0.1:
            dut.in_valid.value = 0
            await FallingEdge(dut.clk)
        dut.in_valid.value = 1
        dut.in_vec.value = mode
        dut.in_x.value = x
        dut.in_y.value = y
        dut.in_z.value = z
        dut.in_tag.value = tag
    await FallingEdge(dut.clk)
    dut.in_valid.value = 0
    await ClockCycles(dut.clk, P.cordic_latency + 4)

    assert len(got) == len(ops), f"expected {len(ops)} results, got {len(got)}"
    bad = 0
    for tag, *res in got:
        exp = expected(ops[tag])
        if tuple(res) != exp:
            bad += 1
            if bad <= 10:
                dut._log.error(f"op {ops[tag]}: rtl {tuple(res)} != model {exp}")
    assert bad == 0, f"{bad} mismatches out of {len(ops)}"
    dut._log.info(f"{len(ops)} CORDIC operations bit-exact")
