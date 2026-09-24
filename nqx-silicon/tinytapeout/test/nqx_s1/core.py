"""Bit-accurate transaction-level model of the NQX-S1 core.

``S1Core.run(stream)`` consumes the same host byte stream the chip consumes and
returns the bytes the chip emits. It is the golden model for RTL verification
and for the silicon bring-up vectors. It is not cycle-accurate; the CYCLES and
BUSY_CYCLES counters are therefore not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from nqx_s1 import cordic, quant
from nqx_s1.params import CHIP_ID, DEFAULT, INC_DEFAULT, PHASE_W, VERSION, S1Params


class Op:
    NOP = 0x00
    LDV = 0x01
    STV = 0x02
    STVR = 0x03
    GVNS = 0x10
    GVNS_INV = 0x11
    POLAR = 0x20
    IPOLAR = 0x21
    QUANT = 0x30
    DEQUANT = 0x31
    ENC = 0x60
    DEC = 0x61
    SYNC = 0x70
    CSRW = 0x71
    CSRR = 0x72


class Csr:
    ID = 0x00
    VERSION = 0x01
    PARAMS = 0x02
    CTRL = 0x03
    STATUS = 0x04
    INC0 = 0x05
    INC1 = 0x06
    INC2 = 0x07
    RMIN = 0x08
    RMAX = 0x09
    SCRATCH = 0x0A
    FEATURES = 0x0B
    CYCLES = 0x10
    BUSY_CYCLES = 0x11
    CORDIC_OPS = 0x12
    ENC_COUNT = 0x13
    DEC_COUNT = 0x14


STATUS_ILLEGAL = 1 << 0
STATUS_SAT = 1 << 1
CTRL_REFINE = 1 << 0
SYNC_BYTE = 0xA5
MASK32 = 0xFFFFFFFF

ENC_UOPS = ("LDV", ("ROT", 0, False), ("ROT", 1, False), ("ROT", 2, False), "POLAR", "QUANT")
DEC_UOPS = ("DEQUANT", "IPOLAR", ("ROT", 2, True), ("ROT", 1, True), ("ROT", 0, True), "STV")


@dataclass
class S1Core:
    p: S1Params = DEFAULT
    rf: list[int] = field(default_factory=list)
    inc: list[int] = field(default_factory=lambda: list(INC_DEFAULT))
    ctrl: int = CTRL_REFINE
    status: int = 0
    rmin: int = 0
    rmax: int = 0
    scratch: int = 0
    cordic_ops: int = 0
    enc_count: int = 0
    dec_count: int = 0
    features: int = 0

    def __post_init__(self) -> None:
        if not self.rf:
            self.rf = [0] * self.p.dim

    # ------------------------------------------------------------------ uops
    def _phase_z(self, layer: int, k: int, inverse: bool) -> int:
        p = self.p
        phase = ((k + 1) * self.inc[layer]) & MASK32
        sh = PHASE_W - p.zw
        z = ((phase + (1 << (sh - 1))) >> sh) & ((1 << p.zw) - 1)
        if inverse:
            z = (-z) & ((1 << p.zw) - 1)
        return z

    def rot_layer(self, layer: int, inverse: bool) -> None:
        if self.inc[layer] == 0:
            return
        new = list(self.rf)
        for i, j, k in self.p.layer_pairs(layer):
            z = self._phase_z(layer, k, inverse)
            a, b, sat = cordic.rotate(self.p, self.rf[i], self.rf[j], z)
            new[i], new[j] = a, b
            self.cordic_ops += 1
            if sat:
                self.status |= STATUS_SAT
        self.rf = new

    def polar(self) -> None:
        first = True
        for k in range(self.p.n_pairs):
            r, t, sat = cordic.vector(self.p, self.rf[2 * k], self.rf[2 * k + 1])
            self.rf[2 * k], self.rf[2 * k + 1] = r, t
            self.cordic_ops += 1
            if sat:
                self.status |= STATUS_SAT
            if first:
                self.rmin = self.rmax = r
                first = False
            else:
                self.rmin = min(self.rmin, r)
                self.rmax = max(self.rmax, r)

    def ipolar(self) -> None:
        for k in range(self.p.n_pairs):
            z = self.rf[2 * k + 1] & ((1 << self.p.zw) - 1)
            a, b, sat = cordic.rotate(self.p, self.rf[2 * k], 0, z)
            self.rf[2 * k], self.rf[2 * k + 1] = a, b
            self.cordic_ops += 1
            if sat:
                self.status |= STATUS_SAT

    def quant_bytes(self) -> bytes:
        p = self.p
        out = bytearray()
        out += (self.rmin & ((1 << (8 * p.hb)) - 1)).to_bytes(p.hb, "little")
        out += (self.rmax & ((1 << (8 * p.hb)) - 1)).to_bytes(p.hb, "little")
        for k in range(p.n_pairs):
            out.append(
                quant.encode_pair(p, self.rf[2 * k], self.rf[2 * k + 1], self.rmin, self.rmax)
            )
        self.enc_count += 1
        return bytes(out)

    def dequant_bytes(self, packet: bytes) -> None:
        p = self.p
        lim = (1 << (p.dw - 1)) - 1
        mask = (1 << p.dw) - 1
        rmin = int.from_bytes(packet[: p.hb], "little") & mask
        rmax = int.from_bytes(packet[p.hb : 2 * p.hb], "little") & mask
        if rmin > lim or rmax > lim:
            self.status |= STATUS_SAT
        rmin, rmax = min(rmin, lim), min(rmax, lim)
        step = quant.radius_step(p, rmin, rmax)
        refine = bool(self.ctrl & CTRL_REFINE)
        for k in range(p.n_pairs):
            r, t, sat = quant.decode_pair(p, packet[2 * p.hb + k], rmin, step, refine)
            self.rf[2 * k], self.rf[2 * k + 1] = r, t
            if sat:
                self.status |= STATUS_SAT
        self.dec_count += 1

    def load(self, data: bytes) -> None:
        for i in range(self.p.dim):
            v = int.from_bytes(data[2 * i : 2 * i + 2], "little", signed=True)
            self.rf[i] = v << self.p.fw

    def store16(self) -> bytes:
        out = bytearray()
        fw = self.p.fw
        for v in self.rf:
            o, sat = cordic.saturate((v + (1 << (fw - 1))) >> fw, 16)
            if sat:
                self.status |= STATUS_SAT
            out += (o & 0xFFFF).to_bytes(2, "little")
        return bytes(out)

    def store_raw(self) -> bytes:
        hb = self.p.hb
        return b"".join((v & ((1 << (8 * hb)) - 1)).to_bytes(hb, "little") for v in self.rf)

    # ------------------------------------------------------------------ CSRs
    def csr_read(self, addr: int) -> int:
        p = self.p
        table = {
            Csr.ID: CHIP_ID,
            Csr.VERSION: VERSION,
            Csr.PARAMS: p.params_word(),
            Csr.CTRL: self.ctrl,
            Csr.STATUS: self.status,
            Csr.INC0: self.inc[0],
            Csr.INC1: self.inc[1],
            Csr.INC2: self.inc[2],
            Csr.RMIN: self.rmin,
            Csr.RMAX: self.rmax,
            Csr.SCRATCH: self.scratch,
            Csr.FEATURES: self.features,
            Csr.CORDIC_OPS: self.cordic_ops,
            Csr.ENC_COUNT: self.enc_count,
            Csr.DEC_COUNT: self.dec_count,
        }
        return table.get(addr, 0) & MASK32

    def csr_write(self, addr: int, data: int) -> None:
        data &= MASK32
        if addr == Csr.CTRL:
            self.ctrl = data & CTRL_REFINE
        elif addr == Csr.STATUS:
            self.status &= ~data
        elif addr in (Csr.INC0, Csr.INC1, Csr.INC2):
            self.inc[addr - Csr.INC0] = data
        elif addr == Csr.SCRATCH:
            self.scratch = data

    # ------------------------------------------------------------------ stream
    def run(self, stream: bytes) -> bytes:
        it: Iterator[int] = iter(stream)
        out = bytearray()
        p = self.p

        def take(n: int) -> bytes:
            b = bytes(next(it) for _ in range(n))
            return b

        def uop(u) -> None:
            if u == "LDV":
                self.load(take(2 * p.dim))
            elif u == "STV":
                out.extend(self.store16())
            elif u == "POLAR":
                self.polar()
            elif u == "IPOLAR":
                self.ipolar()
            elif u == "QUANT":
                out.extend(self.quant_bytes())
            elif u == "DEQUANT":
                self.dequant_bytes(take(p.packet_bytes))
            else:
                _, layer, inverse = u
                self.rot_layer(layer, inverse)

        for op in it:
            if op == Op.NOP:
                continue
            if op == Op.LDV:
                uop("LDV")
            elif op == Op.STV:
                uop("STV")
            elif op == Op.STVR:
                out.extend(self.store_raw())
            elif op in (Op.GVNS, Op.GVNS_INV):
                layer = take(1)[0]
                if layer > 2:
                    self.status |= STATUS_ILLEGAL
                else:
                    self.rot_layer(layer, op == Op.GVNS_INV)
            elif op == Op.POLAR:
                uop("POLAR")
            elif op == Op.IPOLAR:
                uop("IPOLAR")
            elif op == Op.QUANT:
                uop("QUANT")
            elif op == Op.DEQUANT:
                uop("DEQUANT")
            elif op == Op.ENC:
                for u in ENC_UOPS:
                    uop(u)
            elif op == Op.DEC:
                for u in DEC_UOPS:
                    uop(u)
            elif op == Op.SYNC:
                out.append(SYNC_BYTE)
            elif op == Op.CSRW:
                addr = take(1)[0]
                self.csr_write(addr, int.from_bytes(take(4), "little"))
            elif op == Op.CSRR:
                addr = take(1)[0]
                out.extend(self.csr_read(addr).to_bytes(4, "little"))
            else:
                self.status |= STATUS_ILLEGAL
        return bytes(out)
