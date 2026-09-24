"""Host-side command builders for the NQX-S1 byte protocol."""

from __future__ import annotations

from typing import Iterable, Sequence

from nqx_s1.core import Op


def _vec_bytes(vec: Sequence[int]) -> bytes:
    return b"".join((int(v) & 0xFFFF).to_bytes(2, "little") for v in vec)


def ldv(vec: Sequence[int]) -> bytes:
    return bytes([Op.LDV]) + _vec_bytes(vec)


def stv() -> bytes:
    return bytes([Op.STV])


def stvr() -> bytes:
    return bytes([Op.STVR])


def gvns(layer: int, inverse: bool = False) -> bytes:
    return bytes([Op.GVNS_INV if inverse else Op.GVNS, layer])


def polar() -> bytes:
    return bytes([Op.POLAR])


def ipolar() -> bytes:
    return bytes([Op.IPOLAR])


def quant() -> bytes:
    return bytes([Op.QUANT])


def dequant(packet: bytes) -> bytes:
    return bytes([Op.DEQUANT]) + packet


def enc(vec: Sequence[int]) -> bytes:
    return bytes([Op.ENC]) + _vec_bytes(vec)


def dec(packet: bytes) -> bytes:
    return bytes([Op.DEC]) + packet


def sync() -> bytes:
    return bytes([Op.SYNC])


def csrw(addr: int, data: int) -> bytes:
    return bytes([Op.CSRW, addr]) + (data & 0xFFFFFFFF).to_bytes(4, "little")


def csrr(addr: int) -> bytes:
    return bytes([Op.CSRR, addr])


def join(parts: Iterable[bytes]) -> bytes:
    return b"".join(parts)


def unpack16(data: bytes) -> list[int]:
    return [int.from_bytes(data[i : i + 2], "little", signed=True) for i in range(0, len(data), 2)]
