"""Stimulus shared by the cocotb tests and the silicon test-vector generator."""

from __future__ import annotations

import random

from nqx_s1 import Csr, S1Params, host
from nqx_s1.params import INC_DEFAULT

I16_MIN, I16_MAX = -32768, 32767


def kv_vector(rng: random.Random, dim: int) -> list[int]:
    """KV-cache-like vector: Gaussian body, a few large outlier channels."""
    scale = rng.choice([64, 512, 2048, 4096])
    out = [round(rng.gauss(0.0, 1.0) * scale) for _ in range(dim)]
    for c in rng.sample(range(dim), max(1, dim // 20)):
        if rng.random() < 0.5:
            out[c] = round(rng.gauss(0.0, 1.0) * scale * 30)
    return [max(I16_MIN, min(I16_MAX, v)) for v in out]


def full_scale_vector(rng: random.Random, dim: int) -> list[int]:
    return [rng.randint(I16_MIN, I16_MAX) for _ in range(dim)]


def edge_vectors(dim: int) -> list[list[int]]:
    spike = [0] * dim
    spike[dim // 3] = I16_MAX
    return [
        [0] * dim,
        [I16_MAX] * dim,
        [I16_MIN] * dim,
        [I16_MIN if i % 2 else I16_MAX for i in range(dim)],
        [1 if i % 2 else -1 for i in range(dim)],
        spike,
        list(range(-dim // 2, dim // 2)),
    ]


def random_vector(rng: random.Random, dim: int) -> list[int]:
    kind = rng.random()
    if kind < 0.6:
        return kv_vector(rng, dim)
    if kind < 0.9:
        return full_scale_vector(rng, dim)
    return rng.choice(edge_vectors(dim))


def random_packet(rng: random.Random, p: S1Params) -> bytes:
    lim = (1 << (p.dw - 1)) - 1
    if rng.random() < 0.2:
        rmin = rng.randint(0, (1 << p.dw) - 1)
        rmax = rng.randint(0, (1 << p.dw) - 1)
    else:
        rmin = rng.randint(0, lim // 64)
        rmax = rng.randint(rmin, lim // 8)
    body = bytes(rng.randrange(256) for _ in range(p.n_pairs))
    return rmin.to_bytes(p.hb, "little") + rmax.to_bytes(p.hb, "little") + body


def csr_smoke() -> bytes:
    return host.join(
        [
            host.csrr(Csr.ID),
            host.csrr(Csr.VERSION),
            host.csrr(Csr.PARAMS),
            host.csrr(Csr.CTRL),
            host.csrr(Csr.STATUS),
            host.csrr(Csr.INC0),
            host.csrr(Csr.INC1),
            host.csrr(Csr.INC2),
            host.csrw(Csr.SCRATCH, 0xDEADBEEF),
            host.csrr(Csr.SCRATCH),
            host.csrr(0x55),
            host.sync(),
        ]
    )


def random_program(rng: random.Random, p: S1Params, n_ops: int) -> bytes:
    """Constrained-random command stream. Never reads CYCLES / BUSY_CYCLES."""
    parts = [host.ldv(random_vector(rng, p.dim))]
    for _ in range(n_ops):
        r = rng.random()
        if r < 0.12:
            parts.append(host.ldv(random_vector(rng, p.dim)))
        elif r < 0.30:
            parts.append(host.gvns(rng.randrange(3), inverse=rng.random() < 0.5))
        elif r < 0.38:
            parts.append(host.polar())
        elif r < 0.44:
            parts.append(host.ipolar())
        elif r < 0.50:
            parts.append(host.quant())
        elif r < 0.55:
            parts.append(host.dequant(random_packet(rng, p)))
        elif r < 0.62:
            parts.append(host.enc(random_vector(rng, p.dim)))
        elif r < 0.66:
            parts.append(host.dec(random_packet(rng, p)))
        elif r < 0.72:
            parts.append(host.stvr())
        elif r < 0.78:
            parts.append(host.stv())
        elif r < 0.82:
            layer = rng.randrange(3)
            inc = rng.choice([0, rng.getrandbits(32), INC_DEFAULT[layer]])
            parts.append(host.csrw(Csr.INC0 + layer, inc))
        elif r < 0.85:
            parts.append(host.csrw(Csr.CTRL, rng.getrandbits(32)))
        elif r < 0.88:
            parts.append(host.csrw(Csr.STATUS, rng.getrandbits(2)))
        elif r < 0.92:
            parts.append(
                host.csrr(
                    rng.choice(
                        [Csr.STATUS, Csr.RMIN, Csr.RMAX, Csr.CORDIC_OPS, Csr.ENC_COUNT, Csr.DEC_COUNT]
                    )
                )
            )
        elif r < 0.95:
            parts.append(bytes([rng.choice([0x04, 0x12, 0x40, 0x7F, 0xFF])]))
        elif r < 0.97:
            parts.append(bytes([0x10, rng.randrange(3, 256)]))
        else:
            parts.append(host.sync())
    parts.append(host.stvr())
    parts.append(host.csrr(Csr.STATUS))
    parts.append(host.csrr(Csr.CORDIC_OPS))
    return host.join(parts)
