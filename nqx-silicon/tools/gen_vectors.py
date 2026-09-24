#!/usr/bin/env python3
"""Generate golden test vectors for NQX-S1 silicon bring-up and production test.

Each file is a sequence of transactions:

    W <hex>      bytes the host writes (continued on following W lines)
    R <hex>      bytes the chip must return for the preceding W bytes
    .            end of transaction

Lines starting with '#' are comments. Expected output never contains
CYCLES/BUSY_CYCLES reads (timing-dependent), so files compare bit-exactly.

Usage: python tools/gen_vectors.py [--out vectors]
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "model"), str(ROOT / "verif" / "cocotb")]

from nqx_s1 import Csr, S1Core, S1Params, host  # noqa: E402
from nqx_s1.params import CHIP_ID, VERSION  # noqa: E402
from stimulus import (  # noqa: E402
    csr_smoke,
    edge_vectors,
    full_scale_vector,
    kv_vector,
    random_packet,
    random_program,
)

P = S1Params()
LINE = 32


def patterns() -> list[list[int]]:
    """Vector-register march: every VR bit that LDV can reach toggles both ways."""
    d = P.dim
    return [
        [0] * d,
        [-1] * d,
        [0x5555] * d,
        [-0x5556] * d,  # 0xAAAA
        [0x5555 if i % 2 else -0x5556 for i in range(d)],
        [-0x5556 if i % 2 else 0x5555 for i in range(d)],
        [1 << (i % 16) if (i % 16) != 15 else -32768 for i in range(d)],
        [i * 257 - 16384 for i in range(d)],
        [0] * d,
    ]


def suite() -> dict[str, list[bytes]]:
    rng = random.Random(2026)
    s: dict[str, list[bytes]] = {}

    s["00_identity"] = [csr_smoke(), host.csrr(Csr.CORDIC_OPS) + host.csrr(Csr.ENC_COUNT)]

    s["01_vector_register"] = [host.ldv(v) + host.stvr() + host.stv() for v in patterns()]

    rot = []
    for v in edge_vectors(P.dim)[:4] + [kv_vector(rng, P.dim) for _ in range(4)]:
        t = host.ldv(v)
        for layer in (0, 1, 2):
            t += host.gvns(layer) + host.stvr()
        for layer in (2, 1, 0):
            t += host.gvns(layer, inverse=True) + host.stvr()
        rot.append(t + host.stv())
    s["02_rotation"] = rot

    s["03_polar"] = [
        host.ldv(v) + host.gvns(0) + host.gvns(1) + host.polar() + host.stvr()
        + host.csrr(Csr.RMIN) + host.csrr(Csr.RMAX) + host.ipolar() + host.stvr()
        for v in [kv_vector(rng, P.dim) for _ in range(6)] + edge_vectors(P.dim)[1:4]
    ]

    enc = []
    vecs = edge_vectors(P.dim) + [kv_vector(rng, P.dim) for _ in range(24)]
    vecs += [full_scale_vector(rng, P.dim) for _ in range(8)]
    for v in vecs:
        pkt = S1Core().run(host.enc(v))
        enc.append(host.enc(v) + host.dec(pkt))
    s["04_encode_decode"] = enc

    s["05_refine_off"] = [host.csrw(Csr.CTRL, 0)] + [
        host.dec(random_packet(rng, P)) for _ in range(6)
    ] + [host.csrw(Csr.CTRL, 1) + host.csrr(Csr.CTRL)]

    inc = []
    for incs in [(0x2F1BBCDC, 0x5A827999, 0x6ED9EBA1), (0, 0, 0), (0xFFFFFFFF, 1, 0x80000000)]:
        t = b"".join(host.csrw(Csr.INC0 + i, x) for i, x in enumerate(incs))
        v = kv_vector(rng, P.dim)
        t += host.enc(v) + host.ldv(v) + host.gvns(2) + host.stvr()
        inc.append(t)
    inc.append(
        b"".join(host.csrw(Csr.INC0 + i, x) for i, x in enumerate((0x61C88647, 0x9E3779B9, 0)))
    )
    s["06_programmable_rotation"] = inc

    big = ((1 << P.dw) - 1).to_bytes(P.hb, "little")
    s["07_errors"] = [
        bytes([0x7F]) + host.csrr(Csr.STATUS),
        bytes([0x10, 0x05]) + host.csrr(Csr.STATUS),
        host.csrw(Csr.STATUS, 3) + host.csrr(Csr.STATUS),
        host.dequant(bytes(P.hb) + big + bytes([0x77] * P.n_pairs)) + host.csrr(Csr.STATUS),
        host.ipolar() + host.stv() + host.csrr(Csr.STATUS),
        host.csrw(Csr.STATUS, 3) + host.csrr(Csr.STATUS),
    ]

    s["08_random_programs"] = [random_program(rng, P, 30) for _ in range(6)]
    return s


def write_file(path: Path, name: str, transactions: list[bytes]) -> str:
    core = S1Core()
    digest = hashlib.sha256()
    lines = [
        f"# NQX-S1 golden vectors: {name}",
        f"# chip id 0x{CHIP_ID:08X}, version 0x{VERSION:08X}, DIM={P.dim}",
        "# start from reset; W = host writes, R = chip must return, '.' ends a transaction",
    ]
    for t in transactions:
        exp = core.run(t)
        digest.update(t)
        digest.update(exp)
        for i in range(0, len(t), LINE):
            lines.append("W " + t[i:i + LINE].hex())
        for i in range(0, len(exp), LINE):
            lines.append("R " + exp[i:i + LINE].hex())
        lines.append(".")
    path.write_text("\n".join(lines) + "\n")
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "vectors")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = ["# sha256 over (write, expected) bytes of every transaction", ""]
    for name, tx in suite().items():
        h = write_file(args.out / f"{name}.txt", name, tx)
        n_w = sum(len(t) for t in tx)
        manifest.append(f"{h}  {name}.txt  ({len(tx)} transactions, {n_w} bytes written)")
        print(manifest[-1])
    (args.out / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main()
