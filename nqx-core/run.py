#!/usr/bin/env python3
"""NQX-Core CLI: assemble & run programs, benchmark, verify."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nqx.constants import NQXConfig
from nqx.cpu import NQXCore
from nqx.assembler import assemble


def _make_data(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32) * 0.5
    for col in (0, 15, 31, 63, 95, 127):
        if col < d:
            mask = rng.random(n) < 0.75
            x[mask, col] = rng.standard_normal(int(mask.sum())).astype(np.float32) * 30.0
    return x


def cmd_run(args: argparse.Namespace) -> int:
    cfg = NQXConfig(dim=args.dim, bits=args.bits)
    core = NQXCore(cfg)
    x = _make_data(args.vectors, args.dim, seed=args.seed)
    core.load_vectors_to_hbm(0, x)

    with open(args.program, "r") as f:
        program = assemble(f.read())
    if program and program[0].opcode.name == "LDV":
        program[0].extra["count"] = args.vectors

    t0 = time.perf_counter()
    res = core.execute_program(program)
    dt = time.perf_counter() - t0

    print(f"--- Program: {args.program} ---")
    print(f"halted: {res['halted']}")
    print(f"vectors: {args.vectors}, dim: {args.dim}, bits: {args.bits}")
    print(f"wall time:    {dt*1000:.1f} ms")
    print(core.cycles.report())
    print(core.energy.report())
    if args.trace:
        print("\nTrace:")
        for line in core.trace_log[:50]:
            print(f"  {line}")
        if len(core.trace_log) > 50:
            print(f"  ... ({len(core.trace_log) - 50} more)")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    cfg = NQXConfig(dim=args.dim, bits=args.bits)
    core = NQXCore(cfg)
    x = _make_data(args.vectors, args.dim, seed=args.seed)

    print(f"=== NQX bench: dim={args.dim} bits={args.bits} batch={args.vectors} ===")
    print(core.lut.summary())

    t0 = time.perf_counter()
    enc = core.encode(x)
    enc_dt = time.perf_counter() - t0

    print(f"\nEncode:")
    print(f"  wall time:        {enc_dt*1000:.1f} ms")
    print(f"  cycles:           {enc.cycles}")
    print(f"  ipc throughput:   {args.vectors / max(enc.cycles, 1):.4f} vec/cycle")
    print(f"  packed bytes:     {len(enc.packed_bytes)}")
    print(f"  raw FP16 bytes:   {x.size * 2}")
    print(f"  compression:      {(x.size * 2) / max(len(enc.packed_bytes), 1):.2f}x")

    t0 = time.perf_counter()
    dec = core.decode(enc)
    dec_dt = time.perf_counter() - t0

    rmse = float(np.sqrt(((x - dec.reconstructed) ** 2).mean()))
    norm_in = np.linalg.norm(x, axis=-1).mean()
    norm_out = np.linalg.norm(dec.reconstructed, axis=-1).mean()
    print(f"\nDecode:")
    print(f"  wall time:        {dec_dt*1000:.1f} ms")
    print(f"  cycles:           {dec.cycles}")
    print(f"  RMSE vs original: {rmse:.6f}")
    print(f"  norm preservation:{norm_in:.4f} -> {norm_out:.4f}")

    print()
    if not args.quiet:
        print(core.energy.report())
        print(f"\nEnergy per vector: {core.energy.total_nj()/args.vectors:.3f} nJ/vec")
        naive_fp16_nj = (args.vectors * args.dim * 2 * 2 * cfg.pj_hbm_byte) / 1000.0
        print(f"Naive FP16 RW HBM: {naive_fp16_nj:.1f} nJ ({naive_fp16_nj/args.vectors:.3f} nJ/vec)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    cfg = NQXConfig(dim=args.dim, bits=args.bits)
    core = NQXCore(cfg)
    print(f"=== Verifying NQX dim={args.dim} ===")

    T = core.rotation_matrix()
    err = np.abs(T.T @ T - np.eye(args.dim)).max()
    print(f"Orthogonality T^T*T = I:  max err = {err:.3e}")
    assert err < 1e-4, "FAILED: orthogonality"

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((50, args.dim)).astype(np.float32)
    y = core.forward_rotation(x)
    n_in = np.linalg.norm(x, axis=-1).mean()
    n_out = np.linalg.norm(y, axis=-1).mean()
    print(f"Norm preservation:        {n_in:.4f} -> {n_out:.4f} (err {abs(n_in-n_out):.2e})")

    x_back = core.inverse_rotation(y)
    rt_rmse = float(np.sqrt(((x - x_back) ** 2).mean()))
    print(f"Roundtrip (no quant):     RMSE = {rt_rmse:.3e}")
    assert rt_rmse < 1e-4, "FAILED: roundtrip"

    enc = core.encode(x)
    dec = core.decode(enc)
    quant_rmse = float(np.sqrt(((x - dec.reconstructed) ** 2).mean()))
    print(f"Roundtrip (3-bit quant):  RMSE = {quant_rmse:.4f}")

    if args.vs_reference:
        try:
            sys.path.insert(0, "/tmp/naut/reference")
            import torch
            from nautilus_triton import NautilusConfig, NautilusQuantPyTorch
            ref = NautilusQuantPyTorch(NautilusConfig(dim=args.dim, bits=args.bits))
            x_t = torch.from_numpy(x)
            y_ref = ref.forward(x_t).numpy()
            y_nqx = core.forward_rotation(x)
            diff = np.abs(y_ref - y_nqx).max()
            print(f"vs reference forward:    max diff = {diff:.3e}")
            assert diff < 1e-4, "FAILED: reference mismatch"
        except ImportError:
            print("vs reference: torch / reference repo not available; skipping")
    print("\nAll checks PASSED")
    return 0


def main():
    parser = argparse.ArgumentParser(prog="nqx", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run an NQ-ASM program")
    p_run.add_argument("program")
    p_run.add_argument("--vectors", type=int, default=128)
    p_run.add_argument("--dim", type=int, default=128)
    p_run.add_argument("--bits", type=int, default=3)
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--trace", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_b = sub.add_parser("bench", help="benchmark encode/decode pipeline")
    p_b.add_argument("--vectors", type=int, default=4096)
    p_b.add_argument("--dim", type=int, default=128)
    p_b.add_argument("--bits", type=int, default=3)
    p_b.add_argument("--seed", type=int, default=42)
    p_b.add_argument("--quiet", action="store_true", help="suppress energy report")
    p_b.set_defaults(func=cmd_bench)

    p_v = sub.add_parser("verify", help="run correctness checks")
    p_v.add_argument("--dim", type=int, default=128)
    p_v.add_argument("--bits", type=int, default=3)
    p_v.add_argument("--seed", type=int, default=0)
    p_v.add_argument("--quiet", action="store_true", help="suppress energy report")
    p_v.add_argument("--vs-reference", action="store_true",
                     help="cross-check against /tmp/naut NautilusQuant reference")
    p_v.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
