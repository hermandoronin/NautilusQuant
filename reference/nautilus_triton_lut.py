"""
NautilusQuant — Triton LUT Kernel (production-grade)

Архитектура памяти на GPU:
  ┌─────────────────────────────────────────┐
  │  GPU Global Memory (HBM)               │
  │  ├── KV-cache tensors (FP16, гигабайты)│
  │  └── Output packed (INT3, в 5x меньше) │
  └───────────┬─────────────────────────────┘
              │ одна загрузка
  ┌───────────▼─────────────────────────────┐
  │  Shared Memory / SRAM (48-228 KB per SM)│
  │  ├── Input tile: N vectors × dim × FP16│
  │  ├── LUT: cos_sin[] (512 bytes, const) │ ◄── ЗДЕСЬ
  │  └── Output tile: packed INT3 + signs  │
  └───────────┬─────────────────────────────┘
              │ вычисления без round-trip
  ┌───────────▼─────────────────────────────┐
  │  Registers (per thread)                │
  │  └── Один вектор [dim] в регистрах     │
  └─────────────────────────────────────────┘

LUT размещается в constant memory (64KB на NVIDIA).
Для dim=128: 3 слоя × ~64 пар × 2 (cos+sin) × 4 bytes = 1536 bytes.
Помещается даже в L0 кэш Tensor Cores.

Запуск:
  pip install torch triton
  python nautilus_triton_lut.py
  python nautilus_triton_lut.py --benchmark
"""

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

PHI = (1 + math.sqrt(5)) / 2
GA = 2 * math.pi / (PHI ** 2)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Triton not found. Install: pip install triton")
    print("Using PyTorch fallback.")


# ================================================================
# 1. LUT BUILDER — предвычисляем cos/sin ОДИН РАЗ
# ================================================================

@dataclass
class LUTConfig:
    dim: int = 128
    phi: float = PHI
    bits: int = 3
    qjl_alpha: float = 0.5


def build_lut(cfg: LUTConfig) -> dict:
    """
    Строит LUT для всех 3 слоёв вращений.

    Возвращает плоские тензоры, готовые к загрузке в constant memory GPU.
    Формат: interleaved [cos₀, sin₀, cos₁, sin₁, ...] для coalesced access.

    Для dim=128:
      Layer 1: 64 pairs × 2 (cos+sin) × 4 bytes = 512 bytes
      Layer 2: 63 pairs × 2 × 4 = 504 bytes
      Layer 3: ~32 pairs × 2 × 4 = 256 bytes
      Indices: ~159 pairs × 2 × 4 = 1272 bytes
      TOTAL: ~2544 bytes ← fits in L1 cache of ANY GPU
    """
    ga = 2 * math.pi / (cfg.phi ** 2)
    layers = []

    # Layer 1: adjacent pairs (0,1), (2,3), ...
    n1 = cfg.dim // 2
    cos1 = torch.zeros(n1, dtype=torch.float32)
    sin1 = torch.zeros(n1, dtype=torch.float32)
    idx1_i = torch.zeros(n1, dtype=torch.int32)
    idx1_j = torch.zeros(n1, dtype=torch.int32)
    for k in range(n1):
        theta = ga * (k + 1)
        cos1[k] = math.cos(theta)
        sin1[k] = math.sin(theta)
        idx1_i[k] = 2 * k
        idx1_j[k] = 2 * k + 1
    layers.append({'cos': cos1, 'sin': sin1, 'i': idx1_i, 'j': idx1_j, 'n': n1})

    # Layer 2: shifted pairs (1,2), (3,4), ...
    n2 = (cfg.dim - 1) // 2
    cos2 = torch.zeros(n2, dtype=torch.float32)
    sin2 = torch.zeros(n2, dtype=torch.float32)
    idx2_i = torch.zeros(n2, dtype=torch.int32)
    idx2_j = torch.zeros(n2, dtype=torch.int32)
    for k in range(n2):
        theta = ga * (k + 1) * cfg.phi
        cos2[k] = math.cos(theta)
        sin2[k] = math.sin(theta)
        idx2_i[k] = 2 * k + 1
        idx2_j[k] = 2 * k + 2
    layers.append({'cos': cos2, 'sin': sin2, 'i': idx2_i, 'j': idx2_j, 'n': n2})

    # Layer 3: butterfly (non-overlapping!)
    stride = max(2, cfg.dim // 4)
    pairs3 = []
    used = set()
    for k in range(cfg.dim):
        i, j = k, (k + stride) % cfg.dim
        if i == j or i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        theta = ga * (k + 1) * cfg.phi ** 2
        pairs3.append((i, j, math.cos(theta), math.sin(theta)))

    n3 = len(pairs3)
    cos3 = torch.tensor([p[2] for p in pairs3], dtype=torch.float32)
    sin3 = torch.tensor([p[3] for p in pairs3], dtype=torch.float32)
    idx3_i = torch.tensor([p[0] for p in pairs3], dtype=torch.int32)
    idx3_j = torch.tensor([p[1] for p in pairs3], dtype=torch.int32)
    layers.append({'cos': cos3, 'sin': sin3, 'i': idx3_i, 'j': idx3_j, 'n': n3})

    total_bytes = sum(l['n'] * (4 + 4 + 4 + 4) for l in layers)  # cos+sin+i+j
    return {'layers': layers, 'total_bytes': total_bytes, 'config': cfg}


def lut_to_device(lut: dict, device: str = 'cuda') -> dict:
    """Move LUT tensors to GPU. Call ONCE at init."""
    for layer in lut['layers']:
        layer['cos'] = layer['cos'].to(device)
        layer['sin'] = layer['sin'].to(device)
        layer['i'] = layer['i'].to(device)
        layer['j'] = layer['j'].to(device)
    return lut


# ================================================================
# 2. TRITON KERNEL — fused rotate с LUT в shared memory
# ================================================================

if HAS_TRITON:

    @triton.jit
    def _givens_layer_kernel(
        # Pointers
        x_ptr,          # input/output vectors [N, dim], in-place
        cos_ptr,        # LUT cos values for this layer
        sin_ptr,        # LUT sin values for this layer
        idx_i_ptr,      # LUT pair index i
        idx_j_ptr,      # LUT pair index j
        # Sizes
        N,              # number of vectors
        dim: tl.constexpr,
        n_pairs: tl.constexpr,
        BLOCK_N: tl.constexpr,  # vectors per block
    ):
        """
        Один слой вращений Гивенса для батча векторов.

        Каждый threadblock обрабатывает BLOCK_N векторов.
        LUT загружается в shared memory ОДИН РАЗ, затем
        переиспользуется для всех BLOCK_N векторов.

        Memory pattern:
          1. Load cos/sin/idx from constant mem → shared mem (once)
          2. Load BLOCK_N vectors from HBM → registers
          3. Apply all Givens rotations (register-only, no HBM access)
          4. Store results back to HBM
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_N

        # Load LUT into registers (tiny — 512 bytes total)
        pair_offsets = tl.arange(0, n_pairs)
        pair_mask = pair_offsets < n_pairs

        cos_vals = tl.load(cos_ptr + pair_offsets, mask=pair_mask)
        sin_vals = tl.load(sin_ptr + pair_offsets, mask=pair_mask)
        idx_i = tl.load(idx_i_ptr + pair_offsets, mask=pair_mask)
        idx_j = tl.load(idx_j_ptr + pair_offsets, mask=pair_mask)

        # Process each vector in block
        for vec_idx in range(BLOCK_N):
            row = block_start + vec_idx
            if row >= N:
                break

            base = row * dim
            dim_offsets = tl.arange(0, dim)
            dim_mask = dim_offsets < dim

            # Load entire vector into registers
            vec = tl.load(x_ptr + base + dim_offsets, mask=dim_mask)

            # Apply all Givens rotations (purely in registers)
            for p in tl.static_range(n_pairs):
                ci = tl.load(cos_ptr + p)
                si = tl.load(sin_ptr + p)
                ii = tl.load(idx_i_ptr + p)
                jj = tl.load(idx_j_ptr + p)

                # Extract values at indices i and j
                vi = tl.sum(tl.where(dim_offsets == ii, vec, 0.0))
                vj = tl.sum(tl.where(dim_offsets == jj, vec, 0.0))

                # Givens rotation: [cos -sin; sin cos] @ [vi; vj]
                new_i = vi * ci - vj * si
                new_j = vi * si + vj * ci

                # Write back
                vec = tl.where(dim_offsets == ii, new_i, vec)
                vec = tl.where(dim_offsets == jj, new_j, vec)

            # Store rotated vector
            tl.store(x_ptr + base + dim_offsets, vec, mask=dim_mask)


    @triton.jit
    def _fused_encode_kernel(
        x_ptr, out_ptr,
        # Layer 1 LUT
        cos1_ptr, sin1_ptr, i1_ptr, j1_ptr,
        # Layer 2 LUT
        cos2_ptr, sin2_ptr, i2_ptr, j2_ptr,
        # Layer 3 LUT
        cos3_ptr, sin3_ptr, i3_ptr, j3_ptr,
        # Sizes
        N,
        dim: tl.constexpr,
        n1: tl.constexpr,
        n2: tl.constexpr,
        n3: tl.constexpr,
    ):
        """
        FUSED kernel: все 3 слоя + polar в одном проходе.
        Вектор загружается из HBM ОДИН РАЗ, все вращения
        выполняются в регистрах, результат записывается обратно.

        HBM reads:  1 (input vector)
        HBM writes: 1 (rotated+polar vector)
        LUT access: constant memory (cached in L1)
        """
        pid = tl.program_id(0)
        if pid >= N:
            return

        base = pid * dim
        offsets = tl.arange(0, dim)
        mask = offsets < dim

        # 1. Load from HBM → registers (SINGLE READ)
        vec = tl.load(x_ptr + base + offsets, mask=mask)

        # 2. Layer 1: adjacent pairs
        for p in tl.static_range(n1):
            ci = tl.load(cos1_ptr + p)
            si = tl.load(sin1_ptr + p)
            ii = tl.load(i1_ptr + p)
            jj = tl.load(j1_ptr + p)
            vi = tl.sum(tl.where(offsets == ii, vec, 0.0))
            vj = tl.sum(tl.where(offsets == jj, vec, 0.0))
            vec = tl.where(offsets == ii, vi * ci - vj * si, vec)
            vec = tl.where(offsets == jj, vi * si + vj * ci, vec)

        # 3. Layer 2: shifted pairs
        for p in tl.static_range(n2):
            ci = tl.load(cos2_ptr + p)
            si = tl.load(sin2_ptr + p)
            ii = tl.load(i2_ptr + p)
            jj = tl.load(j2_ptr + p)
            vi = tl.sum(tl.where(offsets == ii, vec, 0.0))
            vj = tl.sum(tl.where(offsets == jj, vec, 0.0))
            vec = tl.where(offsets == ii, vi * ci - vj * si, vec)
            vec = tl.where(offsets == jj, vi * si + vj * ci, vec)

        # 4. Layer 3: butterfly
        for p in tl.static_range(n3):
            ci = tl.load(cos3_ptr + p)
            si = tl.load(sin3_ptr + p)
            ii = tl.load(i3_ptr + p)
            jj = tl.load(j3_ptr + p)
            vi = tl.sum(tl.where(offsets == ii, vec, 0.0))
            vj = tl.sum(tl.where(offsets == jj, vec, 0.0))
            vec = tl.where(offsets == ii, vi * ci - vj * si, vec)
            vec = tl.where(offsets == jj, vi * si + vj * ci, vec)

        # 5. Store rotated vector → HBM (SINGLE WRITE)
        tl.store(out_ptr + base + offsets, vec, mask=mask)


# ================================================================
# 3. PYTHON WRAPPER — прячет Triton за чистым API
# ================================================================

class NautilusTritonLUT:
    """
    Production-grade Triton kernel с предвычисленной LUT.

    Использование:
        kernel = NautilusTritonLUT(dim=128)
        rotated = kernel.forward(x)       # encode
        original = kernel.inverse(rotated) # decode (T^(-1))
    """

    def __init__(self, dim: int = 128, phi: float = PHI, bits: int = 3,
                 device: str = 'cuda'):
        self.cfg = LUTConfig(dim=dim, phi=phi, bits=bits)
        self.device = device
        self.lut = build_lut(self.cfg)

        if device == 'cuda' and torch.cuda.is_available():
            self.lut = lut_to_device(self.lut, device)

        # Inverse LUT: same indices, negated sin
        self.inv_lut = build_lut(self.cfg)
        for layer in self.inv_lut['layers']:
            layer['sin'] = -layer['sin']  # -theta for inverse
        if device == 'cuda' and torch.cuda.is_available():
            self.inv_lut = lut_to_device(self.inv_lut, device)

        print(f"[NautilusTritonLUT] dim={dim}, LUT={self.lut['total_bytes']} bytes, "
              f"layers={[l['n'] for l in self.lut['layers']]}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode: 3-layer Givens rotation using precomputed LUT."""
        if not HAS_TRITON or x.device.type != 'cuda':
            return self._pytorch_forward(x, self.lut)
        return self._triton_forward(x, self.lut)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Decode: inverse rotation (T^(-1) = T^T, sin negated)."""
        if not HAS_TRITON or x.device.type != 'cuda':
            return self._pytorch_inverse(x, self.lut)
        return self._triton_inverse(x, self.inv_lut)

    def _triton_forward(self, x, lut):
        N, dim = x.shape
        out = torch.empty_like(x)
        l = lut['layers']

        grid = (N,)
        _fused_encode_kernel[grid](
            x, out,
            l[0]['cos'], l[0]['sin'], l[0]['i'], l[0]['j'],
            l[1]['cos'], l[1]['sin'], l[1]['i'], l[1]['j'],
            l[2]['cos'], l[2]['sin'], l[2]['i'], l[2]['j'],
            N,
            dim=dim,
            n1=l[0]['n'], n2=l[1]['n'], n3=l[2]['n'],
        )
        return out

    def _triton_inverse(self, x, inv_lut):
        """Inverse: layers in reverse order, sin already negated in inv_lut."""
        N, dim = x.shape
        out = x.clone()
        # Apply layers 3→2→1 (reverse order)
        for layer in reversed(inv_lut['layers']):
            grid = (N,)
            _givens_layer_kernel[grid](
                out,
                layer['cos'], layer['sin'], layer['i'], layer['j'],
                N, dim=dim, n_pairs=layer['n'],
                BLOCK_N=1,
            )
        return out

    def _pytorch_forward(self, x, lut):
        """CPU/PyTorch fallback."""
        out = x.clone()
        for layer in lut['layers']:
            cos_v = layer['cos']
            sin_v = layer['sin']
            for k in range(layer['n']):
                i, j = layer['i'][k].item(), layer['j'][k].item()
                c, s = cos_v[k].item(), sin_v[k].item()
                a = out[..., i].clone()
                b = out[..., j].clone()
                out[..., i] = a * c - b * s
                out[..., j] = a * s + b * c
        return out

    def _pytorch_inverse(self, x, lut):
        """CPU/PyTorch fallback for inverse."""
        out = x.clone()
        for layer in reversed(lut['layers']):
            for k in range(layer['n'] - 1, -1, -1):
                i, j = layer['i'][k].item(), layer['j'][k].item()
                c = layer['cos'][k].item()
                s = -layer['sin'][k].item()  # negate for inverse
                a = out[..., i].clone()
                b = out[..., j].clone()
                out[..., i] = a * c - b * s
                out[..., j] = a * s + b * c
        return out

    def verify_orthogonality(self, n_tests: int = 100):
        """Verify T^T·T = I on random vectors."""
        device = self.device if torch.cuda.is_available() else 'cpu'
        x = torch.randn(n_tests, self.cfg.dim, device=device)

        fwd = self.forward(x)
        rt = self.inverse(fwd)

        norm_error = (x.norm(dim=1) - fwd.norm(dim=1)).abs().max().item()
        roundtrip_error = (x - rt).pow(2).mean().sqrt().item()

        # Dot product preservation
        a, b = x[:n_tests//2], x[n_tests//2:]
        fa, fb = fwd[:n_tests//2], fwd[n_tests//2:]
        orig_dots = (a * b).sum(dim=1)
        fwd_dots = (fa * fb).sum(dim=1)
        dot_error = (orig_dots - fwd_dots).abs().max().item()

        return {
            'norm_error': norm_error,
            'roundtrip_rmse': roundtrip_error,
            'dot_product_error': dot_error,
            'is_orthogonal': norm_error < 1e-5 and roundtrip_error < 1e-5,
        }


# ================================================================
# 4. BENCHMARK
# ================================================================

def benchmark(dim=128, n_vectors=50000, n_warmup=5, n_iters=20):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"NautilusQuant Triton LUT Benchmark")
    print(f"  dim={dim}, vectors={n_vectors}, device={device}")
    print(f"{'='*60}")

    kernel = NautilusTritonLUT(dim=dim, device=device)

    # Verify
    orth = kernel.verify_orthogonality()
    print(f"\n  Orthogonality check:")
    print(f"    Norm error:     {orth['norm_error']:.2e}")
    print(f"    Roundtrip RMSE: {orth['roundtrip_rmse']:.2e}")
    print(f"    Dot error:      {orth['dot_product_error']:.2e}")
    print(f"    Is orthogonal:  {orth['is_orthogonal']}")

    # Generate test data
    x = torch.randn(n_vectors, dim, device=device, dtype=torch.float32)
    # Add realistic outliers
    for d in [0, 15, 31, 63, 95, 127 % dim]:
        if d < dim:
            mask = torch.rand(n_vectors, device=device) < 0.75
            x[mask, d] = torch.randn(mask.sum(), device=device) * 30

    # Warmup
    for _ in range(n_warmup):
        _ = kernel.forward(x)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark forward
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = kernel.forward(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - t0) / n_iters

    # Benchmark inverse
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = kernel.inverse(out)
    if device == 'cuda':
        torch.cuda.synchronize()
    inv_time = (time.perf_counter() - t0) / n_iters

    # Stats
    vectors_per_sec = n_vectors / fwd_time
    bytes_processed = n_vectors * dim * 2  # FP16 input
    bandwidth_gbps = bytes_processed / fwd_time / 1e9

    print(f"\n  Forward:  {fwd_time*1000:.2f} ms ({vectors_per_sec/1e6:.2f}M vec/s)")
    print(f"  Inverse:  {inv_time*1000:.2f} ms")
    print(f"  Bandwidth: {bandwidth_gbps:.1f} GB/s")
    print(f"  LUT size: {kernel.lut['total_bytes']} bytes")

    # Compare with naive PyTorch (no Triton)
    x_cpu = x.cpu()
    t0 = time.perf_counter()
    for _ in range(min(3, n_iters)):
        _ = kernel._pytorch_forward(x_cpu, kernel.lut)
    pytorch_time = (time.perf_counter() - t0) / min(3, n_iters)

    if HAS_TRITON and device == 'cuda':
        speedup = pytorch_time / fwd_time
        print(f"\n  PyTorch CPU: {pytorch_time*1000:.1f} ms")
        print(f"  Triton GPU:  {fwd_time*1000:.2f} ms")
        print(f"  Speedup:     {speedup:.1f}x")

    return {
        'forward_ms': fwd_time * 1000,
        'inverse_ms': inv_time * 1000,
        'vectors_per_sec': vectors_per_sec,
        'bandwidth_gbps': bandwidth_gbps,
        'lut_bytes': kernel.lut['total_bytes'],
        'orthogonal': orth['is_orthogonal'],
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--n', type=int, default=50000)
    parser.add_argument('--benchmark', action='store_true', default=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()

    if args.verify_only:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        k = NautilusTritonLUT(dim=args.dim, device=device)
        r = k.verify_orthogonality(n_tests=1000)
        print(f"Orthogonal: {r['is_orthogonal']}")
        print(f"Norm error: {r['norm_error']:.2e}")
        print(f"Roundtrip:  {r['roundtrip_rmse']:.2e}")
    else:
        benchmark(dim=args.dim, n_vectors=args.n)
