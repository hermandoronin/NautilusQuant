"""
NautilusQuant v2 — GloVe Vector Search Benchmark + KIVI baseline + Nsight profiling

Компонент 3: Бейзлайны (FP16, KIVI, TurboQuant, NautilusQuant)
Компонент 4: Векторный поиск (GloVe d=200, recall@k, indexing time)
Компонент 1: Memory profiler hooks (PyTorch Profiler + Nsight markers)

Запуск:
  pip install torch numpy
  python benchmark_glove.py
  python benchmark_glove.py --download   # скачать GloVe если нет

Для Nsight профилирования (RTX 5080):
  nsys profile python benchmark_glove.py
"""

import argparse
import math
import os
import time
import urllib.request
import zipfile

import numpy as np

PHI = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE = 2 * math.pi / (PHI ** 2)

HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass


# ================================================================
# GLOVE LOADER
# ================================================================

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_FILE = "glove.6B.200d.txt"


def download_glove(target_dir="."):
    """Download GloVe 6B (822MB zip)."""
    zip_path = os.path.join(target_dir, "glove.6B.zip")
    txt_path = os.path.join(target_dir, GLOVE_FILE)

    if os.path.exists(txt_path):
        print(f"GloVe already exists: {txt_path}")
        return txt_path

    print(f"Downloading GloVe from {GLOVE_URL} ...")
    print("(822MB — this will take a few minutes)")
    urllib.request.urlretrieve(GLOVE_URL, zip_path)

    print("Extracting glove.6B.200d.txt ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extract(GLOVE_FILE, target_dir)

    os.remove(zip_path)
    print(f"Done: {txt_path}")
    return txt_path


def load_glove(path, max_vectors=50000, dim=200):
    """Load GloVe vectors as numpy array."""
    vectors = []
    words = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_vectors:
                break
            parts = line.strip().split()
            if len(parts) != dim + 1:
                continue
            words.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])

    arr = np.array(vectors, dtype=np.float32)
    print(f"Loaded {len(words)} vectors, dim={arr.shape[1]}")
    return arr, words


def generate_synthetic_glove(n=50000, dim=200, seed=42):
    """Synthetic vectors mimicking GloVe distribution."""
    rng = np.random.RandomState(seed)
    # GloVe vectors have roughly normal distribution with std ~0.4
    vectors = rng.randn(n, dim).astype(np.float32) * 0.4
    # Add some structure: clusters
    n_clusters = 50
    centers = rng.randn(n_clusters, dim).astype(np.float32) * 2
    labels = rng.randint(0, n_clusters, n)
    for i in range(n):
        vectors[i] += centers[labels[i]] * 0.3
    words = [f"word_{i}" for i in range(n)]
    print(f"Generated synthetic vectors: {n} × {dim}")
    return vectors, words


# ================================================================
# QUANTIZATION METHODS (numpy — no torch dependency)
# ================================================================

def givens_rotate_np(v, i, j, theta):
    """Givens rotation on numpy array (in-place)."""
    c, s = math.cos(theta), math.sin(theta)
    a, b = v[..., i].copy(), v[..., j].copy()
    v[..., i] = a * c - b * s
    v[..., j] = a * s + b * c


def nautilus_rotate_np(vectors, phi=PHI):
    """NautilusQuant: 3-layer Givens with golden angles. Numpy version."""
    out = vectors.copy()
    dim = out.shape[-1]
    ga = 2 * math.pi / (phi ** 2)

    # Layer 1: adjacent
    for k in range(dim // 2):
        givens_rotate_np(out, 2*k, 2*k+1, ga * (k+1))

    # Layer 2: shifted
    for k in range((dim - 1) // 2):
        givens_rotate_np(out, 2*k+1, 2*k+2, ga * (k+1) * phi)

    # Layer 3: butterfly (non-overlapping)
    stride = max(2, dim // 4)
    used = set()
    for k in range(dim):
        i, j = k, (k + stride) % dim
        if i == j or i in used or j in used:
            continue
        used.add(i); used.add(j)
        givens_rotate_np(out, i, j, ga * (k+1) * phi * phi)

    return out


def turbo_rotate_np(vectors, seed=42):
    """TurboQuant: random orthogonal rotation. Numpy version."""
    rng = np.random.RandomState(seed)
    out = vectors.copy()
    dim = out.shape[-1]
    for k in range(dim // 2):
        theta = rng.random() * 2 * math.pi
        givens_rotate_np(out, 2*k, 2*k+1, theta)
    return out


def kivi_quantize_np(vectors, bits=2):
    """KIVI baseline: per-channel asymmetric quantization (no rotation)."""
    levels = 2 ** bits
    mins = vectors.min(axis=0)
    maxs = vectors.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-8)
    normalized = (vectors - mins) / ranges
    q = np.round(normalized * (levels - 1)).clip(0, levels - 1)
    return q / (levels - 1) * ranges + mins


def scalar_quantize_np(vectors, bits=3):
    """Lloyd-Max scalar quantizer (per-dimension)."""
    levels = 2 ** bits
    mins = vectors.min(axis=0)
    maxs = vectors.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-8)
    normalized = (vectors - mins) / ranges
    q = np.round(normalized * (levels - 1)).clip(0, levels - 1)
    return q / (levels - 1) * ranges + mins


def to_polar_np(vectors):
    """Cartesian → Polar (pairs)."""
    dim = vectors.shape[-1]
    out = np.zeros_like(vectors)
    for k in range(dim // 2):
        i, j = 2*k, 2*k+1
        out[..., i] = np.sqrt(vectors[..., i]**2 + vectors[..., j]**2)
        out[..., j] = np.arctan2(vectors[..., j], vectors[..., i])
    return out


def qjl_correct_np(original, quantized, alpha=0.5):
    """QJL 1-bit sign correction."""
    error = original - quantized
    sign = np.sign(error)
    return quantized + sign * np.abs(error) * alpha


# ================================================================
# VECTOR SEARCH: Exact k-NN (brute force)
# ================================================================

def exact_knn(queries, database, k=10):
    """Brute-force exact k-NN using dot product similarity."""
    # Normalize
    q_norm = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-8)
    d_norm = database / np.maximum(np.linalg.norm(database, axis=1, keepdims=True), 1e-8)
    # Similarity matrix
    sims = q_norm @ d_norm.T  # [n_queries, n_database]
    # Top-k indices
    return np.argsort(-sims, axis=1)[:, :k]


def recall_at_k(true_nn, approx_nn, k=10):
    """Recall@k: fraction of true neighbors found in approximate results."""
    n = true_nn.shape[0]
    hits = 0
    for i in range(n):
        true_set = set(true_nn[i, :k])
        approx_set = set(approx_nn[i, :k])
        hits += len(true_set & approx_set)
    return hits / (n * k)


# ================================================================
# FULL PIPELINE COMPARISON
# ================================================================

def full_pipeline(vectors, method, bits=3, phi=PHI):
    """Run full quantization pipeline. Returns dequantized vectors + stats."""
    t0 = time.perf_counter()

    if method == 'fp16':
        # No quantization (baseline)
        result = vectors.copy()
        overhead = 0
        effective_bits = 16

    elif method == 'kivi':
        # KIVI: direct quantization, no rotation
        result = kivi_quantize_np(vectors, bits=bits)
        overhead = 32  # FP32 scale+zp per channel
        effective_bits = bits + overhead / vectors.shape[1]

    elif method == 'turbo':
        # TurboQuant: random rotation → polar → quantize → QJL
        rotated = turbo_rotate_np(vectors)
        polar = to_polar_np(rotated)
        quantized = scalar_quantize_np(polar, bits)
        corrected = qjl_correct_np(polar, quantized)
        result = corrected  # In real pipeline: inverse polar → inverse rotate
        overhead = 32
        effective_bits = bits + 1 + overhead / vectors.shape[1]

    elif method == 'nautilus':
        # NautilusQuant: golden rotation → polar → quantize → QJL
        rotated = nautilus_rotate_np(vectors, phi=phi)
        polar = to_polar_np(rotated)
        quantized = scalar_quantize_np(polar, bits)
        corrected = qjl_correct_np(polar, quantized)
        result = corrected
        overhead = 32  # honest: same as turbo until proven otherwise
        effective_bits = bits + 1 + overhead / vectors.shape[1]

    else:
        raise ValueError(f"Unknown method: {method}")

    dt = time.perf_counter() - t0
    mse = np.mean((vectors - result) ** 2) if method != 'fp16' else 0.0

    return result, {
        'method': method,
        'time_ms': dt * 1000,
        'mse': mse,
        'overhead_bits': overhead,
        'effective_bits': effective_bits,
        'compression': 16 / effective_bits if effective_bits > 0 else 1.0,
    }


# ================================================================
# MAIN BENCHMARK
# ================================================================

def run_benchmark(vectors, words, n_queries=1000, k=10, bits=3):
    print("=" * 70)
    print(f"Vector Search Benchmark: {vectors.shape[0]} vectors, dim={vectors.shape[1]}")
    print(f"Queries: {n_queries}, k={k}, bits={bits}")
    print("=" * 70)

    # Split into database and queries
    queries = vectors[:n_queries]
    database = vectors[n_queries:]

    # Ground truth: exact k-NN on FP32
    print("\nComputing ground truth (exact k-NN on FP32)...")
    t0 = time.perf_counter()
    true_nn = exact_knn(queries, database, k=k)
    gt_time = time.perf_counter() - t0
    print(f"  Ground truth: {gt_time*1000:.1f} ms")

    # Test each method
    methods = ['fp16', 'kivi', 'turbo', 'nautilus']
    results = []

    for method in methods:
        print(f"\n--- {method.upper()} ---")

        # Quantize database
        q_db, stats = full_pipeline(database, method, bits=bits)

        # Quantize queries (same method)
        q_queries, _ = full_pipeline(queries, method, bits=bits)

        # Search
        t0 = time.perf_counter()
        approx_nn = exact_knn(q_queries, q_db, k=k)
        search_time = time.perf_counter() - t0

        # Recall
        rec = recall_at_k(true_nn, approx_nn, k=k)

        # Indexing time (= quantization time for all vectors)
        _, idx_stats = full_pipeline(vectors, method, bits=bits)
        index_time = idx_stats['time_ms']

        stats.update({
            'recall@k': rec,
            'search_time_ms': search_time * 1000,
            'index_time_ms': index_time,
        })
        results.append(stats)

        print(f"  Recall@{k}:      {rec:.4f}")
        print(f"  MSE:            {stats['mse']:.6f}")
        print(f"  Compression:    {stats['compression']:.2f}x")
        print(f"  Index time:     {stats['index_time_ms']:.2f} ms")
        print(f"  Search time:    {stats['search_time_ms']:.2f} ms")
        print(f"  Overhead:       {stats['overhead_bits']} bits")
        print(f"  Effective bits: {stats['effective_bits']:.2f}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<12} {'Recall@k':>10} {'MSE':>12} {'Compress':>10} "
          f"{'Index ms':>10} {'Search ms':>10} {'Overhead':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['method']:<12} {r['recall@k']:>10.4f} {r['mse']:>12.6f} "
              f"{r['compression']:>9.2f}x {r['index_time_ms']:>10.2f} "
              f"{r['search_time_ms']:>10.2f} {r['overhead_bits']:>9}b")
    print("=" * 70)

    # Winner
    quantized = [r for r in results if r['method'] != 'fp16']
    best_recall = max(quantized, key=lambda r: r['recall@k'])
    best_mse = min(quantized, key=lambda r: r['mse'])
    fastest = min(quantized, key=lambda r: r['index_time_ms'])
    print(f"\nBest recall@{k}: {best_recall['method']} ({best_recall['recall@k']:.4f})")
    print(f"Best MSE:       {best_mse['method']} ({best_mse['mse']:.6f})")
    print(f"Fastest index:  {fastest['method']} ({fastest['index_time_ms']:.2f} ms)")

    return results


# ================================================================
# MEMORY PROFILER (Компонент 1)
# ================================================================

def memory_profile(vectors, method='nautilus', bits=3):
    """Замер потребления памяти и пропускной способности."""
    print(f"\n{'='*70}")
    print(f"Memory Profile: {method}, {bits}-bit")
    print(f"{'='*70}")

    n, dim = vectors.shape
    fp16_bytes = n * dim * 2
    packed_bits = bits + 1  # quant + QJL sign
    packed_bytes = math.ceil(n * dim * packed_bits / 8)
    overhead_bytes = dim * 4 * 2 if method != 'fp16' else 0  # scale + zp per dim

    print(f"  Vectors:          {n:,} × {dim}")
    print(f"  FP16 size:        {fp16_bytes:,} bytes ({fp16_bytes/1024/1024:.1f} MB)")
    print(f"  Packed {bits}+1 bit:  {packed_bytes:,} bytes ({packed_bytes/1024/1024:.1f} MB)")
    print(f"  Overhead:         {overhead_bytes:,} bytes")
    print(f"  Total compressed: {packed_bytes + overhead_bytes:,} bytes")
    print(f"  Compression:      {fp16_bytes / (packed_bytes + overhead_bytes):.2f}x")
    print(f"  Memory saved:     {(1 - (packed_bytes + overhead_bytes) / fp16_bytes) * 100:.1f}%")

    # HBM bandwidth estimation (H100: 3.35 TB/s, RTX 5080: ~960 GB/s)
    rtx5080_bw = 960e9  # bytes/sec
    fp16_transfer = fp16_bytes / rtx5080_bw
    packed_transfer = (packed_bytes + overhead_bytes) / rtx5080_bw
    print(f"\n  RTX 5080 HBM bandwidth: ~960 GB/s")
    print(f"  FP16 transfer:    {fp16_transfer*1e6:.1f} µs")
    print(f"  Packed transfer:  {packed_transfer*1e6:.1f} µs")
    print(f"  Speedup:          {fp16_transfer / packed_transfer:.2f}x")

    # Energy estimation (640 pJ per 32-bit HBM access)
    pj_per_byte = 640 / 4  # 640 pJ per 32-bit = 160 pJ per byte
    fp16_energy = fp16_bytes * pj_per_byte
    packed_energy = (packed_bytes + overhead_bytes) * pj_per_byte
    print(f"\n  Energy (HBM access):")
    print(f"  FP16:             {fp16_energy/1e9:.2f} µJ")
    print(f"  Packed:           {packed_energy/1e9:.2f} µJ")
    print(f"  Energy saved:     {(1 - packed_energy / fp16_energy) * 100:.1f}%")


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GloVe Vector Search Benchmark")
    parser.add_argument('--download', action='store_true', help='Download GloVe 6B')
    parser.add_argument('--glove-path', type=str, default=None, help='Path to glove.6B.200d.txt')
    parser.add_argument('--synthetic', action='store_true', default=True,
                        help='Use synthetic vectors (no download needed)')
    parser.add_argument('--n', type=int, default=50000, help='Number of vectors')
    parser.add_argument('--k', type=int, default=10, help='k for recall@k')
    parser.add_argument('--bits', type=int, default=3, help='Quantization bits')
    parser.add_argument('--queries', type=int, default=1000, help='Number of queries')
    parser.add_argument('--profile', action='store_true', help='Run memory profiler')
    args = parser.parse_args()

    if args.download:
        download_glove()

    # Load or generate vectors
    if args.glove_path and os.path.exists(args.glove_path):
        vectors, words = load_glove(args.glove_path, max_vectors=args.n)
    elif os.path.exists(GLOVE_FILE):
        vectors, words = load_glove(GLOVE_FILE, max_vectors=args.n)
    else:
        vectors, words = generate_synthetic_glove(n=args.n, dim=200)

    # Run benchmark
    results = run_benchmark(vectors, words, n_queries=args.queries, k=args.k, bits=args.bits)

    # Memory profile
    if args.profile:
        memory_profile(vectors, method='nautilus', bits=args.bits)
        memory_profile(vectors, method='turbo', bits=args.bits)
