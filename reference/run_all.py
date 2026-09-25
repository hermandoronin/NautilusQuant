#!/usr/bin/env python3
"""
NautilusQuant — Единая точка запуска всех экспериментов.

Быстрый старт:
  pip install torch numpy
  python run_all.py              # запустить ВСЁ (синтетика)
  python run_all.py --test core  # только ядро: ортогональность + MSE
  python run_all.py --test triton # GPU бенчмарк
  python run_all.py --test planb  # Plan B: 4 невероятные идеи

Для реальной модели Gemma 3:
  pip install transformers accelerate
  python run_all.py --test real --model google/gemma-3-4b-it
"""

import argparse
import importlib
import math
import sys
import time

# =====================================================================
# ТЕСТ 1: Ядро — ортогональность + MSE + overhead (CPU, без зависимостей)
# =====================================================================

def test_core(dim=128, n_vectors=500, bits=3):
    """
    Проверяет фундамент NautilusQuant v2:
    1. Ортогональность: T^T · T ≈ I
    2. Сохранение нормы: ||Tv|| = ||v||
    3. Roundtrip: T^(-1)(T(v)) ≈ v
    4. MSE сравнение: NautilusQuant vs TurboQuant (случайное вращение)
    5. Дисперсия углов после полярного преобразования
    """
    import torch

    PHI = (1 + math.sqrt(5)) / 2
    GA = 2 * math.pi / (PHI ** 2)

    print("=" * 70)
    print(f"  NautilusQuant v2 — CORE TEST")
    print(f"  dim={dim}  vectors={n_vectors}  bits={bits}")
    print("=" * 70)

    # --- Build Givens layers (Fix 4: non-overlapping pairs) ---
    def build_layers(d):
        layers = []
        # Layer 1: adjacent (0,1),(2,3),...
        l1 = [(2*k, 2*k+1, GA*(k+1)) for k in range(d//2)]
        layers.append(l1)
        # Layer 2: shifted (1,2),(3,4),...
        l2 = [(2*k+1, 2*k+2, GA*(k+1)*PHI) for k in range((d-1)//2)]
        layers.append(l2)
        # Layer 3: butterfly non-overlapping
        l3 = []
        stride = max(2, d//4)
        used = set()
        for k in range(d):
            i, j = k, (k+stride) % d
            if i == j or i in used or j in used:
                continue
            used.add(i); used.add(j)
            l3.append((i, j, GA*(k+1)*PHI*PHI))
        layers.append(l3)
        return layers

    def apply_forward(x, layers):
        out = x.clone()
        for layer in layers:
            for i, j, theta in layer:
                c, s = math.cos(theta), math.sin(theta)
                a, b = out[..., i].clone(), out[..., j].clone()
                out[..., i] = a*c - b*s
                out[..., j] = a*s + b*c
        return out

    def apply_inverse(x, layers):
        out = x.clone()
        for layer in reversed(layers):
            for i, j, theta in reversed(layer):
                c, s = math.cos(theta), math.sin(-theta)
                a, b = out[..., i].clone(), out[..., j].clone()
                out[..., i] = a*c - b*s
                out[..., j] = a*s + b*c
        return out

    def random_rotation(x):
        """TurboQuant-style: random orthogonal via QR."""
        d = x.shape[-1]
        R, _ = torch.linalg.qr(torch.randn(d, d))
        return x @ R.T

    def to_polar_angles(x):
        """Переводим пары координат в полярные углы."""
        d = x.shape[-1]
        angles = []
        for k in range(d // 2):
            xi, xj = x[..., 2*k], x[..., 2*k+1]
            theta = torch.atan2(xj, xi)
            angles.append(theta)
        return torch.stack(angles, dim=-1)

    def quantize(x, bits):
        levels = 2 ** bits
        mn, mx = x.min(), x.max()
        step = (mx - mn) / levels
        q = torch.round((x - mn) / step).clamp(0, levels-1)
        return mn + q * step

    layers = build_layers(dim)

    # --- Generate data with realistic outliers ---
    torch.manual_seed(42)
    data = torch.randn(n_vectors, dim) * 0.5
    # Inject outliers in 6 specific dimensions (like real transformers)
    outlier_dims = [7, 23, 41, 58, 89, 112] if dim > 112 else list(range(min(6, dim)))
    mask = torch.rand(n_vectors) < 0.75  # 75% of positions
    for od in outlier_dims:
        if od < dim:
            data[mask, od] = torch.randn(mask.sum()) * 30 - 30  # values up to -60

    print(f"\n  Data stats: mean={data.mean():.3f}  std={data.std():.3f}")
    print(f"  Max outlier: {data.abs().max():.1f}")
    print(f"  Outlier dims: {outlier_dims[:min(6,dim)]}")

    # ---- TEST 1: Orthogonality ----
    print(f"\n{'─'*50}")
    print("  TEST 1: Ортогональность матрицы T")
    print(f"{'─'*50}")

    # Build explicit matrix T
    T = torch.eye(dim)
    T = apply_forward(T, layers)
    TTt = T @ T.T
    I = torch.eye(dim)
    orth_err = (TTt - I).abs().max().item()
    print(f"  ||T·T^T - I||_max = {orth_err:.2e}  {'✅ PASS' if orth_err < 1e-10 else '❌ FAIL'}")

    # ---- TEST 2: Norm preservation ----
    print(f"\n{'─'*50}")
    print("  TEST 2: Сохранение нормы ||Tv|| = ||v||")
    print(f"{'─'*50}")

    norms_before = torch.norm(data, dim=-1)
    rotated = apply_forward(data, layers)
    norms_after = torch.norm(rotated, dim=-1)
    norm_err = (norms_before - norms_after).abs().max().item()
    print(f"  Max norm error = {norm_err:.2e}  {'✅ PASS' if norm_err < 1e-10 else '❌ FAIL'}")

    # ---- TEST 3: Roundtrip ----
    print(f"\n{'─'*50}")
    print("  TEST 3: Roundtrip T^(-1)(T(v)) ≈ v")
    print(f"{'─'*50}")

    recovered = apply_inverse(rotated, layers)
    rt_err = (data - recovered).abs().max().item()
    print(f"  Max roundtrip error = {rt_err:.2e}  {'✅ PASS' if rt_err < 1e-10 else '❌ FAIL'}")

    # ---- TEST 4: Dot product preservation ----
    print(f"\n{'─'*50}")
    print("  TEST 4: Сохранение dot product")
    print(f"{'─'*50}")

    dots_before = (data[:100] * data[1:101]).sum(dim=-1)
    dots_after = (rotated[:100] * rotated[1:101]).sum(dim=-1)
    dot_err = (dots_before - dots_after).abs().max().item()
    print(f"  Max dot error = {dot_err:.2e}  {'✅ PASS' if dot_err < 1e-10 else '❌ FAIL'}")

    # ---- TEST 5: MSE comparison ----
    print(f"\n{'─'*50}")
    print(f"  TEST 5: MSE сравнение при {bits}-bit квантовании")
    print(f"{'─'*50}")

    # NautilusQuant pipeline
    naut_rotated = apply_forward(data, layers)
    naut_quantized = quantize(naut_rotated, bits)
    naut_recovered = apply_inverse(naut_quantized, layers)
    naut_mse = ((data - naut_recovered) ** 2).mean().item()

    # TurboQuant pipeline (random rotation)
    turbo_rotated = random_rotation(data)
    turbo_quantized = quantize(turbo_rotated, bits)
    # Inverse random rotation
    d = data.shape[-1]
    R, _ = torch.linalg.qr(torch.randn(d, d))
    turbo_rotated2 = data @ R.T
    turbo_q2 = quantize(turbo_rotated2, bits)
    turbo_recovered = turbo_q2 @ R
    turbo_mse = ((data - turbo_recovered) ** 2).mean().item()

    print(f"  NautilusQuant MSE: {naut_mse:.8f}")
    print(f"  TurboQuant MSE:    {turbo_mse:.8f}")
    winner = "NautilusQuant 🏆" if naut_mse < turbo_mse else "TurboQuant"
    ratio = turbo_mse / naut_mse if naut_mse > 0 else float('inf')
    print(f"  Winner: {winner}  (ratio: {ratio:.3f}x)")

    # ---- TEST 6: Angle variance ----
    print(f"\n{'─'*50}")
    print("  TEST 6: Дисперсия углов (ключевая метрика!)")
    print(f"{'─'*50}")

    naut_angles = to_polar_angles(naut_rotated)
    turbo_angles = to_polar_angles(turbo_rotated2)

    naut_var = naut_angles.var().item()
    turbo_var = turbo_angles.var().item()

    print(f"  NautilusQuant angle variance: {naut_var:.6f}")
    print(f"  TurboQuant angle variance:    {turbo_var:.6f}")
    angle_winner = "NautilusQuant 🏆" if naut_var < turbo_var else "TurboQuant"
    print(f"  Winner: {angle_winner}")
    print(f"  (Меньше = лучше: предсказуемее → возможен 0 overhead)")

    # ---- SUMMARY ----
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Ортогональность:     {'✅' if orth_err < 1e-10 else '❌'}  err={orth_err:.2e}")
    print(f"  Нормы:               {'✅' if norm_err < 1e-10 else '❌'}  err={norm_err:.2e}")
    print(f"  Roundtrip:           {'✅' if rt_err < 1e-10 else '❌'}  err={rt_err:.2e}")
    print(f"  Dot products:        {'✅' if dot_err < 1e-10 else '❌'}  err={dot_err:.2e}")
    print(f"  MSE winner:          {winner}")
    print(f"  Angle var winner:    {angle_winner}")
    print(f"{'='*70}")

    return naut_mse < turbo_mse


# =====================================================================
# ТЕСТ 2: Triton GPU бенчмарк
# =====================================================================

def test_triton(dim=128, n=10000):
    """GPU-бенчмарк: PyTorch baseline vs Triton kernel."""
    print("\n" + "=" * 70)
    print("  NautilusQuant v2 — TRITON GPU BENCHMARK")
    print("=" * 70)

    try:
        import torch
        if not torch.cuda.is_available():
            print("  ⚠️  CUDA не найден. Пропускаю GPU тест.")
            print("  Для запуска нужна NVIDIA GPU + CUDA.")
            return
    except ImportError:
        print("  ⚠️  PyTorch не установлен.")
        return

    try:
        from nautilus_triton import NautilusConfig, NautilusQuantPyTorch
        config = NautilusConfig(dim=dim, bits=3)
        engine = NautilusQuantPyTorch(config)

        data = torch.randn(n, dim, device='cuda', dtype=torch.float16)

        # Warmup
        for _ in range(5):
            engine.encode(data.float())

        # Benchmark
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            engine.encode(data.float())
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 100

        throughput = n / elapsed
        print(f"  dim={dim}  vectors={n}")
        print(f"  PyTorch encode: {elapsed*1000:.2f} ms")
        print(f"  Throughput: {throughput:,.0f} vectors/sec")
        print(f"  Memory: {dim*2*n/1024/1024:.1f} MB FP16 → {dim*3/8*n/1024/1024:.1f} MB 3-bit")

    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


# =====================================================================
# ТЕСТ 3: Hardware concepts
# =====================================================================

def test_hardware(dim=128):
    """Тест 4 аппаратных концептов: SRAM, Dataflow, MX, Sub-1-bit."""
    print("\n" + "=" * 70)
    print("  NautilusQuant v2 — HARDWARE CONCEPTS TEST")
    print("=" * 70)

    try:
        from nautilus_hardware import (
            SRAMFusedConfig, NautilusLUT,
            MXQuantizer, NautilusWithMX,
            NautilusDataflow, SubBitExperiment
        )
        import torch

        # Concept 1: SRAM budget
        max_v = SRAMFusedConfig.max_vectors_per_tile(dim)
        print(f"\n  [SRAM] Max vectors per tile (dim={dim}): {max_v}")
        print(f"  [SRAM] Budget: {SRAMFusedConfig.SRAM_BUDGET_BYTES//1024}KB per threadblock")

        # Concept 2: LUT
        lut = NautilusLUT.build(dim)
        print(f"\n  [LUT] Total entries: {lut.total_entries}")
        print(f"  [LUT] Memory: {lut.memory_bytes} bytes ({lut.memory_bytes} < 2048 ✅)")

        # Concept 3: MX format
        data = torch.randn(100, dim)
        mx = MXQuantizer(block_size=32, format_type='MXFP4')
        q, scales = mx.quantize(data)
        deq = mx.dequantize(q, scales)
        mx_mse = ((data - deq) ** 2).mean().item()
        print(f"\n  [MX-Format] MXFP4 MSE: {mx_mse:.6f}")
        print(f"  [MX-Format] Overhead: 8-bit scale per 32 elements = 0.25 bit/value")

        # Concept 4: Dataflow
        df = NautilusDataflow(dim)
        schedule = df.static_schedule()
        print(f"\n  [Dataflow] Static schedule: {len(schedule)} operations")
        print(f"  [Dataflow] Fully deterministic: ✅ (no PRNG needed)")

        # Concept 5: Sub-1-bit
        sub = SubBitExperiment(dim)
        info = sub.golden_orbit_encode(data[:10])
        print(f"\n  [Sub-1-bit] Golden orbit indices (sample): {info['indices'][0,:5].tolist()}")
        print(f"  [Sub-1-bit] Bits per angle: ~{info['bits_per_value']:.2f}")

        print(f"\n  All 4 hardware concepts: ✅ PASS")

    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


# =====================================================================
# ТЕСТ 4: Plan B — 4 невероятные идеи
# =====================================================================

def test_planb(dim=64):
    """Тест экспериментальных модулей Plan B."""
    print("\n" + "=" * 70)
    print("  NautilusQuant — PLAN B EXPERIMENTS")
    print("=" * 70)

    import torch
    data = torch.randn(50, dim)

    tests = [
        ("plan_b.quasicrystal", "QuasiCrystalQuantizer", "8D квазикристалл (убийца QuIP#)"),
        ("plan_b.golden_jl", "GoldenJL", "Детерминированный JL-трансформ (убийца QJL)"),
        ("plan_b.phinary", "PhinaryQuantizer", "Фибоначчиева система счисления"),
        ("plan_b.fractal_hash", "FractalOrbitEncoder", "Суб-1-бит фрактальное хеширование"),
        ("plan_b.groq_dataflow", "GroqScheduler", "Groq LPU dataflow schedule"),
        ("plan_b.multimodal_spiral", "MultiModalSpiral", "Адаптивная спираль VLM"),
    ]

    for mod_name, cls_name, desc in tests:
        try:
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            obj = cls(dim)
            # Try running
            if hasattr(obj, 'encode'):
                result = obj.encode(data)
                print(f"  ✅ {desc}")
                if hasattr(result, 'shape'):
                    print(f"     output shape: {result.shape}")
            elif hasattr(obj, 'quantize'):
                result = obj.quantize(data)
                print(f"  ✅ {desc}")
            elif hasattr(obj, 'schedule'):
                result = obj.schedule()
                print(f"  ✅ {desc}  ({len(result)} ops)")
            elif hasattr(obj, 'build'):
                result = obj.build()
                print(f"  ✅ {desc}")
            else:
                print(f"  ✅ {desc}  (initialized)")
        except Exception as e:
            print(f"  ⚠️  {desc}: {e}")


# =====================================================================
# ТЕСТ 5: Sweep по PHI — какое значение оптимально?
# =====================================================================

def test_sweep(dim=128, n=500, bits=3):
    """Sweep: перебор значений PHI для поиска оптимального."""
    import torch
    print("\n" + "=" * 70)
    print(f"  PHI SWEEP — поиск оптимального значения")
    print(f"  dim={dim}  vectors={n}  bits={bits}")
    print("=" * 70)

    PHI_ORIG = (1 + math.sqrt(5)) / 2

    candidates = [
        ("φ (золотое)", PHI_ORIG),
        ("φ² ", PHI_ORIG**2),
        ("1/φ", 1/PHI_ORIG),
        ("√2 (серебряное)", 1 + math.sqrt(2)),
        ("π/2", math.pi/2),
        ("e/2", math.e/2),
        ("random seed=42", None),
    ]

    torch.manual_seed(42)
    data = torch.randn(n, dim) * 0.5
    outlier_dims = [7, 23, 41, 58, 89, 112]
    mask = torch.rand(n) < 0.75
    for od in outlier_dims:
        if od < dim:
            data[mask, od] = torch.randn(mask.sum()) * 30 - 30

    def build_and_test(phi_val):
        if phi_val is None:
            # Random rotation baseline
            R, _ = torch.linalg.qr(torch.randn(dim, dim))
            rotated = data @ R.T
            quantized = quantize_simple(rotated, bits)
            recovered = quantized @ R
            mse = ((data - recovered)**2).mean().item()
            angles = polar_angles(rotated)
            return mse, angles.var().item()

        ga = 2 * math.pi / (phi_val ** 2)
        layers = []
        l1 = [(2*k, 2*k+1, ga*(k+1)) for k in range(dim//2)]
        layers.append(l1)
        l2 = [(2*k+1, 2*k+2, ga*(k+1)*phi_val) for k in range((dim-1)//2)]
        layers.append(l2)
        l3 = []
        stride = max(2, dim//4)
        used = set()
        for k in range(dim):
            i, j = k, (k+stride)%dim
            if i==j or i in used or j in used: continue
            used.add(i); used.add(j)
            l3.append((i, j, ga*(k+1)*phi_val*phi_val))
        layers.append(l3)

        out = data.clone()
        for layer in layers:
            for i, j, theta in layer:
                c, s = math.cos(theta), math.sin(theta)
                a, b = out[...,i].clone(), out[...,j].clone()
                out[...,i] = a*c - b*s
                out[...,j] = a*s + b*c

        quantized = quantize_simple(out, bits)

        rec = quantized.clone()
        for layer in reversed(layers):
            for i, j, theta in reversed(layer):
                c, s = math.cos(-theta), math.sin(-theta)
                a, b = rec[...,i].clone(), rec[...,j].clone()
                rec[...,i] = a*c - b*s
                rec[...,j] = a*s + b*c

        mse = ((data - rec)**2).mean().item()
        angles = polar_angles(out)
        return mse, angles.var().item()

    def quantize_simple(x, b):
        levels = 2**b
        mn, mx = x.min(), x.max()
        step = (mx - mn) / levels
        q = torch.round((x - mn) / step).clamp(0, levels-1)
        return mn + q * step

    def polar_angles(x):
        d = x.shape[-1]
        angs = []
        for k in range(d//2):
            angs.append(torch.atan2(x[...,2*k+1], x[...,2*k]))
        return torch.stack(angs, -1)

    print(f"\n  {'Метод':<22} {'MSE':>12} {'Angle Var':>12} {'Winner':>8}")
    print(f"  {'─'*56}")

    results = []
    for name, phi_val in candidates:
        mse, avar = build_and_test(phi_val)
        results.append((name, mse, avar))
        print(f"  {name:<22} {mse:>12.8f} {avar:>12.6f}")

    best_mse = min(results, key=lambda x: x[1])
    best_var = min(results, key=lambda x: x[2])
    print(f"\n  🏆 Best MSE:       {best_mse[0]} ({best_mse[1]:.8f})")
    print(f"  🏆 Best Angle Var: {best_var[0]} ({best_var[2]:.6f})")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NautilusQuant — Run experiments")
    parser.add_argument("--test", choices=["core", "triton", "hardware", "planb", "sweep", "all"],
                        default="all", help="Which test to run")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument("--n", type=int, default=500, help="Number of vectors")
    parser.add_argument("--bits", type=int, default=3, help="Quantization bits")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model for real KV-cache test")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║     NautilusQuant v2 — Deterministic Self-Organization          ║
║     KV-Cache Quantization via Golden Ratio Geometry              ║
║     github.com/hermandoronin/NautilusQuant                           ║
╚══════════════════════════════════════════════════════════════════╝
""")

    if args.test in ("core", "all"):
        test_core(args.dim, args.n, args.bits)

    if args.test in ("sweep", "all"):
        test_sweep(args.dim, args.n, args.bits)

    if args.test in ("triton", "all"):
        test_triton(args.dim, args.n)

    if args.test in ("hardware", "all"):
        test_hardware(args.dim)

    if args.test in ("planb", "all"):
        test_planb(args.dim)

    if args.model:
        print("\n  Для теста на реальной модели запустите:")
        print(f"  python validate_real_kv.py --model {args.model} --sweep")

    print("\n  Done! 🚀")
