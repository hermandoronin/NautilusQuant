"""
NautilusQuant v2 — Валидация на реальных KV-кэш тензорах
Fixes 1, 3, 6: Тестирование на реальных данных, а не random()

Запуск:
  pip install torch transformers
  python validate_real_kv.py

Для Gemma 3 (рекомендуется):
  pip install torch transformers accelerate
  python validate_real_kv.py --model google/gemma-3-4b-it
"""

import argparse
import math
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

PHI = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE = 2 * math.pi / (PHI ** 2)


# ============ ОРТОГОНАЛЬНАЯ СПИРАЛЬНАЯ МАТРИЦА ============

def build_givens_layers(dim: int, phi: float = PHI) -> list:
    """
    Строит 3 слоя вращений Гивенса с непересекающимися парами.
    Fix 4: каждый индекс участвует в слое РОВНО один раз.
    """
    ga = 2 * math.pi / (phi ** 2)
    layers = []

    # Слой 1: соседние пары (0,1), (2,3), ...
    layer1 = []
    for k in range(dim // 2):
        i, j = 2 * k, 2 * k + 1
        theta = ga * (k + 1)
        layer1.append((i, j, theta))
    layers.append(layer1)

    # Слой 2: сдвинутые пары (1,2), (3,4), ...
    layer2 = []
    for k in range((dim - 1) // 2):
        i, j = 2 * k + 1, 2 * k + 2
        theta = ga * (k + 1) * phi
        layer2.append((i, j, theta))
    layers.append(layer2)

    # Слой 3: butterfly (непересекающиеся!)
    layer3 = []
    stride = max(2, dim // 4)
    used = set()
    for k in range(dim):
        i = k
        j = (k + stride) % dim
        if i == j or i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        theta = ga * (k + 1) * phi * phi
        layer3.append((i, j, theta))
    layers.append(layer3)

    return layers


def apply_givens_forward(x: torch.Tensor, layers: list) -> torch.Tensor:
    """Прямое преобразование: T·x. Ортогонально."""
    out = x.clone()
    for layer in layers:
        for i, j, theta in layer:
            c, s = math.cos(theta), math.sin(theta)
            a = out[..., i].clone()
            b = out[..., j].clone()
            out[..., i] = a * c - b * s
            out[..., j] = a * s + b * c
    return out


def apply_givens_inverse(x: torch.Tensor, layers: list) -> torch.Tensor:
    """Fix 5: Обратное преобразование T^(-1)·x = T^T·x."""
    out = x.clone()
    for layer in reversed(layers):
        for i, j, theta in reversed(layer):
            c, s = math.cos(-theta), math.sin(-theta)
            a = out[..., i].clone()
            b = out[..., j].clone()
            out[..., i] = a * c - b * s
            out[..., j] = a * s + b * c
    return out


def random_rotation(x: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """TurboQuant baseline: случайные углы."""
    gen = torch.Generator().manual_seed(seed)
    out = x.clone()
    dim = x.shape[-1]
    for k in range(dim // 2):
        i, j = 2 * k, 2 * k + 1
        theta = torch.rand(1, generator=gen).item() * 2 * math.pi
        c, s = math.cos(theta), math.sin(theta)
        a = out[..., i].clone()
        b = out[..., j].clone()
        out[..., i] = a * c - b * s
        out[..., j] = a * s + b * c
    return out


# ============ КВАНТОВАНИЕ (Fix 7: идентичное для обоих) ============

def to_polar(x: torch.Tensor) -> torch.Tensor:
    """Пары (x,y) → (r, theta)."""
    dim = x.shape[-1]
    pairs = dim // 2
    out = torch.zeros_like(x)
    for k in range(pairs):
        i, j = 2 * k, 2 * k + 1
        out[..., i] = torch.sqrt(x[..., i] ** 2 + x[..., j] ** 2)  # r
        out[..., j] = torch.atan2(x[..., j], x[..., i])              # theta
    if dim % 2:
        out[..., -1] = x[..., -1]
    return out


def quantize(x: torch.Tensor, bits: int) -> tuple:
    """Скалярное квантование Lloyd-Max. Одинаковое для обоих методов."""
    levels = 2 ** bits
    x_min = x.min(dim=0).values
    x_max = x.max(dim=0).values
    x_range = (x_max - x_min).clamp(min=1e-8)
    normalized = (x - x_min) / x_range
    q = torch.round(normalized * (levels - 1))
    dequant = q / (levels - 1) * x_range + x_min
    return dequant, x_min, x_max


def qjl_correct(original: torch.Tensor, quantized: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """QJL 1-bit correction."""
    error = original - quantized
    sign = torch.sign(error)
    return quantized + sign * error.abs() * alpha


# ============ МЕТРИКИ ============

@dataclass
class PipelineResult:
    name: str
    mse: float
    angle_variance: float
    radius_variance: float
    norm_preservation: float
    roundtrip_error: float
    overhead_bits: int


def analyze_polar_distribution(polar: torch.Tensor) -> tuple:
    """Анализ распределения углов и радиусов."""
    dim = polar.shape[-1]
    angles = polar[..., 1::2].reshape(-1)  # все theta
    radii = polar[..., 0::2].reshape(-1)   # все r

    # Дисперсия углов (гистограмма)
    bins = 32
    hist = torch.histc(angles, bins=bins, min=-math.pi, max=math.pi)
    mean_count = angles.numel() / bins
    angle_var = ((hist - mean_count) ** 2).mean().item()

    # Дисперсия радиусов
    radius_var = radii.var().item()

    return angle_var, radius_var


def run_pipeline(x: torch.Tensor, name: str, rotate_fn, layers=None,
                 bits: int = 3, phi: float = PHI) -> PipelineResult:
    """Полный пайплайн: rotate → polar → quantize → QJL → metrics."""
    # Rotate
    if layers is not None:
        rotated = rotate_fn(x, layers)
    else:
        rotated = rotate_fn(x)

    # Проверка нормы
    orig_norms = x.norm(dim=-1)
    rot_norms = rotated.norm(dim=-1)
    norm_error = (orig_norms - rot_norms).abs().mean().item()

    # Roundtrip (Fix 5)
    if layers is not None:
        roundtrip = apply_givens_inverse(rotated, layers)
        rt_error = (x - roundtrip).pow(2).mean().item()
    else:
        rt_error = float('nan')  # TurboQuant: inverse not implemented

    # Polar
    polar = to_polar(rotated)

    # Distribution analysis
    angle_var, radius_var = analyze_polar_distribution(polar)

    # Quantize (Fix 7: одинаковое)
    dequant, _, _ = quantize(polar, bits)

    # QJL
    corrected = qjl_correct(polar, dequant)

    # MSE
    mse = (polar - corrected).pow(2).mean().item()

    return PipelineResult(
        name=name,
        mse=mse,
        angle_variance=angle_var,
        radius_variance=radius_var,
        norm_preservation=norm_error,
        roundtrip_error=rt_error,
        overhead_bits=32  # Fix 7: честно для обоих
    )


# ============ KV-CACHE EXTRACTION ============

def extract_kv_cache(model_name: str, prompt: str, max_tokens: int = 128):
    """Извлекает реальные KV-кэш тензоры из модели."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Input tokens: {inputs.input_ids.shape[1]}")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    kv_cache = outputs.past_key_values
    keys, values = [], []
    for layer_kv in kv_cache:
        k, v = layer_kv[0], layer_kv[1]  # [batch, heads, seq, dim]
        keys.append(k.squeeze(0).float().cpu())    # [heads, seq, dim]
        values.append(v.squeeze(0).float().cpu())

    print(f"Extracted KV-cache: {len(keys)} layers, shape per layer: {keys[0].shape}")
    return keys, values


def generate_synthetic_kv(count: int = 200, dim: int = 128, seed: int = 42):
    """Синтетические данные с реалистичным распределением (тяжёлые хвосты)."""
    torch.manual_seed(seed)
    # Основная масса: нормальное распределение ±0.5
    x = torch.randn(count, dim) * 0.5
    # 6 конкретных измерений с выбросами до ±60 (как в реальных LLM)
    outlier_dims = [0, 15, 31, 63, 95, 127 % dim]
    for d in outlier_dims:
        if d < dim:
            mask = torch.rand(count) < 0.75  # 75% позиций активны
            x[mask, d] = torch.randn(mask.sum()) * 30  # до ±60
    return x


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description="NautilusQuant v2 validation")
    parser.add_argument("--model", type=str, default=None,
                        help="HuggingFace model (e.g. google/gemma-3-4b-it)")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension")
    parser.add_argument("--count", type=int, default=500, help="Number of vectors")
    parser.add_argument("--bits", type=int, default=3, help="Quantization bits")
    parser.add_argument("--sweep", action="store_true", help="Sweep PHI values")
    args = parser.parse_args()

    print("=" * 60)
    print("NautilusQuant v2 — Validation")
    print("=" * 60)

    # --- Получаем данные ---
    if args.model:
        prompt = "The transformer architecture uses multi-head attention to process sequences of tokens efficiently."
        keys, values = extract_kv_cache(args.model, prompt)
        # Берём первый слой, все головы, все позиции
        x = keys[0].reshape(-1, keys[0].shape[-1])  # [heads*seq, dim]
        args.dim = x.shape[-1]
        print(f"Using real KV-cache: {x.shape}")
    else:
        print(f"Using synthetic data with realistic outliers (dim={args.dim}, n={args.count})")
        x = generate_synthetic_kv(args.count, args.dim)

    print(f"Data shape: {x.shape}, norm range: [{x.norm(dim=-1).min():.2f}, {x.norm(dim=-1).max():.2f}]")
    print(f"Outlier dims: max abs value per dim = {x.abs().max(dim=0).values.topk(6).values.tolist()}")
    print()

    # --- Строим слои ---
    layers = build_givens_layers(args.dim, PHI)
    print(f"Nautilus layers: {len(layers)} layers, "
          f"pairs per layer: {[len(l) for l in layers]}")

    # --- Проверка ортогональности ---
    print("\n--- Orthogonality Check ---")
    test_vec = x[0]
    fwd = apply_givens_forward(test_vec.unsqueeze(0), layers).squeeze(0)
    rt = apply_givens_inverse(fwd.unsqueeze(0), layers).squeeze(0)
    print(f"  |v|     = {test_vec.norm():.6f}")
    print(f"  |T·v|   = {fwd.norm():.6f}")
    print(f"  |T⁻¹Tv| = {rt.norm():.6f}")
    print(f"  Norm error:      {abs(test_vec.norm().item() - fwd.norm().item()):.2e}")
    print(f"  Roundtrip error: {(test_vec - rt).pow(2).sum().sqrt().item():.2e}")

    # dot product preservation
    v2 = x[1] if len(x) > 1 else torch.randn_like(test_vec)
    fwd2 = apply_givens_forward(v2.unsqueeze(0), layers).squeeze(0)
    orig_dot = (test_vec * v2).sum().item()
    fwd_dot = (fwd * fwd2).sum().item()
    print(f"  dot(a,b)     = {orig_dot:.6f}")
    print(f"  dot(Ta, Tb)  = {fwd_dot:.6f}")
    print(f"  Dot error:   {abs(orig_dot - fwd_dot):.2e}")

    # --- Запускаем пайплайны ---
    print(f"\n--- Pipeline Comparison (bits={args.bits}) ---")

    turbo = run_pipeline(x, "TurboQuant", random_rotation, bits=args.bits)
    nautilus = run_pipeline(x, "NautilusQuant", apply_givens_forward,
                           layers=layers, bits=args.bits)

    for r in [turbo, nautilus]:
        print(f"\n  {r.name}:")
        print(f"    MSE:              {r.mse:.8f}")
        print(f"    Angle variance:   {r.angle_variance:.4f}")
        print(f"    Radius variance:  {r.radius_variance:.4f}")
        print(f"    Norm preservation:{r.norm_preservation:.2e}")
        print(f"    Roundtrip error:  {r.roundtrip_error:.2e}")
        print(f"    Overhead:         {r.overhead_bits} bits/group")

    print(f"\n  >>> MSE diff: {turbo.mse - nautilus.mse:.8f}")
    print(f"  >>> Angle var diff: {turbo.angle_variance - nautilus.angle_variance:.4f}")
    winner = "NautilusQuant" if nautilus.mse < turbo.mse else "TurboQuant"
    print(f"  >>> Winner by MSE: {winner}")

    # --- PHI Sweep ---
    if args.sweep:
        print(f"\n--- PHI Sweep ---")
        phi_values = [1.2, 1.3, 1.414, 1.5, 1.618, 1.732, 2.0, 2.236, 2.718, 3.14159]
        results = []
        for phi in phi_values:
            ly = build_givens_layers(args.dim, phi)
            r = run_pipeline(x, f"φ={phi:.3f}", apply_givens_forward,
                             layers=ly, bits=args.bits)
            results.append((phi, r))
            label = "φ" if abs(phi - PHI) < 0.01 else ""
            print(f"  φ={phi:.4f} {label:2s} MSE={r.mse:.8f}  angleVar={r.angle_variance:.2f}")

        best = min(results, key=lambda x: x[1].mse)
        print(f"\n  >>> Best PHI: {best[0]:.4f} (MSE={best[1].mse:.8f})")

    # --- Plan B experiments (numpy, no extra deps) ---
    print(f"\n--- Plan B: Experimental Modules ---")
    x_np = x.cpu().numpy()

    try:
        from plan_b.golden_jl import GoldenJLTransform
        gjl = GoldenJLTransform(dim=args.dim)
        jl_result = gjl.test_jl_property(x_np, bits=args.bits)
        print(f"\n  [Idea 2] Golden JL-Transform (can we kill QJL?):")
        print(f"    WITHOUT QJL: bias={jl_result['without_qjl']['bias']:.6f}, "
              f"corr={jl_result['without_qjl']['correlation']:.6f}, "
              f"compression={jl_result['without_qjl']['compression']:.1f}x")
        print(f"    WITH QJL:    bias={jl_result['with_qjl']['bias']:.6f}, "
              f"corr={jl_result['with_qjl']['correlation']:.6f}, "
              f"compression={jl_result['with_qjl']['compression']:.1f}x")
        print(f"    >>> {jl_result['verdict']}")
    except Exception as e:
        print(f"  [Idea 2] Skipped: {e}")

    try:
        from plan_b.phinary import PhinaryQuantizer
        pq = PhinaryQuantizer(bits=args.bits)
        flat = x_np.flatten()
        ph_result = pq.encode(flat)
        print(f"\n  [Idea 3] Phinary Quantization (base φ):")
        print(f"    MSE: {ph_result['mse']:.6f} ({ph_result['method']})")
    except Exception as e:
        print(f"  [Idea 3] Skipped: {e}")

    try:
        from plan_b.fractal_hash import DeltaOrbitEncoder
        # Get polar angles from nautilus-rotated data
        rot_np = x_np.copy()
        ga = 2 * math.pi / (PHI ** 2)
        for k in range(args.dim // 2):
            i, j = 2*k, 2*k+1
            theta = ga * (k+1)
            c, s = math.cos(theta), math.sin(theta)
            a, b = rot_np[:, i].copy(), rot_np[:, j].copy()
            rot_np[:, i] = a * c - b * s
            rot_np[:, j] = a * s + b * c
        angles = np.arctan2(rot_np[:, 1], rot_np[:, 0])

        enc = DeltaOrbitEncoder(n_steps=256)
        fr_result = enc.encode(angles)
        print(f"\n  [Idea 4] Fractal Sub-1-bit Hashing:")
        print(f"    Direct: {fr_result['direct_bits']:.1f} bits/angle")
        print(f"    Delta entropy: {fr_result['delta_entropy']:.2f} bits/angle")
        print(f"    Max error: {fr_result['max_error_deg']:.2f}°")
    except Exception as e:
        print(f"  [Idea 4] Skipped: {e}")

    print("\n" + "=" * 60)
    print("Done. For real validation, run with --model google/gemma-3-4b-it")


if __name__ == "__main__":
    import numpy as np
    main()
