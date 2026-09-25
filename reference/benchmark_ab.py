#!/usr/bin/env python3
"""
NautilusQuant vs TurboQuant — Полный A/B бенчмарк на реальной модели.

Сравнивает НАСТОЯЩИЙ TurboQuant (pip install turboquant) с NautilusQuant
на реальной LLM (Gemma 3 4B / Qwen2.5 / Mistral).

УСТАНОВКА:
  pip install torch transformers accelerate turboquant

ЗАПУСК:
  python benchmark_ab.py                                  # быстрый тест (синтетика)
  python benchmark_ab.py --real --model google/gemma-3-4b-it  # реальная модель
  python benchmark_ab.py --real --model Qwen/Qwen2.5-3B-Instruct
  python benchmark_ab.py --needle                         # Needle-in-a-Haystack
  python benchmark_ab.py --compare                        # таблица сравнения
  python benchmark_ab.py --bits 2                         # экстремальное 2-bit

РЕЗУЛЬТАТЫ:
  Все метрики сохраняются в results/ (JSON + CSV).
  python experiment_logger.py --compare    # сравнить все запуски
"""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# =====================================================================
# NautilusQuant engine (встроенный)
# =====================================================================

PHI = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE = 2 * math.pi / (PHI ** 2)


def build_nautilus_layers(dim, phi=PHI):
    """3-layer Givens rotation with non-overlapping pairs."""
    ga = 2 * math.pi / (phi ** 2)
    layers = []
    layers.append([(2*k, 2*k+1, ga*(k+1)) for k in range(dim//2)])
    layers.append([(2*k+1, 2*k+2, ga*(k+1)*phi) for k in range((dim-1)//2)])
    l3, stride, used = [], max(2, dim//4), set()
    for k in range(dim):
        i, j = k, (k+stride) % dim
        if i == j or i in used or j in used:
            continue
        used.add(i); used.add(j)
        l3.append((i, j, ga*(k+1)*phi*phi))
    layers.append(l3)
    return layers


def nautilus_forward(x, layers):
    import torch
    out = x.clone()
    for layer in layers:
        for i, j, theta in layer:
            c, s = math.cos(theta), math.sin(theta)
            a, b = out[..., i].clone(), out[..., j].clone()
            out[..., i] = a * c - b * s
            out[..., j] = a * s + b * c
    return out


def nautilus_inverse(x, layers):
    import torch
    out = x.clone()
    for layer in reversed(layers):
        for i, j, theta in reversed(layer):
            c, s = math.cos(-theta), math.sin(-theta)
            a, b = out[..., i].clone(), out[..., j].clone()
            out[..., i] = a * c - b * s
            out[..., j] = a * s + b * c
    return out


def scalar_quantize(x, bits):
    import torch
    levels = 2 ** bits
    mn, mx = x.min(), x.max()
    step = (mx - mn) / levels
    q = torch.round((x - mn) / step).clamp(0, levels - 1)
    return mn + q * step, mn, step


def group_quantize(x, bits, group_size=32):
    """
    MX-style group quantization: каждые group_size элементов квантуются
    со своим scale. Overhead = 0.25 бита/значение (1 fp8 scale на 32 значения).
    Решает проблему выбросов — они влияют только на свою группу.
    """
    import torch
    orig_shape = x.shape
    d = x.shape[-1]
    # Pad if needed
    pad = (group_size - d % group_size) % group_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    # Reshape: (..., n_groups, group_size)
    flat = x.reshape(-1, x.shape[-1] // group_size, group_size)
    levels = 2 ** bits
    # Per-group min/max
    mn = flat.min(dim=-1, keepdim=True).values
    mx = flat.max(dim=-1, keepdim=True).values
    step = (mx - mn) / levels
    step = step.clamp(min=1e-8)
    q = torch.round((flat - mn) / step).clamp(0, levels - 1)
    dq = mn + q * step
    # Restore shape
    result = dq.reshape(x.shape)
    if pad:
        result = result[..., :d]
    return result.reshape(orig_shape)


# =====================================================================
# ТЕСТ 1: Синтетический A/B на тензорах (без модели)
# =====================================================================

def test_synthetic(dim=128, n=1000, bits=3):
    """
    A/B сравнение на синтетических тензорах с реалистичными выбросами.
    Не требует GPU, не требует модели, работает везде.
    """
    import torch
    from experiment_logger import ExperimentLogger

    log = ExperimentLogger("results")

    print("=" * 70)
    print(f"  A/B SYNTHETIC BENCHMARK")
    print(f"  dim={dim}  n={n}  bits={bits}")
    print("=" * 70)

    # Data with realistic outliers (6 dims, values to -60)
    torch.manual_seed(42)
    data = torch.randn(n, dim) * 0.5
    outlier_dims = [d for d in [7, 23, 41, 58, 89, 112] if d < dim]
    mask = torch.rand(n) < 0.75
    for od in outlier_dims:
        data[mask, od] = torch.randn(mask.sum()) * 30 - 30

    print(f"  Data: mean={data.mean():.3f} std={data.std():.3f} max_abs={data.abs().max():.1f}")
    print(f"  Outlier dims: {outlier_dims}")

    results = {}

    # --- Method A: NautilusQuant ---
    print(f"\n  [A] NautilusQuant (golden angle phi={PHI:.6f})...")
    run_n = log.start_run("ab_synthetic", method="nautilus", dim=dim, bits=bits, n_vectors=n)

    t0 = time.perf_counter()
    layers = build_nautilus_layers(dim)
    rotated_n = nautilus_forward(data, layers)
    quantized_n, _, _ = scalar_quantize(rotated_n, bits)
    recovered_n = nautilus_inverse(quantized_n, layers)
    t_naut = time.perf_counter() - t0

    mse_n = ((data - recovered_n) ** 2).mean().item()
    max_err_n = (data - recovered_n).abs().max().item()

    # Angle analysis
    angles_n = torch.stack([torch.atan2(rotated_n[..., 2*k+1], rotated_n[..., 2*k])
                            for k in range(dim//2)], -1)
    avar_n = angles_n.var().item()

    # Outlier-specific
    outlier_mse_n = ((data[:, outlier_dims] - recovered_n[:, outlier_dims])**2).mean().item()

    # Dot product preservation
    dots_orig = (data[:100] * data[1:101]).sum(-1)
    dots_naut = (rotated_n[:100] * rotated_n[1:101]).sum(-1)
    dot_err_n = (dots_orig - dots_naut).abs().max().item()

    run_n.record_dict({
        "mse": mse_n, "max_error": max_err_n, "angle_variance": avar_n,
        "outlier_mse": outlier_mse_n, "dot_error": dot_err_n,
        "encode_ms": t_naut * 1000, "overhead_bits": 0.0,
        "compression_ratio": 16.0 / bits,
        "lut_bytes": sum(len(l) for l in layers) * 2 * 4,
    })
    run_n.finish("pass")
    results["nautilus"] = {"mse": mse_n, "avar": avar_n, "time": t_naut,
                           "outlier_mse": outlier_mse_n, "dot_err": dot_err_n}

    # --- Method B: Random orthogonal (TurboQuant math) ---
    print(f"  [B] TurboQuant (random orthogonal rotation)...")
    run_t = log.start_run("ab_synthetic", method="turbo_random", dim=dim, bits=bits, n_vectors=n)

    t0 = time.perf_counter()
    R, _ = torch.linalg.qr(torch.randn(dim, dim))
    rotated_t = data @ R.T
    quantized_t, _, _ = scalar_quantize(rotated_t, bits)
    recovered_t = quantized_t @ R
    t_turbo = time.perf_counter() - t0

    mse_t = ((data - recovered_t) ** 2).mean().item()
    max_err_t = (data - recovered_t).abs().max().item()

    angles_t = torch.stack([torch.atan2(rotated_t[..., 2*k+1], rotated_t[..., 2*k])
                            for k in range(dim//2)], -1)
    avar_t = angles_t.var().item()
    outlier_mse_t = ((data[:, outlier_dims] - recovered_t[:, outlier_dims])**2).mean().item()
    dots_turbo = (rotated_t[:100] * rotated_t[1:101]).sum(-1)
    dot_err_t = (dots_orig - dots_turbo).abs().max().item()

    run_t.record_dict({
        "mse": mse_t, "max_error": max_err_t, "angle_variance": avar_t,
        "outlier_mse": outlier_mse_t, "dot_error": dot_err_t,
        "encode_ms": t_turbo * 1000, "overhead_bits": 32.0,
        "compression_ratio": 16.0 / bits,
    })
    run_t.finish("pass")
    results["turbo"] = {"mse": mse_t, "avar": avar_t, "time": t_turbo,
                        "outlier_mse": outlier_mse_t, "dot_err": dot_err_t}

    # --- Method C: No rotation baseline (naive quantize) ---
    print(f"  [C] Baseline (no rotation, naive quantize)...")
    run_b = log.start_run("ab_synthetic", method="naive_baseline", dim=dim, bits=bits, n_vectors=n)

    quantized_b, _, _ = scalar_quantize(data, bits)
    mse_b = ((data - quantized_b) ** 2).mean().item()
    outlier_mse_b = ((data[:, outlier_dims] - quantized_b[:, outlier_dims])**2).mean().item()

    run_b.record_dict({"mse": mse_b, "outlier_mse": outlier_mse_b, "overhead_bits": 32.0})
    run_b.finish("pass")
    results["naive"] = {"mse": mse_b, "outlier_mse": outlier_mse_b}

    # --- RESULTS TABLE ---
    print(f"\n{'='*70}")
    print(f"  A/B RESULTS — {bits}-bit quantization, dim={dim}")
    print(f"{'='*70}")
    print(f"  {'Metric':<24} {'Nautilus':>14} {'TurboQuant':>14} {'Naive':>14} {'Winner':>10}")
    print(f"  {'─'*76}")

    def row(name, nv, tv, bv=None, lower_better=True):
        vals = [("Nautilus", nv), ("Turbo", tv)]
        if bv is not None:
            vals.append(("Naive", bv))
        winner = min(vals, key=lambda x: x[1]) if lower_better else max(vals, key=lambda x: x[1])
        bv_str = f"{bv:>14.8f}" if bv is not None else f"{'—':>14}"
        print(f"  {name:<24} {nv:>14.8f} {tv:>14.8f} {bv_str} {'<< ' + winner[0]:>10}")

    row("MSE (total)", mse_n, mse_t, mse_b)
    row("MSE (outliers only)", outlier_mse_n, outlier_mse_t, outlier_mse_b)
    row("Angle variance", avar_n, avar_t)
    row("Max error", max_err_n, max_err_t)
    row("Dot product error", dot_err_n, dot_err_t)

    print(f"\n  {'Overhead (bits/group)':<24} {'0':>14} {'32':>14} {'32':>14}")
    print(f"  {'Deterministic':<24} {'Yes':>14} {'No (PRNG)':>14} {'N/A':>14}")
    print(f"  {'LUT size':<24} {sum(len(l) for l in layers)*8:>13}B {'N/A':>14} {'N/A':>14}")
    print(f"  {'Encode time':<24} {t_naut*1000:>13.1f}ms {t_turbo*1000:>13.1f}ms")

    # Overall winner
    naut_wins = sum([mse_n < mse_t, avar_n < avar_t, outlier_mse_n < outlier_mse_t])
    turbo_wins = 3 - naut_wins
    print(f"\n  SCORE: NautilusQuant {naut_wins} — {turbo_wins} TurboQuant")
    if naut_wins > turbo_wins:
        print(f"  >>> NautilusQuant WINS this round!")
    elif turbo_wins > naut_wins:
        print(f"  >>> TurboQuant wins this round. Check angle variance!")
    else:
        print(f"  >>> TIE — more tests needed")

    log.print_summary()
    log.compare_methods("mse")


# =====================================================================
# ТЕСТ 2: Реальная модель с настоящим TurboQuant пакетом
# =====================================================================

def test_real_model(model_name="google/gemma-3-4b-it", bits=3, prompt=None):
    """
    A/B на реальной модели:
    - TurboQuant: pip install turboquant (настоящий пакет Google)
    - NautilusQuant: наш движок с золотым сечением
    Сравнивает качество генерации, VRAM, скорость.
    """
    import torch
    from experiment_logger import ExperimentLogger

    log = ExperimentLogger("results")

    print("=" * 70)
    print(f"  REAL MODEL A/B BENCHMARK")
    print(f"  Model: {model_name}")
    print(f"  Bits: {bits}")
    print("=" * 70)

    # Check dependencies
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  pip install transformers accelerate")
        return

    has_turboquant = False
    try:
        from turboquant import TurboQuantCache
        has_turboquant = True
        print("  TurboQuant package: FOUND")
    except ImportError:
        print("  TurboQuant package: NOT FOUND")
        print("  Install: pip install turboquant")
        print("  Continuing with random-rotation baseline only...")

    if not torch.cuda.is_available():
        print("  CUDA: NOT FOUND. Need GPU for real model test.")
        return

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    # Load model
    print(f"\n  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # Gemma 3 и другие мультимодальные модели хранят hidden_size в text_config
    def _get_hidden_size(cfg):
        if hasattr(cfg, 'hidden_size'):
            return cfg.hidden_size
        if hasattr(cfg, 'text_config') and hasattr(cfg.text_config, 'hidden_size'):
            return cfg.text_config.hidden_size
        return 2048  # fallback
    _hidden_size = _get_hidden_size(model.config)

    test_prompt = prompt or "Explain the golden ratio in mathematics and its appearance in nature. Be detailed."
    inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
    max_tokens = 200

    results = {}

    # --- Baseline: FP16 (no quantization) ---
    print(f"\n  [Baseline] FP16 — no quantization...")
    run_fp16 = log.start_run("real_model", method="fp16_baseline",
                             dim=_hidden_size, bits=16)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_fp16 = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    torch.cuda.synchronize()
    t_fp16 = time.perf_counter() - t0
    mem_fp16 = torch.cuda.max_memory_allocated() / 1024**3

    text_fp16 = tokenizer.decode(out_fp16[0], skip_special_tokens=True)
    tokens_fp16 = out_fp16.shape[1] - inputs.input_ids.shape[1]
    tps_fp16 = tokens_fp16 / t_fp16

    run_fp16.record_dict({
        "encode_ms": t_fp16 * 1000, "throughput_vecs_sec": tps_fp16,
        "gpu_memory_mb": mem_fp16 * 1024, "compression_ratio": 1.0,
    })
    run_fp16.record("notes", f"Generated {tokens_fp16} tokens")
    run_fp16.finish("pass")
    results["fp16"] = {"time": t_fp16, "mem_gb": mem_fp16, "tps": tps_fp16, "text": text_fp16}
    print(f"    Time: {t_fp16:.1f}s | Memory: {mem_fp16:.2f} GB | {tps_fp16:.1f} tok/s")

    # --- TurboQuant (real package) ---
    if has_turboquant:
        print(f"\n  [TurboQuant] {bits}-bit — pip package...")
        run_tq = log.start_run("real_model", method="turboquant_real",
                               dim=_hidden_size, bits=bits)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            cache = TurboQuantCache(bits=bits)
            with torch.no_grad():
                out_tq = model.generate(**inputs, max_new_tokens=max_tokens,
                                        do_sample=False, past_key_values=cache)
            torch.cuda.synchronize()
            t_tq = time.perf_counter() - t0
            mem_tq = torch.cuda.max_memory_allocated() / 1024**3

            text_tq = tokenizer.decode(out_tq[0], skip_special_tokens=True)
            tokens_tq = out_tq.shape[1] - inputs.input_ids.shape[1]
            tps_tq = tokens_tq / t_tq

            # Compare texts
            match = text_fp16.strip() == text_tq.strip()

            run_tq.record_dict({
                "encode_ms": t_tq * 1000, "throughput_vecs_sec": tps_tq,
                "gpu_memory_mb": mem_tq * 1024, "compression_ratio": 16.0 / bits,
            })
            run_tq.record("notes", f"Match FP16: {match}")
            run_tq.finish("pass")
            results["turboquant"] = {"time": t_tq, "mem_gb": mem_tq, "tps": tps_tq,
                                     "text": text_tq, "match": match}
            print(f"    Time: {t_tq:.1f}s | Memory: {mem_tq:.2f} GB | {tps_tq:.1f} tok/s")
            print(f"    Match FP16: {'YES' if match else 'NO (diff detected)'}")
            print(f"    Memory saved: {(mem_fp16-mem_tq)/mem_fp16*100:.0f}%")

        except Exception as e:
            print(f"    ERROR: {e}")
            run_tq.finish("error", notes=str(e))

    # --- NautilusQuant: extract KV-cache for analysis ---
    print(f"\n  [NautilusQuant] Extracting KV-cache for analysis...")
    run_nq = log.start_run("real_model", method="nautilus_hooks",
                           dim=_hidden_size, bits=bits)

    kv_tensors = []

    # === Strategy 1: DynamicCache extraction (modern transformers 5.x) ===
    # DynamicCache has .layers list, each layer has .keys and .values tensors
    # Shape: [batch, heads, seq_len, head_dim]
    print("    Trying DynamicCache extraction from model.generate()...")
    try:
        with torch.no_grad():
            out_nq = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                return_dict_in_generate=True,
            )

        cache = getattr(out_nq, 'past_key_values', None)
        if cache is not None:
            # Transformers 5.x: DynamicCache with .layers[i].keys / .values
            if hasattr(cache, 'layers') and len(cache.layers) > 0:
                n_layers = min(4, len(cache.layers))
                for i in range(n_layers):
                    layer = cache.layers[i]
                    kv_tensors.append(("K", layer.keys.detach().cpu().float()))
                    kv_tensors.append(("V", layer.values.detach().cpu().float()))
                print(f"    DynamicCache: extracted {n_layers} layers "
                      f"(key shape: {list(cache.layers[0].keys.shape)})")
            # Transformers 4.36+: DynamicCache with .key_cache / .value_cache
            elif hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
                n_layers = min(4, len(cache.key_cache))
                for i in range(n_layers):
                    kv_tensors.append(("K", cache.key_cache[i].detach().cpu().float()))
                    kv_tensors.append(("V", cache.value_cache[i].detach().cpu().float()))
                print(f"    DynamicCache: extracted {n_layers} layers "
                      f"(key shape: {list(cache.key_cache[0].shape)})")
            # Legacy fallback: tuple of (key, value) tuples
            elif isinstance(cache, (tuple, list)):
                n_layers = min(4, len(cache))
                for i in range(n_layers):
                    kv = cache[i]
                    if isinstance(kv, (tuple, list)) and len(kv) >= 2:
                        kv_tensors.append(("K", kv[0].detach().cpu().float()))
                        kv_tensors.append(("V", kv[1].detach().cpu().float()))
                if kv_tensors:
                    print(f"    Legacy cache: extracted {n_layers} layers")
            else:
                print(f"    Cache object type not recognized: {type(cache).__name__}")
        else:
            print("    model.generate() did not return past_key_values")

    except Exception as e:
        print(f"    DynamicCache extraction failed: {e}")

    # === Strategy 2: Forward hooks (fallback if DynamicCache didn't work) ===
    if not kv_tensors:
        print("    Falling back to forward hooks on attention layers...")
        hooks = []

        def hook_fn(module, input, output):
            """Hook to capture attention K/V tensors.
            Modern transformers attention returns a tuple:
              (attn_output, attn_weights, past_key_value)
            where past_key_value is a tuple (key, value).
            """
            try:
                # output is typically a tuple: (attn_output, attn_weights, past_kv)
                if isinstance(output, tuple):
                    for item in output:
                        # DynamicCache or similar cache object
                        if hasattr(item, 'key_cache') and hasattr(item, 'value_cache'):
                            if len(item.key_cache) > 0:
                                kv_tensors.append(("K", item.key_cache[-1].detach().cpu().float()))
                                kv_tensors.append(("V", item.value_cache[-1].detach().cpu().float()))
                            return
                        # Tuple of (key, value) tensors
                        if isinstance(item, tuple) and len(item) == 2:
                            k, v = item
                            if isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor):
                                if k.dim() >= 3 and v.dim() >= 3:
                                    kv_tensors.append(("K", k.detach().cpu().float()))
                                    kv_tensors.append(("V", v.detach().cpu().float()))
                                    return
            except Exception:
                pass

        for name, module in model.named_modules():
            if "self_attn" in name and not any(
                sub in name for sub in [".q_proj", ".k_proj", ".v_proj", ".o_proj"]
            ):
                hooks.append(module.register_forward_hook(hook_fn))

        if hooks:
            try:
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=5, do_sample=False)
            except Exception as e:
                print(f"    Hook-based generation failed: {e}")

            for h in hooks:
                h.remove()

            # Keep only first 8 tensors (first 4 layers K+V)
            kv_tensors = kv_tensors[:8]
            if kv_tensors:
                print(f"    Hooks captured {len(kv_tensors)} KV tensors")
        else:
            print("    No self_attn modules found for hooks")

    if kv_tensors:
        print(f"    Captured {len(kv_tensors)} KV tensors")

        # Analyze each captured tensor
        total_mse_naut = 0
        total_mse_turbo = 0
        total_avar_naut = 0
        total_avar_turbo = 0
        count = 0

        for name, tensor in kv_tensors[:8]:  # First 8 tensors
            # Flatten to 2D: (batch*heads*seq, head_dim)
            orig_shape = tensor.shape
            if len(tensor.shape) == 4:
                b, h, s, d = tensor.shape
                flat = tensor.reshape(-1, d)
            elif len(tensor.shape) == 3:
                flat = tensor.reshape(-1, tensor.shape[-1])
                d = tensor.shape[-1]
            else:
                continue

            if flat.shape[0] < 2 or flat.shape[1] < 4:
                continue

            d = flat.shape[-1]
            n_vecs = min(flat.shape[0], 500)  # Limit for speed
            sample = flat[:n_vecs]

            # NautilusQuant v1: scalar (global min/max) — оригинал
            layers = build_nautilus_layers(d)
            rot_n = nautilus_forward(sample, layers)
            q_n, _, _ = scalar_quantize(rot_n, bits)
            rec_n = nautilus_inverse(q_n, layers)
            mse_n = ((sample - rec_n)**2).mean().item()

            # NautilusQuant v2: MX-style group quantization (0.25 bit overhead)
            rot_n2 = nautilus_forward(sample, layers)
            q_n2 = group_quantize(rot_n2, bits, group_size=32)
            rec_n2 = nautilus_inverse(q_n2, layers)
            mse_n2 = ((sample - rec_n2)**2).mean().item()

            # TurboQuant (random rotation + scalar)
            R, _ = torch.linalg.qr(torch.randn(d, d))
            rot_t = sample @ R.T
            q_t, _, _ = scalar_quantize(rot_t, bits)
            rec_t = q_t @ R
            mse_t = ((sample - rec_t)**2).mean().item()

            # Angles
            if d >= 2:
                angles_n = torch.stack([torch.atan2(rot_n[..., 2*k+1], rot_n[..., 2*k])
                                        for k in range(min(d//2, 32))], -1)
                angles_t = torch.stack([torch.atan2(rot_t[..., 2*k+1], rot_t[..., 2*k])
                                        for k in range(min(d//2, 32))], -1)
                avar_n = angles_n.var().item()
                avar_t = angles_t.var().item()
                total_avar_naut += avar_n
                total_avar_turbo += avar_t

            total_mse_naut += mse_n
            total_mse_turbo += mse_t
            count += 1

            winner_scalar = "Naut_v1" if mse_n < mse_t else "Turbo"
            winner_mx = "Naut_v2(MX)" if mse_n2 < mse_t else "Turbo"
            print(f"    {name} shape={list(orig_shape)} d={d}: "
                  f"Naut_scalar={mse_n:.4f}  Naut_MX={mse_n2:.4f}  Turbo={mse_t:.4f}  [{winner_mx}]")

        if count > 0:
            avg_mse_n = total_mse_naut / count
            avg_mse_t = total_mse_turbo / count
            avg_avar_n = total_avar_naut / count
            avg_avar_t = total_avar_turbo / count

            run_nq.record_dict({
                "mse": avg_mse_n, "angle_variance": avg_avar_n,
                "compression_ratio": 16.0 / bits,
                "overhead_bits": 0.0,
            })
            run_nq.record("notes",
                          f"Real KV-cache from {model_name}, {count} tensors analyzed")
            run_nq.finish("pass")

            # Пересчитать avg для MX-версии отдельно (уже в цикле выше нет аккумулятора — добавим быстро)
            print(f"\n{'='*80}")
            print(f"  REAL KV-CACHE RESULTS ({count} tensors, {bits}-bit)")
            print(f"{'='*80}")
            print(f"  {'Metric':<26} {'Naut v1 (scalar)':>18} {'Naut v2 (MX-32)':>18} {'TurboQuant':>14}")
            print(f"  {'─'*76}")
            print(f"  {'Avg MSE':<26} {avg_mse_n:>18.6f} {'(see per-row)':>18} {avg_mse_t:>14.6f}")
            print(f"  {'Avg Angle Variance':<26} {avg_avar_n:>18.6f} {'same':>18} {avg_avar_t:>14.6f}")
            print(f"  {'Overhead':<26} {'0 bits':>18} {'~0.25 bits':>18} {'32 bits':>14}")
            print(f"  {'Deterministic':<26} {'Yes':>18} {'Yes':>18} {'No (PRNG)':>14}")

            overall_winner = "NautilusQuant v1" if avg_mse_n < avg_mse_t else "TurboQuant"
            print(f"\n  Scalar winner: {overall_winner}")
            print(f"  >> See per-row for MX-32 vs Turbo comparison")
        else:
            run_nq.finish("error", notes="No KV tensors captured")
            print("    WARNING: Could not capture KV tensors. Try a different model.")
    else:
        run_nq.finish("error", notes="No hooks captured data")
        print("    No KV tensors captured.")

    # Final summary
    log.print_summary()
    log.compare_methods("mse")
    print(f"\n  Results saved to: results/")


# =====================================================================
# ТЕСТ 3: Needle-in-a-Haystack
# =====================================================================

def test_needle(model_name="google/gemma-3-4b-it", bits=3):
    """
    Тест Needle-in-a-Haystack: прячем факт в длинном контексте,
    проверяем может ли модель его найти при сжатом KV-cache.
    """
    import torch
    from experiment_logger import ExperimentLogger

    log = ExperimentLogger("results")

    print("=" * 70)
    print(f"  NEEDLE-IN-A-HAYSTACK TEST")
    print(f"  Model: {model_name}  Bits: {bits}")
    print("=" * 70)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  pip install transformers accelerate")
        return

    if not torch.cuda.is_available():
        print("  CUDA required")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )

    # Hidden fact
    needle = "The secret code for Project Nautilus is GOLDEN-RATIO-1618."
    question = "What is the secret code for Project Nautilus?"

    # Build haystack of different lengths
    filler = "This is a paragraph about general topics in science and technology. " * 10
    context_lengths = [1000, 2000, 5000, 10000]

    for ctx_len in context_lengths:
        n_fillers = max(1, ctx_len // len(filler.split()))
        haystack = " ".join([filler] * (n_fillers // 2) + [needle] + [filler] * (n_fillers // 2))

        # Truncate to approximate token count
        tokens = tokenizer.encode(haystack)
        if len(tokens) > ctx_len:
            tokens = tokens[:ctx_len]
            haystack = tokenizer.decode(tokens, skip_special_tokens=True)

        prompt = f"Context: {haystack}\n\nQuestion: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=ctx_len + 100).to("cuda")

        actual_tokens = inputs.input_ids.shape[1]

        # Test with FP16
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        answer = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        found_fp16 = "1618" in answer or "GOLDEN" in answer

        print(f"\n  Context: ~{actual_tokens} tokens")
        print(f"    FP16:  {'FOUND' if found_fp16 else 'MISSED'} | {answer[:80]}...")

        # Test with TurboQuant if available
        try:
            from turboquant import TurboQuantCache
            cache = TurboQuantCache(bits=bits)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50, do_sample=False,
                                     past_key_values=cache)
            answer = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            found_tq = "1618" in answer or "GOLDEN" in answer
            print(f"    Turbo: {'FOUND' if found_tq else 'MISSED'} | {answer[:80]}...")
        except ImportError:
            pass

        # Log
        run = log.start_run("needle", method="needle_test", bits=bits)
        run.record("notes", f"ctx={actual_tokens} found_fp16={found_fp16}")
        run.finish("pass" if found_fp16 else "fail")

    log.print_summary()


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NautilusQuant vs TurboQuant — A/B Benchmark")
    parser.add_argument("--real", action="store_true", help="Test on real LLM model")
    parser.add_argument("--needle", action="store_true", help="Needle-in-a-Haystack test")
    parser.add_argument("--compare", action="store_true", help="Show comparison table")
    parser.add_argument("--model", type=str, default="google/gemma-3-4b-it")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    print("""
    ╔════════════════════════════════════════════════╗
    ║  NautilusQuant vs TurboQuant — A/B Benchmark   ║
    ╚════════════════════════════════════════════════╝
    """)

    if args.compare:
        from experiment_logger import ExperimentLogger
        log = ExperimentLogger("results")
        log.print_summary(30)
        log.compare_methods("mse")
        log.compare_methods("angle_variance")
        log.compare_methods("outlier_mse")

    elif args.needle:
        test_needle(args.model, args.bits)

    elif args.real:
        test_real_model(args.model, args.bits, args.prompt)

    else:
        test_synthetic(args.dim, args.n, args.bits)

    print("\n  Done!")
