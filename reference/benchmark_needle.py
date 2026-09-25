"""
NautilusQuant v2 — Фаза 4: Needle-in-a-Haystack + LongBench бенчмарки

Тестирует: сжатый KV-кэш (3 бит) не ломает ли "мозг" нейросети?

Запуск:
  pip install torch transformers datasets
  python benchmark_needle.py --model google/gemma-3-4b-it --method nautilus
  python benchmark_needle.py --model google/gemma-3-4b-it --method turbo
  python benchmark_needle.py --model google/gemma-3-4b-it --method both
"""

import argparse
import math
import time

import torch

PHI = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE = 2 * math.pi / (PHI ** 2)


# ============ KV-CACHE HOOKS ============

class KVCacheQuantizer:
    """Перехватывает KV-кэш и квантует на лету."""

    def __init__(self, method='nautilus', bits=3, dim=128, phi=PHI):
        self.method = method
        self.bits = bits
        self.dim = dim
        self.phi = phi
        self.stats = {'total_mse': 0, 'n_calls': 0, 'total_time': 0}

        if method == 'nautilus':
            from nautilus_triton import NautilusQuantPyTorch, NautilusConfig
            config = NautilusConfig(dim=dim, bits=bits, phi=phi)
            self.quantizer = NautilusQuantPyTorch(config)
        self._build_turbo_angles()

    def _build_turbo_angles(self):
        """Pre-generate random angles for TurboQuant baseline."""
        gen = torch.Generator().manual_seed(42)
        self.turbo_angles = []
        for k in range(self.dim // 2):
            self.turbo_angles.append(torch.rand(1, generator=gen).item() * 2 * math.pi)

    def quantize_kv(self, key_or_value: torch.Tensor) -> torch.Tensor:
        """Quantize a KV tensor: rotate → polar → quantize → QJL → inverse."""
        t0 = time.perf_counter()
        original_shape = key_or_value.shape
        x = key_or_value.float().reshape(-1, key_or_value.shape[-1])

        if self.method == 'nautilus':
            enc = self.quantizer.encode(x)
            result = self.quantizer.decode(enc['corrected'])
            mse = enc['mse']
        elif self.method == 'turbo':
            rotated = self._turbo_rotate(x)
            polar = self._to_polar(rotated)
            dequant, _, _ = self._quantize(polar)
            corrected = self._qjl(polar, dequant)
            cartesian = self._from_polar(corrected)
            result = self._turbo_unrotate(cartesian)
            mse = (x - result).pow(2).mean().item()
        else:
            return key_or_value  # no quantization

        dt = time.perf_counter() - t0
        self.stats['total_mse'] += mse
        self.stats['n_calls'] += 1
        self.stats['total_time'] += dt

        return result.to(key_or_value.dtype).reshape(original_shape)

    def _turbo_rotate(self, x):
        out = x.clone()
        for k in range(min(x.shape[-1] // 2, len(self.turbo_angles))):
            theta = self.turbo_angles[k]
            c, s = math.cos(theta), math.sin(theta)
            a, b = out[..., 2*k].clone(), out[..., 2*k+1].clone()
            out[..., 2*k] = a * c - b * s
            out[..., 2*k+1] = a * s + b * c
        return out

    def _turbo_unrotate(self, x):
        out = x.clone()
        for k in reversed(range(min(x.shape[-1] // 2, len(self.turbo_angles)))):
            theta = -self.turbo_angles[k]
            c, s = math.cos(theta), math.sin(theta)
            a, b = out[..., 2*k].clone(), out[..., 2*k+1].clone()
            out[..., 2*k] = a * c - b * s
            out[..., 2*k+1] = a * s + b * c
        return out

    def _to_polar(self, x):
        dim = x.shape[-1]
        out = torch.zeros_like(x)
        for k in range(dim // 2):
            out[..., 2*k] = torch.sqrt(x[..., 2*k]**2 + x[..., 2*k+1]**2)
            out[..., 2*k+1] = torch.atan2(x[..., 2*k+1], x[..., 2*k])
        return out

    def _from_polar(self, p):
        dim = p.shape[-1]
        out = torch.zeros_like(p)
        for k in range(dim // 2):
            out[..., 2*k] = p[..., 2*k] * torch.cos(p[..., 2*k+1])
            out[..., 2*k+1] = p[..., 2*k] * torch.sin(p[..., 2*k+1])
        return out

    def _quantize(self, x):
        levels = 2 ** self.bits
        mins = x.min(dim=0).values
        maxs = x.max(dim=0).values
        ranges = (maxs - mins).clamp(min=1e-8)
        normalized = (x - mins) / ranges
        q = torch.round(normalized * (levels - 1))
        dequant = q / (levels - 1) * ranges + mins
        return dequant, ranges, mins

    def _qjl(self, original, quantized, alpha=0.5):
        error = original - quantized
        return quantized + torch.sign(error) * error.abs() * alpha

    def report(self):
        n = self.stats['n_calls'] or 1
        print(f"\n  [{self.method.upper()}] Stats:")
        print(f"    Total calls:    {self.stats['n_calls']}")
        print(f"    Avg MSE:        {self.stats['total_mse']/n:.8f}")
        print(f"    Total time:     {self.stats['total_time']*1000:.1f} ms")
        print(f"    Avg time/call:  {self.stats['total_time']/n*1000:.2f} ms")


# ============ NEEDLE-IN-A-HAYSTACK TEST ============

def needle_in_haystack(model_name, method='nautilus', bits=3,
                       haystack_length=4096, needle_depth=0.5):
    """
    Прячем факт в длинном тексте и проверяем, найдёт ли модель после квантования KV-кэша.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"Needle-in-a-Haystack Test")
    print(f"  Model: {model_name}")
    print(f"  Method: {method}, bits: {bits}")
    print(f"  Haystack: {haystack_length} tokens, needle at {needle_depth*100:.0f}%")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()

    # Build haystack
    filler = "The weather in London has been quite variable this year with temperatures ranging from cold winters to warm summers. Many tourists visit the city throughout the year to see landmarks like the Tower of London and Buckingham Palace. "
    needle = "The secret code for the vault is NAUTILUS-PHI-137."
    question = "\nWhat is the secret code for the vault?"

    # Repeat filler to target length
    filler_tokens = tokenizer(filler, return_tensors="pt").input_ids.shape[1]
    n_repeats = max(1, haystack_length // filler_tokens)
    needle_pos = int(n_repeats * needle_depth)

    parts = []
    for i in range(n_repeats):
        if i == needle_pos:
            parts.append(needle)
        parts.append(filler)
    haystack = " ".join(parts) + question

    inputs = tokenizer(haystack, return_tensors="pt", truncation=True,
                       max_length=haystack_length).to(model.device)
    actual_len = inputs.input_ids.shape[1]
    print(f"  Actual context: {actual_len} tokens")

    # Generate with quantized KV-cache
    dim = model.config.hidden_size // getattr(model.config, 'num_attention_heads',
                                               getattr(model.config, 'num_heads', 8))
    quantizer = KVCacheQuantizer(method=method, bits=bits, dim=min(dim, 128))

    with torch.no_grad():
        # First pass: get KV-cache
        outputs = model(**inputs, use_cache=True)

        # Quantize KV-cache
        if method != 'none':
            quantized_kv = []
            for layer_kv in outputs.past_key_values:
                k, v = layer_kv
                qk = quantizer.quantize_kv(k)
                qv = quantizer.quantize_kv(v)
                quantized_kv.append((qk, qv))
            past = tuple(quantized_kv)
        else:
            past = outputs.past_key_values

        # Generate answer
        gen_ids = model.generate(
            inputs.input_ids,
            past_key_values=past,
            max_new_tokens=30,
            do_sample=False,
            temperature=1.0,
        )

    answer = tokenizer.decode(gen_ids[0][actual_len:], skip_special_tokens=True)
    print(f"\n  Answer: {answer.strip()}")

    # Check if needle was found
    found = "NAUTILUS-PHI-137" in answer or "nautilus" in answer.lower()
    print(f"  Needle found: {'YES ✓' if found else 'NO ✗'}")

    quantizer.report()
    return found


# ============ SIMPLE ACCURACY TEST ============

def accuracy_test(model_name, method='nautilus', bits=3):
    """Quick accuracy test: can the model still answer basic questions?"""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"Accuracy Test: {method}, {bits}-bit")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    model.eval()

    questions = [
        ("What is 2+2?", ["4"]),
        ("What is the capital of France?", ["Paris", "paris"]),
        ("Complete: The quick brown fox jumps over the", ["lazy", "dog"]),
    ]

    dim = model.config.hidden_size // getattr(model.config, 'num_attention_heads',
                                               getattr(model.config, 'num_heads', 8))
    quantizer = KVCacheQuantizer(method=method, bits=bits, dim=min(dim, 128))
    correct = 0

    for q, expected_keywords in questions:
        inputs = tokenizer(q, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            if method != 'none':
                quantized_kv = []
                for layer_kv in outputs.past_key_values:
                    k, v = layer_kv
                    quantized_kv.append((quantizer.quantize_kv(k), quantizer.quantize_kv(v)))
                past = tuple(quantized_kv)
            else:
                past = outputs.past_key_values

            gen = model.generate(inputs.input_ids, past_key_values=past,
                                 max_new_tokens=20, do_sample=False)
        answer = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        hit = any(kw in answer for kw in expected_keywords)
        correct += hit
        print(f"  Q: {q}")
        print(f"  A: {answer.strip()} {'✓' if hit else '✗'}")

    print(f"\n  Score: {correct}/{len(questions)}")
    quantizer.report()
    return correct / len(questions)


# ============ MAIN ============

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NautilusQuant Phase 4 Benchmarks")
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-it')
    parser.add_argument('--method', type=str, default='both',
                        choices=['nautilus', 'turbo', 'none', 'both'])
    parser.add_argument('--bits', type=int, default=3)
    parser.add_argument('--haystack', type=int, default=4096)
    parser.add_argument('--test', type=str, default='all',
                        choices=['needle', 'accuracy', 'all'])
    args = parser.parse_args()

    methods = ['turbo', 'nautilus'] if args.method == 'both' else [args.method]

    for method in methods:
        if args.test in ('accuracy', 'all'):
            accuracy_test(args.model, method=method, bits=args.bits)

        if args.test in ('needle', 'all'):
            needle_in_haystack(args.model, method=method, bits=args.bits,
                               haystack_length=args.haystack)

    print("\n" + "="*60)
    print("All benchmarks complete.")
