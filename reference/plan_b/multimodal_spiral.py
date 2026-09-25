"""
Plan B — Idea 6: Адаптивная спираль для мультимодальных моделей (VLM)

Визуальные и текстовые токены имеют РАЗНУЮ чувствительность к квантизации.
Нельзя крутить всё одним углом.

Решение: разные математические константы для разных модальностей:
  - Текст:   φ = (1+√5)/2 ≈ 1.618  (золотое сечение, максимальная точность)
  - Визуал:  δs = 1+√2   ≈ 2.414  (серебряное сечение, агрессивное сжатие)
  - Аудио:   δb = 1+√3   ≈ 2.732  (бронзовое сечение, баланс)
  - Код:     φ² = φ+1     ≈ 2.618  (квадрат золотого, для структурных данных)

Каждое «металлическое сечение» δn = (n + √(n²+4)) / 2 имеет
свой «металлический угол» = 2π/δn² с уникальными свойствами покрытия.

CVPR 2025: визуальные токены на 40% менее чувствительны к квантизации.
→ Можно сжимать их до 2 бит вместо 3, экономя ~30% памяти на VLM.
"""

import math
import numpy as np

# Metallic ratios family
GOLDEN = (1 + math.sqrt(5)) / 2        # φ ≈ 1.618 (n=1)
SILVER = 1 + math.sqrt(2)              # δs ≈ 2.414 (n=2)
BRONZE = (3 + math.sqrt(13)) / 2       # δb ≈ 3.303 (n=3)
GOLDEN_SQ = GOLDEN + 1                 # φ² ≈ 2.618

METALLIC_RATIOS = {
    'golden':    {'value': GOLDEN,    'symbol': 'φ',  'name': 'Golden Ratio',  'n': 1},
    'silver':    {'value': SILVER,    'symbol': 'δs', 'name': 'Silver Ratio',  'n': 2},
    'bronze':    {'value': BRONZE,    'symbol': 'δb', 'name': 'Bronze Ratio',  'n': 3},
    'golden_sq': {'value': GOLDEN_SQ, 'symbol': 'φ²', 'name': 'Golden Square', 'n': 0},
}


class ModalityConfig:
    """Конфигурация квантизации для каждой модальности."""

    # Оптимальные настройки (на основе CVPR 2025 sensitivity analysis)
    PRESETS = {
        'text': {
            'ratio': 'golden',       # φ — максимальная точность для attention
            'bits_radius': 3,
            'bits_angle': 2,
            'qjl_alpha': 0.5,
            'description': 'Text tokens: high precision, golden angle 137.5°',
            'sensitivity': 1.0,      # baseline sensitivity
        },
        'image': {
            'ratio': 'silver',       # δs — агрессивнее, визуал устойчивее
            'bits_radius': 2,
            'bits_angle': 1,
            'qjl_alpha': 0.3,
            'description': 'Visual tokens: aggressive compression, silver angle 105.4°',
            'sensitivity': 0.6,      # 40% less sensitive (CVPR 2025)
        },
        'audio': {
            'ratio': 'bronze',       # δb — баланс для спектральных данных
            'bits_radius': 3,
            'bits_angle': 1,
            'qjl_alpha': 0.4,
            'description': 'Audio tokens: balanced, bronze angle for spectral features',
            'sensitivity': 0.75,
        },
        'code': {
            'ratio': 'golden_sq',    # φ² — для структурных паттернов
            'bits_radius': 3,
            'bits_angle': 2,
            'qjl_alpha': 0.5,
            'description': 'Code tokens: structured patterns, golden square angle',
            'sensitivity': 0.9,
        },
        'video': {
            'ratio': 'silver',       # Видео ≈ изображения, но с темпоральностью
            'bits_radius': 2,
            'bits_angle': 1,
            'qjl_alpha': 0.35,
            'description': 'Video tokens: like image but with temporal redundancy',
            'sensitivity': 0.55,
        },
    }

    @classmethod
    def get(cls, modality: str) -> dict:
        config = cls.PRESETS.get(modality, cls.PRESETS['text'])
        ratio_info = METALLIC_RATIOS[config['ratio']]
        phi = ratio_info['value']
        metallic_angle = 2 * math.pi / (phi ** 2)
        return {
            **config,
            'phi': phi,
            'metallic_angle_rad': metallic_angle,
            'metallic_angle_deg': math.degrees(metallic_angle),
            'ratio_name': ratio_info['name'],
            'ratio_symbol': ratio_info['symbol'],
            'effective_bits': (config['bits_radius'] + config['bits_angle']) / 2 + 1,
        }


class AdaptiveSpiralRotator:
    """
    Многомодальный ротатор: разные углы для разных типов токенов.

    В VLM (Vision-Language Model) один batch содержит:
    - Текстовые токены (prompt, response)
    - Визуальные токены (из Vision Encoder)
    - Иногда: аудио, код, видео

    Каждый тип получает свою матрицу вращения с оптимальным
    металлическим сечением.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.rotators = {}
        for modality in ModalityConfig.PRESETS:
            config = ModalityConfig.get(modality)
            self.rotators[modality] = self._build_layers(config['phi'])

    def _build_layers(self, phi):
        ga = 2 * math.pi / (phi ** 2)
        layers = []

        # Layer 1
        layer1 = []
        for k in range(self.dim // 2):
            theta = ga * (k + 1)
            layer1.append((2*k, 2*k+1, math.cos(theta), math.sin(theta)))
        layers.append(layer1)

        # Layer 2
        layer2 = []
        for k in range((self.dim - 1) // 2):
            theta = ga * (k + 1) * phi
            layer2.append((2*k+1, 2*k+2, math.cos(theta), math.sin(theta)))
        layers.append(layer2)

        # Layer 3
        layer3 = []
        stride = max(2, self.dim // 4)
        used = set()
        for k in range(self.dim):
            i, j = k, (k + stride) % self.dim
            if i == j or i in used or j in used:
                continue
            used.add(i); used.add(j)
            theta = ga * (k + 1) * phi ** 2
            layer3.append((i, j, math.cos(theta), math.sin(theta)))
        layers.append(layer3)

        return layers

    def rotate(self, vectors, modality='text'):
        layers = self.rotators.get(modality, self.rotators['text'])
        out = vectors.copy()
        for layer in layers:
            for i, j, c, s in layer:
                a, b = out[..., i].copy(), out[..., j].copy()
                out[..., i] = a * c - b * s
                out[..., j] = a * s + b * c
        return out

    def quantize_adaptive(self, vectors, modality='text'):
        """Full adaptive pipeline: rotate → polar → quantize per modality config."""
        config = ModalityConfig.get(modality)
        rotated = self.rotate(vectors, modality)

        # Polar
        dim = vectors.shape[-1]
        polar = np.zeros_like(rotated)
        for k in range(dim // 2):
            polar[..., 2*k] = np.sqrt(rotated[..., 2*k]**2 + rotated[..., 2*k+1]**2)
            polar[..., 2*k+1] = np.arctan2(rotated[..., 2*k+1], rotated[..., 2*k])

        # Quantize radius and angle separately
        radii = polar[..., 0::2]
        angles = polar[..., 1::2]

        q_radii = self._scalar_quant(radii, config['bits_radius'])
        q_angles = self._scalar_quant(angles, config['bits_angle'])

        reconstructed = np.zeros_like(polar)
        reconstructed[..., 0::2] = q_radii
        reconstructed[..., 1::2] = q_angles

        mse = np.mean((polar - reconstructed) ** 2)

        return {
            'reconstructed': reconstructed,
            'mse': mse,
            'modality': modality,
            'config': config,
            'bits_per_value': config['effective_bits'],
            'compression': 16 / config['effective_bits'],
        }

    def _scalar_quant(self, x, bits):
        levels = 2 ** bits
        mins = x.min(axis=0) if x.ndim > 1 else x.min()
        maxs = x.max(axis=0) if x.ndim > 1 else x.max()
        ranges = np.maximum(maxs - mins, 1e-8)
        norm = (x - mins) / ranges
        q = np.round(norm * (levels - 1)).clip(0, levels - 1)
        return q / (levels - 1) * ranges + mins


def estimate_vlm_savings(token_distribution: dict, dim: int = 128):
    """
    Estimate memory savings for a VLM with mixed modalities.

    token_distribution: {'text': 8000, 'image': 4000, 'audio': 500, ...}
    """
    rotator = AdaptiveSpiralRotator(dim)

    uniform_bits = 4  # 3+1 (standard NautilusQuant)
    results = {}
    total_uniform = 0
    total_adaptive = 0

    for modality, n_tokens in token_distribution.items():
        config = ModalityConfig.get(modality)
        bits = config['effective_bits']
        uniform_total = n_tokens * dim * uniform_bits
        adaptive_total = n_tokens * dim * bits

        total_uniform += uniform_total
        total_adaptive += adaptive_total

        results[modality] = {
            'tokens': n_tokens,
            'ratio': config['ratio_symbol'],
            'angle': f"{config['metallic_angle_deg']:.1f}°",
            'bits_per_value': bits,
            'memory_uniform_kb': uniform_total / 8 / 1024,
            'memory_adaptive_kb': adaptive_total / 8 / 1024,
            'savings_pct': (1 - bits / uniform_bits) * 100,
        }

    return {
        'per_modality': results,
        'total_uniform_kb': total_uniform / 8 / 1024,
        'total_adaptive_kb': total_adaptive / 8 / 1024,
        'total_savings_pct': (1 - total_adaptive / total_uniform) * 100,
    }


def test():
    np.random.seed(42)
    N, D = 500, 128

    print("=" * 60)
    print("Plan B — Idea 6: Multimodal Adaptive Spiral")
    print("=" * 60)

    # Show metallic ratios
    print("\n  Metallic Ratios Family:")
    for name, info in METALLIC_RATIOS.items():
        angle = 2 * math.pi / (info['value'] ** 2)
        print(f"    {info['symbol']:3s} = {info['value']:.6f}  "
              f"angle = {math.degrees(angle):.2f}°  ({info['name']})")

    # Generate test data
    x = np.random.randn(N, D).astype(np.float32) * 0.5
    for d in [0, 15, 31, 63, 95, 127]:
        mask = np.random.rand(N) < 0.75
        x[mask, d] = np.random.randn(mask.sum()) * 30

    # Test each modality
    rotator = AdaptiveSpiralRotator(D)
    print(f"\n  Per-Modality Quantization (dim={D}, N={N}):")
    for modality in ['text', 'image', 'audio', 'code', 'video']:
        result = rotator.quantize_adaptive(x, modality)
        config = result['config']
        print(f"\n    [{modality.upper()}] {config['ratio_symbol']} = {config['phi']:.4f}, "
              f"angle = {config['metallic_angle_deg']:.1f}°")
        print(f"      Bits: r={config['bits_radius']} + θ={config['bits_angle']} "
              f"→ {result['bits_per_value']:.1f} bit/val")
        print(f"      MSE:  {result['mse']:.6f}")
        print(f"      Compression: {result['compression']:.1f}x")
        print(f"      Sensitivity: {config['sensitivity']}")

    # VLM memory estimation
    print("\n  --- VLM Memory Estimation ---")
    print("  Scenario: Gemma 3 4B VLM, 128K context")
    vlm_tokens = {
        'text': 80000,    # 80K text tokens
        'image': 40000,   # 40K visual tokens (high-res images)
        'code': 8000,     # 8K code tokens
    }
    savings = estimate_vlm_savings(vlm_tokens, D)

    for mod, info in savings['per_modality'].items():
        print(f"    {mod:6s}: {info['tokens']:6d} tokens × {info['ratio']:3s} "
              f"({info['angle']:>7s}) → {info['bits_per_value']:.1f}b, "
              f"savings: {info['savings_pct']:.0f}%")

    print(f"\n    Uniform (4 bit):  {savings['total_uniform_kb']:.0f} KB")
    print(f"    Adaptive:         {savings['total_adaptive_kb']:.0f} KB")
    print(f"    Total savings:    {savings['total_savings_pct']:.1f}%")


if __name__ == '__main__':
    test()
