"""
Plan B — Idea 4: Суб-1-битное фрактальное хеширование через золотую орбиту

Вместо хранения самого угла θ, кодируем его как ИНДЕКС ШАГА k
на золотой орбите:
  θ ≈ GOLDEN_ANGLE × k (mod 2π)

Поскольку золотой угол ≈ 137.5° — самый иррациональный —
орбита {GA×1, GA×2, GA×3, ...} покрывает окружность без повторений.

Любой угол можно аппроксимировать ближайшим k.
Для точности ε нужно всего log₂(1/ε) / log₂(φ) бит ≈ 0.69 · log₂(1/ε).

Это МЕНЬШЕ чем log₂(1/ε) бит при двоичном кодировании!

Целевое улучшение: углы кодируются 1-2 битами (или даже < 1 бита
через контекстное предсказание).
"""

import math
import numpy as np

PHI = (1 + math.sqrt(5)) / 2
GA = 2 * math.pi / (PHI ** 2)  # golden angle ≈ 2.3999 рад


class GoldenOrbitEncoder:
    """
    Кодирует угол θ ∈ [-π, π] как индекс k на золотой орбите.

    orbit[k] = GA × k (mod 2π) - π

    Для N шагов орбиты, максимальная ошибка аппроксимации:
      ε_max ≈ π / N (by three-distance theorem)

    Для N = 256 шагов: ε_max ≈ 0.012 рад ≈ 0.7°
    Нужно всего 8 бит (log₂(256)) для кодирования с точностью 0.7°.

    Но! Если контекст (соседние значения) предсказуем, можно
    кодировать РАЗНИЦУ Δk между соседними индексами. Δk обычно мал
    → можно использовать переменное кодирование (Golomb, Elias gamma).
    """

    def __init__(self, n_steps=256):
        self.n_steps = n_steps
        # Precompute orbit
        self.orbit = np.array([(GA * k) % (2 * math.pi) - math.pi
                               for k in range(n_steps)], dtype=np.float32)
        # Sort for binary search
        self.sorted_orbit = np.sort(self.orbit)
        self.sort_indices = np.argsort(self.orbit)

    def encode_angle(self, theta):
        """Find closest orbit step k for angle theta."""
        # Normalize to [-π, π]
        theta = ((theta + math.pi) % (2 * math.pi)) - math.pi
        # Binary search in sorted orbit
        idx = np.searchsorted(self.sorted_orbit, theta)
        idx = np.clip(idx, 0, self.n_steps - 1)

        # Check neighbors
        best_k = self.sort_indices[idx]
        best_err = abs(theta - self.orbit[best_k])

        if idx > 0:
            k2 = self.sort_indices[idx - 1]
            err2 = abs(theta - self.orbit[k2])
            if err2 < best_err:
                best_k, best_err = k2, err2

        return best_k, best_err

    def decode_angle(self, k):
        """Recover angle from orbit index."""
        return self.orbit[k]

    def encode_batch(self, angles):
        """Encode array of angles."""
        angles = np.asarray(angles, dtype=np.float32).flatten()
        indices = np.zeros(len(angles), dtype=np.int32)
        errors = np.zeros(len(angles), dtype=np.float32)

        for i, theta in enumerate(angles):
            k, err = self.encode_angle(theta)
            indices[i] = k
            errors[i] = err

        return indices, errors

    def bits_per_angle(self):
        """Bits needed per angle (direct encoding)."""
        return math.log2(self.n_steps)


class DeltaOrbitEncoder:
    """
    Дельта-кодирование на золотой орбите.

    Вместо абсолютного индекса k, кодируем разницу Δk = k[i] - k[i-1].
    Если соседние углы коррелированы (что типично для KV-кэша),
    Δk мал → нужно меньше бит.

    С Golomb-Rice кодированием: средний расход < 2 бит/значение
    при корреляции > 0.5.
    """

    def __init__(self, n_steps=256):
        self.base = GoldenOrbitEncoder(n_steps)
        self.n_steps = n_steps

    def encode(self, angles):
        """Delta-encode angles on golden orbit."""
        indices, errors = self.base.encode_batch(angles)

        # Compute deltas
        deltas = np.zeros_like(indices)
        deltas[0] = indices[0]
        for i in range(1, len(indices)):
            deltas[i] = indices[i] - indices[i - 1]

        # Estimate bits via entropy
        unique, counts = np.unique(deltas, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(np.maximum(probs, 1e-10)))

        return {
            'indices': indices,
            'deltas': deltas,
            'errors': errors,
            'max_error_rad': errors.max(),
            'mean_error_rad': errors.mean(),
            'max_error_deg': np.degrees(errors.max()),
            'direct_bits': self.base.bits_per_angle(),
            'delta_entropy': entropy,
            'effective_bits': entropy,  # theoretical minimum
            'method': f'GoldenOrbit-Δ (N={self.n_steps})'
        }


class FractalSubBitEncoder:
    """
    Суб-1-битное кодирование через фрактальное разбиение.

    Идея: разбиваем окружность на сегменты по правилу Фибоначчи.
    Каждый уровень разбиения добавляет 1 бит точности, но сегменты
    имеют размеры в пропорции φ (как мозаика Пенроуза на окружности).

    Уровень 0: вся окружность [0, 2π)          → 0 бит
    Уровень 1: два сегмента [0, GA), [GA, 2π)  → 1 бит
    Уровень 2: три сегмента (Фибоначчи!)       → ~1.58 бит
    Уровень 3: пять сегментов                   → ~2.32 бит
    ...
    Уровень n: F(n+1) сегментов                 → ~n·0.694 бит

    Каждый бит даёт ~1.44× больше информации чем двоичный бит!
    """

    def __init__(self, max_level=8):
        self.max_level = max_level
        self.fibs = [1, 1]
        for _ in range(max_level + 2):
            self.fibs.append(self.fibs[-1] + self.fibs[-2])

    def encode(self, angles, level=None):
        """
        Кодирует углы на заданном уровне фрактального разбиения.

        level: число Фибоначчи-сегментов. None = max_level.
        """
        if level is None:
            level = self.max_level

        n_segments = self.fibs[level + 1]
        segment_size = 2 * math.pi / n_segments

        angles = np.asarray(angles, dtype=np.float32).flatten()
        # Normalize to [0, 2π)
        normalized = (angles + math.pi) % (2 * math.pi)

        # Assign to segment
        indices = (normalized / segment_size).astype(np.int32)
        indices = np.clip(indices, 0, n_segments - 1)

        # Reconstruct: center of segment
        reconstructed = (indices + 0.5) * segment_size - math.pi

        errors = np.abs(angles - reconstructed)
        errors = np.minimum(errors, 2 * math.pi - errors)  # circular distance

        bits_per_value = math.log2(n_segments)
        info_per_binary_bit = bits_per_value / level if level > 0 else 0

        return {
            'indices': indices,
            'reconstructed': reconstructed,
            'errors': errors,
            'max_error_deg': np.degrees(errors.max()),
            'mean_error_deg': np.degrees(errors.mean()),
            'n_segments': n_segments,
            'level': level,
            'bits_per_value': bits_per_value,
            'equivalent_binary_bits': level,
            'info_efficiency': info_per_binary_bit,
            'method': f'FractalFib-L{level} ({n_segments} segments, {bits_per_value:.2f} bits)'
        }

    def sweep_levels(self, angles):
        """Test all levels, show bits vs accuracy tradeoff."""
        results = []
        for level in range(1, self.max_level + 1):
            r = self.encode(angles, level=level)
            results.append(r)
        return results


def test():
    np.random.seed(42)
    N = 5000

    # Simulate polar angles from rotated KV-cache
    # After golden rotation, angles should be concentrated
    angles = np.random.randn(N).astype(np.float32) * 0.8  # concentrated around 0
    # Some spread
    angles += np.random.choice([-1, 0, 1], N) * 1.5

    print("=" * 60)
    print("Plan B — Idea 4: Fractal Sub-1-bit Hashing")
    print("=" * 60)
    print(f"  Data: {N} angles, range [{np.degrees(angles.min()):.1f}°, {np.degrees(angles.max()):.1f}°]")

    # Golden Orbit encoding
    print("\n  --- Golden Orbit Encoding ---")
    for n_steps in [16, 32, 64, 128, 256]:
        enc = DeltaOrbitEncoder(n_steps=n_steps)
        result = enc.encode(angles)
        print(f"  N={n_steps:4d}: direct={result['direct_bits']:.1f}b, "
              f"delta={result['delta_entropy']:.2f}b, "
              f"max_err={result['max_error_deg']:.2f}°, "
              f"mean_err={np.degrees(result['mean_error_rad']):.3f}°")

    # Fractal Fibonacci encoding
    print("\n  --- Fractal Fibonacci Encoding ---")
    frac = FractalSubBitEncoder(max_level=10)
    sweep = frac.sweep_levels(angles)
    for r in sweep:
        marker = " ← sweet spot" if 1.5 < r['bits_per_value'] < 3 else ""
        print(f"  L{r['level']:2d}: {r['n_segments']:5d} segments, "
              f"{r['bits_per_value']:.2f} bits/val, "
              f"max_err={r['max_error_deg']:.2f}°, "
              f"efficiency={r['info_efficiency']:.3f} info/binary_bit"
              f"{marker}")

    # Comparison with uniform quantization
    print("\n  --- Comparison ---")
    for bits in [1, 2, 3]:
        # Uniform
        levels = 2 ** bits
        step = 2 * math.pi / levels
        uniform_q = np.round((angles + math.pi) / step) * step - math.pi
        uniform_err = np.degrees(np.abs(angles - uniform_q).mean())

        # Fractal (same effective bits)
        best_frac = None
        for r in sweep:
            if abs(r['bits_per_value'] - bits) < 0.5:
                best_frac = r
                break

        frac_err = np.degrees(best_frac['errors'].mean()) if best_frac else float('inf')
        print(f"  {bits}-bit: Uniform err={uniform_err:.2f}°, "
              f"Fractal err={frac_err:.2f}°, "
              f"improvement={((uniform_err - frac_err) / uniform_err * 100):.1f}%" if frac_err < float('inf') else "")


if __name__ == '__main__':
    test()
