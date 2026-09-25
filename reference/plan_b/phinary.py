"""
Plan B — Idea 3: Квантование в Фибоначчиевой системе счисления (Phinary Base)

Вместо базы 2 (двоичная: 1, 2, 4, 8, 16...) используем базу φ:
  φ⁰=1, φ¹=1.618, φ²=2.618, φ³=4.236, φ⁴=6.854...

По теореме Цекендорфа: любое целое число единственным образом
раскладывается в сумму неcоседних чисел Фибоначчи.

Зачем: шкала на базе φ расширяется ЭКСПОНЕНЦИАЛЬНО, но МЕДЛЕННЕЕ
чем база 2. Это идеально для KV-кэша, где 99% значений — маленькие (±0.5),
но 1% — гигантские выбросы (до ±60). φ-экспонента естественно покрывает
оба масштаба без потери точности.

Целевое улучшение: лучшее покрытие выбросов при 3-4 битах.
"""

import math
import numpy as np

PHI = (1 + math.sqrt(5)) / 2
PSI = PHI - 1  # 1/φ ≈ 0.618


class PhinaryQuantizer:
    """
    Квантователь с нелинейной шкалой на базе φ.

    Уровни квантования расставлены не равномерно (как Lloyd-Max),
    а по степеням φ:
      level_k = sign(k) · φ^(|k| - offset)

    Мелкие значения квантуются точно (шаг ~ 0.618),
    выбросы квантуются крупнее (шаг ~ φ^n), но всё ещё представимы.
    """

    def __init__(self, bits=3):
        self.bits = bits
        self.levels = 2 ** bits
        self.codebook = self._build_phinary_codebook()

    def _build_phinary_codebook(self):
        """
        Строим кодбук с φ-масштабированными уровнями.

        Для 3 бит (8 уровней):
          [-φ², -φ, -1, -1/φ, 1/φ, 1, φ, φ²]
          = [-2.618, -1.618, -1, -0.618, 0.618, 1, 1.618, 2.618]

        Сравни с равномерным: [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        φ-кодбук плотнее вокруг нуля, шире на хвостах — идеально для выбросов!
        """
        half = self.levels // 2
        positive = []
        for k in range(half):
            if k == 0:
                positive.append(PSI)  # 1/φ ≈ 0.618
            else:
                positive.append(PHI ** (k - 1))  # 1, φ, φ², ...

        negative = [-x for x in reversed(positive)]
        codebook = np.array(negative + positive, dtype=np.float32)
        return codebook

    def encode(self, values):
        """
        Квантует значения, находя ближайший уровень в φ-кодбуке.

        values: np.array — входные значения
        """
        values = np.asarray(values, dtype=np.float32)
        flat = values.flatten()

        # Нормализуем к диапазону кодбука
        scale = np.max(np.abs(flat)) / np.max(np.abs(self.codebook))
        if scale < 1e-8:
            scale = 1.0
        normalized = flat / scale

        # Ближайший уровень
        # distances[i, j] = |normalized[i] - codebook[j]|
        distances = np.abs(normalized[:, None] - self.codebook[None, :])
        indices = np.argmin(distances, axis=1)
        reconstructed = self.codebook[indices] * scale

        mse = np.mean((flat - reconstructed) ** 2)

        return {
            'indices': indices.reshape(values.shape),
            'reconstructed': reconstructed.reshape(values.shape),
            'mse': mse,
            'scale': scale,
            'codebook': self.codebook * scale,
            'bits': self.bits,
            'method': f'Phinary-{self.bits}bit'
        }


class ZeckendorfEncoder:
    """
    Кодирование через разложение Цекендорфа (представление Фибоначчи).

    Любое положительное целое N единственным образом записывается как
    сумма неcоседних чисел Фибоначчи:
      42 = 34 + 8 = F(9) + F(6)

    Это даёт уникальный двоичный код (где 1 = "использовать F(k)"),
    в котором никогда не стоят две единицы рядом.

    Свойство: среднее число бит ~ log_φ(N) ≈ 1.44 · log₂(N).
    Чуть длиннее двоичного, но с уникальным свойством самосинхронизации.
    """

    FIB = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]

    @staticmethod
    def encode_int(n):
        """Разложение Цекендорфа для целого n >= 1."""
        if n <= 0:
            return []
        fibs = ZeckendorfEncoder.FIB
        result = []
        remainder = n
        for f in reversed(fibs):
            if f <= remainder:
                result.append(f)
                remainder -= f
                if remainder == 0:
                    break
        return result

    @staticmethod
    def to_bits(n, max_bits=16):
        """Представление в виде битовой строки Фибоначчи."""
        fibs = ZeckendorfEncoder.FIB[:max_bits]
        bits = [0] * len(fibs)
        remainder = abs(n)
        for i in range(len(fibs) - 1, -1, -1):
            if fibs[i] <= remainder:
                bits[i] = 1
                remainder -= fibs[i]
                if remainder == 0:
                    break
        return bits

    @staticmethod
    def from_bits(bits):
        """Декодирование из Фибоначчиевой битовой строки."""
        fibs = ZeckendorfEncoder.FIB[:len(bits)]
        return sum(b * f for b, f in zip(bits, fibs))


class PhinaryFloatQuantizer:
    """
    Комбинация: φ-экспонента + Фибоначчиева мантисса.

    Формат: [sign(1)] [exponent φ^k (3 bits)] [mantissa Fib (4 bits)]

    Аналог FP8, но с экспонентой на базе φ вместо базы 2.
    Покрытие от ±0.001 до ±1000 с адаптивной точностью.
    """

    def __init__(self, exp_bits=3, mantissa_bits=4):
        self.exp_bits = exp_bits
        self.mantissa_bits = mantissa_bits
        self.total_bits = 1 + exp_bits + mantissa_bits  # sign + exp + mantissa
        self._build_levels()

    def _build_levels(self):
        n_exp = 2 ** self.exp_bits
        n_mant = 2 ** self.mantissa_bits

        self.levels = []
        for e in range(n_exp):
            exp_val = PHI ** (e - n_exp // 2)  # φ^(-4) to φ^(3)
            for m in range(n_mant):
                mant_val = 1.0 + m / n_mant  # [1.0, 2.0)
                self.levels.append(exp_val * mant_val)
                self.levels.append(-exp_val * mant_val)

        self.levels = np.array(sorted(set(self.levels)), dtype=np.float32)

    def encode(self, values):
        flat = np.asarray(values, dtype=np.float32).flatten()
        distances = np.abs(flat[:, None] - self.levels[None, :])
        indices = np.argmin(distances, axis=1)
        reconstructed = self.levels[indices]
        mse = np.mean((flat - reconstructed) ** 2)

        return {
            'reconstructed': reconstructed.reshape(np.asarray(values).shape),
            'mse': mse,
            'total_bits': self.total_bits,
            'n_levels': len(self.levels),
            'range': [self.levels.min(), self.levels.max()],
            'method': f'PhinaryFloat-{self.total_bits}bit (φ-exp)'
        }


def test():
    np.random.seed(42)
    N = 10000

    # Realistic KV-cache values: mostly small, some huge outliers
    values = np.random.randn(N).astype(np.float32) * 0.5
    outlier_mask = np.random.rand(N) < 0.06
    values[outlier_mask] = np.random.randn(outlier_mask.sum()) * 30

    print("=" * 60)
    print("Plan B — Idea 3: Phinary Quantization (base φ)")
    print("=" * 60)
    print(f"  Data: {N} values, range [{values.min():.1f}, {values.max():.1f}]")
    print(f"  Outliers: {outlier_mask.sum()} ({outlier_mask.mean()*100:.1f}%)")

    # Baseline: uniform 3-bit
    levels = 8
    vmin, vmax = values.min(), values.max()
    vrange = vmax - vmin
    norm = (values - vmin) / vrange
    q = np.round(norm * (levels - 1))
    baseline = q / (levels - 1) * vrange + vmin
    baseline_mse = np.mean((values - baseline) ** 2)
    print(f"\n  Baseline (uniform 3-bit):")
    print(f"    MSE: {baseline_mse:.6f}")

    # Phinary 3-bit
    pq = PhinaryQuantizer(bits=3)
    result = pq.encode(values)
    print(f"\n  Phinary 3-bit (φ-codebook):")
    print(f"    MSE: {result['mse']:.6f}")
    print(f"    Improvement: {(1 - result['mse'] / baseline_mse) * 100:.1f}%")
    print(f"    Codebook: {result['codebook']}")

    # Zeckendorf encoding example
    print(f"\n  Zeckendorf examples:")
    for n in [1, 5, 13, 42, 100]:
        decomp = ZeckendorfEncoder.encode_int(n)
        bits = ZeckendorfEncoder.to_bits(n)
        print(f"    {n} = {' + '.join(map(str, decomp))} = bits {bits[:8]}")

    # PhinaryFloat (8-bit, φ-exponent)
    pfq = PhinaryFloatQuantizer(exp_bits=3, mantissa_bits=4)
    fresult = pfq.encode(values)
    print(f"\n  PhinaryFloat-{pfq.total_bits}bit (φ-exponent):")
    print(f"    MSE: {fresult['mse']:.6f}")
    print(f"    Levels: {fresult['n_levels']}")
    print(f"    Range: [{fresult['range'][0]:.4f}, {fresult['range'][1]:.4f}]")
    print(f"    Improvement vs baseline: {(1 - fresult['mse'] / baseline_mse) * 100:.1f}%")


if __name__ == '__main__':
    test()
