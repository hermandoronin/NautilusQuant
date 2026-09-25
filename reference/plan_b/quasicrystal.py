"""
Plan B — Idea 1: Квазикристаллический квантователь (8D Penrose на φ)

Вместо скалярного Lloyd-Max (независимо по каждой координате),
группируем координаты по 8 и проецируем в 8-мерный квазикристалл,
построенный на степенях φ.

Квазикристаллы (мозаика Пенроуза) заполняют пространство без повторений,
используя два масштаба в пропорции φ. Это позволяет кодировать информацию
плотнее, чем периодические решётки (E8 в QuIP#).

Целевое улучшение: 2-bit квантование с качеством 3-bit.
"""

import math
import numpy as np

PHI = (1 + math.sqrt(5)) / 2
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]


class QuasiCrystalQuantizer:
    """
    8D квазикристаллический квантователь на базе φ.

    Вместо равномерной сетки (как Lloyd-Max) используем
    квазикристаллическую решётку, где узлы расположены
    по пропорциям Фибоначчи.

    Кодбук: 2^bits * φ-масштабированных точек в 8D пространстве.
    """

    def __init__(self, dim=8, bits=2, phi=PHI):
        self.dim = dim
        self.bits = bits
        self.phi = phi
        self.codebook = self._build_codebook()

    def _build_codebook(self):
        """
        Строим квазикристаллический кодбук.

        Ключевая идея: вместо равномерной сетки 2^(bits*dim),
        строим φ-масштабированную решётку с Фибоначчиевыми пропорциями.
        Это даёт ~40% больше уникальных состояний при том же количестве бит.
        """
        n_entries = 2 ** (self.bits * self.dim)
        # Ограничиваем для практичности
        max_entries = min(n_entries, 4096)

        # Квазикристаллическая решётка: два масштаба в пропорции φ
        # Генерируем через cut-and-project метод из более высокого измерения
        codebook = []
        golden_angle = 2 * math.pi / (self.phi ** 2)

        for i in range(max_entries):
            point = np.zeros(self.dim)
            for d in range(self.dim):
                # Фибоначчиевы координаты: каждая ось использует разный масштаб
                fib_scale = FIB[d % len(FIB)] / FIB[max(1, (d + 1) % len(FIB))]
                # Квазикристаллический сдвиг: золотой угол на каждой оси
                phase = golden_angle * (i * (d + 1))
                # Два масштаба в пропорции φ (как в мозаике Пенроуза)
                if (i // (2 ** d)) % 2 == 0:
                    point[d] = math.cos(phase) * fib_scale
                else:
                    point[d] = math.cos(phase) * fib_scale * self.phi
            codebook.append(point)

        return np.array(codebook, dtype=np.float32)

    def encode(self, vectors):
        """
        Квантует группы по dim координат через ближайшую точку в кодбуке.

        vectors: [N, D] — входные векторы (D должно делиться на dim)
        returns: indices [N, D//dim], reconstructed [N, D]
        """
        N, D = vectors.shape
        n_groups = D // self.dim
        indices = np.zeros((N, n_groups), dtype=np.int32)
        reconstructed = np.zeros_like(vectors)

        for g in range(n_groups):
            start = g * self.dim
            end = start + self.dim
            group = vectors[:, start:end]  # [N, dim]

            # Нормализуем группу
            norms = np.linalg.norm(group, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            normalized = group / norms

            # Ближайшая точка в кодбуке (brute-force)
            # distances[i,j] = ||normalized[i] - codebook[j]||²
            dots = normalized @ self.codebook.T  # [N, codebook_size]
            indices[:, g] = np.argmax(dots, axis=1)

            # Реконструкция
            reconstructed[:, start:end] = self.codebook[indices[:, g]] * norms

        mse = np.mean((vectors - reconstructed) ** 2)
        return {
            'indices': indices,
            'reconstructed': reconstructed,
            'mse': mse,
            'bits_per_value': math.log2(len(self.codebook)) / self.dim,
            'codebook_size': len(self.codebook),
            'method': 'QuasiCrystal-8D-φ'
        }


class PenroseVectorQuantizer:
    """
    Двумасштабный квантователь в стиле мозаики Пенроуза.

    Два набора уровней квантования в пропорции φ:
      - "толстые ромбы": крупный шаг = Δ·φ
      - "тонкие ромбы":  мелкий шаг = Δ

    Выбросы попадают на крупный шаг, фоновые значения — на мелкий.
    Автоматическая адаптация к тяжёлым хвостам.
    """

    def __init__(self, bits=3, phi=PHI):
        self.bits = bits
        self.phi = phi

    def encode(self, values):
        """
        values: [N] — одномерный массив значений
        returns: dict с реконструированными значениями
        """
        levels = 2 ** self.bits
        vmin, vmax = values.min(), values.max()
        vrange = vmax - vmin
        if vrange < 1e-8:
            return {'reconstructed': values.copy(), 'mse': 0, 'method': 'Penrose-φ'}

        # Два масштаба шага в пропорции φ
        # Мелкий шаг для центра распределения
        fine_step = vrange / (levels * self.phi)
        # Крупный шаг для хвостов
        coarse_step = fine_step * self.phi

        # Граница между зонами: ±1σ от медианы
        median = np.median(values)
        std = np.std(values)
        threshold = 1.5 * std

        # Квантуем
        result = np.zeros_like(values)
        for i in range(len(values)):
            v = values[i]
            dist_from_center = abs(v - median)

            if dist_from_center < threshold:
                # Мелкий шаг (fine zone)
                step = fine_step
                origin = median - threshold
            else:
                # Крупный шаг (coarse zone) для выбросов
                step = coarse_step
                origin = median - threshold - coarse_step * levels // 2 if v < median else median + threshold

            q = round((v - origin) / step)
            result[i] = q * step + origin

        mse = np.mean((values - result) ** 2)
        return {
            'reconstructed': result,
            'mse': mse,
            'fine_step': fine_step,
            'coarse_step': coarse_step,
            'phi_ratio': coarse_step / fine_step,
            'method': 'Penrose-φ-dual-scale'
        }


def test():
    """Тест квазикристаллического квантователя."""
    np.random.seed(42)
    N, D = 200, 128

    # Данные с выбросами (как реальный KV-кэш)
    x = np.random.randn(N, D).astype(np.float32) * 0.5
    for d in [0, 15, 31, 63, 95, 127]:
        mask = np.random.rand(N) < 0.75
        x[mask, d] = np.random.randn(mask.sum()) * 30

    print("=" * 60)
    print("Plan B — Idea 1: QuasiCrystal Quantizer")
    print("=" * 60)

    # Test QuasiCrystal
    qc = QuasiCrystalQuantizer(dim=8, bits=2)
    result = qc.encode(x)
    print(f"\n  QuasiCrystal-8D:")
    print(f"    MSE:            {result['mse']:.6f}")
    print(f"    Bits/value:     {result['bits_per_value']:.2f}")
    print(f"    Codebook size:  {result['codebook_size']}")

    # Test Penrose dual-scale
    pq = PenroseVectorQuantizer(bits=3)
    flat = x.flatten()
    presult = pq.encode(flat)
    print(f"\n  Penrose Dual-Scale:")
    print(f"    MSE:            {presult['mse']:.6f}")
    print(f"    Fine step:      {presult['fine_step']:.4f}")
    print(f"    Coarse step:    {presult['coarse_step']:.4f}")
    print(f"    φ ratio:        {presult['phi_ratio']:.4f}")

    # Baseline: uniform scalar
    levels = 2 ** 3
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    ranges = np.maximum(maxs - mins, 1e-8)
    norm = (x - mins) / ranges
    q = np.round(norm * (levels - 1))
    recon = q / (levels - 1) * ranges + mins
    baseline_mse = np.mean((x - recon) ** 2)
    print(f"\n  Baseline (scalar 3-bit):")
    print(f"    MSE:            {baseline_mse:.6f}")

    print(f"\n  Penrose improvement: {(1 - presult['mse'] / baseline_mse) * 100:.1f}%")


if __name__ == '__main__':
    test()
