"""
Plan B — Idea 2: Детерминированный Golden JL-Transform

Гипотеза: Ортогональная матрица на золотых углах САМА является
проектором Джонсона-Линденштраусса → этап QJL можно ВЫБРОСИТЬ.

Если NautilusRotate уже сохраняет расстояния между векторами
(что гарантировано ортогональностью), и при этом угловое
распределение оптимально для квантования → QJL-коррекция
становится лишним шагом, который добавляет 1 бит без пользы.

Целевое улучшение: убрать 1 бит/значение → 3 бита вместо 3+1.
Сжатие: 16/3 = 5.3x вместо 16/4 = 4.0x.
"""

import math
import numpy as np

PHI = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE = 2 * math.pi / (PHI ** 2)


class GoldenJLTransform:
    """
    Проверяет, является ли NautilusRotate встроенным JL-проектором.

    Лемма Джонсона-Линденштраусса: для N точек в R^d существует
    проекция в R^k (k = O(log N / ε²)) сохраняющая все попарные
    расстояния с точностью (1±ε).

    Для ортогональной матрицы расстояния сохраняются ТОЧНО (ε=0).
    Вопрос: сохраняются ли они после КВАНТОВАНИЯ?

    Если квантованные золотые углы дают меньший bias скалярных
    произведений, чем квантованные случайные углы → QJL не нужен.
    """

    def __init__(self, dim=128, phi=PHI):
        self.dim = dim
        self.phi = phi
        self._build_layers()

    def _build_layers(self):
        ga = GOLDEN_ANGLE
        self.layers = []

        # Layer 1: adjacent
        layer1 = []
        for k in range(self.dim // 2):
            layer1.append((2*k, 2*k+1, ga * (k+1)))
        self.layers.append(layer1)

        # Layer 2: shifted
        layer2 = []
        for k in range((self.dim - 1) // 2):
            layer2.append((2*k+1, 2*k+2, ga * (k+1) * self.phi))
        self.layers.append(layer2)

        # Layer 3: butterfly
        layer3 = []
        stride = max(2, self.dim // 4)
        used = set()
        for k in range(self.dim):
            i, j = k, (k + stride) % self.dim
            if i == j or i in used or j in used:
                continue
            used.add(i); used.add(j)
            layer3.append((i, j, ga * (k+1) * self.phi ** 2))
        self.layers.append(layer3)

    def rotate(self, x):
        out = x.copy()
        for layer in self.layers:
            for i, j, theta in layer:
                c, s = math.cos(theta), math.sin(theta)
                a, b = out[..., i].copy(), out[..., j].copy()
                out[..., i] = a * c - b * s
                out[..., j] = a * s + b * c
        return out

    def unrotate(self, x):
        out = x.copy()
        for layer in reversed(self.layers):
            for i, j, theta in reversed(layer):
                c, s = math.cos(-theta), math.sin(-theta)
                a, b = out[..., i].copy(), out[..., j].copy()
                out[..., i] = a * c - b * s
                out[..., j] = a * s + b * c
        return out

    def quantize_scalar(self, x, bits):
        levels = 2 ** bits
        mins = x.min(axis=0)
        maxs = x.max(axis=0)
        ranges = np.maximum(maxs - mins, 1e-8)
        norm = (x - mins) / ranges
        q = np.round(norm * (levels - 1)).clip(0, levels - 1)
        return q / (levels - 1) * ranges + mins

    def test_jl_property(self, x, bits=3, n_pairs=500):
        """
        Тест: сохраняются ли попарные расстояния после квантования?

        Измеряем bias и correlation скалярных произведений:
        - С QJL-коррекцией (4 бита = 3+1)
        - БЕЗ QJL-коррекции (3 бита) ← если bias ~0, QJL не нужен!
        """
        N = x.shape[0]

        # Rotate
        rotated = self.rotate(x)

        # Polar
        dim = x.shape[-1]
        polar = np.zeros_like(rotated)
        for k in range(dim // 2):
            i, j = 2*k, 2*k+1
            polar[:, i] = np.sqrt(rotated[:, i]**2 + rotated[:, j]**2)
            polar[:, j] = np.arctan2(rotated[:, j], rotated[:, i])

        # Quantize (NO QJL)
        quantized_no_qjl = self.quantize_scalar(polar, bits)

        # Quantize + QJL
        error = polar - quantized_no_qjl
        sign = np.sign(error)
        quantized_with_qjl = quantized_no_qjl + sign * np.abs(error) * 0.5

        # Sample random pairs for dot product test
        rng = np.random.RandomState(42)
        pairs_i = rng.randint(0, N, n_pairs)
        pairs_j = rng.randint(0, N, n_pairs)

        # True dot products (FP32)
        true_dots = np.array([x[pairs_i[k]] @ x[pairs_j[k]] for k in range(n_pairs)])

        # Dot products after quantization WITHOUT QJL
        # (decode: from_polar → inverse_rotate)
        no_qjl_cart = self._from_polar(quantized_no_qjl)
        no_qjl_decoded = self.unrotate(no_qjl_cart)
        no_qjl_dots = np.array([no_qjl_decoded[pairs_i[k]] @ no_qjl_decoded[pairs_j[k]]
                                for k in range(n_pairs)])

        # Dot products after quantization WITH QJL
        qjl_cart = self._from_polar(quantized_with_qjl)
        qjl_decoded = self.unrotate(qjl_cart)
        qjl_dots = np.array([qjl_decoded[pairs_i[k]] @ qjl_decoded[pairs_j[k]]
                             for k in range(n_pairs)])

        # Metrics
        no_qjl_bias = np.mean(no_qjl_dots - true_dots)
        no_qjl_corr = np.corrcoef(true_dots, no_qjl_dots)[0, 1]
        no_qjl_mse = np.mean((true_dots - no_qjl_dots) ** 2)

        qjl_bias = np.mean(qjl_dots - true_dots)
        qjl_corr = np.corrcoef(true_dots, qjl_dots)[0, 1]
        qjl_mse = np.mean((true_dots - qjl_dots) ** 2)

        return {
            'without_qjl': {
                'bits': bits,
                'bias': no_qjl_bias,
                'correlation': no_qjl_corr,
                'dot_mse': no_qjl_mse,
                'compression': 16 / bits,
            },
            'with_qjl': {
                'bits': bits + 1,
                'bias': qjl_bias,
                'correlation': qjl_corr,
                'dot_mse': qjl_mse,
                'compression': 16 / (bits + 1),
            },
            'qjl_needed': abs(no_qjl_bias) > abs(qjl_bias) * 2,
            'verdict': 'QJL UNNECESSARY — golden angles sufficient!'
                       if abs(no_qjl_bias) < 0.1 and no_qjl_corr > 0.995
                       else 'QJL still helpful'
        }

    def _from_polar(self, p):
        dim = p.shape[-1]
        out = np.zeros_like(p)
        for k in range(dim // 2):
            out[:, 2*k] = p[:, 2*k] * np.cos(p[:, 2*k+1])
            out[:, 2*k+1] = p[:, 2*k] * np.sin(p[:, 2*k+1])
        return out


def test():
    np.random.seed(42)
    N, D = 500, 128
    x = np.random.randn(N, D).astype(np.float32) * 0.5
    for d in [0, 15, 31, 63, 95, 127]:
        mask = np.random.rand(N) < 0.75
        x[mask, d] = np.random.randn(mask.sum()) * 30

    print("=" * 60)
    print("Plan B — Idea 2: Golden JL-Transform (Kill QJL?)")
    print("=" * 60)

    gjl = GoldenJLTransform(dim=D)

    for bits in [2, 3, 4]:
        result = gjl.test_jl_property(x, bits=bits)
        print(f"\n  --- {bits}-bit quantization ---")
        print(f"  WITHOUT QJL ({bits} bit, {result['without_qjl']['compression']:.1f}x compression):")
        print(f"    Dot product bias:        {result['without_qjl']['bias']:.6f}")
        print(f"    Dot product correlation: {result['without_qjl']['correlation']:.6f}")
        print(f"    Dot product MSE:         {result['without_qjl']['dot_mse']:.4f}")
        print(f"  WITH QJL ({bits}+1 bit, {result['with_qjl']['compression']:.1f}x compression):")
        print(f"    Dot product bias:        {result['with_qjl']['bias']:.6f}")
        print(f"    Dot product correlation: {result['with_qjl']['correlation']:.6f}")
        print(f"    Dot product MSE:         {result['with_qjl']['dot_mse']:.4f}")
        print(f"  >>> {result['verdict']}")


if __name__ == '__main__':
    test()
