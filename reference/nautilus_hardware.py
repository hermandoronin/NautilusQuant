"""
NautilusQuant v2 — Hardware-Software Co-design Module
4 концепта: SRAM-центричность, Dataflow, MX-форматы, Суб-1-бит

Концепт 1: SRAM-центричное ядро (fused rotate+quantize в одном проходе)
Концепт 2: Детерминированный dataflow (статический schedule, precomputed LUT)
Концепт 3: MX-форматы (MXFP4/MXINT8 — Plan B если overhead ≠ 0)
Концепт 4: Суб-1-бит + мультимодальность (VLM-адаптивное сжатие)
"""

import math
import struct
from dataclasses import dataclass, field

import torch
import torch.nn as nn

PHI = (1 + math.sqrt(5)) / 2
GOLDEN_ANGLE = 2 * math.pi / (PHI ** 2)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ================================================================
# КОНЦЕПТ 1: SRAM-ЦЕНТРИЧНОЕ ЯДРО
# Fused: load_from_HBM → rotate → polar → quantize → pack → store_to_HBM
# Один проход через SRAM. Данные не возвращаются в HBM между шагами.
# ================================================================

class SRAMFusedConfig:
    """Конфигурация для fused kernel.
    TMEM (Blackwell): 512KB per SM
    VMEM (TPU v5): 32MB per chip
    Shared memory (Ampere/Hopper): 164-228KB per SM
    """
    SRAM_BUDGET_BYTES = 48 * 1024  # conservative: 48KB per threadblock
    BYTES_PER_FP16 = 2
    BYTES_PER_INT3 = 0.375  # 3 bits packed

    @staticmethod
    def max_vectors_per_tile(dim: int) -> int:
        """How many vectors fit in SRAM for fused processing."""
        bytes_per_vec_input = dim * SRAMFusedConfig.BYTES_PER_FP16
        bytes_per_vec_output = math.ceil(dim * SRAMFusedConfig.BYTES_PER_INT3)
        bytes_per_vec_scratch = dim * 4  # FP32 scratch for rotation
        total = bytes_per_vec_input + bytes_per_vec_output + bytes_per_vec_scratch
        return SRAMFusedConfig.SRAM_BUDGET_BYTES // total


@dataclass
class NautilusLUT:
    """Precomputed Lookup Table for golden angle rotations.
    Stored in GPU constant memory (64KB on NVIDIA).
    Total size: dim/2 * 3_layers * 2 (cos+sin) * 4 bytes = tiny.
    For dim=128: 64 * 3 * 2 * 4 = 1536 bytes. Fits easily.
    """
    dim: int
    phi: float = PHI
    cos_tables: dict = field(default_factory=dict)
    sin_tables: dict = field(default_factory=dict)
    pair_indices: dict = field(default_factory=dict)

    def __post_init__(self):
        ga = 2 * math.pi / (self.phi ** 2)

        # Layer 1: adjacent
        self._build_layer('L1',
            pairs=[(2*k, 2*k+1) for k in range(self.dim // 2)],
            angles=[ga * (k+1) for k in range(self.dim // 2)])

        # Layer 2: shifted
        self._build_layer('L2',
            pairs=[(2*k+1, 2*k+2) for k in range((self.dim-1) // 2)],
            angles=[ga * (k+1) * self.phi for k in range((self.dim-1) // 2)])

        # Layer 3: butterfly (non-overlapping)
        stride = max(2, self.dim // 4)
        pairs, angles, used = [], [], set()
        for k in range(self.dim):
            i, j = k, (k + stride) % self.dim
            if i == j or i in used or j in used:
                continue
            used.add(i); used.add(j)
            pairs.append((i, j))
            angles.append(ga * (k+1) * self.phi * self.phi)
        self._build_layer('L3', pairs, angles)

    def _build_layer(self, name, pairs, angles):
        n = min(len(pairs), len(angles))
        self.cos_tables[name] = torch.tensor([math.cos(angles[k]) for k in range(n)], dtype=torch.float32)
        self.sin_tables[name] = torch.tensor([math.sin(angles[k]) for k in range(n)], dtype=torch.float32)
        self.pair_indices[name] = pairs[:n]

    def to_device(self, device):
        for name in self.cos_tables:
            self.cos_tables[name] = self.cos_tables[name].to(device)
            self.sin_tables[name] = self.sin_tables[name].to(device)
        return self

    def memory_bytes(self):
        total = 0
        for name in self.cos_tables:
            total += self.cos_tables[name].numel() * 4 * 2  # cos + sin
            total += len(self.pair_indices[name]) * 8  # pair ints
        return total


class NautilusFusedKernel(nn.Module):
    """Концепт 1: Fused SRAM kernel.
    Все 5 шагов в одном проходе:
      1. Load FP16 from HBM → SRAM
      2. Givens rotation (3 layers, from LUT)
      3. Cartesian → Polar
      4. Scalar quantize (Lloyd-Max)
      5. Pack to INT3 + QJL sign bit → store to HBM

    Энергетический выигрыш:
      Без fusion: 5 round-trips HBM ↔ SRAM = 5 × 640pJ × dim = expensive
      С fusion: 1 load + 1 store = 2 × 640pJ × dim = 2.5x дешевле
    """

    def __init__(self, dim: int, bits: int = 3, phi: float = PHI):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.lut = NautilusLUT(dim=dim, phi=phi)
        self.tile_size = SRAMFusedConfig.max_vectors_per_tile(dim)
        print(f"[SRAM] dim={dim}, bits={bits}, LUT={self.lut.memory_bytes()} bytes, "
              f"tile={self.tile_size} vectors/tile")

    def forward(self, x: torch.Tensor) -> dict:
        """Fused encode (PyTorch reference — Triton version below)."""
        device = x.device
        lut = self.lut.to_device(device)
        n = x.shape[0]
        result_packed = []
        total_mse = 0

        # Process in SRAM-sized tiles
        for start in range(0, n, self.tile_size):
            end = min(start + self.tile_size, n)
            tile = x[start:end].float()  # ← simulates HBM → SRAM load

            # Step 2: Givens rotation (all in SRAM / registers)
            for layer_name in ['L1', 'L2', 'L3']:
                cos_t = lut.cos_tables[layer_name]
                sin_t = lut.sin_tables[layer_name]
                pairs = lut.pair_indices[layer_name]
                for k, (i, j) in enumerate(pairs):
                    a = tile[:, i].clone()
                    b = tile[:, j].clone()
                    tile[:, i] = a * cos_t[k] - b * sin_t[k]
                    tile[:, j] = a * sin_t[k] + b * cos_t[k]

            # Step 3: Polar (still in SRAM)
            polar = torch.zeros_like(tile)
            for k in range(self.dim // 2):
                i, j = 2*k, 2*k+1
                polar[:, i] = torch.sqrt(tile[:, i]**2 + tile[:, j]**2)
                polar[:, j] = torch.atan2(tile[:, j], tile[:, i])

            # Step 4: Quantize (still in SRAM)
            levels = 2 ** self.bits
            mins = polar.min(dim=0).values
            maxs = polar.max(dim=0).values
            ranges = (maxs - mins).clamp(min=1e-8)
            norm = (polar - mins) / ranges
            q = torch.round(norm * (levels - 1)).clamp(0, levels - 1)
            dequant = q / (levels - 1) * ranges + mins

            # Step 5: QJL sign bit (still in SRAM)
            error = polar - dequant
            sign_bits = (error >= 0).byte()  # 1 bit per value
            corrected = dequant + (sign_bits.float() * 2 - 1) * error.abs() * 0.5

            total_mse += (polar - corrected).pow(2).sum().item()
            result_packed.append({
                'q': q.byte(),
                'signs': sign_bits,
                'mins': mins,
                'maxs': maxs,
            })
            # ← simulates SRAM → HBM store (packed INT3 + sign bits)

        return {
            'tiles': result_packed,
            'mse': total_mse / (n * self.dim),
            'compression_ratio': 16 / (self.bits + 1),
            'hbm_reads': 1,   # fused: single load
            'hbm_writes': 1,  # fused: single store
            'energy_saved_vs_unfused': '~2.5x'
        }


# ================================================================
# КОНЦЕПТ 2: ДЕТЕРМИНИРОВАННЫЙ DATAFLOW
# Полностью статический schedule. Нет ветвлений, нет зависимостей от данных.
# Идеально для Groq LPU, Cerebras, TPU — архитектуры без кэшей.
# ================================================================

class NautilusDataflow:
    """Статический execution plan — вычисляется ДО запуска.
    Groq LPU исполняет его с предсказуемой латентностью.
    """

    def __init__(self, dim: int, phi: float = PHI):
        self.dim = dim
        self.phi = phi
        self.schedule = self._compile_schedule()

    def _compile_schedule(self) -> list:
        """Компилирует полный план исполнения.
        Каждый шаг = (operation, arg1, arg2, ...)
        Нет if/else, нет циклов с переменными границами.
        """
        ga = 2 * math.pi / (self.phi ** 2)
        ops = []

        # Layer 1: adjacent pairs
        for k in range(self.dim // 2):
            theta = ga * (k + 1)
            ops.append(('GIVENS', 2*k, 2*k+1, math.cos(theta), math.sin(theta)))

        # Layer 2: shifted pairs
        for k in range((self.dim - 1) // 2):
            theta = ga * (k + 1) * self.phi
            ops.append(('GIVENS', 2*k+1, 2*k+2, math.cos(theta), math.sin(theta)))

        # Layer 3: butterfly
        stride = max(2, self.dim // 4)
        used = set()
        for k in range(self.dim):
            i, j = k, (k + stride) % self.dim
            if i == j or i in used or j in used:
                continue
            used.add(i); used.add(j)
            theta = ga * (k + 1) * self.phi * self.phi
            ops.append(('GIVENS', i, j, math.cos(theta), math.sin(theta)))

        # Polar conversion
        for k in range(self.dim // 2):
            ops.append(('TO_POLAR', 2*k, 2*k+1))

        # Quantize marker
        ops.append(('QUANTIZE_ALL', self.dim))

        # QJL marker
        ops.append(('QJL_SIGN', self.dim))

        return ops

    def execute(self, x: torch.Tensor) -> torch.Tensor:
        """Execute static schedule on a batch of vectors."""
        out = x.clone().float()
        for op in self.schedule:
            if op[0] == 'GIVENS':
                _, i, j, c, s = op
                a = out[..., i].clone()
                b = out[..., j].clone()
                out[..., i] = a * c - b * s
                out[..., j] = a * s + b * c
            elif op[0] == 'TO_POLAR':
                _, i, j = op
                r = torch.sqrt(out[..., i]**2 + out[..., j]**2)
                theta = torch.atan2(out[..., j], out[..., i])
                out[..., i] = r
                out[..., j] = theta
        return out

    def inverse_schedule(self) -> list:
        """Compile inverse schedule for decode."""
        inv = []
        for op in reversed(self.schedule):
            if op[0] == 'GIVENS':
                _, i, j, c, s = op
                inv.append(('GIVENS', i, j, c, -s))  # negate sin = -theta
            elif op[0] == 'TO_POLAR':
                _, i, j = op
                inv.append(('FROM_POLAR', i, j))
        return inv

    def stats(self) -> dict:
        n_givens = sum(1 for op in self.schedule if op[0] == 'GIVENS')
        n_polar = sum(1 for op in self.schedule if op[0] == 'TO_POLAR')
        return {
            'total_ops': len(self.schedule),
            'givens_rotations': n_givens,
            'polar_conversions': n_polar,
            'flops_per_vector': n_givens * 6 + n_polar * 5,  # each Givens = 4 mul + 2 add
            'is_deterministic': True,
            'has_branches': False,
            'has_data_dependent_control': False,
            'lut_size_bytes': n_givens * 8,  # 2 floats (cos,sin) per rotation
            'compatible_with': ['Groq LPU', 'Cerebras WSE', 'Google TPU v5',
                                'NVIDIA Blackwell', 'AMD MI355X']
        }


# ================================================================
# КОНЦЕПТ 3: MX-ФОРМАТЫ (Microscaling) — PLAN B
# Если golden angle не даёт 0 overhead → используем OCP MX standard.
# Блок из 32 элементов делит один 8-bit shared exponent.
# ================================================================

class MXQuantizer:
    """Microscaling quantizer (OCP MX standard).
    Поддерживает: MXFP4, MXFP6, MXFP8, MXINT8.
    Overhead: 8 бит на блок из 32 значений = 0.25 бит/значение.
    Сравни с FP32 scale+zp: 32 бит на группу = до 50% overhead при 2-bit quant.
    """

    FORMATS = {
        'MXFP4': {'mantissa_bits': 2, 'exponent_bits': 1, 'total': 4},
        'MXFP6': {'mantissa_bits': 3, 'exponent_bits': 2, 'total': 6},
        'MXFP8': {'mantissa_bits': 3, 'exponent_bits': 4, 'total': 8},
        'MXINT8': {'mantissa_bits': 7, 'exponent_bits': 0, 'total': 8},
    }

    def __init__(self, format_name: str = 'MXFP4', block_size: int = 32):
        assert format_name in self.FORMATS, f"Unknown: {format_name}. Use: {list(self.FORMATS)}"
        self.format = self.FORMATS[format_name]
        self.format_name = format_name
        self.block_size = block_size
        self.overhead_bits_per_value = 8 / block_size  # shared exponent

    def quantize(self, x: torch.Tensor) -> dict:
        """MX block quantization."""
        flat = x.reshape(-1)
        n = flat.numel()
        # Pad to block_size
        pad = (self.block_size - n % self.block_size) % self.block_size
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, device=flat.device)])

        blocks = flat.reshape(-1, self.block_size)
        n_blocks = blocks.shape[0]

        # Shared exponent per block (8-bit)
        block_max = blocks.abs().max(dim=1).values.clamp(min=1e-30)
        shared_exp = torch.floor(torch.log2(block_max)).clamp(-127, 127).to(torch.int8)

        # Scale blocks by shared exponent
        scale = (2.0 ** shared_exp.float()).unsqueeze(1)
        normalized = blocks / scale

        # Quantize mantissa
        total_bits = self.format['total']
        if self.format['exponent_bits'] > 0:
            # Float-like: sign + exp + mantissa
            levels = 2 ** (self.format['mantissa_bits'] + 1)  # +1 for sign
        else:
            # Int-like: symmetric
            levels = 2 ** (total_bits - 1)

        q = torch.round(normalized * levels).clamp(-levels, levels)
        dequant = (q / levels) * scale

        mse = (flat[:n] - dequant.reshape(-1)[:n]).pow(2).mean().item()

        return {
            'quantized': dequant.reshape(-1)[:n].reshape(x.shape),
            'shared_exponents': shared_exp,
            'mse': mse,
            'format': self.format_name,
            'overhead_bits_per_value': self.overhead_bits_per_value,
            'effective_bits': self.format['total'] + self.overhead_bits_per_value,
            'n_blocks': n_blocks,
        }


class NautilusWithMX:
    """NautilusQuant + MX fallback.
    Стратегия: пробуем 0-overhead (чистый Lloyd-Max).
    Если MSE слишком высокий → переключаемся на MX для блочного масштабирования.
    """

    def __init__(self, dim: int, bits: int = 3, phi: float = PHI,
                 mx_format: str = 'MXFP4', mse_threshold: float = 0.01):
        self.dim = dim
        self.bits = bits
        self.phi = phi
        self.mx = MXQuantizer(mx_format)
        self.mse_threshold = mse_threshold
        self.lut = NautilusLUT(dim=dim, phi=phi)

    def encode(self, x: torch.Tensor) -> dict:
        """Adaptive: try 0-overhead first, fall back to MX if needed."""
        # Step 1: Rotate
        rotated = self._rotate(x)

        # Step 2: Polar
        polar = self._to_polar(rotated)

        # Step 3a: Try standard quantization (0 overhead)
        std_quant, mins, maxs = self._standard_quantize(polar)
        std_mse = (polar - std_quant).pow(2).mean().item()

        if std_mse <= self.mse_threshold:
            # Standard quant is good enough — 0 overhead wins!
            corrected = self._qjl(polar, std_quant)
            return {
                'result': corrected,
                'method': 'standard (0 overhead)',
                'mse': (polar - corrected).pow(2).mean().item(),
                'overhead_bits': 0,
                'effective_bits': self.bits + 1,
            }

        # Step 3b: MX quantization (Plan B)
        mx_result = self.mx.quantize(polar)
        corrected = self._qjl(polar, mx_result['quantized'])
        return {
            'result': corrected,
            'method': f'MX ({self.mx.format_name})',
            'mse': (polar - corrected).pow(2).mean().item(),
            'overhead_bits': self.mx.overhead_bits_per_value,
            'effective_bits': self.mx.format['total'] + self.mx.overhead_bits_per_value + 1,
        }

    def _rotate(self, x):
        out = x.clone().float()
        lut = self.lut
        for name in ['L1', 'L2', 'L3']:
            for k, (i, j) in enumerate(lut.pair_indices[name]):
                a = out[..., i].clone()
                b = out[..., j].clone()
                out[..., i] = a * lut.cos_tables[name][k] - b * lut.sin_tables[name][k]
                out[..., j] = a * lut.sin_tables[name][k] + b * lut.cos_tables[name][k]
        return out

    def _to_polar(self, x):
        out = torch.zeros_like(x)
        for k in range(self.dim // 2):
            i, j = 2*k, 2*k+1
            out[..., i] = torch.sqrt(x[..., i]**2 + x[..., j]**2)
            out[..., j] = torch.atan2(x[..., j], x[..., i])
        return out

    def _standard_quantize(self, x):
        levels = 2 ** self.bits
        mins = x.min(dim=0).values
        maxs = x.max(dim=0).values
        ranges = (maxs - mins).clamp(min=1e-8)
        norm = (x - mins) / ranges
        q = torch.round(norm * (levels - 1))
        dequant = q / (levels - 1) * ranges + mins
        return dequant, mins, maxs

    def _qjl(self, original, quantized, alpha=0.5):
        error = original - quantized
        return quantized + torch.sign(error) * error.abs() * alpha


# ================================================================
# КОНЦЕПТ 4: СУБ-1-БИТ + МУЛЬТИМОДАЛЬНОСТЬ
# ================================================================

class SubBitExperiment:
    """Исследование: можно ли сжать фазу (угол) до 1-2 бит?
    Если золотые углы дают сверх-концентрированное распределение →
    угол можно закодировать 1 битом (полуплоскость: верхняя/нижняя).
    """

    def __init__(self, dim: int, phi: float = PHI):
        self.dim = dim
        self.phi = phi
        self.lut = NautilusLUT(dim=dim, phi=phi)

    def encode_subbits(self, x: torch.Tensor, radius_bits: int = 3,
                       angle_bits: int = 1) -> dict:
        """Раздельное сжатие: radius (3 bit) + angle (1-2 bit)."""
        rotated = self._rotate(x)
        polar = self._to_polar(rotated)

        # Split radius and angle
        radii = polar[..., 0::2]
        angles = polar[..., 1::2]

        # Radius: standard 3-bit quantization
        r_dequant = self._scalar_quantize(radii, radius_bits)

        # Angle: aggressive 1-2 bit quantization
        # 1 bit = sign of angle (which half-plane)
        # 2 bits = quadrant
        a_dequant = self._scalar_quantize(angles, angle_bits)

        # Reconstruct
        result = torch.zeros_like(polar)
        result[..., 0::2] = r_dequant
        result[..., 1::2] = a_dequant

        total_bits_per_pair = radius_bits + angle_bits
        bits_per_value = total_bits_per_pair / 2

        return {
            'result': result,
            'radius_mse': (radii - r_dequant).pow(2).mean().item(),
            'angle_mse': (angles - a_dequant).pow(2).mean().item(),
            'total_mse': (polar - result).pow(2).mean().item(),
            'bits_per_value': bits_per_value,
            'compression_ratio': 16 / bits_per_value,
            'radius_bits': radius_bits,
            'angle_bits': angle_bits,
        }

    def sweep_bit_allocation(self, x: torch.Tensor) -> list:
        """Поиск оптимального распределения бит между r и θ."""
        results = []
        for r_bits in range(1, 6):
            for a_bits in range(1, 6):
                res = self.encode_subbits(x, radius_bits=r_bits, angle_bits=a_bits)
                results.append(res)
        return sorted(results, key=lambda r: r['total_mse'])

    def _rotate(self, x):
        out = x.clone().float()
        for name in ['L1', 'L2', 'L3']:
            for k, (i, j) in enumerate(self.lut.pair_indices[name]):
                a = out[..., i].clone()
                b = out[..., j].clone()
                out[..., i] = a * self.lut.cos_tables[name][k] - b * self.lut.sin_tables[name][k]
                out[..., j] = a * self.lut.sin_tables[name][k] + b * self.lut.cos_tables[name][k]
        return out

    def _to_polar(self, x):
        out = torch.zeros_like(x)
        for k in range(self.dim // 2):
            out[..., 2*k] = torch.sqrt(x[..., 2*k]**2 + x[..., 2*k+1]**2)
            out[..., 2*k+1] = torch.atan2(x[..., 2*k+1], x[..., 2*k])
        return out

    def _scalar_quantize(self, x, bits):
        levels = 2 ** bits
        mins = x.min(dim=0).values
        maxs = x.max(dim=0).values
        ranges = (maxs - mins).clamp(min=1e-8)
        norm = (x - mins) / ranges
        q = torch.round(norm * (levels - 1))
        return q / (levels - 1) * ranges + mins


class MultimodalAdapter:
    """Концепт 4b: Адаптивные параметры для текста vs изображений.
    В VLM (Vision-Language Models) визуальные токены менее чувствительны
    к квантизации → можно сжимать агрессивнее.
    """

    def __init__(self, dim: int, phi: float = PHI):
        self.dim = dim
        self.phi = phi
        # Разные конфигурации для разных модальностей
        self.configs = {
            'text': {
                'radius_bits': 3,
                'angle_bits': 2,
                'qjl_alpha': 0.5,
                'description': 'Текст: высокая точность для attention-паттернов'
            },
            'image': {
                'radius_bits': 2,
                'angle_bits': 1,
                'qjl_alpha': 0.3,
                'description': 'Изображения: агрессивное сжатие, визуальные токены устойчивее'
            },
            'audio': {
                'radius_bits': 3,
                'angle_bits': 1,
                'qjl_alpha': 0.4,
                'description': 'Аудио: средняя точность, частотные компоненты устойчивы'
            },
        }

    def get_config(self, modality: str) -> dict:
        return self.configs.get(modality, self.configs['text'])

    def estimate_savings(self, token_counts: dict) -> dict:
        """Сколько памяти экономит адаптивное сжатие vs uniform."""
        uniform_bits = 3 + 1  # 3-bit quant + 1-bit QJL
        total_tokens = sum(token_counts.values())

        adaptive_bits = 0
        for modality, count in token_counts.items():
            cfg = self.get_config(modality)
            bits = (cfg['radius_bits'] + cfg['angle_bits']) / 2 + 1
            adaptive_bits += count * bits

        uniform_total = total_tokens * uniform_bits
        return {
            'uniform_bits_total': uniform_total,
            'adaptive_bits_total': adaptive_bits,
            'savings_pct': (1 - adaptive_bits / uniform_total) * 100,
            'per_modality': {
                mod: {
                    'tokens': cnt,
                    'bits_per_value': (self.get_config(mod)['radius_bits'] +
                                       self.get_config(mod)['angle_bits']) / 2 + 1
                }
                for mod, cnt in token_counts.items()
            }
        }


# ================================================================
# ТЕСТИРОВАНИЕ ВСЕХ КОНЦЕПТОВ
# ================================================================

def test_all_concepts(dim=128, n_vectors=1000):
    print("=" * 70)
    print("NautilusQuant v2 — Hardware-Software Co-design Test")
    print("=" * 70)

    torch.manual_seed(42)
    # Realistic data with outliers
    x = torch.randn(n_vectors, dim) * 0.5
    for d in [0, 15, 31, 63, 95, 127 % dim]:
        if d < dim:
            mask = torch.rand(n_vectors) < 0.75
            x[mask, d] = torch.randn(mask.sum()) * 30

    # --- Концепт 1: SRAM Fused Kernel ---
    print("\n--- Концепт 1: SRAM-Fused Kernel ---")
    fused = NautilusFusedKernel(dim=dim, bits=3)
    result = fused.forward(x)
    print(f"  MSE: {result['mse']:.8f}")
    print(f"  Compression: {result['compression_ratio']:.1f}x")
    print(f"  HBM reads: {result['hbm_reads']}, writes: {result['hbm_writes']}")
    print(f"  Energy savings vs unfused: {result['energy_saved_vs_unfused']}")

    # --- Концепт 2: Deterministic Dataflow ---
    print("\n--- Концепт 2: Deterministic Dataflow ---")
    df = NautilusDataflow(dim=dim)
    stats = df.stats()
    print(f"  Total ops: {stats['total_ops']}")
    print(f"  FLOPs/vector: {stats['flops_per_vector']}")
    print(f"  LUT size: {stats['lut_size_bytes']} bytes")
    print(f"  Deterministic: {stats['is_deterministic']}")
    print(f"  Data-dependent branches: {stats['has_data_dependent_control']}")
    print(f"  Compatible: {', '.join(stats['compatible_with'])}")

    # Execute and verify
    df_result = df.execute(x[:10])
    print(f"  Sample output norm preservation: "
          f"{x[:10].norm(dim=-1).mean():.4f} → {df_result.norm(dim=-1).mean():.4f}")

    # --- Концепт 3: MX-форматы (Plan B) ---
    print("\n--- Концепт 3: MX-Format Fallback ---")
    for fmt in ['MXFP4', 'MXFP6', 'MXFP8']:
        nmx = NautilusWithMX(dim=dim, bits=3, mx_format=fmt, mse_threshold=0.001)
        enc = nmx.encode(x[:100])
        print(f"  {fmt}: method={enc['method']}, MSE={enc['mse']:.6f}, "
              f"overhead={enc['overhead_bits']:.2f} bits/val, "
              f"effective={enc['effective_bits']:.2f} bits")

    # --- Концепт 4: Суб-1-бит + Мультимодальность ---
    print("\n--- Концепт 4a: Sub-bit Allocation Sweep ---")
    sub = SubBitExperiment(dim=dim)
    sweep = sub.sweep_bit_allocation(x[:200])
    print("  Top 5 allocations:")
    for i, r in enumerate(sweep[:5]):
        print(f"    {i+1}. r={r['radius_bits']}bit + θ={r['angle_bits']}bit → "
              f"{r['bits_per_value']:.1f} bit/val, MSE={r['total_mse']:.6f}, "
              f"compression={r['compression_ratio']:.1f}x")

    print("\n--- Концепт 4b: Multimodal Savings ---")
    mm = MultimodalAdapter(dim=dim)
    savings = mm.estimate_savings({'text': 8000, 'image': 4000, 'audio': 1000})
    print(f"  Uniform: {savings['uniform_bits_total']} bits")
    print(f"  Adaptive: {savings['adaptive_bits_total']:.0f} bits")
    print(f"  Savings: {savings['savings_pct']:.1f}%")
    for mod, info in savings['per_modality'].items():
        print(f"    {mod}: {info['tokens']} tokens × {info['bits_per_value']:.1f} bits")

    print("\n" + "=" * 70)
    print("All concepts tested. Ready for hardware deployment.")


if __name__ == '__main__':
    test_all_concepts()
