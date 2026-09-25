"""
NautilusQuant — Experiment Logger & Monitor
Записывает ВСЕ результаты экспериментов в JSON + CSV.
Позволяет сравнивать запуски, строить графики, отслеживать прогресс.

Использование:
  from experiment_logger import ExperimentLogger
  log = ExperimentLogger("results")

  run = log.start_run("core_test", dim=128, bits=3, method="nautilus")
  run.record("mse", 0.00823)
  run.record("angle_variance", 1.234)
  run.finish(status="pass")

  # Посмотреть все результаты:
  log.print_summary()
  log.compare_methods("mse")
  log.export_csv("results/summary.csv")
"""

import csv
import json
import os
import platform
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class RunMetrics:
    """Все метрики одного запуска."""
    # Identity
    run_id: str = ""
    timestamp: str = ""
    experiment: str = ""
    method: str = ""  # "nautilus" | "turbo" | "random" | "planb_*"

    # Config
    dim: int = 128
    bits: int = 3
    n_vectors: int = 500
    phi_value: float = 1.618033988749895

    # Orthogonality checks
    orth_error: float = 0.0
    norm_error: float = 0.0
    roundtrip_error: float = 0.0
    dot_error: float = 0.0

    # Quantization quality
    mse: float = 0.0
    psnr: float = 0.0  # Peak Signal-to-Noise Ratio
    max_error: float = 0.0

    # Angle distribution (KEY metric for 0-overhead)
    angle_variance: float = 0.0
    angle_uniformity: float = 0.0  # 0=clustered, 1=perfectly uniform
    angle_range_min: float = 0.0
    angle_range_max: float = 0.0

    # Radius distribution
    radius_variance: float = 0.0
    radius_preservation: float = 0.0

    # Outlier handling
    outlier_mse: float = 0.0  # MSE only on outlier dimensions
    outlier_dampening: float = 0.0  # how much outliers were smoothed

    # Overhead
    overhead_bits: float = 0.0
    compression_ratio: float = 0.0

    # Performance (GPU)
    encode_ms: float = 0.0
    decode_ms: float = 0.0
    throughput_vecs_sec: float = 0.0
    gpu_memory_mb: float = 0.0

    # Hardware concepts
    lut_bytes: int = 0
    sram_vectors_per_tile: int = 0

    # Status
    status: str = "running"  # "running" | "pass" | "fail" | "error"
    notes: str = ""
    duration_sec: float = 0.0

    # Extra data (flexible)
    extra: dict = field(default_factory=dict)


class RunContext:
    """Контекст одного запуска. Используйте record() для записи метрик."""

    def __init__(self, logger: 'ExperimentLogger', metrics: RunMetrics):
        self.logger = logger
        self.metrics = metrics
        self._start_time = time.time()

    def record(self, key: str, value: Any):
        """Записать метрику."""
        if hasattr(self.metrics, key):
            setattr(self.metrics, key, value)
        else:
            self.metrics.extra[key] = value

    def record_dict(self, d: dict):
        """Записать несколько метрик сразу."""
        for k, v in d.items():
            self.record(k, v)

    def finish(self, status: str = "pass", notes: str = ""):
        """Завершить запуск и сохранить."""
        self.metrics.status = status
        self.metrics.notes = notes
        self.metrics.duration_sec = time.time() - self._start_time
        self.logger._save_run(self.metrics)
        return self.metrics


class ExperimentLogger:
    """
    Главный логгер экспериментов.
    Сохраняет каждый запуск в JSON + поддерживает summary CSV.
    """

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs: list[RunMetrics] = []
        self._load_existing()

    def _load_existing(self):
        """Загрузить предыдущие запуски из файлов."""
        history_file = self.output_dir / "history.jsonl"
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            extra = data.pop("extra", {})
                            # Remove keys not in RunMetrics
                            valid_keys = {f.name for f in RunMetrics.__dataclass_fields__.values()}
                            filtered = {k: v for k, v in data.items() if k in valid_keys}
                            m = RunMetrics(**filtered)
                            m.extra = extra
                            self.runs.append(m)
                        except (json.JSONDecodeError, TypeError):
                            pass
        print(f"  [Logger] Loaded {len(self.runs)} previous runs from {history_file}")

    def start_run(self, experiment: str, **kwargs) -> RunContext:
        """Начать новый запуск."""
        run_id = f"{experiment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.runs)}"

        metrics = RunMetrics(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            experiment=experiment,
        )

        # Apply kwargs
        for k, v in kwargs.items():
            if hasattr(metrics, k):
                setattr(metrics, k, v)
            else:
                metrics.extra[k] = v

        return RunContext(self, metrics)

    def _save_run(self, metrics: RunMetrics):
        """Сохранить завершённый запуск."""
        self.runs.append(metrics)

        # Append to JSONL (one line per run — easy to parse)
        history_file = self.output_dir / "history.jsonl"
        with open(history_file, "a", encoding="utf-8") as f:
            data = asdict(metrics)
            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

        # Save individual run
        run_file = self.output_dir / f"{metrics.run_id}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2, ensure_ascii=False, default=str)

        # Update summary CSV
        self._update_csv()

        status_icon = "✅" if metrics.status == "pass" else "❌" if metrics.status == "fail" else "⚠️"
        print(f"  [Logger] {status_icon} Saved: {metrics.run_id}  MSE={metrics.mse:.8f}  "
              f"AngleVar={metrics.angle_variance:.4f}  {metrics.duration_sec:.1f}s")

    def _update_csv(self):
        """Обновить summary CSV со всеми запусками."""
        csv_file = self.output_dir / "summary.csv"
        fields = [
            "run_id", "timestamp", "experiment", "method",
            "dim", "bits", "n_vectors", "phi_value",
            "mse", "angle_variance", "angle_uniformity",
            "overhead_bits", "compression_ratio",
            "orth_error", "norm_error", "roundtrip_error", "dot_error",
            "outlier_mse", "outlier_dampening",
            "encode_ms", "throughput_vecs_sec",
            "lut_bytes", "status", "duration_sec", "notes"
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in self.runs:
                writer.writerow(asdict(r))

    def print_summary(self, last_n: int = 20):
        """Напечатать таблицу последних запусков."""
        runs = self.runs[-last_n:]
        if not runs:
            print("  Нет записанных запусков.")
            return

        print(f"\n{'='*90}")
        print(f"  EXPERIMENT HISTORY (last {len(runs)} runs)")
        print(f"{'='*90}")
        print(f"  {'Method':<14} {'Dim':>4} {'Bits':>4} {'MSE':>12} {'AngleVar':>10} "
              f"{'Overhead':>8} {'Status':>6} {'Time':>6}")
        print(f"  {'─'*74}")

        for r in runs:
            status = "✅" if r.status == "pass" else "❌" if r.status == "fail" else "⏳"
            print(f"  {r.method:<14} {r.dim:>4} {r.bits:>4} {r.mse:>12.8f} "
                  f"{r.angle_variance:>10.4f} {r.overhead_bits:>7.2f}b "
                  f"{status:>6} {r.duration_sec:>5.1f}s")

    def compare_methods(self, metric: str = "mse"):
        """Сравнить методы по указанной метрике."""
        if not self.runs:
            print("  Нет данных для сравнения.")
            return

        # Group by method
        methods: dict[str, list] = {}
        for r in self.runs:
            m = r.method or "unknown"
            if m not in methods:
                methods[m] = []
            val = getattr(r, metric, r.extra.get(metric, None))
            if val is not None and val != 0:
                methods[m].append(val)

        print(f"\n{'='*60}")
        print(f"  COMPARISON by {metric}")
        print(f"{'='*60}")

        results = []
        for method, values in sorted(methods.items()):
            if not values:
                continue
            avg = sum(values) / len(values)
            best = min(values)
            worst = max(values)
            results.append((method, avg, best, worst, len(values)))

        results.sort(key=lambda x: x[1])

        print(f"  {'Method':<16} {'Avg':>12} {'Best':>12} {'Worst':>12} {'Runs':>5}")
        print(f"  {'─'*59}")
        for i, (method, avg, best, worst, n) in enumerate(results):
            marker = " 🏆" if i == 0 else ""
            print(f"  {method:<16} {avg:>12.8f} {best:>12.8f} {worst:>12.8f} {n:>5}{marker}")

    def export_csv(self, path: str = None):
        """Экспортировать в CSV."""
        path = path or str(self.output_dir / "summary.csv")
        self._update_csv()
        print(f"  [Logger] CSV exported: {path} ({len(self.runs)} runs)")

    def get_best(self, metric: str = "mse", method: str = None) -> Optional[RunMetrics]:
        """Получить лучший запуск по метрике."""
        candidates = self.runs
        if method:
            candidates = [r for r in candidates if r.method == method]
        if not candidates:
            return None
        return min(candidates, key=lambda r: getattr(r, metric, float('inf')))


# =====================================================================
# Интеграция с run_all.py — обёртка для автоматической записи
# =====================================================================

def run_with_logging(output_dir: str = "results"):
    """
    Запуск всех тестов с автоматическим логированием.

    python experiment_logger.py                    # всё
    python experiment_logger.py --test core        # только ядро
    python experiment_logger.py --test sweep       # PHI sweep
    python experiment_logger.py --compare          # сравнить все запуски
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["core", "sweep", "all"], default="all")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--compare", action="store_true", help="Show comparison table")
    parser.add_argument("--history", action="store_true", help="Show run history")
    args = parser.parse_args()

    log = ExperimentLogger(output_dir)

    if args.compare:
        log.compare_methods("mse")
        log.compare_methods("angle_variance")
        return

    if args.history:
        log.print_summary(50)
        return

    import torch
    import math

    PHI = (1 + math.sqrt(5)) / 2
    GA = 2 * math.pi / (PHI ** 2)

    def build_layers(d, phi=PHI):
        ga = 2 * math.pi / (phi ** 2)
        layers = []
        l1 = [(2*k, 2*k+1, ga*(k+1)) for k in range(d//2)]
        layers.append(l1)
        l2 = [(2*k+1, 2*k+2, ga*(k+1)*phi) for k in range((d-1)//2)]
        layers.append(l2)
        l3 = []
        stride = max(2, d//4)
        used = set()
        for k in range(d):
            i, j = k, (k+stride)%d
            if i==j or i in used or j in used: continue
            used.add(i); used.add(j)
            l3.append((i, j, ga*(k+1)*phi*phi))
        layers.append(l3)
        return layers

    def apply_fwd(x, layers):
        out = x.clone()
        for layer in layers:
            for i, j, theta in layer:
                c, s = math.cos(theta), math.sin(theta)
                a, b = out[...,i].clone(), out[...,j].clone()
                out[...,i] = a*c - b*s
                out[...,j] = a*s + b*c
        return out

    def apply_inv(x, layers):
        out = x.clone()
        for layer in reversed(layers):
            for i, j, theta in reversed(layer):
                c, s = math.cos(-theta), math.sin(-theta)
                a, b = out[...,i].clone(), out[...,j].clone()
                out[...,i] = a*c - b*s
                out[...,j] = a*s + b*c
        return out

    def quantize(x, bits):
        levels = 2**bits
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

    def run_method(data, method_name, phi_val=None, bits=3):
        """Run one method, return RunContext with all metrics."""
        run = log.start_run(
            experiment="quant_comparison",
            method=method_name,
            dim=data.shape[-1],
            bits=bits,
            n_vectors=data.shape[0],
            phi_value=phi_val or PHI,
        )

        d = data.shape[-1]

        if method_name == "turbo_random":
            # TurboQuant: random orthogonal
            R, _ = torch.linalg.qr(torch.randn(d, d))
            rotated = data @ R.T
            quantized = quantize(rotated, bits)
            recovered = quantized @ R
        elif method_name == "fp16_baseline":
            # No quantization
            rotated = data.clone()
            quantized = data.clone()
            recovered = data.clone()
        else:
            # NautilusQuant with given phi
            p = phi_val or PHI
            layers = build_layers(d, p)
            rotated = apply_fwd(data, layers)
            quantized = quantize(rotated, bits)
            recovered = apply_inv(quantized, layers)

            # Orthogonality checks
            T = torch.eye(d)
            T = apply_fwd(T, layers)
            orth_err = (T @ T.T - torch.eye(d)).abs().max().item()
            run.record("orth_error", orth_err)

            norms_before = torch.norm(data, dim=-1)
            norms_after = torch.norm(rotated, dim=-1)
            run.record("norm_error", (norms_before - norms_after).abs().max().item())

            rt = apply_inv(rotated, layers)
            run.record("roundtrip_error", (data - rt).abs().max().item())

            dots_b = (data[:50] * data[1:51]).sum(-1)
            dots_a = (rotated[:50] * rotated[1:51]).sum(-1)
            run.record("dot_error", (dots_b - dots_a).abs().max().item())

            run.record("lut_bytes", len(layers[0]) * 3 * 2 * 4)
            run.record("overhead_bits", 0.0)

        # MSE
        mse = ((data - recovered)**2).mean().item()
        max_err = (data - recovered).abs().max().item()
        psnr = 10 * math.log10(data.abs().max().item()**2 / max(mse, 1e-20))

        run.record("mse", mse)
        run.record("max_error", max_err)
        run.record("psnr", psnr)
        run.record("compression_ratio", 16.0 / bits)

        # Angle distribution
        angles = polar_angles(rotated)
        run.record("angle_variance", angles.var().item())
        run.record("angle_range_min", angles.min().item())
        run.record("angle_range_max", angles.max().item())

        # Check uniformity (compare to ideal uniform variance = pi²/3)
        ideal_var = math.pi**2 / 3
        run.record("angle_uniformity", 1.0 - abs(angles.var().item() - ideal_var) / ideal_var)

        # Outlier-specific MSE
        outlier_dims = [7, 23, 41, 58, 89, 112]
        valid_dims = [od for od in outlier_dims if od < d]
        if valid_dims:
            outlier_mse = ((data[:, valid_dims] - recovered[:, valid_dims])**2).mean().item()
            run.record("outlier_mse", outlier_mse)

        if method_name == "turbo_random":
            run.record("overhead_bits", 32.0)  # needs scale/zero per group

        status = "pass" if mse < 1.0 else "fail"
        run.finish(status=status, notes=f"{method_name} dim={d} bits={bits}")
        return mse

    # Generate data with outliers
    torch.manual_seed(42)
    data = torch.randn(args.n, args.dim) * 0.5
    outlier_dims = [7, 23, 41, 58, 89, 112]
    mask = torch.rand(args.n) < 0.75
    for od in outlier_dims:
        if od < args.dim:
            data[mask, od] = torch.randn(mask.sum()) * 30 - 30

    print(f"\n{'='*70}")
    print(f"  LOGGED EXPERIMENT RUN")
    print(f"  dim={args.dim}  n={args.n}  bits={args.bits}")
    print(f"  Output: {log.output_dir}/")
    print(f"{'='*70}\n")

    if args.test in ("core", "all"):
        # Run all methods
        run_method(data, "nautilus_golden", PHI, args.bits)
        run_method(data, "turbo_random", None, args.bits)

        # Also test with different bits
        if args.test == "all":
            for b in [2, 4, 8]:
                if b != args.bits:
                    run_method(data, "nautilus_golden", PHI, b)
                    run_method(data, "turbo_random", None, b)

    if args.test in ("sweep", "all"):
        # PHI sweep
        phi_candidates = [
            ("nautilus_phi", PHI),
            ("nautilus_phi2", PHI**2),
            ("nautilus_inv_phi", 1/PHI),
            ("nautilus_silver", 1 + math.sqrt(2)),
            ("nautilus_pi2", math.pi/2),
            ("nautilus_e2", math.e/2),
        ]
        for name, phi in phi_candidates:
            run_method(data, name, phi, args.bits)

    # Print results
    log.print_summary()
    log.compare_methods("mse")
    log.compare_methods("angle_variance")
    log.export_csv()

    print(f"\n  📁 Все результаты сохранены в: {log.output_dir}/")
    print(f"  📊 CSV: {log.output_dir}/summary.csv")
    print(f"  📋 History: {log.output_dir}/history.jsonl")
    print(f"  📄 Individual runs: {log.output_dir}/*.json")


if __name__ == "__main__":
    run_with_logging()
