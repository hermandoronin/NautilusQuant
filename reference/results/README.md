# results/

Output directory for `validate_real_kv.py`, `benchmark_ab.py` and
`experiment_logger.py`. Runs land here as `<experiment>_<timestamp>_<n>.json`
plus a rolling `summary.csv` / `history.jsonl`.

The directory ships empty on purpose. An earlier batch of ten `real_model`
runs (2026-03-27) was removed: the PyTorch forward hooks never captured KV
tensors, so two runs ended with `status=error` / "No hooks captured data" and
the rest recorded all-zero metrics. Publishing them would have implied
measurements that were never taken.

Regenerate real data with:

```bash
python validate_real_kv.py --model google/gemma-3-4b-it --sweep
```
