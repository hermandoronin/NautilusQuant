# reference/ — version 1 software (PyTorch, Triton, benchmarks)

The first version of NautilusQuant (March–July 2026) as a software
prototype: the three-layer golden-angle Givens rotation, the polar
quantizer, GPU kernels and the benchmark scripts around them. The maintained
code is elsewhere: [`nqx-core/`](../nqx-core/) (emulator and SDK) and
[`nqx-silicon/`](../nqx-silicon/) (the chip, its model and the research
scripts). This directory is kept as the reference the later work was checked
against (`nqx-core/tests/test_vs_reference.py` compares with
`nautilus_triton.py`).

| File | What it is |
|---|---|
| `nautilus_triton.py` | PyTorch reference of the rotation and quantizer, plus a Triton GPU kernel |
| `nautilus_triton_lut.py` | Triton kernel that reads the rotation from the 1.9 KB cos/sin table |
| `nautilus_hardware.py` | Four hardware co-design concepts (SRAM-centric, fused kernel, dataflow, MX formats) |
| `validate_real_kv.py` | Validation on synthetic KV tensors and on real ones via `transformers` |
| `benchmark_ab.py` | A/B comparison with a TurboQuant-style pipeline |
| `benchmark_glove.py` | GloVe vector-search benchmark with a KIVI baseline |
| `benchmark_needle.py` | Needle-in-a-haystack and LongBench harness |
| `run_all.py`, `experiment_logger.py` | One entry point for the experiments; results go to `results/` |
| [`plan_b/`](plan_b/) | Early experimental ideas, untested |

**Status.** No real-model measurement from these scripts is published:
the first attempt's forward hooks captured no KV tensors, and those runs were
removed (see [`results/README.md`](results/README.md)). The measured
comparisons of the project are in `nqx-core/bench/` and
`nqx-silicon/research/`.

```bash
cd reference
pip install -r requirements.txt
python validate_real_kv.py --sweep --dim 128 --count 500       # synthetic
python validate_real_kv.py --model google/gemma-3-4b-it --sweep # real KV, needs transformers + weights
python nautilus_triton.py --dim 128 --n 10000                   # GPU kernel, needs triton
```
