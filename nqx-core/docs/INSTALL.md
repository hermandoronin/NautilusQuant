# Installing NQX-Core

## TL;DR — local CPU-only

```bash
git clone https://github.com/hermandoronin/NautilusQuant
cd NautilusQuant/nqx-core
pip install -r requirements.txt
python -m pytest tests -q     # 229 passing
python run.py verify --dim 128
```

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.11 | 3.12 |
| NumPy | 2.0 | 2.4 |
| FastAPI | 0.110 | 0.128+ |
| Uvicorn | 0.27 | 0.40+ |
| Pydantic | 2.5 | 2.12+ |
| pytest (dev) | 8.0 | 9.0+ |
| Docker (optional) | 20.10 | 24+ with buildx |
| Verilator (RTL) | 5.0 | 5.020+ |
| Yosys (synth) | 0.30 | latest |
| OpenLane2 (MPW) | latest | via Docker |

GPU stack (optional, for `server/backends.GPUBackend`):

| Component | Minimum | Recommended |
|---|---|---|
| PyTorch | 2.2 | 2.5+ |
| Triton | 2.2 | 3.x |
| CUDA | 12.0 | 12.4 |
| GPU | RTX 4090 (FP4 emulated) | RTX 5090 / B200 (MXFP4 native) |

## Platforms tested

| Platform | Status |
|---|---|
| Ubuntu 22.04+ | ✅ supported |
| Arch / Manjaro | ✅ supported |
| Debian 12+ | ✅ supported |
| macOS 14+ (Apple Silicon) | ✅ CPU-only |
| Windows 11 (WSL2) | ✅ via WSL2 |
| Docker (multi-arch) | ✅ amd64 + arm64 |

## Install methods

### 1. Pip (development)

```bash
git clone https://github.com/hermandoronin/NautilusQuant && cd NautilusQuant/nqx-core
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # if pyproject.toml present
```

### 2. Docker (CPU)

No prebuilt image is published. Build it locally:

```bash
docker build -f Dockerfile.cpu -t nqx-server:cpu .
docker run --rm -p 8000:8000 nqx-server:cpu
curl http://localhost:8000/health
```

### 3. Docker (GPU, NVIDIA Container Toolkit required)

No prebuilt image is published. Build it locally:

```bash
docker build -f Dockerfile -t nqx-server:gpu .
docker run --rm --gpus all -p 8000:8000 nqx-server:gpu
```

### 4. CLI launchers (KDE / GNOME)

```bash
bash tools/cli/install.sh
```

Creates symlinks in `~/.local/bin/` and a `.desktop` shortcut on the desktop:
- `nqx-claude`, `nqx-deepseek`, `nqx-flash`, `nqx-codex`, `nqx-trio`
- `nqx-heavy`, `nqx-routine`, `nqx-audit`, `nqx-status`, `nqx-debug`
- `nqx-demo`, `nqx-launch-all`
- `~/Desktop/NQX-Core.desktop` (двойной клик → launches Heavy + Routine in two windows)

### 5. Vast.ai (one-command deploy on RTX 5090 / H100)

```bash
docker build -t YOURUSER/nqx-server:gpu . && docker push YOURUSER/nqx-server:gpu
IMAGE=YOURUSER/nqx-server:gpu bash deploy/quickstart-vastai.sh
```

See [`deploy/vastai.md`](../deploy/vastai.md) for full instructions.

## Verify install

```bash
python -m pytest tests -q                       # 229 passing in <20 sec
python run.py verify --dim 128                  # acceptance criteria
python run.py bench --vectors 4096 --quiet      # cycles + throughput
nqx-status                                      # CLI tasks status
```

## Optional: RTL / synth tools

For RTL development you need additional tools:

```bash
# Verilator (simulation)
sudo apt-get install verilator      # or pacman -S verilator (Arch)

# Yosys (synthesis)
sudo apt-get install yosys          # or pacman -S yosys

# SymbiYosys (formal)
pip install symbiyosys

# OpenLane2 (full ASIC flow → Skywater MPW)
docker pull efabless/openlane2:latest

# Run RTL flow
cd rtl
make sim                             # Verilator simulation
cd synth && make synth               # Yosys synthesis
cd ../formal && make formal          # SymbiYosys formal verification
cd ../openlane && openlane2 -i config.json   # full flow
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'nqx'"

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
# or
pip install -e .
```

### "GPUBackend requires nautilus_triton.py"

Either set `NQX_BACKEND=cpu`, or clone the upstream into the search path:

```bash
git clone https://github.com/hermandoronin/NautilusQuant /tmp/naut
export PYTHONPATH=/tmp/naut/reference:$PYTHONPATH   # nautilus_triton.py lives in reference/
```

### Tests fail with "import torch"

Tests for the GPU backend are skipped automatically if torch is not installed.
If you see a torch import error during `pytest`, check that the failing test
isn't shadowing the optional-import logic.

### `nqx-launch-all` doesn't open windows on KDE

```bash
chmod +x ~/Desktop/NQX-Core.desktop
# Or right-click → Properties → Permissions → ☑ Executable
```

## Uninstall

```bash
# Remove CLI launchers
rm ~/.local/bin/nqx-* ~/Desktop/NQX-Core.desktop

# Remove pip install
pip uninstall nqx-core

# Remove Docker images
docker rmi nqx-server:cpu nqx-server:gpu
```

## Got stuck?

Open an issue with the output of:

```bash
python --version
pip list | grep -E 'numpy|fastapi|pydantic|torch'
uname -a
python -m pytest tests -q --tb=short 2>&1 | tail -20
```
