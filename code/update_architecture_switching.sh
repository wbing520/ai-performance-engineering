#!/bin/bash
# Regenerate Blackwell-only configuration artifacts (requirements & docs).

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

cat <<'TEMPLATE' | python3 - "$SCRIPT_DIR" <<'PY'
import sys
from pathlib import Path

TEMPLATE = sys.stdin.read()
root = Path(sys.argv[1])
for path in root.rglob('requirements*.txt'):
    path.write_text(TEMPLATE)
PY
TEMPLATE
# AI Performance Engineering - Blackwell Requirements
# Target: Blackwell B200/B300 (SM100) with PyTorch 2.8, CUDA 12.8, Triton 3.3

--index-url https://download.pytorch.org/whl/nightly/cu128
torch==2.8.0.dev
torchvision==0.19.0.dev
torchaudio==2.8.0.dev

nvidia-cuda-runtime-cu12==12.8.*
nvidia-cuda-nvrtc-cu12==12.8.*
nvidia-cudnn-cu12==9.12.0.46
nvidia-cublas-cu12==12.8.*
nvidia-cufft-cu12==11.3.3.83
nvidia-curand-cu12==10.3.10.19
# nvidia-cusolver-cu12==12.8.*
# nvidia-cusparse-cu12==12.8.*
nvidia-nccl-cu12==2.20.5
nvidia-nvtx-cu12==12.8.*

triton==3.3.1

nvidia-ml-py3==7.352.0
psutil==6.1.0
GPUtil==1.4.0
py-cpuinfo==9.0.0

numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
pillow==10.2.0

matplotlib==3.8.4
seaborn==0.13.2
tensorboard==2.16.2
wandb==0.17.0
plotly==5.18.0
bokeh==3.4.1
dash==2.16.1

jupyter==1.0.0
ipykernel==6.29.5
black==24.2.0
flake8==7.0.0
mypy==1.9.0

transformers==4.40.2
datasets==2.18.0
accelerate==0.29.0
sentencepiece==0.2.0
tokenizers==0.19.1

onnx==1.16.1
onnxruntime-gpu==1.18.0

py-spy==0.3.14
memory-profiler==0.61.0
line-profiler==4.1.2
pyinstrument==5.0.0
snakeviz==2.1.1

optuna==4.0.0
hyperopt==0.2.7
ray[tune]==2.10.0

dask==2024.1.1
xarray==2024.1.0
TEMPLATE

cat <<'MSG'
Regenerated requirements files for Blackwell-only configuration.
Review documentation to ensure any legacy references are removed manually.
MSG
