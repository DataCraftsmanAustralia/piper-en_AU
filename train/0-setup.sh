#!/usr/bin/env bash
# Build the piper1-gpl training environment at the repo root (../).
#
# Linux only. Do not run this under native Windows: the PyPI torch wheel for
# win_amd64 is CPU-only (every nvidia-* dependency is gated behind
# platform_system == "Linux"), and build_monotonic_align.sh assumes a .so.
set -euo pipefail

# This script lives in train/, so the repo root is one level up.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$ROOT/piper1-gpl"
PY="${PIPER_AU_PYTHON:-python3.12}"

# torch 2.8.0 is deliberate, both ends of the range:
#   >= 2.7  -- first version whose CUDA 12.8 build carries sm_120, so one venv
#              covers the newest cards as well as this box. Harmless on older.
#   <  2.9  -- 2.9 defaults the ONNX exporter to dynamo=True and 2.10 deprecates
#              dynamic_axes, which piper1-gpl's export_onnx.py still passes with
#              no dynamo= argument. Training survives it; export may not.
TORCH="${PIPER_AU_TORCH:-2.8.0}"
TORCH_INDEX="${PIPER_AU_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "==> python: $PY"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "!! $PY not found."
    echo "   Ubuntu:  sudo apt install python3.12 python3.12-venv python3.12-dev"
    echo "   Or set PIPER_AU_PYTHON=python3.11"
    exit 1
fi
"$PY" --version

echo "==> build tools"
for tool in cmake ninja gcc git; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "!! missing $tool -- sudo apt install build-essential cmake ninja-build git"
        exit 1
    }
done

echo "==> clone piper1-gpl"
if [ -d "$REPO/.git" ]; then
    echo "    already present at $REPO, leaving it alone"
else
    git clone https://github.com/OHF-voice/piper1-gpl.git "$REPO"
fi

cd "$REPO"

echo "==> venv"
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
# scikit-build is not pulled in automatically and the build fails without it.
python -m pip install scikit-build

echo "==> torch + torchaudio $TORCH from $TORCH_INDEX"
# Installed BEFORE the editable install, so the torch>=2,<3 constraint in
# pyproject is already satisfied and pip does not pull a different build.
#
# torchaudio is NOT optional despite not being a hard dependency. piper1-gpl's
# UTMOS MOS predictor imports it, and without it val_mos is silently disabled --
# but the default ModelCheckpoint callback still monitors val_mos, so training
# crashes at the END of epoch 0 with "could not find the monitored key".
# Versions must match torch exactly.
python -m pip install "torch==$TORCH" "torchaudio==$TORCH" --index-url "$TORCH_INDEX"

echo "==> piper1-gpl (editable, with training extras)"
# Note: 'pip install piper-tts[train]' from PyPI CANNOT train -- that wheel
# ships monotonic_align/__init__.py but no compiled core extension, so it
# ImportErrors at startup. The editable install from the clone is required.
python -m pip install -e '.[train]'

echo "==> build monotonic_align"
./build_monotonic_align.sh
python setup.py build_ext --inplace

echo
echo "==> done"
python - <<'EOF'
import torch
print("    torch      %s" % torch.__version__)
print("    cuda       %s" % torch.version.cuda)
print("    available  %s" % torch.cuda.is_available())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print("    device     %s (sm_%d%d)" % (torch.cuda.get_device_name(0), cap[0], cap[1]))
EOF

echo
echo "Next:"
echo "  source $REPO/.venv/bin/activate"
echo "  python $ROOT/train/1-preflight.py"
echo "  $ROOT/train/2-train.sh single"
