#!/usr/bin/env bash
# Export a trained checkpoint to the .onnx + .onnx.json pair a Piper runtime loads.
#
#   ./deploy/0-export.sh multi                    # newest last.ckpt
#   ./deploy/0-export.sh multi <path.ckpt>        # a specific checkpoint
#
# Profile names are the same as in train/2-train.sh: multi, female, or the label
# of one reader from config/roster.json.
#
# Pick the checkpoint BY EAR. piper1-gpl's own code notes that val_mel saturates
# early while the GAN losses keep improving audibility, so the best-scoring
# checkpoint is not reliably the best-sounding one. The default callbacks keep
# the top 5 on val_mel and the top 5 on val_mos (UTMOS) plus last.ckpt -- render
# the same sentence through a few and listen.
set -euo pipefail

# This script lives in deploy/, so the repo root is one level up.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib/piper-au.sh
. "$ROOT/lib/piper-au.sh"

PROFILE="${1:-}"
CKPT="${2:-}"

profile_vars "$PROFILE" || {
    echo "usage: $0 <multi|female|reader-label> [checkpoint.ckpt]"
    exit 2
}

RUNDIR="$PIPER_AU_RUNS/$RUNKEY"
CONFIG="$RUNDIR/$VOICE.onnx.json"
OUTDIR="$PIPER_AU_VOICES"
mkdir -p "$OUTDIR"

if [ -z "$CKPT" ]; then
    CKPT="$(ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)"
fi
[ -n "$CKPT" ] && [ -f "$CKPT" ] || {
    echo "!! no checkpoint found under $RUNDIR"
    echo "   available:"
    ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null || echo "   (none)"
    exit 1
}

# This is the trap worth repeating: export_onnx writes ONLY the .onnx. The
# companion .onnx.json is the file training wrote to --data.config_path. The
# exporter does not produce it and cannot reconstruct it. Without it the voice
# will not load, and for a multi-speaker model it is also the only record of
# the speaker name -> id mapping.
[ -f "$CONFIG" ] || {
    echo "!! config not found: $CONFIG"
    echo "   That file is written during training (--data.config_path)."
    echo "   Without it the exported voice cannot be loaded."
    exit 1
}

activate_venv

echo "==> checkpoint $CKPT"
python -m piper.train.export_onnx \
    --checkpoint "$CKPT" \
    --output-file "$OUTDIR/$VOICE-medium.onnx"

cp "$CONFIG" "$OUTDIR/$VOICE-medium.onnx.json"

echo
echo "==> wrote"
ls -lh "$OUTDIR/$VOICE-medium.onnx" "$OUTDIR/$VOICE-medium.onnx.json"

echo
echo "==> speaker map"
python "$ROOT/deploy/2-voice-map.py" "$OUTDIR/$VOICE-medium.onnx.json" --write

cat <<EOF

==> test it here, on Linux, NOT with piper.exe on the workstation.
    piper1-gpl issue #260: piper.exe crashes with STATUS_STACK_BUFFER_OVERRUN
    in ucrtbase.dll during onnxruntime init, specifically on Windows 11 build
    26200 -- which is the build the workstation runs. Open, unresolved.

    echo 'Good on ya, love. The arvo turned out alright in the end.' \\
      | python -m piper -m $OUTDIR/$VOICE-medium.onnx -f /tmp/test.wav
EOF
