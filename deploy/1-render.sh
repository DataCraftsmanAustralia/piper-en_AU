#!/usr/bin/env bash
# Render every speaker in a trained model, so a checkpoint can be judged by ear.
#
#   ./deploy/1-render.sh multi                        # newest checkpoint
#   ./deploy/1-render.sh multi archive/multi/epoch=544-val_mos=3.7609.ckpt
#
# Writes out/<label>/<speaker>.wav plus a combined index, then prints the paths.
#
# Why this exists: val_mel and val_dur disagreed on this run -- val_mel kept
# improving to the end while val_dur bottomed near epoch 100 and rose after.
# val_dur is the duration predictor, i.e. rhythm and pacing. So the last
# checkpoint is not automatically the best voice, and the only way to settle it
# is to hear the same sentences from both.
set -uo pipefail

# This script lives in deploy/, so the repo root is one level up.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib/piper-au.sh
. "$ROOT/lib/piper-au.sh"

PROFILE="${1:-}"
CKPT="${2:-}"
[ -n "$PROFILE" ] || { print_usage "$0"; exit 2; }
profile_vars "$PROFILE" || { echo "!! unknown profile $PROFILE"; exit 2; }

RUNDIR="$PIPER_AU_RUNS/$RUNKEY"
cd "$ROOT"

# Sentences deliberately unlike the training data. The corpus is 19th-century
# Australian prose; a voice that handles that and falls apart on modern
# punctuation, numerals and abbreviations is not deployable. The last two exist
# to break it on purpose.
read -r -d '' SENTENCES <<'EOF'
Good on ya. The arvo turned out alright in the end.
I reckon we should head down to the beach before it gets too hot.
Your meeting with Ana starts at half past three, and the report is due Friday.
The balance is forty-two thousand, one hundred and seven dollars.
Right -- deploy at 09:45, check the logs, then we'll know if it worked.
Dr. Chen said the new GPU costs approx. $8,500 (incl. GST), so e.g. two would be 17k.
EOF

CKPT_LABEL="latest"
if [ -n "$CKPT" ]; then
    # Lightning names checkpoints after the MONITORED METRIC, not the step:
    # epoch=150-val_mos=3.3685.ckpt, epoch=200-val_mel=0.4683.ckpt.
    # An unmatched glob arrives here as a literal string, so check before use --
    # otherwise the export fails on a path that was never a path.
    if [ ! -f "$CKPT" ]; then
        echo "!! no such checkpoint: $CKPT"
        echo
        echo "   available in archive/$RUNKEY:"
        ls -1 "$PIPER_AU_ARCHIVE/$RUNKEY"/*.ckpt 2>/dev/null | sed 's/^/     /' || echo "     (none)"
        echo
        echo "   available in runs/$RUNKEY (top-k retained, live):"
        ls -1 "$RUNDIR"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null |
            sed 's/^/     /' || echo "     (none)"
        exit 1
    fi
    CKPT_LABEL="$(basename "$CKPT" .ckpt | tr '=' '-' | tr -cd 'A-Za-z0-9._-')"
fi

echo "==> exporting $PROFILE ($CKPT_LABEL)"
# Capture rather than discard: a swallowed error message costs more time than
# the noise it saves.
EXPORT_LOG="$(mktemp)"
if [ -n "$CKPT" ]; then
    "$ROOT/deploy/0-export.sh" "$PROFILE" "$CKPT" >"$EXPORT_LOG" 2>&1 || EXPORT_FAILED=1
else
    "$ROOT/deploy/0-export.sh" "$PROFILE" >"$EXPORT_LOG" 2>&1 || EXPORT_FAILED=1
fi
if [ -n "${EXPORT_FAILED:-}" ]; then
    echo "!! export failed:"
    sed 's/^/   /' "$EXPORT_LOG"
    rm -f "$EXPORT_LOG"
    exit 1
fi
rm -f "$EXPORT_LOG"

MODEL="$PIPER_AU_VOICES/$VOICE-medium.onnx"
CONFIG="$MODEL.json"
[ -f "$MODEL" ] || { echo "!! $MODEL not found"; exit 1; }

# Keep a copy per checkpoint, otherwise the next export overwrites the one you
# are trying to compare against.
KEEP="$PIPER_AU_VOICES/$VOICE-medium.$CKPT_LABEL.onnx"
cp "$MODEL" "$KEEP"; cp "$CONFIG" "$KEEP.json"

OUT="$ROOT/out/$CKPT_LABEL"
mkdir -p "$OUT"

activate_venv

# Speaker ids come from the trained config, never from roster order.
IDS=$(python - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
smap = cfg.get("speaker_id_map") or {}
if not smap:
    print("default 0")
else:
    for name, sid in sorted(smap.items(), key=lambda kv: kv[1]):
        print("%s %d" % (name, sid))
PY
)

n=0
while read -r NAME SID; do
    [ -n "$NAME" ] || continue
    i=0
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        i=$((i + 1))
        f="$OUT/${NAME}_$(printf '%02d' $i).wav"
        if [ "$SID" = "0" ] && [ "$NAME" = "default" ]; then
            printf '%s\n' "$line" | python -m piper -m "$MODEL" -f "$f" 2>/dev/null
        else
            printf '%s\n' "$line" | python -m piper -m "$MODEL" -s "$SID" -f "$f" 2>/dev/null
        fi
        [ -s "$f" ] || echo "  !! failed: $f  (check 'python -m piper --help' for the speaker flag)"
    done <<< "$SENTENCES"
    n=$((n + 1))
    echo "  rendered $NAME (id $SID)"
done <<< "$IDS"

echo
echo "==> $n speaker(s) x $(printf '%s\n' "$SENTENCES" | grep -c .) sentences -> $OUT"
echo "    model kept as $KEEP"
echo
# Lightning names checkpoints after the monitored metric, so the filenames are
# epoch=150-val_mos=3.3685.ckpt, not epoch=150-step=NNNNN.ckpt.
echo "Compare two checkpoints by rendering both, then playing the same file from each:"
echo "  ls $PIPER_AU_ARCHIVE/$RUNKEY/ $RUNDIR/lightning_logs/version_*/checkpoints/"
echo "  ./deploy/1-render.sh $PROFILE 'archive/$RUNKEY/epoch=150-val_mos=3.3685.ckpt'"
echo "  ./deploy/1-render.sh $PROFILE"
echo "  # then e.g.  out/epoch-150-val_mos-3.3685/ophelia_darcy_01.wav  vs  out/latest/..."
