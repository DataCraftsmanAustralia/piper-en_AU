#!/usr/bin/env bash
# Everything from a clean clone to a training run, in one command.
#
#   ./build.sh                       # one reader (PIPER_AU_SPEAKER), best free GPU
#   ./build.sh magdalena 24          # a named reader, on a 24 GB card
#   ./build.sh multi   96            # the whole roster, on a bigger card
#   ./build.sh multi --force         # rebuild the dataset even if it exists
#
# It runs the stages in order and skips any whose output is already there:
#
#   1  prepare/0-download.sh    corpus parquet   (~14 GB,  one-off)
#   2  prepare/2-export.py      wav + manifest   (~20 min, one-off)
#   3  prepare/3-project.py     the training csvs (seconds, always re-run)
#   4  train/0-setup.sh         piper1-gpl + venv (one-off)
#   5  train/1-preflight.py     GPU, espeak, csv integrity
#   6  train/2-train.sh         trains until YOU stop it
#
# Step 6 is exec'd, so this script is not in the signal path: Ctrl-C stops
# training cleanly, and nothing between you and tmux. Export afterwards with
# ./deploy/0-export.sh -- the script prints the command when the run stops.
#
# PIPER_AU_EPOCHS caps an unattended run. PIPER_AU_NORMALISE=1 loudness-normalises
# the clips, which fixes the volume drift LibriVox home recordings have; it
# changes the audio, so it has to be decided before step 2, not after.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/piper-au.sh
. "$ROOT/lib/piper-au.sh"

FORCE=0
ARGS=()
for a in "$@"; do
    [ "$a" = "--force" ] && FORCE=1 || ARGS+=("$a")
done
PROFILE="${ARGS[0]:-single}"
CARD="${ARGS[1]:-auto}"

profile_vars "$PROFILE" || {
    echo "!! unknown profile '$PROFILE'"
    echo "   use multi, female, or a reader label from dataset/single-speaker-clean/"
    exit 2
}

step() { printf '\n=== %s ===\n' "$*"; }
have() { [ -e "$1" ]; }
fresh() { [ "$FORCE" = 1 ] || ! have "$1"; }

# --- 1. corpus -------------------------------------------------------------
SHARDS="$ROOT/source-data/australian-english-speech/data"
if ! ls "$SHARDS"/train-*.parquet >/dev/null 2>&1 &&
   ! ls "$ROOT"/source-data/australian-english-speech/train-*.parquet >/dev/null 2>&1; then
    step "1/6 download the corpus"
    ./prepare/0-download.sh
else
    step "1/6 corpus already downloaded (pass --force to refetch)"
fi

# --- 2. audio + manifest ---------------------------------------------------
if fresh "$PIPER_AU_DATASET/manifest.jsonl"; then
    step "2/6 export the audio and manifest (~20 min)"
    # --out writes manifest.jsonl, metadata.csv and speakers.json together; the
    # roster in config/roster.json decides who is in them.
    python3 prepare/2-export.py --multi ${PIPER_AU_NORMALISE:+--normalise}
else
    step "2/6 dataset/manifest.jsonl exists (--force to rebuild the audio)"
fi

# --- 3. training csvs ------------------------------------------------------
# Cheap, so always re-run: this is what picks up a roster or filter change.
step "3/6 project the training csvs"
python3 prepare/3-project.py --suffix=-clean --max-chars 400

# --- 4. training environment ----------------------------------------------
if [ ! -f "$PIPER_AU_VENV/bin/activate" ] || [ "$FORCE" = 1 ] && [ "${PIPER_AU_REBUILD_VENV:-}" = "1" ]; then
    step "4/6 build the piper1-gpl environment"
    ./train/0-setup.sh
else
    step "4/6 environment already built"
fi
activate_venv

# --- 5. preflight ----------------------------------------------------------
# --quick skips the per-file wav scan; a full scan costs a couple of minutes on
# 17,571 files and the training csvs were just written from the manifest.
step "5/6 preflight"
python3 "$ROOT/train/1-preflight.py" --quick

# --- 6. train --------------------------------------------------------------
mkdir -p "$PIPER_AU_LOGS"
step "6/6 training $PROFILE on '$CARD' (Ctrl-C to stop, it is safe)"
echo "    log:   $PIPER_AU_LOGS/$RUNKEY.log"
echo "    watch: ./train/3-watch.sh -f          (from another shell)"
echo "    graphs: tensorboard --logdir $PIPER_AU_RUNS"
# pipefail is on and Ctrl-C makes the trainer exit 130, so take the status from
# the pipeline rather than letting set -e skip the steps after it.
set +e
./train/2-train.sh "$PROFILE" "$CARD" 2>&1 | tee "$PIPER_AU_LOGS/$RUNKEY.log"
STATUS="${PIPESTATUS[0]}"
set -e

echo
[ "$STATUS" = 0 ] || echo "!! trainer exited $STATUS -- the newest checkpoint is still usable"
echo "==> listen before trusting any score:   ./deploy/1-render.sh $PROFILE"
echo "==> then export the voice:              ./deploy/0-export.sh $PROFILE"
