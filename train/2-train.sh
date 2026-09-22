#!/usr/bin/env bash
# Train an Australian Piper voice.
#
#   ./train/2-train.sh <profile> [card]
#
# profile:
#   single      one reader from config/roster.json, named by PIPER_AU_SPEAKER
#               (default ophelia_darcy, ~1,900 clips / 3.2 h). RUN THIS FIRST:
#               it is the smallest build and its job is proving the toolchain
#               end to end, where a failure costs an hour instead of a day.
#   female      the female half of the roster (~10,100 clips / 17 h)
#   multi       the whole roster (~17,450 clips / 31 h)
#
# Any narrator label is a profile too, so `magdalena` trains that reader alone;
# `single` is the same thing with PIPER_AU_SPEAKER naming the reader. Each one
# keys its own runs/<label>/, cache/<label>/ and archive/<label/>.
#
# card:  a VRAM capacity in GB, never a model name
#   24          a card with at least 24 GB of VRAM -- the SMALLEST one that
#               qualifies, so asking for 24 does not tie up the big card
#   96          a card with at least 96 GB
#   auto        whichever card has the most free VRAM (default)
#   gpu:<n>     an explicit PCI index, for when you already know the box
#
# Batch and worker counts are derived from the capacity of whatever card the
# run lands on, so nothing in here needs editing when the hardware changes.
# The curve it derives from is measured in docs/notes.md.
#
# Cards are resolved out of nvidia-smi rather than by index. CUDA's default
# device order is fastest-first while nvidia-smi and docker order by PCI bus,
# so index numbers disagree between tools and are not safe to hardcode.
#
# Resumes automatically if the profile already has a last.ckpt.
# Stop it yourself when the voice sounds right -- max_epochs defaults to -1
# (forever) upstream and this script does not change that.
#
# Env overrides (win over the card defaults):
#   PIPER_AU_BATCH     batch size
#   PIPER_AU_WORKERS   dataloader workers. piper1-gpl ships a default of 1,
#                      which starves the GPU on any card -- usually a bigger
#                      win than a faster GPU.
#   PIPER_AU_EPOCHS    stop at this ABSOLUTE epoch number, not "N more". The
#                      script works out where the run starts and refuses early
#                      if the target is already behind it. Use this to cap an
#                      unattended run so it does not train for days.
#   PIPER_AU_BENCH=1   benchmark mode: warmstart (fresh epoch counter), a
#                      separate runs/<runkey>-bench-<card>/ dir so the real run
#                      cannot resume from throwaway weights, and 5 epochs unless
#                      PIPER_AU_EPOCHS says otherwise.
#   PIPER_AU_DATASET   default ./dataset
#   PIPER_AU_SPEAKER   which reader the single profile trains (ophelia_darcy)
#   PIPER_AU_BASE_CKPT default en_GB jenny_dioco medium
set -euo pipefail

# This script lives in train/, so the repo root is one level up.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib/piper-au.sh
. "$ROOT/lib/piper-au.sh"

PROFILE="${1:-}"
CARD="${2:-auto}"

# Makes torch agree with nvidia-smi and docker.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# --- profile ---------------------------------------------------------------
# CSV and VOICE come from lib/piper-au.sh, the same source the export and render
# steps read, so a profile cannot mean one thing to training and another to
# deployment.
profile_vars "$PROFILE" || { print_usage "$0"; exit 2; }
DATASET="$PIPER_AU_DATASET"

# --- card ------------------------------------------------------------------
gpu_table() { nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits 2>/dev/null; }

# resolve_by_vram <GB> -- the smallest card at or above the requested capacity,
# so asking for 24 GB leaves a bigger card free for somebody else. The 5% slack
# is for cards that report a hair under their nominal size.
resolve_by_vram() {
    gpu_table | awk -F', *' -v want="$1" '
        $4 >= want * 1024 * 0.95 { if (best == "" || $4 < best) { best = $4; hit = $1 } }
        END { if (hit != "") print hit }'
}
emptiest() {
    gpu_table | sort -t, -k3 -n -r | head -1 | awk -F', *' '{print $1}'
}
gpu_field() {  # gpu_field <index> <1..4>
    gpu_table | awk -F', *' -v i="$1" -v f="$2" '$1 == i {print $f; exit}'
}

command -v nvidia-smi >/dev/null 2>&1 || { echo "!! nvidia-smi not found"; exit 1; }

# The measured VRAM curve, docs/notes.md: about 535 MiB per unit of batch plus
# about 2 GB fixed. Used both to size the default batch and to check that the
# run will actually fit in what is free.
VRAM_PER_UNIT=535
VRAM_FIXED=2000

# default_batch <MiB total> -- the batch this card can hold, from its capacity.
# Three quarters of the card, rounded to a multiple of 8, never above 64. Not
# maxed on purpose: VITS is a GAN, and past a point a larger batch costs
# convergence rather than buying speed, so the room on a big card is better kept
# as headroom than spent. On the small single-reader profiles a huge batch also
# means very few optimiser steps per epoch. Raise it with PIPER_AU_BATCH if the
# loss curves stay healthy.
default_batch() {
    local fit=$(( ($1 * 3 / 4 - VRAM_FIXED) / VRAM_PER_UNIT ))
    fit=$(( (fit + 4) / 8 * 8 ))
    [ "$fit" -gt 64 ] && fit=64
    [ "$fit" -lt 8 ] && fit=8
    echo "$fit"
}

case "$CARD" in
  auto)         GPU="$(emptiest)" ;;
  gpu:*)        GPU="${CARD#gpu:}" ;;
  ''|*[!0-9]*)  GPU="" ;;
  *)            GPU="$(resolve_by_vram "$CARD")" ;;
esac

case "${GPU:-}" in
  ''|*[!0-9]*)
    echo "!! no card picked from '$CARD'. Give a VRAM size in GB (24, 48, 96 ...),"
    echo "   'auto', or gpu:<index>. Cards on this box (index, name, free, total):"
    gpu_table | sed 's/^/     /'
    exit 2 ;;
esac

GPU_NAME="$(gpu_field "$GPU" 2)"
GPU_FREE="$(gpu_field "$GPU" 3)"
GPU_TOTAL="$(gpu_field "$GPU" 4)"

case "${GPU_TOTAL:-}" in
    ''|*[!0-9]*) echo "!! nvidia-smi gave no memory total for GPU $GPU"; exit 1 ;;
esac

# Both defaults come out of the card this run actually landed on, which is what
# makes the same script behave sensibly on a 24 GB card and a 96 GB one.
DEF_BATCH="$(default_batch "$GPU_TOTAL")"
DEF_WORKERS=$(( DEF_BATCH / 4 ))
[ "$DEF_WORKERS" -gt 12 ] && DEF_WORKERS=12

BATCH="${PIPER_AU_BATCH:-$DEF_BATCH}"
WORKERS="${PIPER_AU_WORKERS:-$DEF_WORKERS}"

export CUDA_VISIBLE_DEVICES="$GPU"

# --- is it actually free? --------------------------------------------------
# MEASURED 2026-09-18 over full epochs: batch 24 -> 14.5 GB and batch 32 ->
# 18.4 GB on a 24 GB card, batch 64 -> 35.1 GB on a 96 GB card. Solving the last
# two gives ~535 MiB per unit of batch plus ~1.7 GB fixed; 2 GB fixed is used
# here so the estimate stays a shade conservative rather than optimistic. vLLM
# pre-allocates its KV cache, so an idle-looking lane still owns the card and
# will not yield.
NEED=$(( BATCH * VRAM_PER_UNIT + VRAM_FIXED ))

# What would fit instead, so the advice is a number and not a guess.
SUGGEST=$(( (${GPU_FREE:-0} - VRAM_FIXED) / VRAM_PER_UNIT / 8 * 8 ))
[ "$SUGGEST" -lt 8 ] && SUGGEST=8

if [ "${GPU_FREE:-0}" -lt "$NEED" ]; then
    cat <<EOF
!! GPU $GPU ($GPU_NAME) has ${GPU_FREE} MiB free; this run wants about ${NEED} MiB.

   Holding it now:
$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory,process_name --format=csv,noheader 2>/dev/null | sed 's/^/     /')

   All cards:
$(gpu_table | sed 's/^/     /')

   Inference servers pre-allocate their VRAM, so a lane at low utilisation
   still owns the card. Stop the containers holding it:
     docker compose stop <inference-service>

   Restart afterwards with the matching 'docker compose start'.
   Or size the run to what is free:  PIPER_AU_BATCH=${SUGGEST} $0 $PROFILE $CARD
EOF
    exit 1
fi

# --- paths -----------------------------------------------------------------
# A benchmark run gets its own run dir. Otherwise the real run would later find
# the benchmark's last.ckpt and "resume" from throwaway weights. The cache is
# shared deliberately -- it is derived data, not training state, so the second
# card does not pay to rebuild it.
BENCH=0
[ -n "${PIPER_AU_BENCH:-}" ] && BENCH=1
# A benchmark is short by definition; default it so PIPER_AU_BENCH=1 alone works.
[ "$BENCH" = "1" ] && PIPER_AU_EPOCHS="${PIPER_AU_EPOCHS:-5}"

if [ "$BENCH" = "1" ]; then
    RUNDIR="$PIPER_AU_RUNS/$RUNKEY-bench-$CARD"
else
    RUNDIR="$PIPER_AU_RUNS/$RUNKEY"
fi
CACHE="$ROOT/cache/$RUNKEY"
CONFIG="$RUNDIR/$VOICE.onnx.json"

[ -f "$CSV" ] || {
    echo "!! csv not found: $CSV"
    echo "   Build the dataset first (this takes ~20 min for the audio):"
    echo "     ./prepare/0-download.sh"
    echo "     python3 prepare/2-export.py --multi"
    echo "     python3 prepare/3-project.py --suffix=-clean --max-chars 400"
    echo "   Or point at a dataset that already exists: PIPER_AU_DATASET=/path"
    if [ -d "$PIPER_AU_DATASET/single-speaker-clean" ]; then
        echo "   Readers available for the single profile:"
        ls -1 "$PIPER_AU_DATASET/single-speaker-clean" 2>/dev/null |
            sed 's/\.csv$//;s/^/     /'
    fi
    exit 1
}
[ -d "$DATASET/wav" ] || { echo "!! wav dir not found: $DATASET/wav"; exit 1; }

# Counted out of the csv instead of being written down here, so editing
# config/roster.json does not require editing this script to match. Single
# speaker csvs have no speaker column, which is what makes them single speaker.
NUM_SPEAKERS="$(count_speakers "$CSV")"

activate_venv
mkdir -p "$RUNDIR" "$CACHE"

# --- resume or start -------------------------------------------------------
# Lightning resumes from --ckpt_path. If a run already exists, point at its own
# last.ckpt and drop the base checkpoint; mixing the two silently restarts from
# the base and throws the run away.
# --ckpt_path is Lightning's RESUME: it restores the epoch and step counters
# along with the weights. The jenny_dioco base sits at epoch 2748, so any
# --trainer.max_epochs below that means "stop in the past" and Lightning raises
# MisconfigurationException before a single batch runs. A benchmark therefore
# always starts fresh via warmstart -- which is also what makes the two cards
# comparable, since both get a fresh optimizer.
if [ "$BENCH" = "1" ]; then
    LAST=""
else
    LAST="$(ls -t "$RUNDIR"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)"
fi
BASE_CKPT_URL="${PIPER_AU_BASE_CKPT:-https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_GB/jenny_dioco/medium/epoch%3D2748-step%3D1729300.ckpt}"

RESUME_ARGS=()
if [ -n "$LAST" ]; then
    echo "==> resuming from $LAST"
    RESUME_ARGS=(--ckpt_path "$LAST")
elif [ "$BENCH" = "1" ] || [ "$NUM_SPEAKERS" -gt 0 ]; then
    # warmstart does load_state_dict(strict=False): copies every parameter whose
    # shape matches, fresh optimizer, epoch counter at 0.
    #
    # Multi-speaker needs it because the single-speaker base has no speaker
    # embedding layer, so a strict load fails on shape mismatch -- warmstart
    # initialises emb_g fresh instead. Benchmarking needs it because the epoch
    # counter has to start at 0 for --trainer.max_epochs to mean anything.
    # It also skips the jsonargparse hparam re-validation behind the known
    # --ckpt_path failures (piper1-gpl #76, #81, #132).
    # base-ckpt/ is the only path shared between concurrently running profiles,
    # so the download is made safe for that: a PID-unique temp file, an atomic
    # rename, and a re-check afterwards in case another run won the race.
    mkdir -p "$ROOT/base-ckpt"
    LOCAL_CKPT="$ROOT/base-ckpt/en_GB-jenny_dioco-medium.ckpt"
    if [ ! -f "$LOCAL_CKPT" ]; then
        echo "==> downloading base checkpoint (~846 MB)"
        TMP_CKPT="$LOCAL_CKPT.$$.part"
        curl -L --fail --progress-bar -o "$TMP_CKPT" "$BASE_CKPT_URL"
        if [ -f "$LOCAL_CKPT" ]; then
            echo "    another run finished it first; discarding this copy"
            rm -f "$TMP_CKPT"
        else
            mv -n "$TMP_CKPT" "$LOCAL_CKPT" || rm -f "$TMP_CKPT"
        fi
    fi
    if [ "$BENCH" = "1" ] && [ "$NUM_SPEAKERS" -eq 0 ]; then
        echo "==> BENCHMARK MODE: warmstart (fresh epoch counter) so max_epochs applies."
        echo "    Throughput is representative; the resulting weights are not a"
        echo "    real fine-tune. Delete runs/$RUNKEY before the actual run."
    fi
    echo "==> warmstart from $LOCAL_CKPT"
    RESUME_ARGS=(--model.warmstart_ckpt "$LOCAL_CKPT")
else
    # Single speaker: the documented path. Pass the URL rather than a local file
    # -- Lightning's remote loader upgrades the old checkpoint format on the way
    # in, and the same file downloaded locally often fails where the URL works.
    echo "==> fine-tuning from the en_GB jenny_dioco medium checkpoint (URL)"
    RESUME_ARGS=(--ckpt_path "$BASE_CKPT_URL")
fi

SPEAKER_ARGS=()
[ "$NUM_SPEAKERS" -gt 0 ] && SPEAKER_ARGS=(--model.num_speakers "$NUM_SPEAKERS")

EPOCH_ARGS=()
if [ -n "${PIPER_AU_EPOCHS:-}" ]; then
    # max_epochs is an ABSOLUTE target, not "train N more". Lightning refuses to
    # start if it is at or below the epoch a checkpoint restored -- which is how
    # the jenny_dioco base (epoch 2748) produced
    #   MisconfigurationException: You restored a checkpoint with
    #   current_epoch=2748, but you have set Trainer(max_epochs=5)
    # Work out the floor and refuse early with a useful message instead.
    FLOOR=0
    if [ -n "$LAST" ]; then
        FLOOR=$(ls "$RUNDIR"/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null |
                sed -n 's/.*epoch=\([0-9]\+\).*/\1/p' | sort -n | tail -1)
        FLOOR=${FLOOR:-0}
    elif [ "$NUM_SPEAKERS" -eq 0 ] && [ "$BENCH" != "1" ]; then
        FLOOR=2748   # the en_GB jenny_dioco base this profile fine-tunes from
    fi
    if [ "$PIPER_AU_EPOCHS" -le "$FLOOR" ]; then
        echo "!! PIPER_AU_EPOCHS=$PIPER_AU_EPOCHS but this run starts at epoch $FLOOR."
        echo "   max_epochs is an absolute target, not 'train N more', and Lightning"
        echo "   refuses to start when it is already behind. Set it above $FLOOR"
        echo "   (e.g. $((FLOOR + 200)) to train 200 more), or unset it to run until"
        echo "   you stop it."
        exit 2
    fi
    echo "==> will stop at epoch $PIPER_AU_EPOCHS (currently $FLOOR)"
    EPOCH_ARGS=(--trainer.max_epochs "$PIPER_AU_EPOCHS")
fi

cat <<EOF

profile      $PROFILE${SPEAKER:+  (speaker $SPEAKER)}
card         $GPU  $GPU_NAME  $(( GPU_TOTAL / 1024 )) GB  (${GPU_FREE} of ${GPU_TOTAL} MiB free)
csv          $CSV  ($(wc -l < "$CSV") rows)
speakers     $NUM_SPEAKERS
batch        $BATCH  (sized from the card's capacity; PIPER_AU_BATCH wins)
workers      $WORKERS
run dir      $RUNDIR
config out   $CONFIG

EOF

# --data.espeak_voice: espeak-ng has no Australian voice. Australian English is
# non-rhotic, so RP is the right approximation -- en-us would be actively wrong.
set -x
python -m piper.train fit \
    --data.voice_name "$VOICE" \
    --data.csv_path "$CSV" \
    --data.audio_dir "$DATASET/wav/" \
    --data.espeak_voice "en-gb-x-rp" \
    --data.cache_dir "$CACHE" \
    --data.config_path "$CONFIG" \
    --data.batch_size "$BATCH" \
    --data.num_workers "$WORKERS" \
    --model.sample_rate 22050 \
    --trainer.accelerator gpu \
    --trainer.devices 1 \
    --trainer.precision 16-mixed \
    --trainer.default_root_dir "$RUNDIR" \
    "${SPEAKER_ARGS[@]}" \
    "${EPOCH_ARGS[@]}" \
    "${RESUME_ARGS[@]}"
set +x

echo
echo "==> training stopped. Export with:"
echo "    ./deploy/0-export.sh $PROFILE"
