#!/usr/bin/env bash
# Archive training checkpoints so they cannot be pruned.
#
#   ./train/4-snapshot.sh multi              every 25 epochs, check every 10 min
#   ./train/4-snapshot.sh multi 10           every 10 epochs
#   ./train/4-snapshot.sh multi 25 300       every 25 epochs, check every 5 min
#
# Why this exists: piper's default callbacks keep save_top_k=5 on val_mel (min)
# and save_top_k=5 on val_mos (max). "Top 5" means older checkpoints are DELETED
# as better-scoring ones appear. Over a long run that is fine while the metric
# tracks quality -- and a problem when it stops, which piper's own code warns
# about (val_mel saturates before audible quality does). The checkpoint that
# sounded best can be pruned because a later one scored better.
#
# Copies are never deleted by this script. ~850 MB each; at every 25 epochs over
# a 540-epoch run that is ~18 GB.
set -uo pipefail

# This script lives in train/, so the repo root is one level up.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib/piper-au.sh
. "$ROOT/lib/piper-au.sh"
cd "$ROOT"

PROFILE="${1:-}"
STRIDE="${2:-25}"
INTERVAL="${3:-600}"

[ -n "$PROFILE" ] || { print_usage "$0"; exit 2; }
profile_vars "$PROFILE" || { echo "!! unknown profile '$PROFILE'"; exit 2; }

RUNDIR="$PIPER_AU_RUNS/$RUNKEY"
ARCHIVE="$PIPER_AU_ARCHIVE/$RUNKEY"
mkdir -p "$ARCHIVE"

echo "archiving every $STRIDE epochs from $RUNDIR -> $ARCHIVE (checking every ${INTERVAL}s)"
echo "Ctrl-C to stop. Copies here are never pruned."
echo

# Highest epoch already archived, so a restart of this script picks up correctly.
LAST_ARCHIVED=-1
for f in "$ARCHIVE"/epoch=*.ckpt; do
    [ -f "$f" ] || continue
    e="$(basename "$f" | sed -n 's/^epoch=\([0-9]\+\).*/\1/p')"
    [ -n "$e" ] && [ $((10#$e)) -gt "$LAST_ARCHIVED" ] && LAST_ARCHIVED=$((10#$e))
done
[ "$LAST_ARCHIVED" -ge 0 ] && echo "resuming after epoch $LAST_ARCHIVED"

while true; do
    # Take the NEWEST checkpoint on disk once we are STRIDE epochs past the last
    # one archived -- rather than insisting on exact multiples of STRIDE.
    #
    # Lightning only WRITES a checkpoint when it beats the current top-k. Early
    # in training the metrics improve most epochs, so multiples of STRIDE get
    # written and this worked. Once the run converges and the metric fluctuates
    # on a plateau, only occasional epochs beat the top-k and those epoch
    # numbers are arbitrary -- version_8 finished holding epochs 224, 254, 266,
    # 272, 274, 278, 291, 293, 294 and 299, NOT ONE of them a multiple of 25.
    # The old rule was asking for files that were never written; it silently
    # archived nothing from epoch 350 onward. Taking whatever is newest cannot
    # have that failure.
    best_ck=""; best_epoch=-1
    for ck in "$RUNDIR"/lightning_logs/version_*/checkpoints/epoch=*.ckpt; do
        [ -f "$ck" ] || continue
        e="$(basename "$ck" | sed -n 's/^epoch=\([0-9]\+\).*/\1/p')"
        [ -n "$e" ] || continue
        # 10# forces base 10 -- otherwise 0042 is octal and 0008 errors.
        if [ $((10#$e)) -gt "$best_epoch" ]; then
            best_epoch=$((10#$e)); best_ck="$ck"
        fi
    done

    if [ -n "$best_ck" ] && [ "$best_epoch" -ge $((LAST_ARCHIVED + STRIDE)) ]; then
        base="$(basename "$best_ck")"
        dest="$ARCHIVE/$base"
        if [ ! -f "$dest" ]; then
            # Copy to a temp name first: the trainer may still be writing this
            # file, and a half-copied checkpoint that looks complete is worse
            # than no copy at all.
            if cp "$best_ck" "$dest.part" 2>/dev/null && mv "$dest.part" "$dest"; then
                echo "$(date '+%H:%M:%S')  archived $base ($(du -h "$dest" | cut -f1))"
                LAST_ARCHIVED=$best_epoch
            else
                rm -f "$dest.part"
            fi
        else
            LAST_ARCHIVED=$best_epoch
        fi
    fi
    sleep "$INTERVAL"
done
