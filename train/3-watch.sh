#!/usr/bin/env bash
# Status of the Australian Piper training runs.
#
#   ./train/3-watch.sh          one snapshot
#   ./train/3-watch.sh -f       refresh every 10s until Ctrl-C
#   ./train/3-watch.sh -t       tail every log live, in tmux
#
# tqdm redraws its progress bar with carriage returns, so every log read here
# goes through `tr '\r' '\n'` -- without it the last line is unreadable.
set -uo pipefail

# This script lives in train/, so the repo root is one level up.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib/piper-au.sh
. "$ROOT/lib/piper-au.sh"
cd "$ROOT"

last_line() {
    [ -f "$1" ] || { echo "no log yet"; return; }
    tr '\r' '\n' < "$1" | grep -E "Epoch [0-9]+:" | tail -1 | cut -c1-96
}

# Epoch numbers are NOT progress, and the two profiles do not mean the same
# thing by them:
#   ophelia  fine-tunes with --ckpt_path, which RESUMES the base checkpoint's
#            counter, so it starts at 2748 (jenny_dioco's epoch), not 0.
#   multi    warmstarts, so it starts at 0.
# Epoch sizes differ too: ophelia is 53 steps, multi is 245. So this reports
# epochs actually TRAINED -- last epoch seen minus the first epoch seen.
epochs_done() {
    [ -f "$1" ] || { echo "-"; return; }
    local nums first last
    nums=$(tr '\r' '\n' < "$1" | grep -oE "^Epoch [0-9]+:" | grep -oE "[0-9]+")
    [ -n "$nums" ] || { echo "-"; return; }
    first=$(echo "$nums" | head -1)
    last=$(echo "$nums" | tail -1)
    if [ "$first" -gt 0 ]; then
        echo "$((last - first + 1)) (from base epoch $first)"
    else
        echo "$((last - first + 1))"
    fi
}

snapshot() {
    echo "=== $(date '+%H:%M:%S') ==================================================="
    echo
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null |
        awk -F', *' '{printf "  GPU %s  %-44s %3s%% util  %6s/%-6s MiB  %s C\n",$1,substr($2,1,44),$3,$4,$5,$6}'
    echo

    # Every profile that has a run dir, including single/<reader> names and the
    # -bench-<card> dirs (folded back onto the profile they benchmark).
    for prof in $(ls -1 "$PIPER_AU_RUNS" 2>/dev/null |
                  sed "s/-bench-.*//" | sort -u); do
        log="$PIPER_AU_LOGS/$prof.log"
        run="$PIPER_AU_RUNS/$prof"
        [ -f "$log" ] || [ -d "$run" ] || continue

        alive="stopped"
        pgrep -f "voice_name en_AU.*$prof" >/dev/null 2>&1 && alive="RUNNING"
        # the voice_name does not always contain the profile; fall back to tmux
        tmux has-session -t "$prof" 2>/dev/null && alive="RUNNING"

        ckpts=$(ls -1 "$run"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | wc -l)
        printf "  %-9s %-8s  trained: %-24s ckpts:%s\n" \
            "$prof" "$alive" "$(epochs_done "$log")" "$ckpts"
        printf "  %-9s %s\n" "" "$(last_line "$log")"
        echo
    done
    echo "  'trained' is epochs YOU have run. The raw Epoch number in a log is not"
    echo "  comparable between profiles: a single voice fine-tunes with --ckpt_path"
    echo "  and inherits the base checkpoint's counter (2748), while a warmstarted"
    echo "  multi-speaker run starts at 0. Epoch sizes differ too."
    echo
    echo "  attach: tmux attach -t <profile>   (Ctrl-B D to detach)"
    echo "  graphs: tensorboard --logdir $PIPER_AU_RUNS"
}

case "${1:-}" in
    -f) while true; do clear; snapshot; sleep 10; done ;;
    -t)
        # One pane per log that actually exists, rather than two hardcoded names.
        tmux kill-session -t pipwatch 2>/dev/null
        first=1
        for log in "$PIPER_AU_LOGS"/*.log; do
            [ -f "$log" ] || continue
            cmd="tail -f $log | tr '\\r' '\\n'"
            if [ "$first" = 1 ]; then
                tmux new-session -d -s pipwatch "$cmd"
                first=0
            else
                tmux split-window -t pipwatch "$cmd"
            fi
        done
        [ "$first" = 0 ] || { echo "!! nothing in logs/ to tail yet"; exit 1; }
        tmux select-layout -t pipwatch even-vertical
        tmux attach -t pipwatch
        ;;
    *)  snapshot ;;
esac
