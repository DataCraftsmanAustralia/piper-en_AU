# Shared paths and profile definitions for the Australian Piper build.
# Sourced, not run.
#
# Each stage script sits one level below the repo root, so the root is always the
# directory above the script. Nothing here is absolute: the folder can live
# anywhere and be run from anywhere. Point PIPER_AU_DATASET somewhere else if the
# audio is on another disk.

PIPER_AU_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPER_AU_REPO="$PIPER_AU_ROOT/piper1-gpl"
PIPER_AU_VENV="$PIPER_AU_REPO/.venv"
PIPER_AU_DATASET="${PIPER_AU_DATASET:-$PIPER_AU_ROOT/dataset}"
PIPER_AU_RUNS="$PIPER_AU_ROOT/runs"
PIPER_AU_ARCHIVE="$PIPER_AU_ROOT/archive"
PIPER_AU_LOGS="$PIPER_AU_ROOT/logs"
PIPER_AU_VOICES="$PIPER_AU_ROOT/voices"
PIPER_AU_CONFIG="$PIPER_AU_ROOT/config"

# activate_venv -- the environment train/0-setup.sh built.
activate_venv() {
    if [ ! -f "$PIPER_AU_VENV/bin/activate" ]; then
        echo "!! no virtualenv at $PIPER_AU_VENV" >&2
        echo "   build it with ./train/0-setup.sh" >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "$PIPER_AU_VENV/bin/activate"
}

# profile_vars <profile> -- sets CSV, VOICE, SPEAKER (empty when multi-speaker)
# and RUNKEY, the directory name under runs/, cache/ and archive/.
#
#   multi                    the whole roster
#   female                   the female half of the roster
#   <reader label>           any single reader, e.g. `magdalena` or `ophelia_darcy`,
#                            read from dataset/single-speaker-clean/<label>.csv
#   single                   PIPER_AU_SPEAKER, defaulting to ophelia_darcy
#
# A single-speaker profile keys its run dir on the reader rather than on the word
# "single", so training two readers separately gives two runs that cannot see
# each other's checkpoints. Every stage resolves profiles through this one
# function, so a profile cannot mean one thing to training and another to export.
profile_vars() {
    local profile="${1:-}"
    SPEAKER=""
    RUNKEY="$profile"
    CSV=""
    VOICE=""

    case "$profile" in
      multi)
        CSV="$PIPER_AU_DATASET/metadata-clean.csv"
        VOICE="en_AU-librivox"
        ;;
      female)
        CSV="$PIPER_AU_DATASET/metadata-female-clean.csv"
        VOICE="en_AU-librivox-female"
        ;;
      single)
        SPEAKER="${PIPER_AU_SPEAKER:-ophelia_darcy}"
        ;;
      *)
        SPEAKER="$profile"
        ;;
    esac

    if [ -n "$SPEAKER" ]; then
        CSV="$PIPER_AU_DATASET/single-speaker-clean/$SPEAKER.csv"
        VOICE="en_AU-$SPEAKER"
        RUNKEY="$SPEAKER"
    fi

    [ -n "$CSV" ] || return 1
}

# count_speakers <csv> -- piper1-gpl is told how many speakers a model has with
# --model.num_speakers, and the honest answer is how many distinct names appear
# in column 2 of the csv. Deriving it here means editing config/roster.json does
# not also mean editing this repo. A single-speaker csv has no speaker column, so
# this prints 0 for those, which is what train/2-train.sh wants.
count_speakers() {
    if [ ! -f "$1" ]; then
        echo 0
        return 0
    fi
    awk -F'|' 'NF >= 3 && $2 != "" { seen[$2]=1 } END { n=0; for (k in seen) n++; print n }' "$1"
}
# print_usage <script> -- echo a script's own leading comment block, which is
# where every stage documents itself. sed with a fixed line range silently
# truncates as soon as a header grows past it.
print_usage() {
    awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$1"
}
