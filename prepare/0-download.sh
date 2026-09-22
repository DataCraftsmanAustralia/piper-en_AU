#!/usr/bin/env bash
# Stage 1, step 0: fetch the source corpus (~14 GB of parquet shards).
#
#   ./prepare/0-download.sh [destination]
#
# The corpus is ablmontazer/australian-english-speech: 61,662 clips of public
# domain LibriVox audiobook reading, CC0, no token needed. Everything the model
# is built from comes out of these shards, so this is the only download the
# pipeline needs.
#
# Needs the Hugging Face CLI:  pip install -U huggingface_hub
# The steps after this one also need pyarrow and ffmpeg:
#   pip install -r prepare/requirements.txt   and   apt install ffmpeg
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="ablmontazer/australian-english-speech"
DEST="${1:-$ROOT/source-data/australian-english-speech}"

if command -v hf >/dev/null 2>&1; then
    DL=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
    DL=(huggingface-cli download)
else
    echo "!! neither 'hf' nor 'huggingface-cli' is on the PATH"
    echo "   pip install -U huggingface_hub"
    exit 1
fi

command -v ffmpeg >/dev/null 2>&1 ||
    echo " note: ffmpeg not found. The download works, but prepare/2-export.py" \
         " needs it to decode the audio (apt install ffmpeg)."
python3 -c 'import pyarrow' 2>/dev/null ||
    echo " note: pyarrow not found. pip install -r prepare/requirements.txt"

echo "==> $REPO -> $DEST"
"${DL[@]}" "$REPO" --repo-type dataset --local-dir "$DEST"

DATA="$DEST/data"
[ -d "$DATA" ] || DATA="$DEST"
echo
echo "==> shards in $DATA"
ls -1 "$DATA"/*.parquet 2>/dev/null | sed 's/^/     /' || \
    echo "   no .parquet found there -- check the layout before continuing"

echo
echo "Next:"
echo "  export AU_SPEECH_DATA=$DATA"
echo "  python3 prepare/1-census.py          # who is in it, and how much audio"
