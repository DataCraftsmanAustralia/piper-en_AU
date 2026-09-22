# Australian Piper voice

Trains an Australian English [Piper](https://github.com/OHF-Voice/piper1-gpl)
TTS voice from public domain LibriVox recordings: 10 accent-verified narrators,
about 31 hours, 22,050 Hz, one ONNX model with a speaker id per narrator.

Nothing but the scripts is in git. The audio, the manifest, every csv, the
checkpoints and the exported voice are all rebuilt by the stages below, so a
clean clone plus `./build.sh` is the whole model.

## Pretrained model

The shipped voice is published at
[DataCraftsmanAustralia/piper-en_AU-librivox-medium](https://huggingface.co/DataCraftsmanAustralia/piper-en_AU-librivox-medium):
the ONNX and its `.onnx.json`, the `voice_to_speaker.yaml` a serving layer reads,
two ready made `length_scale` variants in `configs/`, and a sample of each of the
ten narrators.

```sh
hf download DataCraftsmanAustralia/piper-en_AU-librivox-medium \
    en_AU-librivox-medium.onnx en_AU-librivox-medium.onnx.json --local-dir voices

echo 'Good on ya, love. The arvo turned out alright in the end.' \
  | python -m piper -m voices/en_AU-librivox-medium.onnx -s 5 -f out.wav
```

`deploy/0-export.sh` writes the model and its config, `deploy/2-voice-map.py
--write` writes the serving yaml, and `--write-rates` writes the `configs/`
variants, so shipping the published voice and shipping one you trained yourself
is the same shape. The speaker ids on the hub are the ones training assigned to
it; your own run will pick its own, so read them out of your config with
`deploy/2-voice-map.py` rather than assuming.

## What you need

| | |
|---|---|
| Linux with an NVIDIA GPU | 24 GB of VRAM trains one voice comfortably. More VRAM buys a bigger batch for the full roster, not a better voice. |
| Python 3.11 or 3.12 | not 3.13+; the stack does not support it |
| `git cmake ninja-build build-essential ffmpeg` | `sudo apt install git cmake ninja-build build-essential ffmpeg` |
| ~60 GB free disk | 14 GB corpus, 4.7 GB audio, phoneme cache, checkpoints |
| tmux | the run has to outlive your ssh session |

Native Windows will not train: the PyPI torch wheel for `win_amd64` is CPU-only,
with every `nvidia-*` dependency gated behind `platform_system == "Linux"`.
Prepare the dataset anywhere with Python; train on Linux.

## Layout

| | |
|---|---|
| `prepare/` | stage 1: fetch the corpus, census the readers, audition them, export audio and csv |
| `train/` | stage 2: build the environment, preflight, train, watch, archive checkpoints |
| `deploy/` | stage 3: checkpoint to ONNX, render clips to judge by ear, speaker map and serving yaml |
| `config/roster.json` | which narrators are in the model |
| `config/voice-names.json` | friendly serving names and speech rates |
| `lib/piper-au.sh` | shared paths and profile definitions, sourced by every stage |
| `dataset/` `runs/` `voices/` `logs/` | built by the stages, not tracked |

## Quick start

```sh
git clone <this repo> && cd piper-en_AU
pip install -r prepare/requirements.txt

./build.sh                        # one reader, best free GPU
./build.sh multi  24              # the whole roster, on a 24 GB card
./build.sh magdalena 96           # any narrator, on a bigger card
```

The second argument is the VRAM size to train on, in GB: the smallest card that
meets it wins, so asking for 24 leaves a bigger card free, and batch size is
worked out from whatever the run lands on. `auto` takes the card with the most
free VRAM.

`build.sh` runs the stages in order, skips any whose output already exists, and
hands you the export command when training stops. Do the audition in stage 1
before trusting the end-to-end build: it is the one step a machine cannot do
for you.

## Stage 1 - prepare the dataset

Runs on any machine with Python; no GPU involved.

```sh
./prepare/0-download.sh                      # ~14 GB of parquet, CC0, no token
python3 prepare/1-census.py                  # hours, clips and transcript
python3 prepare/1-census.py --min-hours 0.1  # quality, per reader

# The ear gate. The corpus is Australian TITLES read by international LibriVox
# volunteers, so an Australian book is no guarantee of an Australian narrator.
# This writes clips plus AUDITION.md; listen, then keep only the real ones in
# config/roster.json. The largest reader in this corpus (11.79 h) was rejected
# here after being the obvious pick on every number available.
python3 prepare/2-export.py --audition --out source-data/_audition

python3 prepare/2-export.py --multi          # wav/ + manifest.jsonl + speakers.json
python3 prepare/3-project.py --suffix=-clean --max-chars 400
```

`--multi` takes about 20 minutes and writes `dataset/` (17,571 wav, 4.7 GB, plus
the manifest). Add `--normalise` to loudness-normalise each clip: LibriVox
narrators record at home and the level drifts between sessions, which comes out
of training as volume that shifts mid-paragraph. Decide before you export,
because it changes the audio.

**Always train on the `-clean` csvs.** `3-project.py` is where filtering
happens, and the raw csvs still contain 117 LibriVox announcement clips that the
upstream dataset's own filter missed. Repeated identical short utterances are
exactly what VITS latches onto. Re-filtering is always a re-projection from
`manifest.jsonl` (seconds), never another decode of the audio:

```sh
python3 prepare/3-project.py --suffix=-punct --require-endpunct      # drop clips with no terminal punctuation
python3 prepare/3-project.py --suffix=-fem --speakers magdalena jenno
python3 prepare/3-project.py --keep-boilerplate                      # put the announcements back
```

## Stage 2 - train

```sh
./train/0-setup.sh                 # clone piper1-gpl, venv, torch 2.8.0+cu128, monotonic_align
source piper1-gpl/.venv/bin/activate
python3 train/1-preflight.py       # GPUs, free VRAM and who holds it, espeak, csv integrity
```

Preflight is not a formality. Read its GPU table: a card that looks idle may be
held by an inference server that pre-allocated its VRAM, and training then dies
with an OOM that reads exactly like a batch-size problem and is not. Stop
whatever is holding the card you want before you start.

```sh
mkdir -p logs
tmux new -s run -d './train/2-train.sh multi 24 2>&1 | tee logs/multi.log'
tmux attach -t run                 # Ctrl-B then D to detach
```

| profile | trains | csv |
|---|---|---|
| `single` | one narrator, named by `PIPER_AU_SPEAKER` (default `ophelia_darcy`) | `dataset/single-speaker-clean/<name>.csv` |
| `female` | the female half of the roster | `dataset/metadata-female-clean.csv` |
| `multi` | the whole roster | `dataset/metadata-clean.csv` |

Run the `single` profile first even if you want the multi-speaker voice. It is
the smallest build, and its real job is proving the toolchain end to end - the
compiled alignment extension, the espeak bridge, the checkpoint load, ONNX
export - on a run where a failure costs an hour instead of a day.

```sh
./train/3-watch.sh -f                          # GPUs, epochs trained, checkpoint counts
./train/4-snapshot.sh multi 25                 # archive checkpoints so pruning cannot delete the good one
tensorboard --logdir runs/ --bind_all
```

It trains until you stop it, which is the correct default: the stopping point is
a judgement about how it sounds. Watch `val_mel` (falls, then flattens) and
`val_mos` (rises, then plateaus, and is noisy: ignore moves under 0.1).
`loss_d` is a health check, not progress - VITS is a GAN, so a healthy run holds
`loss_d` oscillating in a band. Turning TensorBoard's smoothing up to about 0.8
makes the trend visible.

| env | effect |
|---|---|
| `PIPER_AU_EPOCHS=900` | stop at this absolute epoch. Lightning counts from the base checkpoint, so this is not "train N more" |
| `PIPER_AU_BATCH=32` | batch size, which the script otherwise works out from the card's VRAM. Bigger is not better: past about 64 a GAN can lose convergence while gaining throughput |
| `PIPER_AU_WORKERS=16` | loader workers. The upstream default of 1 starves any GPU, and fixing that beats a faster card |
| `PIPER_AU_BENCH=1` | 5-epoch throughput test in its own run dir, to compare cards in ten minutes |

Do not resume a run you had basically finished. Resuming a converged GAN cost
this model 0.9 `val_mos` and nine hours to climb back, with a complete
checkpoint and no bug involved. Cap high with `PIPER_AU_EPOCHS` and stop with
Ctrl-C instead: stopping early is free, extending is expensive.

## Stage 3 - deploy

Pick the checkpoint by ear. `val_mel` saturates early while the GAN losses keep
improving audibility, so the best-scoring checkpoint is not reliably the
best-sounding one. The checkpoint filenames carry their own scores, which makes
`ls runs/multi/lightning_logs/version_*/checkpoints/` the leaderboard.

```sh
./deploy/1-render.sh multi                              # every speaker, six fixed sentences
./deploy/1-render.sh multi 'archive/multi/epoch=544-val_mos=3.7609.ckpt'
./deploy/0-export.sh multi                              # writes voices/ + the serving yaml
```

Then test it and ship both files:

```sh
echo 'Good on ya, love.' | python -m piper -m voices/en_AU-librivox-medium.onnx -s 7 -f /tmp/test.wav
```

The `.onnx.json` is required, and the exporter does not produce it: it is what
training wrote, and for a multi-speaker model it is the only record of the
speaker name to id map. Speaker ids come from that file and nowhere else - piper
assigns them during training, and they do not follow roster order.
`deploy/2-voice-map.py` reads them out and prints `voice_to_speaker.yaml`;
names and rates come from `config/voice-names.json`.

## When it misbehaves

- `pip install piper-tts[train]` cannot train: the wheel ships `monotonic_align`
  with no compiled extension. `train/0-setup.sh` clones and builds it.
- `torchaudio` is required but is not a declared dependency. Without it `val_mos`
  is silently never logged while `ModelCheckpoint` still monitors it, so the run
  dies at the end of epoch 0 with "could not find the monitored key".
- Multi-speaker is `--model.num_speakers`, not `--data.num_speakers`. Speaker
  names go in csv column 2 as strings, not integer ids.
- The espeak voice is `en-gb-x-rp`: espeak-ng has no Australian voice, and
  Australian English is non-rhotic, so RP is the right approximation and `en-us`
  would be actively wrong.
- Never hand-edit a csv, and never validate one by splitting on `|`. piper reads
  metadata with `csv.reader`, whose quotechar is `"`, so one transcript that
  starts with a quote merges every following line into a single 25,000 character
  "utterance" and training then asks for 437 GiB of VRAM. `3-project.py` strips
  quotes and round-trips each file back through `csv.reader`; preflight re-checks
  it.
- Only **medium** checkpoints fine-tune without extra vocoder flags, and medium
  is 22,050 Hz. That is why the audio is resampled from its native 24 kHz.
- `rhasspy/piper` is archived and its separate `preprocess` step is dead, but
  most guides online still use it. piper1-gpl caches phonemes and spectrograms
  inside `fit`.
- Test the finished voice on Linux. `piper.exe` crashes in onnxruntime init on
  Windows 11 build 26200 (piper1-gpl #260, open).

## More

- `docs/notes.md` - measured VRAM and throughput, the TensorBoard detail, and the
  list of traps that cost hours to find.
- `docs/model-card.md` - the model card, which is also the README of the hub repo.
- `ATTRIBUTION.md` - provenance, the narrators, licensing and limitations.
