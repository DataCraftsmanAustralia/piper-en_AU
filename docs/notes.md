# Notes from building this voice

The parts of the README that are worth knowing but not worth reading before your
first run: measured numbers, the training-graph detail, and the traps that cost
hours to find.

## GPUs

**Device numbering disagrees between tools.** CUDA's default order is
fastest-first, not PCI-bus order, so on a mixed box torch called the bigger card
device 0 while docker compose gave the same card `--device-ids 1`. Everything
here sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` so torch, `nvidia-smi` and docker agree.
`train/2-train.sh` asks `nvidia-smi` for a card by *VRAM capacity*, so neither an
index nor a model name is hardcoded anywhere.

**A GPU that looks idle may not be.** vLLM and similar inference servers
pre-allocate their KV cache, so a lane at low utilisation still owns the card and
will not yield under pressure. `train/1-preflight.py` prints every card with its
free VRAM and the processes holding it; read that before blaming the batch size.

### Measured, 2026-09-18

`ophelia` profile, 1,710 training clips, 5 epochs each:

| card | batch | workers | steps/epoch | s/epoch | samples/s | steps/s | GPU util | VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 GB | 32 | 8 | 53 | 37 | 45.8 | 1.43 | 77% | 18.4 GB |
| 96 GB | 64 | 12 | 26 | 22 | 75.6 | 1.18 | 61% | 35.1 GB |

The 96 GB card moved 1.65x the audio per second but performed 0.83x the optimiser
steps: doubling the batch cost only 21% more per-step time, so it processes more
samples while updating the weights less often. To a fixed 50,000 steps that is
9.7 h on the 24 GB card and 11.8 h on the 96 GB one.

The caveat is that the table varies two things at once, card and batch, so it
measures "card + batch". `PIPER_AU_BATCH=32 PIPER_AU_EPOCHS=5 ./train/2-train.sh single 96`
is the fair comparison.

The 96 GB card could not be saturated: 61% utilisation against the 24 GB card's
77%, with aggregate CPU never above 15%. So it is not dataloader starvation and more
workers will not help. Either a single-threaded section serialises each step (the
monotonic alignment search is CPU-side Cython; one pinned core is about 3% of a
32-thread box, so low aggregate CPU does not rule this out), or a 70M parameter
model is simply too small to keep a card that size busy and the step is
launch-bound. `mpstat -P ALL 1` during a run says which: one core at 100% is
serialisation, all cores low means the model is too small.

**Split that followed:** the single voice on the 24 GB card, which is ahead on
optimiser steps per hour and that is what matters on a 1,900 clip set. `multi` on
the 96 GB card, where 15,709 clips per epoch make raw sample throughput the
binding constraint instead.

Settle it on your own box in ten minutes rather than arguing from spec sheets:

```sh
PIPER_AU_BENCH=1 ./train/2-train.sh single 24
PIPER_AU_BENCH=1 ./train/2-train.sh single 96
```

Benchmark mode warmstarts instead of resuming, because `--ckpt_path` restores the
base checkpoint's epoch counter (2748) and any `max_epochs` below that makes
Lightning raise `MisconfigurationException` before the first batch. It also writes
to `runs/<runkey>-bench-<card>/` so the real run cannot later "resume" from
throwaway weights. The phoneme cache is shared, since it is derived data.
Benchmark weights are not a fine-tune; delete the `-bench-*` dirs.

### VRAM

Measured over full epochs: batch 24 to 14.5 GB and batch 32 to 18.4 GB on a 24 GB
card, batch 64 to 35.1 GB on a 96 GB card. The last two solve to about 535 MiB per
unit of batch plus 1.7 GB fixed; `train/2-train.sh` assumes 2 GB fixed so the
estimate stays conservative, and refuses a run that will not fit.

That same curve also picks the default batch: three quarters of whatever card the
run lands on, rounded to a multiple of 8 and capped at 64, which is how the
defaults ended up equal to the two measured rows above. Ask for a capacity the box
has and you get the tuned number without writing it down anywhere.

| batch | predicted | 24 GB card | 96 GB card |
|---:|---:|:--|:--|
| 24 | 12.8 GB | fits | fits |
| 32 | 18.7 GB | default | fits |
| 48 | 27.1 GB | no | fits |
| 64 | 35.4 GB | no | default |
| 96 | 52.2 GB | no | fits |
| 128 | 61.1 GB | no | fits |

### Running two at once

Supported, and a good use of a box. Every path that matters is keyed by run, so
`runs/<runkey>/`, `cache/<runkey>/` and the config never collide. `base-ckpt/` is
the only shared directory and its download is race-safe (PID-unique temp file,
atomic rename, re-check).

```sh
tmux new -s ophelia -d './train/2-train.sh ophelia_darcy 24 2>&1 | tee logs/ophelia_darcy.log'
tmux new -s multi   -d './train/2-train.sh multi        96 2>&1 | tee logs/multi.log'
```

Headroom with both running: about 19 GB of 24 on the small card, about 36 GB of 96
on the big one, some
20 loader processes and about 30% CPU across 32 threads. The only contention is
at the start, while `multi` builds its cache for 17,454 utterances. That is CPU
and disk heavy and slows the other run while it lasts, which is once.

Budget about 10 GB of checkpoints per run (Lightning keeps up to 11 at ~850 MB)
plus the cache, which is roughly the size of the audio.

## Reading TensorBoard

The metric names are piper1-gpl's own, not the old rhasspy ones. There is no
`loss_disc_all` here. The set is `loss_g`, `loss_d`, `val_loss` and
`{train,val}_{mel,kl,dur,fm,gen,disc}`, plus `val_mos`.

| graph | want | what it means |
|---|---|---|
| `val_mel` | falls, then flattens | Mel reconstruction on held-out clips, the closest thing to "does it sound like her". Main progress signal. |
| `val_mos` | rises, then plateaus | UTMOS predicted opinion score, 1-5. The perceptual proxy and one of the two metrics checkpointing keeps. Noisy, about +/-0.08 run to run, so ignore moves under 0.1. |
| `loss_d` | oscillates in a band | Health check, not progress. |
| `val_dur` | falls | Duration predictor: rhythm and pacing. Matters more than usual here. |

`loss_d` is not supposed to fall. Generator and discriminator are adversaries and
a healthy run holds them in tension, so you are watching for failure: `loss_d`
toward 0 means the discriminator won and the audio stops improving or degrades;
`loss_d` climbing away means the generator won, usually with artefacts.

`val_dur` matters extra here because espeak drives prosody off punctuation and
these transcripts are inconsistently punctuated (Ophelia 80%, Timothy 21%). If
`val_dur` refuses to come down, that is the punctuation problem surfacing, and
the `--require-endpunct` re-projection becomes worth trying.

`val_kl` and `val_fm` broadly track `val_mel`; no need to watch them separately
unless something looks wrong.

Two practical notes. Turn smoothing up to about 0.8, because GAN losses are noisy
per step and the trend is invisible at the default. And two runs of different
profiles do not share an x-axis: a fine-tune resumes the base checkpoint's counter
and starts near step 1,729,300, while a warmstarted multi-speaker run starts at 0.
They will not overlay on one chart.

## Transferring a dataset

`dataset/wav/` is 17,571 small files, which copies badly over SMB or scp:

```sh
tar -C dataset -cf - wav | ssh <server> 'mkdir -p ~/piper-au/dataset && tar -C ~/piper-au/dataset -xf -'
rsync -a --info=progress2 dataset/ <user>@<server>:~/piper-au/dataset/
```

## Known traps

1. `rhasspy/piper` is archived (last push 2025-08-26). `piper_train` and its
   separate `preprocess` step are dead, and most guides online still use them.
   piper1-gpl caches phonemes and spectrograms inside `fit`.
2. `pip install piper-tts[train]` cannot train: the PyPI wheel ships
   `monotonic_align/__init__.py` with no compiled `core` extension. Clone required.
3. The csv format changed from the rhasspy one. Column 1 is the wav *filename*;
   for multi-speaker, column 2 is the speaker *name string*, not an integer id.
4. Multi-speaker is declared with `--model.num_speakers`, not
   `--data.num_speakers`.
5. Only **medium** checkpoints fine-tune without extra vocoder flags. Medium is
   22,050 Hz, which is why the audio is resampled from its native 24 kHz. Training
   at 24000 would orphan the run from every pretrained checkpoint.
6. `--ckpt_path` against the HuggingFace checkpoints has known strict-load
   failures (piper1-gpl #76, #81, #132; fix PR still open). Passing the **URL**
   often works where the same file downloaded locally does not.
   `--model.warmstart_ckpt` sidesteps the loader entirely, and is what
   `train/2-train.sh` uses for multi-speaker.
7. The espeak voice is `en-gb-x-rp`. espeak-ng has no Australian voice;
   Australian English is non-rhotic, so RP is the right approximation and `en-us`
   would be actively wrong.
8. Python 3.11 or 3.12, not 3.13+.
9. torch pinned to 2.8.0, both ends of the range deliberate: >= 2.7 for the CUDA
   12.8 build that sm_120 cards need, < 2.9 because 2.9 defaults the ONNX exporter
   to `dynamo=True` and 2.10
   deprecates `dynamic_axes`, which `export_onnx.py` still passes. Training
   survives that; export may not.
10. Do not train under native Windows: the `win_amd64` torch wheel is CPU-only.
11. Test the finished voice on Linux. piper1-gpl #260: `piper.exe` dies with
    `STATUS_STACK_BUFFER_OVERRUN` in `ucrtbase.dll` during onnxruntime init on
    Windows 11 build 26200. Open and unresolved; Linux is unaffected.
12. Pick the final checkpoint by ear. piper1-gpl's own code notes `val_mel`
    saturates early while the GAN losses keep improving audibility.
13. Never put a raw `"` in a metadata csv, and never validate a csv by splitting
    on the delimiter. piper reads metadata with `csv.reader`, default
    `quotechar='"'`. A transcript that *begins* with a double quote opens a
    quoted field and the parser swallows every following line until the closing
    quote. (A quote mid-transcript is harmless; `csv` only honours it at the start
    of a field.) Audiobook dialogue opens with `"` constantly: 25 of Ophelia's
    transcripts contain a quote and 18 begin with one. Measured: her 1,900 rows
    parsed as 1,240, one field 25,890 characters; the multi-speaker file had a
    71,987 character field. Training died with `Tried to allocate 437.46 GiB`
    inside `attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)`, a
    `[B,1,T,T]` tensor with T about 70,000. It reads like a batch-size problem and
    is not. `prepare/3-project.py` strips quotes and round-trips every file
    through `csv.reader`; `train/1-preflight.py` checks row count equals line
    count and that no transcript exceeds 1,000 characters.
14. `torchaudio` is required and is not a declared dependency. piper's UTMOS
    predictor imports it; if that import fails, `val_mos` logging is silently
    disabled while the default `ModelCheckpoint(monitor='val_mos')` still expects
    the key, so the run dies at the end of epoch 0 with `could not find the
    monitored key`. `train/0-setup.sh` installs the matching version and preflight
    checks it.
15. `export_onnx` writes only the `.onnx`. The companion `.onnx.json` is the file
    training wrote to `--data.config_path`; the exporter cannot reconstruct it.
    Without it the voice will not load, and for a multi-speaker model it is also
    the only record of the speaker name to id map.

## Doing this again

**Cap high, stop manually.** Stopping early is free, Lightning handles Ctrl-C and
closes the checkpoint cleanly. Extending is expensive. Set `PIPER_AU_EPOCHS`
absurdly high and stop when the top-5 `val_mos` checkpoints stop being the newest
epochs.

**Never resume a converged run.** The resume penalty scales with how settled the
model is. Mid-training resumes cost a few thousand steps. Resuming this run at
epoch 299 cost 0.9 `val_mos` and nine hours to climb back. The checkpoint was
complete, optimiser states, schedulers and scaler all present, so this is
inherent to perturbing a converged GAN rather than a bug to fix.

**Audition before you export.** Accent, gender and voice quality are ear
problems. The largest reader in this corpus (11.79 h) was rejected at that gate
after being the obvious pick on every number available.

**Check transcripts, not just hours.** Punctuation rate varied 0%-82% between
narrators here, and espeak drives prosody off punctuation. Two large readers were
excluded on that alone.

**Validate a file with the parser its consumer uses.** The worst bug in this
project was invisible to a `str.split('|')` check because piper reads with
`csv.reader`. Round-trip every generated file through the real parser.

**Make the expensive artefact reusable.** Decoding 4.7 GB of audio took 20
minutes; every data fix afterwards was a re-projection from `manifest.jsonl`
taking seconds. That paid for itself three times over, which is why
`dataset/manifest.jsonl` is the durable artefact and every csv is a projection.

**Snapshot on "newest past N", never "epoch % N == 0".** Lightning only writes a
checkpoint when it beats the current top-k, and once a run converges those epoch
numbers are arbitrary: one version finished holding epochs 224, 254, 266, 272,
274, 278, 291, 293, 294 and 299, not one of them a multiple of 25. The old rule
silently archived nothing from epoch 350 onward.

**Stop when it sounds right, not when a metric is still moving.** This model was
perceptually converged by about epoch 300. A second 16-hour run added about 0.14
`val_mos` and produced audio that could not reliably be told apart from the
13-hour checkpoint. `val_mos` was still climbing at epoch 599 and never flattened;
the ear plateaued at less than half the training time. Budget the next voice at
about 13 h, not about 30 h.

**`val_mos` decides, `val_mel` corroborates, `val_dur` is a warning light.** They
diverged here: `val_dur` turned at epoch about 150 while `val_mos` kept climbing
to the end. `val_mos` correlates with human judgement so it wins, but it is noisy,
so treat a single high score as suspect and listen before trusting it. The highest
`val_mos` of the project, 3.7609 at epoch 544, was an outlier against its
neighbours at 3.60-3.65 and was not shipped.

**Measure the card, don't read the spec sheet.** See the GPU table above. Which
throughput metric matters depends on how many clips are in an epoch.

## Diagnostics

```sh
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader
nvidia-smi --query-gpu=index,power.draw,clocks.sm,temperature.gpu --format=csv -l 1 > ~/power.csv
mpstat -P ALL 1                             # one core pinned = serialisation; all low = model too small
grep -i xid /var/log/syslog | tail -20      # Xid = GPU fault; none plus hard reset = PSU/board
df -h /home

tail -f logs/multi.log | tr '\r' '\n'       # the tr is not optional: tqdm redraws with \r
tmux capture-pane -t snap -p | tail         # is the snapshotter actually archiving?
tmux send-keys -t multi C-c                 # graceful stop from anywhere
```

What is actually inside a checkpoint:

```python
import torch
ck = torch.load("runs/multi/lightning_logs/version_9/checkpoints/last.ckpt",
                map_location="cpu", weights_only=False)
print("keys:", sorted(ck.keys()))
print("optimizer_states:", len(ck.get("optimizer_states", [])))
print("lr_schedulers:",   len(ck.get("lr_schedulers", [])))
print("epoch/step:",      ck.get("epoch"), ck.get("global_step"))
```
