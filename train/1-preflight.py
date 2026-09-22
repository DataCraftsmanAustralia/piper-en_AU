#!/usr/bin/env python3
"""Preflight for the Australian Piper training run.

Every check here exists because failing it produces a confusing error hours
later instead of a clear one now. Run it before every training run, not just
the first -- the VRAM check is about what else is on the box at this moment,
and that changes.

    python3 train/1-preflight.py             # every GPU, pick the emptiest
    python3 train/1-preflight.py --gpu 0     # check a specific GPU (PCI index)
    python3 train/1-preflight.py --quick     # skip the wav existence scan

Run it before every training run, from the repo root.
"""
import argparse
import collections
import csv
import importlib
import json
import os
import shutil
import struct
import subprocess
import sys

# MUST be set before torch initialises CUDA. CUDA's default device order is
# FASTEST_FIRST, so on a mixed box torch numbers the fastest card 0 while
# nvidia-smi and docker --device-ids number by PCI bus. On one box that
# makes torch call the 96 GB card 0 and the 24 GB one 1, while docker calls
# them 1 and 0. Forcing PCI_BUS_ID makes every tool agree.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# csv defaults to a 128 KiB field limit. A merged-row file can blow straight
# past it, and the resulting _csv.Error would abort the rest of the checks.
# 10 MB is far beyond any legitimate transcript and safe on every platform.
csv.field_size_limit(10_000_000)

# This script lives in train/, so the repo root is one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.environ.get("PIPER_AU_DATASET", os.path.join(ROOT, "dataset"))

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "
_results = []


def report(level, title, detail=""):
    _results.append((level, title))
    print("[%s] %s" % (level, title))
    if detail:
        for d in str(detail).rstrip().split("\n"):
            print("         " + d)


# --------------------------------------------------------------------------
def check_python():
    v = sys.version_info
    s = "%d.%d.%d" % (v.major, v.minor, v.micro)
    if v.major == 3 and v.minor in (11, 12):
        report(OK, "python %s" % s)
    elif v.major == 3 and v.minor in (9, 10, 13):
        report(WARN, "python %s" % s, "piper1-gpl targets 3.9-3.13; 3.11/3.12 are the safe ones.")
    else:
        report(FAIL, "python %s" % s, "Too new for this stack. Build the venv with python3.12.")


def check_torch():
    try:
        import torch
    except ImportError:
        report(FAIL, "torch not importable",
               "Activate the venv:  source piper1-gpl/.venv/bin/activate   "
               "(or run ./train/0-setup.sh)")
        return None

    tv = torch.__version__
    mm = tuple(int(x) for x in tv.split("+")[0].split(".")[:2])
    if mm >= (2, 9):
        report(WARN, "torch %s" % tv,
               "torch >= 2.9 defaults the ONNX exporter to dynamo=True and\n"
               "deprecates dynamic_axes, which piper1-gpl's export_onnx.py still\n"
               "passes. Training is fine; ONNX export may fail.")
    elif mm < (2, 7):
        report(WARN, "torch %s" % tv,
               "Too old for sm_120 cards, which need a CUDA 12.8 build.\n"
               "Fine on older architectures.")
    else:
        report(OK, "torch %s" % tv)

    if not torch.cuda.is_available():
        report(FAIL, "CUDA not available to torch",
               "pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128")
        return None
    return torch


# --------------------------------------------------------------------------
def nvidia_processes():
    """{gpu_index: [(pid, used, name), ...]} attributed via GPU UUID."""
    by_uuid = {}
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        for line in q.stdout.strip().splitlines():
            idx, uuid = [p.strip() for p in line.split(",", 1)]
            by_uuid[uuid] = int(idx)
    except Exception:
        return {}

    out = collections.defaultdict(list)
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        for line in q.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            uuid, pid, used, name = parts[0], parts[1], parts[2], ",".join(parts[3:])
            if uuid in by_uuid:
                out[by_uuid[uuid]].append((pid, used, name))
    except Exception:
        pass
    return out


def check_gpus(torch, want, need_gb):
    n = torch.cuda.device_count()
    if n == 0:
        report(FAIL, "no CUDA devices visible")
        return None

    procs = nvidia_processes()
    rows = []
    print("[%s] %d CUDA device(s), PCI_BUS_ID order" % (OK, n))
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:
            free, total = 0, props.total_memory
        rows.append((i, props.name, free / 1e9, total / 1e9))
        print("         GPU %d  %-46s  free %6.1f / %6.1f GB  sm_%d%d"
              % (i, props.name[:46], free / 1e9, total / 1e9, props.major, props.minor))
        for pid, used, name in procs.get(i, []):
            print("                holding: pid %s  %s  %s" % (pid, used, name))
    _results.append((OK, "gpu enumeration"))

    if want is not None:
        cand = [r for r in rows if r[0] == want]
        if not cand:
            report(FAIL, "GPU %d does not exist" % want)
            return None
        target = cand[0]
    else:
        target = max(rows, key=lambda r: r[2])

    idx, name, free_gb, total_gb = target
    print()

    if free_gb >= need_gb:
        report(OK, "train on GPU %d -- %s, %.1f GB free" % (idx, name, free_gb),
               "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=%d\n"
               "or:  PIPER_AU_GPU=%d ./train/2-train.sh single" % (idx, idx))
        return idx

    # Nothing has room. Say which tenant is on which card and what may be stopped.
    lines = ["No GPU has %.0f GB free. What is holding them:" % need_gb, ""]
    for i, nm, fg, tg in rows:
        lines.append("  GPU %d  %-42s  free %.1f / %.1f GB" % (i, nm[:42], fg, tg))
        for pid, used, pname in procs.get(i, []):
            lines.append("           pid %s  %s  %s" % (pid, used, pname))
    lines += [
        "",
        "vLLM pre-allocates its KV cache, so a lane that looks idle still holds",
        "the card and will not yield under pressure.",
        "",
        "Check which processes hold each card in the table above, then stop",
        "the inference containers on the one you want to train on:",
        "",
        "    docker compose stop <inference-service>",
        "",
        "and start them again when the run finishes:",
        "",
        "    docker compose start <inference-service>",
    ]
    report(FAIL, "no GPU with %.0f GB free" % need_gb, "\n".join(lines))
    return None


# --------------------------------------------------------------------------
def check_piper():
    try:
        import piper  # noqa: F401
    except ImportError:
        report(FAIL, "piper not importable", "Run ./train/0-setup.sh first.")
        return

    try:
        from piper.train.vits.monotonic_align import maximum_path  # noqa: F401
        report(OK, "monotonic_align compiled")
    except Exception as e:
        report(FAIL, "monotonic_align not built", str(e) + "\n"
               "  cd piper1-gpl && ./build_monotonic_align.sh && "
               "python3 setup.py build_ext --inplace")

    # torchaudio is not a declared dependency but training dies without it.
    # piper's UTMOS predictor imports it; if the import fails, val_mos logging is
    # silently disabled -- and the default ModelCheckpoint still monitors
    # val_mos, so the run crashes at the END of epoch 0 with "could not find the
    # monitored key". An expensive way to find out, so check it up front.
    try:
        import torch
        import torchaudio
        if torchaudio.__version__.split("+")[0] != torch.__version__.split("+")[0]:
            report(WARN, "torchaudio %s vs torch %s" % (torchaudio.__version__,
                                                        torch.__version__),
                   "Versions should match exactly.")
        else:
            report(OK, "torchaudio %s" % torchaudio.__version__)
    except ImportError:
        import torch
        v = torch.__version__.split("+")[0]
        report(FAIL, "torchaudio missing -- training will crash at the end of epoch 0",
               "piper's UTMOS predictor needs it. Without it val_mos is never\n"
               "logged, but ModelCheckpoint(monitor='val_mos') still expects it:\n"
               "  MisconfigurationException: could not find the monitored key\n\n"
               "  pip install torchaudio==%s --index-url https://download.pytorch.org/whl/cu128"
               % v)

    # The phonemizer's import shape has moved between versions, and in some of
    # them `piper.phonemize_espeak` is the MODULE, not the function inside it.
    # Probe rather than assume, and report what was actually found.
    fn, where = None, None
    for mod, attr in [
        ("piper", "phonemize_espeak"),
        ("piper.phonemize_espeak", "phonemize_espeak"),
        ("piper.phonemize", "phonemize_espeak"),
        ("piper.phoneme_ids", "phonemize_espeak"),
    ]:
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        c = getattr(m, attr, None)
        if callable(c):
            fn, where = c, "%s.%s" % (mod, attr)
            break
        # module-shaped: look for a callable of the same name one level in
        if c is not None and hasattr(c, attr) and callable(getattr(c, attr)):
            fn, where = getattr(c, attr), "%s.%s.%s" % (mod, attr, attr)
            break

    if fn is None:
        try:
            importlib.import_module("piper.espeakbridge")
            report(OK, "espeak bridge present (piper.espeakbridge)",
                   "Could not locate a phonemize_espeak callable to smoke-test,\n"
                   "but the embedded espeak extension imports. Training does its\n"
                   "own phonemization; this is informational only.")
        except Exception as e:
            report(WARN, "espeak bridge not found", e)
        return

    try:
        ph = fn("Good on ya.", "en-gb-x-rp")
        flat = "".join(ph[0]) if ph and isinstance(ph[0], (list, tuple)) else str(ph)
        report(OK, "espeak en-gb-x-rp via %s" % where, "phonemes: %s" % flat[:60])
    except Exception as e:
        report(WARN, "espeak smoke test failed (%s)" % where, e)


# --------------------------------------------------------------------------
def wav_header(path):
    with open(path, "rb") as f:
        if f.read(4) != b"RIFF":
            return None
        f.read(4)
        if f.read(4) != b"WAVE":
            return None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                return None
            cid, size = struct.unpack("<4sI", hdr)
            if cid == b"fmt ":
                fmt = f.read(size)
                _, ch, rate, _, _, bits = struct.unpack("<HHIIHH", fmt[:16])
                return rate, ch, bits
            f.seek(size + (size & 1), 1)


def check_dataset(quick):
    if not os.path.isdir(DATASET):
        report(FAIL, "dataset not found at %s" % DATASET,
               "Build it first -- ./prepare/0-download.sh, then\n"
               "  python3 prepare/2-export.py --multi\n"
               "  python3 prepare/3-project.py --suffix=-clean --max-chars 400\n"
               "Or point at a dataset that already exists: PIPER_AU_DATASET=/path")
        return
    wav_dir = os.path.join(DATASET, "wav")
    if not os.path.isdir(wav_dir):
        report(FAIL, "no wav/ directory in %s" % DATASET,
               "python3 prepare/2-export.py --multi --out %s" % DATASET)
        return

    on_disk = set(os.listdir(wav_dir))
    report(OK, "wav/ holds %s files" % format(len(on_disk), ","))

    csvs = [
        ("single (ophelia_darcy)", os.path.join(DATASET, "single-speaker-clean", "ophelia_darcy.csv"), 2),
        ("female", os.path.join(DATASET, "metadata-female-clean.csv"), 3),
        ("multi", os.path.join(DATASET, "metadata-clean.csv"), 3),
    ]
    for label, path, want in csvs:
        if not os.path.exists(path):
            report(FAIL, "missing csv: %s" % path)
            continue
        # Parse with csv.reader -- the SAME parser piper1-gpl uses. Checking a
        # csv by splitting on the delimiter is what let a broken file through
        # once already: ASCII double quote is csv.reader's quotechar, so one
        # unpaired quote swallows every following line until the next one.
        # Ophelia's 1,900 rows became 1,240, one of them 25,890 characters
        # long, and training died asking for a 437 GiB attention mask.
        # A line count and a csv row count MUST agree.
        n_lines = sum(1 for l in open(path, encoding="utf-8") if l.strip())
        try:
            with open(path, encoding="utf-8", newline="") as f:
                rows = [r for r in csv.reader(f, delimiter="|") if r]
        except csv.Error as e:
            # A merged file can exceed csv's field size limit outright. That is
            # the same defect as a row-count mismatch, only worse, so report it
            # rather than letting the traceback abort the remaining checks.
            report(FAIL, "%s: csv parse failed -- %s" % (label, e),
                   "The file has %s lines. A field larger than the limit means\n"
                   "rows have been merged by an unescaped double quote.\n"
                   "Regenerate and re-copy:\n"
                   "  python3 prepare/3-project.py --suffix=-clean --max-chars 400"
                   % format(n_lines, ","))
            continue

        if len(rows) != n_lines:
            report(FAIL, "%s: csv.reader sees %s rows but the file has %s lines"
                   % (label, format(len(rows), ","), format(n_lines, ",")),
                   "Rows are being merged by the csv parser. Almost always an\n"
                   "unescaped double quote. Regenerate with:\n"
                   "  python3 prepare/3-project.py --suffix=-clean --max-chars 400")
            continue

        widths = collections.Counter(len(r) for r in rows)
        if set(widths) != {want}:
            report(FAIL, "%s: column count %s, want %d" % (label, dict(widths), want))
            continue

        speakers, missing = set(), []
        longest = 0
        for r in rows:
            if len(r) == 3:
                speakers.add(r[1])
            longest = max(longest, len(r[-1]))
            if not quick and r[0] not in on_disk:
                missing.append(r[0])

        if missing:
            report(FAIL, "%s: %d referenced wavs missing" % (label, len(missing)),
                   "first few: " + ", ".join(missing[:5]))
        elif longest > 1000:
            report(FAIL, "%s: longest transcript is %d chars" % (label, longest),
                   "No 15-second clip has a transcript that long -- rows have been\n"
                   "merged. Phonemizing it will exhaust VRAM.")
        else:
            d = "%s rows, longest transcript %d chars" % (format(len(rows), ","), longest)
            if len(speakers) > 1:
                d += ", --model.num_speakers %d" % len(speakers)
            report(OK, "%s: %s" % (label, d))

    sample = sorted(on_disk)[:: max(1, len(on_disk) // 200)][:200]
    rates = collections.Counter()
    for fn in sample:
        h = wav_header(os.path.join(wav_dir, fn))
        if h:
            rates[h] += 1
    if rates and set(rates) == {(22050, 1, 16)}:
        report(OK, "audio 22050 Hz mono 16-bit (sampled %d)" % len(sample))
    else:
        report(FAIL, "unexpected audio format", dict(rates))

    spk = os.path.join(DATASET, "speakers.json")
    if os.path.exists(spk):
        d = json.load(open(spk, encoding="utf-8"))
        report(OK, "speakers.json: %d speakers, %.2f h"
               % (d.get("num_speakers", 0), d.get("total_hours", 0)))


def check_disk():
    try:
        free_gb = shutil.disk_usage(ROOT).free / 1e9
    except Exception:
        return
    if free_gb < 40:
        report(WARN, "disk free %.0f GB" % free_gb,
               "Cache is roughly the size of the audio; Lightning keeps up to\n"
               "11 checkpoints at ~850 MB each. Budget ~40 GB per run.")
    else:
        report(OK, "disk free %.0f GB" % free_gb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gpu", type=int, default=None,
                    help="PCI index of the GPU to train on (default: emptiest)")
    ap.add_argument("--need-gb", type=float, default=10.0)
    args = ap.parse_args()

    want = args.gpu
    if want is None and os.environ.get("PIPER_AU_GPU"):
        want = int(os.environ["PIPER_AU_GPU"])

    print("Australian Piper preflight")
    print("  root:    %s" % ROOT)
    print("  dataset: %s" % DATASET)
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("  note:    CUDA_VISIBLE_DEVICES=%s is set, so only those GPUs are"
              % os.environ["CUDA_VISIBLE_DEVICES"])
        print("           visible below. Unset it to survey the whole box.")
    print()

    check_python()
    torch = check_torch()
    if torch is not None:
        check_gpus(torch, want, args.need_gb)
    check_piper()
    check_dataset(args.quick)
    check_disk()

    fails = sum(1 for lvl, _ in _results if lvl == FAIL)
    warns = sum(1 for lvl, _ in _results if lvl == WARN)
    print()
    if fails:
        print("%d FAILED, %d warnings -- fix the failures before training." % (fails, warns))
        sys.exit(1)
    print("all checks passed (%d warnings). Ready: ./train/2-train.sh single" % warns)


if __name__ == "__main__":
    main()
