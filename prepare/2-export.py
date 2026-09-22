"""Export the australian-english-speech parquet shards into a Piper training set.

    python3 prepare/2-export.py --audition            # clips to listen to
    python3 prepare/2-export.py --multi               # the training set

Fetch the corpus first with ./prepare/0-download.sh. Needs pyarrow
(pip install -r prepare/requirements.txt) and ffmpeg on the PATH.

Modes:

  --audition          pull a few clips from each candidate reader into one
                      folder, so the accent can be checked by ear before
                      anything is trained. LibriVox volunteers are
                      international and the dataset card does not verify accent
                      per reader, so this gate is not optional.

  --multi             export the accent-approved speakers as one multi-speaker
                      set: wav/ + manifest.jsonl + metadata.csv + speakers.json.

  --reader "<name>"   export a single reader on their own.

Audio in the parquet is 24 kHz mono FLAC. Piper medium voices train at
22.05 kHz. piper1-gpl resamples on its own during cache build, but we do it
here anyway so the 24000 -> 22050 step happens once, with a tool we control,
and is auditable afterwards.

metadata.csv follows piper1-gpl's format, which is NOT the old rhasspy one:
column 1 is the wav FILENAME, and for multi-speaker column 2 is the speaker
NAME as a string, not an integer id. piper1-gpl builds its own name->id map
during training and writes it into --data.config_path.

The durable artefact is manifest.jsonl. Every csv here is a projection of it,
so a change in what Piper wants on the csv side is a re-projection rather than
a multi-gigabyte re-decode.

Written 2026-09-18 for the Australian Piper voice; step 2 of prepare/.
"""
import argparse
import collections
import glob
import io
import json
import os
import re
import subprocess
import sys

import pyarrow.parquet as pq

META = ["text", "reader", "reader_id", "book", "book_id", "section", "duration"]

# This script lives in prepare/, so the repo root is one level up. Every default
# is built from it, so the stage scripts work from any working directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where the HF corpus parquet shards live. ./prepare/0-download.sh puts them
# here; override with $AU_SPEECH_DATA or --data-dir.
DEFAULT_DATA = os.environ.get(
    "AU_SPEECH_DATA",
    os.path.join(ROOT, "source-data", "australian-english-speech", "data"))
DEFAULT_ROSTER = os.environ.get(
    "PIPER_AU_ROSTER", os.path.join(ROOT, "config", "roster.json"))


def load_roster(path):
    """The approved narrators, from config/roster.json.

    Returns [{"reader": ..., "sex": ...}, ...] in the order written. That file is
    the only place the roster lives -- read it for why accent has to be checked
    by ear before anyone is added to it.
    """
    try:
        with io.open(path, encoding="utf-8") as f:
            data = json.load(f)
    except IOError:
        raise SystemExit("roster not found: %s" % path)
    rows = data.get("readers") or []
    if not rows:
        raise SystemExit('no "readers" entries in %s' % path)
    for r in rows:
        if not r.get("reader"):
            raise SystemExit('roster entry with no "reader": %r' % r)
    return rows

# Trailing quote characters to look past when deciding whether a transcript
# ends a sentence. Written as escapes so the file stays shell-quoting-safe.
_TRAILING = "\u0022\u0027\u2019\u201d"


def slug(s, sep="-"):
    return re.sub(r"[^a-z0-9]+", sep, s.lower()).strip(sep)


def speaker_name(s):
    """CSV-safe speaker label. 'Lucy Burgoyne (1950-2014)' -> lucy_burgoyne_1950_2014"""
    return slug(s, "_")


# ASCII double quote is csv.reader's default quotechar, and piper1-gpl reads the
# metadata with csv.reader. One unpaired quote makes the parser consume every
# following line until the next one, merging hundreds of rows into a single
# utterance. Measured on the first export: Ophelia 1,900 rows -> 1,240, one
# field 25,890 chars, training died on a 437 GiB attention mask. Quote marks
# carry no phonetic content, so they are removed.
CSV_QUOTE = chr(34)


def csv_safe(text):
    text = text.replace("|", " ").replace(CSV_QUOTE, "")
    return " ".join(text.split())


def ends_sentence(text):
    return text.rstrip(_TRAILING).endswith((".", "!", "?"))


def to_wav(flac_bytes, dest, sample_rate, normalise=False):
    """FLAC bytes -> mono 16-bit PCM WAV at sample_rate, via ffmpeg stdin.

    normalise applies EBU R128 loudness normalisation (loudnorm) before the
    resample. LibriVox narrators record at home across many sessions, and the
    level drifts between them -- audible in the trained voice as volume that
    shifts mid-paragraph. Michael on `algy_pug`, 2026-09-20: "voice quality
    isn't the best and sometimes volume changes."

    It is per-clip, which is the honest trade: it fixes session-to-session
    drift, and it flattens deliberate loudness differences BETWEEN clips (a
    whispered line and a shouted one end up at the same integrated level).
    Within a clip the dynamics are untouched. For 3-15 s audiobook clips that
    is worth it; for highly dramatic source it might not be.

    loudnorm emits 192 kHz in single-pass mode, so the aresample must follow it.
    """
    af = "aresample=resampler=soxr:precision=28"
    if normalise:
        af = "loudnorm=I=-23:LRA=7:TP=-2," + af
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", "pipe:0",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-af", af,
        "-sample_fmt", "s16",
        dest,
    ]
    p = subprocess.run(cmd, input=flac_bytes, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed on %s: %s" % (dest, p.stderr.decode(errors="replace")[:400])
        )


def shards(data_dir):
    return sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))


def usable(text, dur, args):
    if len(text) < args.min_chars:
        return False
    if not (args.min_dur <= dur <= args.max_dur):
        return False
    if args.require_endpunct and not ends_sentence(text):
        return False
    return True


def audition(args):
    wanted = {r: [] for r in args.candidates}
    os.makedirs(args.out, exist_ok=True)

    for path in shards(args.data_dir):
        if all(len(v) >= args.per_reader for v in wanted.values()):
            break
        d = pq.read_table(path, columns=["audio"] + META).to_pydict()
        for audio, text, reader, dur in zip(
            d["audio"], d["text"], d["reader"], d["duration"]
        ):
            if reader not in wanted or len(wanted[reader]) >= args.per_reader:
                continue
            # Mid-length clips audition better than 3 s fragments.
            if not (6.0 <= dur <= 12.0):
                continue
            n = len(wanted[reader]) + 1
            dest = os.path.join(args.out, "%s_%02d.wav" % (slug(reader), n))
            to_wav(audio["bytes"], dest, args.sample_rate, args.normalise)
            wanted[reader].append((dest, (text or "").strip()))
            print("  %s  (%.1fs)  %s" % (os.path.basename(dest), dur,
                                         (text or "").strip()[:70]), flush=True)

    with open(os.path.join(args.out, "AUDITION.md"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("# Accent audition\n\n")
        f.write("Play these and mark each reader AU / not-AU. The dataset card does\n")
        f.write("not verify accent per reader -- these are the Australian *titles* in\n")
        f.write("LibriVox, and the volunteers are international.\n\n")
        for reader in args.candidates:
            f.write("## %s\n\n" % reader)
            for dest, text in wanted[reader]:
                f.write("- `%s` -- %s\n" % (os.path.basename(dest), text))
            f.write("\n")

    print("\nWrote %d clips + AUDITION.md to %s"
          % (sum(len(v) for v in wanted.values()), args.out))
    for r, v in wanted.items():
        if len(v) < args.per_reader:
            print("  ! %s: only %d/%d clips matched the length window"
                  % (r, len(v), args.per_reader))


def export_multi(args):
    roster = load_roster(args.roster)
    approved = {r["reader"]: (r.get("sex") or "unknown") for r in roster}
    readers = args.readers or list(approved)
    sex = {r: approved.get(r, "unknown") for r in readers}
    names = {r: speaker_name(r) for r in readers}
    order = {r: i for i, r in enumerate(readers)}

    wav_dir = os.path.join(args.out, "wav")
    os.makedirs(wav_dir, exist_ok=True)

    rows = []
    kept = collections.Counter()
    secs = collections.Counter()
    skipped = collections.Counter()
    files = shards(args.data_dir)

    for i, path in enumerate(files, 1):
        d = pq.read_table(path, columns=["audio"] + META).to_pydict()
        for audio, text, r, book, book_id, sec, dur in zip(
            d["audio"], d["text"], d["reader"], d["book"], d["book_id"],
            d["section"], d["duration"],
        ):
            if r not in order:
                continue
            text = (text or "").strip()
            if not usable(text, dur, args):
                skipped[r] += 1
                continue
            wav = "%02d_%s_%05d.wav" % (order[r], slug(r), kept[r])
            to_wav(audio["bytes"], os.path.join(wav_dir, wav), args.sample_rate,
                   args.normalise)
            rows.append({
                "wav": wav,
                "speaker": names[r],
                "reader": r,
                "sex": sex[r],
                "text": text,
                "book": book,
                "book_id": book_id,
                "section": sec,
                "duration": round(float(dur), 3),
            })
            kept[r] += 1
            secs[r] += dur
        print("  [%d/%d] kept=%s  %.2f h"
              % (i, len(files), format(sum(kept.values()), ","),
                 sum(secs.values()) / 3600), flush=True)

    def write_lines(path, lines):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for line in lines:
                f.write(line + "\n")

    with open(os.path.join(args.out, "manifest.jsonl"), "w",
              encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # piper1-gpl multi-speaker: wav_filename|speaker_name|text
    write_lines(
        os.path.join(args.out, "metadata.csv"),
        ["%s|%s|%s" % (r["wav"], r["speaker"], csv_safe(r["text"]))
         for r in rows],
    )

    # Female-only variant -- the accent backbone without the male voices, in
    # case a female-only model is preferred over speaker selection.
    female_rows = [r for r in rows if r["sex"] == "female"]
    write_lines(
        os.path.join(args.out, "metadata-female.csv"),
        ["%s|%s|%s" % (r["wav"], r["speaker"], csv_safe(r["text"]))
         for r in female_rows],
    )

    # piper1-gpl single speaker: wav_filename|text. One per voice, for
    # fine-tuning a single voice off a pretrained checkpoint instead.
    per = os.path.join(args.out, "single-speaker")
    os.makedirs(per, exist_ok=True)
    for r in readers:
        write_lines(
            os.path.join(per, "%s.csv" % names[r]),
            ["%s|%s" % (row["wav"], csv_safe(row["text"]))
             for row in rows if row["reader"] == r],
        )

    with open(os.path.join(args.out, "speakers.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump({
            "sample_rate": args.sample_rate,
            "require_endpunct": args.require_endpunct,
            "num_speakers": len(readers),
            "num_speakers_female": sum(1 for r in readers
                                       if sex[r] == "female"),
            "total_clips": sum(kept.values()),
            "total_hours": round(sum(secs.values()) / 3600, 3),
            "note": ("speaker ids are assigned by piper1-gpl during training and "
                     "written to --data.config_path; read them from there, never "
                     "from the order below"),
            "speakers": [
                {
                    "speaker": names[r],
                    "reader": r,
                    "sex": sex[r],
                    "clips": kept[r],
                    "hours": round(secs[r] / 3600, 3),
                }
                for r in readers
            ],
        }, f, indent=2, ensure_ascii=False)

    print("\n%-28s %-7s %9s %8s %9s" % ("speaker", "sex", "clips", "hours", "skipped"))
    print("-" * 66)
    for r in readers:
        print("%-28s %-7s %9s %8.2f %9s"
              % (names[r][:28], sex[r], format(kept[r], ","),
                 secs[r] / 3600, format(skipped[r], ",")))
    print("-" * 66)
    print("%-28s %-7s %9s %8.2f %9s"
          % ("TOTAL", "", format(sum(kept.values()), ","),
             sum(secs.values()) / 3600, format(sum(skipped.values()), ",")))
    fem_h = sum(secs[r] for r in readers if sex[r] == "female") / 3600
    print("%-28s %-7s %9s %8.2f" % ("  of which female", "", "", fem_h))

    print("\n  %s" % args.out)
    print("  manifest.jsonl       the durable artefact, one json per clip")
    print("  metadata.csv         wav|speaker|text   -- all %d speakers" % len(readers))
    print("  metadata-female.csv  wav|speaker|text   -- female only")
    print("  single-speaker/*.csv wav|text           -- one voice alone")
    print("  speakers.json        roster and counts")
    print("  %d Hz mono 16-bit" % args.sample_rate)
    print()
    print("  Next -- training reads the -clean projections, not these:")
    print("    python3 prepare/3-project.py --suffix=-clean --max-chars 400")
    print()
    print("  Next -- training reads the -clean projections, not these:")
    print("    python3 prepare/3-project.py --suffix=-clean --max-chars 400")
    print()
    print("  Next -- training reads the -clean projections, not these:")
    print("    python3 prepare/3-project.py --suffix=-clean --max-chars 400")
    print()
    print("  Next -- training reads the -clean projections, not these:")
    print("    python3 prepare/3-project.py --suffix=-clean --max-chars 400")
    print()
    print("  Next -- training reads the -clean projections, not these:")
    print("    python3 prepare/3-project.py --suffix=-clean --max-chars 400")
    print()
    print("  Next -- training reads the -clean projections, not these:")
    print("    python3 prepare/3-project.py --suffix=-clean --max-chars 400")
    print()
    print("  Next -- training reads the -clean projections, not these:")
    print("    python3 prepare/3-project.py --suffix=-clean --max-chars 400")
    print()
    print("  Next -- training reads the -clean projections, not these:")
    print("    python3 prepare/3-project.py --suffix=-clean --max-chars 400")


def export_reader(args):
    args.readers = [args.reader]
    export_multi(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA,
                    help="parquet shards (default: source-data/, or $AU_SPEECH_DATA)")
    ap.add_argument("--roster", default=DEFAULT_ROSTER,
                    help="approved narrators (default: config/roster.json)")
    ap.add_argument("--out", default=os.environ.get(
        "PIPER_AU_DATASET", os.path.join(ROOT, "dataset")),
        help="where wav/, manifest.jsonl and the csvs go (default: dataset/)")
    ap.add_argument("--sample-rate", type=int, default=22050,
                    help="22050 for Piper medium/high, 16000 for low/x-low")
    ap.add_argument("--audition", action="store_true")
    ap.add_argument("--per-reader", type=int, default=5)
    ap.add_argument("--candidates", nargs="*",
                    help="audition mode; defaults to the roster in config/roster.json")
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--readers", nargs="*",
                    help="multi-speaker export; defaults to the accent-approved set")
    ap.add_argument("--reader")
    ap.add_argument("--normalise", action="store_true",
                    help="EBU R128 loudness-normalise each clip. Recommended: "
                         "LibriVox level drifts between recording sessions and "
                         "it is audible in the trained voice as volume shifts")
    ap.add_argument("--require-endpunct", action="store_true",
                    help="keep only clips whose transcript ends in . ! or ?")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--min-chars", type=int, default=4)
    args = ap.parse_args()

    if args.audition:
        if not args.candidates:
            args.candidates = [r["reader"] for r in load_roster(args.roster)]
        audition(args)
    elif args.multi or args.readers:
        export_multi(args)
    elif args.reader:
        export_reader(args)
    else:
        sys.exit("give --audition, --multi, or --reader <name>")


if __name__ == "__main__":
    main()
