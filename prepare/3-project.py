"""Re-project the Piper training csvs from manifest.jsonl.

    python3 prepare/3-project.py --suffix=-clean --max-chars 400
    python3 prepare/3-project.py --suffix=-punct --require-endpunct
    python3 prepare/3-project.py --suffix=-fem --speakers magdalena jenno

Every csv that train/2-train.sh reads is written here, so re-filtering the
training set is a re-projection (seconds) rather than another multi-gigabyte
decode of the source audio.

The export script decodes 4.7 GB of audio once and writes manifest.jsonl beside
it. Every csv Piper reads is a projection of that manifest, so changing what
goes into training -- dropping boilerplate, requiring punctuation, holding a
speaker out -- is this script, not another multi-gigabyte decode.

Boilerplate: the dataset card claims LibriVox intro/outro clips were dropped.
Measured 2026-09-18, 117 of 17,571 survived (0.67%, 10.8 min) -- "Recording by
Son of the Exiles" alone appears 11 times identically. Repeated identical short
utterances are exactly what a VITS model latches onto, and chapter headers are
read in a different register from the prose, so they come out.

Written 2026-09-18 for the Australian Piper voice; step 3 of prepare/.
"""
import argparse
import collections
import csv
import json
import os
import re

# LibriVox boilerplate. Deliberately conservative -- these match announcements,
# not prose that merely mentions a chapter.
BOILERPLATE = [
    r"librivox\.org",
    r"\blibrivox recording\b",
    r"^(this is a librivox|recording by|read by|recorded by)\b",
    r"\b(read|recorded) by [A-Z][\w'. -]+\.?$",
    r"^end of\b",
    r"^(section|chapter|part|book) (\d+|[ivxlc]+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    r"\bdedicated to the public domain\b",
    r"\bto volunteer\b.*\blibrivox\b",
]
_BP = [re.compile(p, re.I) for p in BOILERPLATE]

# ASCII double quote is csv.reader's default quotechar, and piper1-gpl reads the
# metadata with csv.reader. ONE unpaired quote in a transcript makes the parser
# consume every following line until the next quote, merging hundreds of rows
# into a single utterance. Not theoretical: it silently took Ophelia from 1,900
# rows to 1,240 and produced a 25,890-character "utterance" whose attention mask
# asked for 437 GiB of VRAM. Quote marks carry no phonetic content -- espeak
# does not pronounce them -- so they are removed outright.
CSV_QUOTE = chr(34)


def csv_safe(text):
    """Make a transcript safe for the csv.reader on the other end."""
    text = text.replace("|", " ").replace(CSV_QUOTE, "")
    return " ".join(text.split())

_TRAILING = "\u0022\u0027\u2019\u201d"


def is_boilerplate(text):
    return any(p.search(text) for p in _BP)


def ends_sentence(text):
    return text.rstrip(_TRAILING).endswith((".", "!", "?"))


def write_lines(path, lines, fields=None):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            f.write(line + "\n")
    if fields:
        verify_roundtrip(path, len(lines), fields)


def verify_roundtrip(path, expected_rows, expected_fields):
    """Re-read with the SAME parser piper uses, and insist it agrees.

    The bug this exists to catch was invisible to a str.split("|") check: the
    validator and the consumer were using different parsers, so the validator
    passed a file the consumer could not read. Never validate a csv by splitting
    on the delimiter.
    """
    with open(path, encoding="utf-8", newline="") as f:
        rows = [r for r in csv.reader(f, delimiter="|") if r]
    bad = [i for i, r in enumerate(rows, 1) if len(r) != expected_fields]
    if len(rows) != expected_rows or bad:
        raise SystemExit(
            "csv round-trip FAILED for %s\n"
            "  wrote %d lines, csv.reader sees %d rows\n"
            "  %d row(s) with != %d fields (first: %s)"
            % (path, expected_rows, len(rows), len(bad), expected_fields,
               bad[:5] or "-")
        )


# This script lives in prepare/, so the repo root is one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",
                    default=os.environ.get("PIPER_AU_DATASET",
                                           os.path.join(ROOT, "dataset")))
    ap.add_argument("--suffix", default="", help="e.g. -clean -> metadata-clean.csv")
    ap.add_argument("--keep-boilerplate", action="store_true")
    ap.add_argument("--require-endpunct", action="store_true")
    ap.add_argument("--speakers", nargs="*", help="restrict to these speaker labels")
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=0,
                    help="drop transcripts longer than this (0 = no cap); guards "
                         "against a single long utterance OOMing training, since "
                         "piper1-gpl has no --max-phoneme-ids escape hatch")
    args = ap.parse_args()

    manifest = os.path.join(args.dataset, "manifest.jsonl")
    rows = [json.loads(line) for line in open(manifest, encoding="utf-8")]

    dropped = collections.Counter()
    kept = []
    for r in rows:
        t = r["text"]
        if args.speakers and r["speaker"] not in args.speakers:
            dropped["speaker not selected"] += 1
            continue
        if not args.keep_boilerplate and is_boilerplate(t):
            dropped["boilerplate"] += 1
            continue
        if len(t) < args.min_chars:
            dropped["too short"] += 1
            continue
        if args.max_chars and len(t) > args.max_chars:
            dropped["too long"] += 1
            continue
        if args.require_endpunct and not ends_sentence(t):
            dropped["no end punctuation"] += 1
            continue
        kept.append(r)

    sfx = args.suffix
    out = args.dataset

    write_lines(
        os.path.join(out, "metadata%s.csv" % sfx),
        ["%s|%s|%s" % (r["wav"], r["speaker"], csv_safe(r["text"])) for r in kept],
        fields=3,
    )
    fem = [r for r in kept if r["sex"] == "female"]
    write_lines(
        os.path.join(out, "metadata-female%s.csv" % sfx),
        ["%s|%s|%s" % (r["wav"], r["speaker"], csv_safe(r["text"])) for r in fem],
        fields=3,
    )

    per_dir = os.path.join(out, "single-speaker%s" % sfx)
    os.makedirs(per_dir, exist_ok=True)
    by_spk = collections.defaultdict(list)
    for r in kept:
        by_spk[r["speaker"]].append(r)
    for spk, rs in by_spk.items():
        write_lines(
            os.path.join(per_dir, "%s.csv" % spk),
            ["%s|%s" % (r["wav"], csv_safe(r["text"])) for r in rs],
            fields=2,
        )

    hours = collections.Counter()
    clips = collections.Counter()
    sex = {}
    for r in kept:
        hours[r["speaker"]] += r["duration"]
        clips[r["speaker"]] += 1
        sex[r["speaker"]] = r["sex"]

    print("dropped:")
    for k, n in dropped.most_common():
        print("  %-22s %6s" % (k, format(n, ",")))
    print("  %-22s %6s" % ("TOTAL DROPPED", format(sum(dropped.values()), ",")))

    print("\n%-28s %-7s %9s %8s" % ("speaker", "sex", "clips", "hours"))
    print("-" * 56)
    order = sorted(by_spk, key=lambda s: (sex[s] != "female", -hours[s]))
    for s in order:
        print("%-28s %-7s %9s %8.2f"
              % (s, sex[s], format(clips[s], ","), hours[s] / 3600))
    print("-" * 56)
    print("%-28s %-7s %9s %8.2f"
          % ("TOTAL", "", format(len(kept), ","), sum(hours.values()) / 3600))
    fh = sum(hours[s] for s in order if sex[s] == "female") / 3600
    print("%-28s %-7s %9s %8.2f" % ("  of which female", "", format(len(fem), ","), fh))

    print("\n  metadata%s.csv / metadata-female%s.csv / single-speaker%s/"
          % (sfx, sfx, sfx))
    print("  num_speakers for training: %d" % len(by_spk))


if __name__ == "__main__":
    main()
