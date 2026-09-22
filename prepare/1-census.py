"""Per-reader census of the australian-english-speech dataset.

    python3 prepare/1-census.py                  # readers with >= 0.5 h
    python3 prepare/1-census.py --min-hours 0.1  # include the small ones

Reads only the metadata columns out of the parquet shards (never the audio
blobs), so it is fast and cheap. Answers the one question that decides the
Piper build: which readers have enough audio to train a single-speaker voice on.

Written 2026-09-18 for the Australian Piper voice; step 1 of prepare/.
"""
import argparse
import collections
import glob
import os

import pyarrow.parquet as pq

# This script lives in prepare/, so the repo root is one level up. Every default
# is built from it, which is what makes "clone, then run the stages in order"
# work from any working directory.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA = os.environ.get(
    "AU_SPEECH_DATA",
    os.path.join(ROOT, "source-data", "australian-english-speech", "data"))

COLS = ["reader", "reader_id", "book", "author", "book_id", "section", "duration"]


def census(data_dir):
    secs = collections.Counter()
    clips = collections.Counter()
    books = collections.defaultdict(set)
    rid = {}
    durations = collections.defaultdict(list)

    files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    for i, path in enumerate(files, 1):
        t = pq.read_table(path, columns=COLS)
        d = t.to_pydict()
        for r, i_, b, dur in zip(d["reader"], d["reader_id"], d["book"], d["duration"]):
            secs[r] += dur
            clips[r] += 1
            books[r].add(b)
            rid.setdefault(r, i_)
            durations[r].append(dur)
        print(f"  [{i}/{len(files)}] {os.path.basename(path)}  rows={t.num_rows}", flush=True)

    return secs, clips, books, rid, durations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA,
                    help="parquet shards (default: source-data/, or $AU_SPEECH_DATA)")
    ap.add_argument("--min-hours", type=float, default=0.5)
    args = ap.parse_args()

    secs, clips, books, rid, durations = census(args.data_dir)

    total_h = sum(secs.values()) / 3600
    print(f"\n{'='*92}")
    print(f"TOTAL: {len(secs)} readers, {sum(clips.values()):,} clips, {total_h:.1f} h")
    print(f"{'='*92}\n")

    ranked = secs.most_common()
    print(f"{'#':>3}  {'reader':<34} {'id':>7} {'hours':>7} {'clips':>7} {'bk':>3} {'avg s':>6}")
    print("-" * 92)
    for n, (reader, s) in enumerate(ranked, 1):
        h = s / 3600
        if h < args.min_hours:
            break
        avg = s / clips[reader]
        print(f"{n:>3}  {reader[:34]:<34} {rid[reader]:>7} {h:>7.2f} {clips[reader]:>7,} "
              f"{len(books[reader]):>3} {avg:>6.2f}")

    # How much of the corpus sits in the top-N readers
    print("\nCumulative share of the corpus:")
    run = 0.0
    for n, (reader, s) in enumerate(ranked, 1):
        run += s
        if n in (1, 3, 5, 10, 20, 50, 100, len(ranked)):
            print(f"  top {n:>3}: {run/3600:>7.1f} h  ({run/sum(secs.values())*100:>5.1f}%)")

    # Training-viability buckets. Piper single-speaker wants hours, not minutes.
    print("\nReaders by trainable volume:")
    for lo, hi, label in [(10, 1e9, ">= 10 h  (comfortable single-speaker)"),
                          (5, 10, " 5 - 10 h (viable fine-tune)"),
                          (2, 5, " 2 -  5 h (thin, fine-tune only)"),
                          (1, 2, " 1 -  2 h (marginal)"),
                          (0, 1, " < 1 h    (multi-speaker pool only)")]:
        sel = [r for r, s in secs.items() if lo <= s / 3600 < hi]
        tot = sum(secs[r] for r in sel) / 3600
        print(f"  {label:<40} {len(sel):>4} readers  {tot:>7.1f} h")

    # Book/author spread for the top readers -- LibriVox readers of Australian
    # titles are not necessarily Australian, so the title is a weak prior only.
    print("\nBooks read by the top 12 readers:")
    for reader, s in ranked[:12]:
        print(f"  {reader} ({s/3600:.2f} h): {', '.join(sorted(books[reader])[:4])}"
              + (" ..." if len(books[reader]) > 4 else ""))


if __name__ == "__main__":
    main()
