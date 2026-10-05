"""
base/dataset/build_subset.py

Pick an evaluation subset of AVI-Personality participants, copy their clips
into input/videos/ (so the existing pipeline and all 3 approaches pick them up
unchanged), and rebuild input/metadata/metadata.json with AVI info.

Sampling is stratified on the primary target (hireability) into 5 quantile bins,
so the subset covers weak, average and strong candidates — otherwise Spearman
on a small sample can look good or bad by luck.

Usage
-----
    # see what would be selected, copy nothing
    python base/dataset/build_subset.py --dry-run

    # default: 50 test participants x generic questions (q1, q2) = 100 clips
    python base/dataset/build_subset.py

    # development subset for tuning bands/weights (never tune on test)
    python base/dataset/build_subset.py --split val --n 30 --name dev

    # all six questions per participant (6x the processing / API cost)
    python base/dataset/build_subset.py --questions 1 2 3 4 5 6

    # move videos that are not in this subset out to input/videos_archive/
    python base/dataset/build_subset.py --archive-others
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base._paths import VIDEOS_DIR, INPUT_DIR, GROUND_TRUTH_DIR, METADATA_FILE
from base.dataset.avi import (
    load_labels, clip_path, make_key, parse_key, load_questions,
    PRIMARY_TARGET, GENERIC_QUESTIONS,
)

import pandas as pd


# ---------- UTF-8 fix Windows ----------
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# SELECT
# ============================================================

def select_participants(split: str, n: int, seed: int, n_bins: int = 5,
                        exclude=()) -> pd.DataFrame:
    labels = load_labels(splits=(split,))
    labels = labels.dropna(subset=[PRIMARY_TARGET])
    labels = labels[~labels.participant_id.isin(set(exclude))]

    if n >= len(labels):
        chosen = labels.copy()
        chosen["stratum"] = pd.qcut(chosen[PRIMARY_TARGET].rank(method="first"),
                                    n_bins, labels=False)
        return chosen.reset_index(drop=True)

    labels = labels.copy()
    # rank first so ties (many 3.0s) don't collapse bins
    labels["stratum"] = pd.qcut(labels[PRIMARY_TARGET].rank(method="first"),
                                n_bins, labels=False)

    # equal share per bin, remainder spread over the first bins
    base, extra = divmod(n, n_bins)
    parts = []
    for b in range(n_bins):
        k = base + (1 if b < extra else 0)
        pool = labels[labels.stratum == b]
        parts.append(pool.sample(n=min(k, len(pool)), random_state=seed + b))
    return pd.concat(parts).sort_values(PRIMARY_TARGET).reset_index(drop=True)


# ============================================================
# COPY
# ============================================================

def copy_clips(chosen: pd.DataFrame, split: str, questions, dry_run: bool):
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    copied = skipped = missing = 0
    keys = []
    for pid in chosen.participant_id:
        for q in questions:
            src = clip_path(pid, q, split)
            dst = VIDEOS_DIR / src.name
            keys.append(src.stem)
            if not src.exists():
                print(f"[MISS] {src}")
                missing += 1
                continue
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                skipped += 1
                continue
            if not dry_run:
                shutil.copy2(src, dst)
            copied += 1
    return keys, copied, skipped, missing


def archive_others(keep_keys, dry_run: bool):
    archive = INPUT_DIR / "videos_archive"
    moved = 0
    for p in VIDEOS_DIR.iterdir():
        if p.is_file() and p.stem not in keep_keys:
            if not dry_run:
                archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(archive / p.name))
            print(f"[ARCHIVE] {p.name}")
            moved += 1
    return moved


# ============================================================
# METADATA
# ============================================================

def rebuild_metadata():
    """Run the normal metadata builder, then add AVI fields to AVI clips."""
    from base.preprocessing.generate_metadata import build_metadata, save_metadata

    metadata = build_metadata()
    questions = load_questions()
    # look the split up per participant: input/videos may hold clips from
    # more than one subset (e.g. dev + eval)
    labels = load_labels()
    split_of = dict(zip(labels.participant_id, labels.split))
    for key, info in metadata.items():
        avi = parse_key(key)
        if avi is None:
            continue
        q = questions[f"q{avi['question_no']}"]
        info.update({
            "dataset": "AVI-Personality",
            "split": split_of.get(avi["participant_id"]),
            "participant_id": avi["participant_id"],
            "question_no": avi["question_no"],
            "question_type": avi["question_type"],
            "question_text": q["text"],
            # asynchronous one-way interview: nobody else is speaking
            "interviewer_present": False,
            "language": "en",
        })
    save_metadata(metadata)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=50, help="number of participants")
    ap.add_argument("--questions", type=int, nargs="+", default=list(GENERIC_QUESTIONS),
                    help="question numbers to include (default: 1 2 = generic)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default="eval", help="manifest name -> avi_<name>_subset.csv")
    ap.add_argument("--exclude-manifest", type=Path, nargs="+", default=None,
                    help="CSV manifest whose participants must not be reused")
    ap.add_argument("--archive-others", action="store_true",
                    help="move videos not in this subset to input/videos_archive/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    exclude = ()
    if args.exclude_manifest:
        exclude = [pid for m in args.exclude_manifest
                   for pid in pd.read_csv(m, dtype=str).participant_id.tolist()]

    chosen = select_participants(args.split, args.n, args.seed, exclude=exclude)
    questions = sorted(set(args.questions))

    print("=" * 60)
    print(f"  AVI subset '{args.name}' — split={args.split}  "
          f"participants={len(chosen)}  questions={questions}")
    print("=" * 60)
    print(chosen.groupby("stratum")[PRIMARY_TARGET]
          .agg(["count", "min", "max"]).round(2).to_string())
    print()

    keys, copied, skipped, missing = copy_clips(chosen, args.split, questions, args.dry_run)
    print(f"Clips: {len(keys)}  copied={copied}  already there={skipped}  missing={missing}"
          + ("  (dry run)" if args.dry_run else ""))

    if args.archive_others:
        moved = archive_others(set(keys), args.dry_run)
        print(f"Archived {moved} other video(s)" + (" (dry run)" if args.dry_run else ""))

    manifest = chosen[["participant_id", "split", "stratum", PRIMARY_TARGET]].copy()
    manifest["questions"] = " ".join(str(q) for q in questions)
    manifest["clip_keys"] = [" ".join(make_key(p, q) for q in questions)
                             for p in manifest.participant_id]
    out = GROUND_TRUTH_DIR / f"avi_{args.name}_subset.csv"
    if not args.dry_run:
        GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out, index=False)
        print(f"Manifest: {out}")
        rebuild_metadata()
        print(f"Metadata: {METADATA_FILE}")
    else:
        print(f"Would write manifest: {out}")


if __name__ == "__main__":
    main()
