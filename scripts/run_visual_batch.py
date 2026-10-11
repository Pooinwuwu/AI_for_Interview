"""
scripts/run_visual_batch.py — visual track for more AVI people, one clip at a time, resumable.

run_pipeline.py redoes every clip after a stop. This script processes clip by clip and skips
clips that are already finished, so it can be stopped (Ctrl+C, shutdown) and started again.

Per clip (each step skipped if its output already exists):
  audio -> prosody -> transcript (English WhisperX) -> frames + landmarks (frames deleted after)
  -> gaze/head/hand/face events + evidence log + visual features -> rule scores (M1)

Usage
  python scripts/run_visual_batch.py --manifests eval2 train4
Clips come from input/ground_truth/avi_<name>_subset.csv and must be in input/videos/
(base/dataset/build_subset.py copies them there). Log: output/visual_batch_log.txt
"""

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd

import pipeline as P                       # app/pipeline.py: the same per-clip steps as the app
from base._paths import (AUDIO_DIR, EVIDENCE_DIR, FEATURES_DIR, GROUND_TRUTH_DIR, LANDMARKS_DIR,
                         METADATA_FILE, OUTPUT_DIR, SCORES_DIR, VIDEOS_DIR)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LOG = OUTPUT_DIR / "visual_batch_log.txt"
BACKUP = METADATA_FILE.with_name("metadata_backup_before_visual300.json")


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def restore_app_metadata():
    """build_subset.py rebuilds metadata.json from input/videos only; put back the entries of
    clips that are not there any more (web-app clips, own clips, earlier AVI clips)."""
    if not BACKUP.exists() or not METADATA_FILE.exists():
        return
    old = json.loads(BACKUP.read_text(encoding="utf-8"))
    cur = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    missing = {k: v for k, v in old.items() if k not in cur}
    if missing:
        cur.update(missing)
        METADATA_FILE.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"metadata: restored {len(missing)} entries from the backup")


def find_video(key):
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".MP4"):
        p = VIDEOS_DIR / f"{key}{ext}"
        if p.exists():
            return p
    return None


def done(key):
    return (SCORES_DIR / f"{key}.json").exists() and (FEATURES_DIR / f"{key}.json").exists()


def run_clip(key, fps):
    video = find_video(key)
    if video is None:
        raise FileNotFoundError(f"{key}: video not in {VIDEOS_DIR}")
    wav = AUDIO_DIR / f"{key}.wav"
    if not wav.exists():
        P.audio(key, video)
    if not (EVIDENCE_DIR / f"{key}_prosody.json").exists():
        P.prosody(key, wav)
    if not (EVIDENCE_DIR / f"{key}_transcript.json").exists() or \
            not (FEATURES_DIR / f"{key}_speech.json").exists():
        P.transcript(key, wav, "en")
    if not (LANDMARKS_DIR / f"{key}.json").exists():
        P.landmarks(key, video, fps)
    P.visual(key)
    P.scores(key)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", nargs="+", default=["eval2", "train4"])
    ap.add_argument("--fps", type=int, default=30, help="keep 30: the first 130 people used 30")
    args = ap.parse_args()

    restore_app_metadata()
    keys = []
    for name in args.manifests:
        m = pd.read_csv(GROUND_TRUTH_DIR / f"avi_{name}_subset.csv", dtype=str)
        keys += [k for ks in m.clip_keys for k in ks.split()]
    todo = [k for k in keys if not done(k)]
    log(f"visual batch: {len(keys)} clips in {args.manifests}, {len(keys) - len(todo)} already done, "
        f"{len(todo)} to do")
    t_all = time.time()
    failed = 0
    for i, key in enumerate(todo, 1):
        t0 = time.time()
        try:
            run_clip(key, args.fps)
            spent = time.time() - t0
            left = (time.time() - t_all) / i * (len(todo) - i)
            log(f"[{i}/{len(todo)}] ok {key}  {spent / 60:.1f} min  (about {left / 3600:.1f} h left)")
        except KeyboardInterrupt:
            log("stopped by user - run the same command again to continue")
            sys.exit(1)
        except Exception as e:
            failed += 1
            log(f"[{i}/{len(todo)}] FAILED {key}: {e}")
            traceback.print_exc()
    log(f"visual batch finished in {(time.time() - t_all) / 3600:.1f} h, failed {failed}")


if __name__ == "__main__":
    main()
