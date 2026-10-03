"""
run_pipeline.py — run Tier 1 + Approach 1 scoring on the clips in input/videos/

One command instead of 13. Typical use:

    python base/dataset/build_subset.py --split train --n 80 --name train --archive-others
    python run_pipeline.py

What it does, in order (stops at the first failing step):
    audio -> frames -> landmarks -> prosody -> transcript -> visual events
    -> evidence log -> Approach 1 features -> speech features -> scores

Disk space: frame images are only needed to make landmarks. After the landmark
step, frame folders whose landmarks exist are deleted (they can always be
re-extracted from the video). This also removes leftover frames from earlier
batches, which detect_face_hand.py would otherwise process again.
Use --keep-frames to keep them.

Options:
    --keep-frames        do not delete frame images
    --with-feedback      also run Approach 1 LLM feedback (uses Gemini quota)
    --from STEP          start at this step (e.g. --from transcript after a crash)
    --fps N              frames per second (default: extract_frames.py default, 30)

A log is appended to output/pipeline_log.txt.
"""

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from base._paths import FRAMES_DIR, LANDMARKS_DIR, VIDEOS_DIR, OUTPUT_DIR

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

A1 = Path("approaches") / "approach_1_rule"
STEPS = [
    ("audio",       Path("base/preprocessing/extract_audio.py")),
    ("frames",      Path("base/preprocessing/extract_frames.py")),
    ("landmarks",   Path("base/preprocessing/detect_face_hand.py")),
    ("prosody",     Path("base/measurement/speech/prosody.py")),
    ("transcript",  Path("base/measurement/speech/transcribe.py")),
    ("gaze",        Path("base/measurement/visual/gaze_events.py")),
    ("head",        Path("base/measurement/visual/head_events.py")),
    ("hand",        Path("base/measurement/visual/hand_events.py")),
    ("face",        Path("base/measurement/visual/face_events.py")),
    ("evidence",    Path("base/measurement/build_evidence.py")),
    ("features",    A1 / "features" / "run_extract_all.py"),
    ("speech",      A1 / "features" / "speech_text.py"),
    ("scores",      A1 / "scoring" / "run_scoring.py"),
]
FEEDBACK_STEP = ("feedback", A1 / "llm" / "run_feedback.py")

LOG = OUTPUT_DIR / "pipeline_log.txt"


def log(msg: str):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def clean_frames(only_done: bool = True) -> int:
    """Delete frame folders whose landmarks file exists. Returns folders removed."""
    if not FRAMES_DIR.exists():
        return 0
    removed = 0
    for d in FRAMES_DIR.iterdir():
        if d.is_dir() and (not only_done or (LANDMARKS_DIR / f"{d.name}.json").exists()):
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


def fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main():
    names = [n for n, _ in STEPS] + [FEEDBACK_STEP[0]]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--with-feedback", action="store_true")
    ap.add_argument("--from", dest="start", choices=names, default=None)
    ap.add_argument("--fps", type=int, default=None)
    args = ap.parse_args()

    steps = list(STEPS) + ([FEEDBACK_STEP] if args.with_feedback else [])
    if args.start:
        steps = steps[[n for n, _ in steps].index(args.start):]

    n_videos = len([p for p in VIDEOS_DIR.glob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}])
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    log("=" * 60)
    log(f"PIPELINE START  videos={n_videos}  free disk={free_gb:.1f} GB  "
        f"steps={', '.join(n for n, _ in steps)}")

    # leftover frames from earlier batches would be re-processed by detect_face_hand.py
    if not args.keep_frames and any(n == "landmarks" for n, _ in steps):
        removed = clean_frames(only_done=True)
        if removed:
            log(f"removed {removed} leftover frame folder(s) that already have landmarks")

    t_all = time.time()
    for name, script in steps:
        cmd = [sys.executable, str(ROOT / script)]
        if name == "frames" and args.fps:
            cmd += ["--fps", str(args.fps)]
        log(f"--> {name:11s} {script.as_posix()}")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            log(f"FAILED at step '{name}' (exit {result.returncode}) after {fmt(time.time() - t0)}")
            log(f"Fix the error above, then resume with:  python run_pipeline.py --from {name}")
            sys.exit(result.returncode)
        log(f"    ok {name} ({fmt(time.time() - t0)})")

        if name == "landmarks" and not args.keep_frames:
            removed = clean_frames(only_done=True)
            log(f"    removed {removed} frame folder(s); "
                f"free disk now {shutil.disk_usage(ROOT).free / 1e9:.1f} GB")

    log(f"PIPELINE DONE in {fmt(time.time() - t_all)}")


if __name__ == "__main__":
    main()
