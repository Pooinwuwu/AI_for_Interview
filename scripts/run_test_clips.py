"""
scripts/run_test_clips.py - run the staged test clips (RQ2 / RQ3) through the web-app pipeline.

Reads the clip list, language and question from a label file in input/video_test_web/, measures
each clip once (same steps as the app), then runs every coach asked for. Resumable: finished
steps are skipped, so it can be stopped with Ctrl+C and started again.

Usage
  python scripts/run_test_clips.py                          # coaches: ollama (main) + local
  python scripts/run_test_clips.py --coaches ollama local gemini groq
  python scripts/run_test_clips.py --only P1_C02 P2_C02     # a few clips first
Clip keys: test_P1_C01 ... ; results: output/feedback_app/by_coach/test_<clip>__<coach>.json
Log: output/test_clips_log.txt
"""

import argparse
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import openpyxl

import pipeline as P
from base._paths import FEATURES_DIR, OUTPUT_DIR, SCORES_DIR

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CLIPS_DIR = ROOT / "input" / "video_test_web"
LOG = OUTPUT_DIR / "test_clips_log.txt"
BY_COACH = OUTPUT_DIR / "feedback_app" / "by_coach"
QUESTION = {"แนะนำตัว": "intro", "จุดแข็ง": "q1", "เพื่อนสนิท": "q2", "ตัดสินใจ": "q3",
            "ทีมมีปัญหา": "q4", "เห็นต่าง": "q5", "งานเร่งด่วน": "q6"}


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def clip_list():
    """[(clip, lang, question_id)] from the first label file (they share the grey columns)."""
    files = sorted(CLIPS_DIR.glob("label_*.xlsx"))
    if not files:
        sys.exit(f"no label_*.xlsx in {CLIPS_DIR}")
    ws = openpyxl.load_workbook(files[0], data_only=True)["label"]
    out = {}
    for row in ws.iter_rows(min_row=3, values_only=True):
        name, lang, q = row[1], row[3], row[4]
        if not name or not lang or not q:
            continue
        clip = Path(str(name)).stem
        qid = next((v for k, v in QUESTION.items() if str(q).startswith(k)), None)
        if qid is None:
            sys.exit(f"unknown question for {clip}: {q}")
        out.setdefault(clip, ("th" if lang == "ไทย" else "en", qid))
    return [(c, *v) for c, v in out.items()]


def find_video(clip):
    for p in CLIPS_DIR.iterdir():
        if p.stem == clip and p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
            return p
    return None


def measured(key):
    return (SCORES_DIR / f"{key}.json").exists() and (FEATURES_DIR / f"{key}.json").exists()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coaches", nargs="+", default=["ollama", "local"], choices=P.COACH_MODES)
    ap.add_argument("--only", nargs="+", default=None, help="clip names, e.g. P1_C02")
    a = ap.parse_args()

    clips = [c for c in clip_list() if not a.only or c[0] in a.only]
    log(f"test clips: {len(clips)}  coaches: {', '.join(a.coaches)}")
    t_all = time.time()
    for n, (clip, lang, qid) in enumerate(clips, 1):
        key = f"test_{clip}"
        try:
            if not measured(key):
                video = find_video(clip)
                if video is None:
                    log(f"[{n}/{len(clips)}] MISSING video for {clip}")
                    continue
                t0 = time.time()
                P.process(video, qid, lang, coach_mode="local", key=key)   # measure + quick local coach
                log(f"[{n}/{len(clips)}] measured {clip} ({lang}, {qid}) in {time.time() - t0:.0f} s")
            for c in a.coaches:
                if (BY_COACH / f"{key}__{c}.json").exists():
                    continue
                t0 = time.time()
                r = P.coach(key, c)
                v = r.get("verification", {})
                log(f"[{n}/{len(clips)}] {clip} coach={c:7s} {time.time() - t0:5.0f} s  "
                    f"strengths={len(r.get('strengths', []))} improvements={len(r.get('improvements', []))} "
                    f"removed={len(v.get('removed', []) or [])}")
        except KeyboardInterrupt:
            log("stopped by user - run the same command again to continue")
            sys.exit(1)
        except Exception as e:
            log(f"[{n}/{len(clips)}] FAILED {clip}: {str(e)[:300]}")
            traceback.print_exc()
    log(f"done in {(time.time() - t_all) / 60:.0f} min. Compare: python scripts/compare_coaches.py")


if __name__ == "__main__":
    main()
