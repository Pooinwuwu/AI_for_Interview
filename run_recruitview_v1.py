"""
run_recruitview.py — audio + text track on a sample of RecruitView (cross-dataset test only)

RecruitView is used ONLY to evaluate models trained on AVI (no training on it, nothing leaves
this computer, results reported only as aggregates). CC BY-NC 4.0 + the authors' terms:
no hiring use, no identity work, no model trained on it in any app or demo.

Steps (each skips clips already done; the speech scripts process every wav in output/audio):
  sample      one clip per participant (user_no), seeded -> input/ground_truth/recruitview_sample.csv
  audio       ffmpeg -> output/audio/rv_<id>.wav
  prosody     -> output/evidence/rv_<id>_prosody.json
  transcript  WhisperX (our own transcript, same as AVI; the dataset's transcript is not used)
  speech      speech features -> output/features/rv_<id>_speech.json
  embed       text embeddings (all models given with --embed-models)

Usage
  python run_recruitview.py --n 300
  python run_recruitview.py --from transcript          # resume
then
  python validation/crosstest_recruitview.py --text-emb minilm bge-small e5-base
"""

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from base._paths import AUDIO_DIR, GROUND_TRUTH_DIR, INPUT_DIR, OUTPUT_DIR

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RV_DIRS = [INPUT_DIR / "input" / "RecruitView", INPUT_DIR / "RecruitView"]
SAMPLE = GROUND_TRUTH_DIR / "recruitview_sample.csv"
LABELS = ["interview_score", "overall_performance", "answer_score", "speaking_skills",
          "confidence_score", "facial_expression"]
LOG = OUTPUT_DIR / "recruitview_log.txt"
STEPS = ["sample", "audio", "prosody", "transcript", "speech", "embed"]
SCRIPTS = {
    "prosody":    ROOT / "base/measurement/speech/prosody.py",
    "transcript": ROOT / "base/measurement/speech/transcribe.py",
    "speech":     ROOT / "approaches/approach_1_rule/features/speech_text.py",
    "embed":      ROOT / "base/embeddings/text_embed.py",
}


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def rv_dir() -> Path:
    for d in RV_DIRS:
        if (d / "metadata.jsonl").exists():
            return d
    sys.exit("[ERROR] RecruitView not found (looked for metadata.jsonl in "
             + ", ".join(str(d) for d in RV_DIRS) + ")")


def step_sample(n, seed):
    d = rv_dir()
    rows = [json.loads(l) for l in open(d / "metadata.jsonl", encoding="utf-8") if l.strip()]
    rng = random.Random(seed)
    by_user = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        by_user.setdefault(r["user_no"], []).append(r)
    users = sorted(by_user)
    rng.shuffle(users)
    picked = [rng.choice(by_user[u]) for u in users[:n]]     # one clip per participant
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "file", "user_no", "question_id", "question",
                                          "duration", *LABELS])
        w.writeheader()
        for r in sorted(picked, key=lambda r: r["id"]):
            w.writerow({"key": f"rv_{r['id']}", "file": r["file_name"], "user_no": r["user_no"],
                        "question_id": r["question_id"], "question": r["question"],
                        "duration": r.get("duration"), **{k: r[k] for k in LABELS}})
    log(f"    sample: {len(picked)} clips from {len(picked)} participants "
        f"({len(users)} participants in the dataset) -> {SAMPLE.name}")
    return True


def step_audio():
    from base.preprocessing.extract_audio import check_ffmpeg, extract_audio
    if not check_ffmpeg():
        sys.exit("[ERROR] ffmpeg not found")
    d = rv_dir()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    made = have = missing = failed = 0
    for r in csv.DictReader(open(SAMPLE, encoding="utf-8")):
        out = AUDIO_DIR / f"{r['key']}.wav"
        if out.exists():
            have += 1
            continue
        video = d / r["file"]
        if not video.exists():
            missing += 1
            continue
        if extract_audio(video, out):
            made += 1
        else:
            failed += 1
    log(f"    audio: new {made}, already {have}, video missing {missing}, failed {failed}")
    return failed == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="participants (one clip each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from", dest="start", choices=STEPS, default="sample")
    ap.add_argument("--embed-models", nargs="+", default=["minilm"])
    args = ap.parse_args()

    if args.start == "sample" and SAMPLE.exists():
        log(f"    {SAMPLE.name} exists - keeping it (delete it to draw a new sample)")
        args.start = "audio"
    steps = STEPS[STEPS.index(args.start):]
    log("=" * 60)
    log(f"RECRUITVIEW (evaluation only)  steps={', '.join(steps)}")
    t_all = time.time()
    for name in steps:
        t0 = time.time()
        log(f"--> {name}")
        if name == "sample":
            ok = step_sample(args.n, args.seed)
        elif name == "audio":
            ok = step_audio()
        elif name == "embed":
            ok = all(subprocess.run([sys.executable, str(SCRIPTS["embed"]), "--model", m],
                                    cwd=str(ROOT)).returncode == 0 for m in args.embed_models)
        else:
            ok = subprocess.run([sys.executable, str(SCRIPTS[name])], cwd=str(ROOT)).returncode == 0
            if not ok and name == "transcript":
                log("    [WARN] some transcripts failed - continuing without those clips")
                ok = True
        if not ok:
            log(f"FAILED at '{name}'. Fix it, then: python run_recruitview.py --from {name}")
            sys.exit(1)
        log(f"    ok {name} ({(time.time() - t0) / 60:.1f} min)")
    log(f"DONE in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
