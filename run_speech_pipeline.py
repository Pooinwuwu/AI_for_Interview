"""
run_speech_pipeline.py — audio + text track for the FULL AVI dataset (no video frames)

Methods M0, M2 (audio_text), M3, M4 (text+audio), M5 need only audio, transcript and
prosody, so they can use every AVI participant (train 452 / dev 64 / test 130) without
the 30-40 GB that frames + landmarks would take. Visual methods stay on the subset.

Steps (each skips clips that are already done, so the run can be stopped and resumed):
  audio       AVI clip (straight from input/AVI-Personality/<split>/) -> output/audio/<key>.wav
  prosody     -> output/evidence/<key>_prosody.json
  transcript  WhisperX -> output/evidence/<key>_transcript.json     (the slow step)
  speech      speech features -> output/features/<key>_speech.json
  embed       text embedding (M3) -> output/embeddings/text_minilm.npz

Usage
  python run_speech_pipeline.py                         # all splits, q1 + q2
  python run_speech_pipeline.py --splits train val      # without the test split
  python run_speech_pipeline.py --from transcript       # resume at a step
Log: output/speech_pipeline_log.txt
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from base._paths import AUDIO_DIR, OUTPUT_DIR
from base.dataset.avi import SPLITS, AVI_DIR, load_labels, make_key

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LOG = OUTPUT_DIR / "speech_pipeline_log.txt"
STEPS = ["audio", "prosody", "transcript", "speech", "embed"]
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


def avi_clips(splits, questions):
    labels = load_labels(tuple(splits))
    for r in labels.itertuples():
        for q in questions:
            key = make_key(r.participant_id, q)
            yield key, AVI_DIR / SPLITS[r.split] / f"{key}.mp4"


def step_audio(splits, questions):
    from base.preprocessing.extract_audio import check_ffmpeg, extract_audio
    if not check_ffmpeg():
        sys.exit("[ERROR] ffmpeg not found")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    made = have = missing = failed = 0
    for key, video in avi_clips(splits, questions):
        out = AUDIO_DIR / f"{key}.wav"
        if out.exists():
            have += 1
            continue
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
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=list(SPLITS))
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--from", dest="start", choices=STEPS, default="audio")
    ap.add_argument("--embed-model", default="minilm")
    args = ap.parse_args()

    steps = STEPS[STEPS.index(args.start):]
    log("=" * 60)
    log(f"SPEECH PIPELINE  splits={args.splits}  questions={args.questions}  steps={', '.join(steps)}")
    t_all = time.time()
    for name in steps:
        t0 = time.time()
        log(f"--> {name}")
        if name == "audio":
            ok = step_audio(args.splits, args.questions)
        else:
            cmd = [sys.executable, str(SCRIPTS[name])]
            if name == "embed":
                cmd += ["--model", args.embed_model]
            ok = subprocess.run(cmd, cwd=str(ROOT)).returncode == 0
            if not ok and name == "transcript":
                # a few clips failing (silence, corrupt audio) should not stop the rest
                log("    [WARN] some transcripts failed (listed above) - continuing; "
                    "those clips are left out of the speech features")
                ok = True
        if not ok:
            log(f"FAILED at '{name}'. Fix the error, then: python run_speech_pipeline.py --from {name}")
            sys.exit(1)
        log(f"    ok {name} ({(time.time() - t0) / 60:.1f} min)")
    log(f"DONE in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
