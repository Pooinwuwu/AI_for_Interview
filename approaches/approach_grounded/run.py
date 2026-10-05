"""
approaches/approach_grounded/run.py

Approach C (default) and D (--with-video): grounded coaching.

  1. judge    scores: dimension bands from Approach 1 measurement, overall from the
              trained model (Approach 1b) when available, otherwise Approach 1's overall
  2. writer   Gemini writes feedback from the cited evidence log (+ video for D)
  3. checker  verify.py checks every point; failures go back to Gemini ONCE to fix;
              points that still fail are removed before the user sees them

Output: output/feedback_grounded/<key>_feedback.json        (C)
        output/feedback_grounded_video/<key>_feedback.json  (D)
  final feedback in the shared format (overall_summary_en, strengths_en, improvements ...)
  plus: ratings (from the judge), evidence_log, draft (before checking), verification
  (problems per round, removed points). draft vs final = effect of the checker (RQ1).

Needs: Tier-1 outputs (run_pipeline.py) and Approach 1 scores. Speech evidence needs the
transcript step; without it only visual evidence is available.

Usage
  python approaches/approach_grounded/run.py                  # C, all clips with evidence
  python approaches/approach_grounded/run.py --with-video     # D
  python approaches/approach_grounded/run.py --only pilot_01 --force

AVI clips are skipped unless --allow-avi (AVI agreement: ask the authors before sending
participant data to Gemini).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "approaches" / "approach_1b_learned"))

from base._paths import EVIDENCE_DIR, OUTPUT_DIR, SCORES_DIR, VIDEOS_DIR
from base.dataset.avi import parse_key
from base.grounded_log import build as build_log, to_prompt_text
from base.learned_features import clip_row
from base.llm_common import (MODEL_NAME, TEMPERATURE, BusyGuard, ModelBusy, call_with_retry,
                             gen_config, BANDS_TH, DIMENSIONS)
from prompt import FEEDBACK_SCHEMA, PROMPT_VERSION, build_prompt, build_revision_prompt
import verify

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUT_C = OUTPUT_DIR / "feedback_grounded"
OUT_D = OUTPUT_DIR / "feedback_grounded_video"
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")


# ============================================================
# JUDGE: scores (not from the LLM)
# ============================================================

def judge(key: str) -> tuple:
    """Returns (ratings, overall_note, details)."""
    rule = json.loads((SCORES_DIR / f"{key}.json").read_text(encoding="utf-8"))
    ratings = {d: rule["dimensions"][d]["band"]["label_th"] for d in DIMENSIONS
               if d in rule.get("dimensions", {})}
    details = {"rule_overall_score": rule.get("fusion", {}).get("overall_score")}
    try:
        import learned_model as m1b
        if m1b.MODEL_FILE.exists():
            o = m1b.score_row(m1b.load(), clip_row(key))
            details["learned"] = o
            if o["overall_score"] is not None:
                ratings["overall"] = o["band"]["label_th"]
                note = (f"trained model: better than {o['overall_score']:.0f}% of practice "
                        f"candidates" + ("" if o["reliable"] else
                                         "; this clip differs from the training data, treat as rough"))
                return ratings, note, details
    except Exception as e:                      # model missing / sklearn missing -> fall back
        details["learned_error"] = str(e)[:200]
    band = rule.get("fusion", {}).get("overall_band", {}).get("label_th")
    ratings["overall"] = band if band in BANDS_TH else "ปานกลาง"
    return ratings, "rule-based measurement (no trained model available)", details


# ============================================================
# WRITER: Gemini
# ============================================================

class Writer:
    def __init__(self, with_video: bool):
        from dotenv import load_dotenv
        from google import genai
        load_dotenv(ROOT / ".env")
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("[ERROR] GEMINI_API_KEY missing in .env")
        self.client = genai.Client(api_key=key)
        self.with_video = with_video
        self.video = None

    def upload(self, path: Path):
        f = self.client.files.upload(file=str(path))
        while f.state.name == "PROCESSING":
            time.sleep(3)
            f = self.client.files.get(name=f.name)
        if f.state.name == "FAILED":
            raise RuntimeError("video processing failed")
        self.video = f

    def cleanup(self):
        if self.video is not None:
            try:
                self.client.files.delete(name=self.video.name)
            finally:
                self.video = None

    def write(self, prompt: str) -> dict:
        contents = [self.video, prompt] if self.video is not None else prompt
        return call_with_retry(lambda: json.loads(self.client.models.generate_content(
            model=MODEL_NAME, contents=contents,
            config=gen_config(FEEDBACK_SCHEMA, TEMPERATURE)).text))


def find_video(key: str):
    for ext in VIDEO_EXT:
        for p in (VIDEOS_DIR / f"{key}{ext}", VIDEOS_DIR / f"{key}{ext.upper()}"):
            if p.exists():
                return p
    return None


# ============================================================
# ONE CLIP
# ============================================================

def to_shared_format(fb: dict) -> dict:
    """Same field names as approaches 1-3, so evaluate/hallucination scripts read it."""
    out = dict(fb)
    out["strengths_en"] = [s["text_en"] for s in fb.get("strengths", [])]
    out["strengths_th"] = [s["text_th"] for s in fb.get("strengths", [])]
    return out


def run_clip(key: str, writer: Writer, with_video: bool, max_rounds: int = 1) -> dict:
    ratings, note, details = judge(key)
    log = build_log(key)
    prompt = build_prompt(key, to_prompt_text(log), ratings, note, with_video)

    if with_video:
        video = find_video(key)
        if video is None:
            raise FileNotFoundError(f"no video for {key} in {VIDEOS_DIR}")
        writer.upload(video)
    try:
        draft = writer.write(prompt)
        rounds = [verify.check(draft, log, allow_video=with_video)]
        current = draft
        for _ in range(max_rounds):
            if not rounds[-1]["problems"]:
                break
            current = writer.write(build_revision_prompt(
                prompt, json.dumps(current, ensure_ascii=False), rounds[-1]["problems"]))
            rounds.append(verify.check(current, log, allow_video=with_video))
    finally:
        writer.cleanup()

    final, removed = verify.drop_failed(current, rounds[-1])
    return {
        **to_shared_format(final),
        "ratings": ratings,
        "key": key,
        "approach": "D_grounded_video" if with_video else "C_grounded",
        "model": MODEL_NAME, "temperature": TEMPERATURE, "prompt_version": PROMPT_VERSION,
        "scoring": {"overall_note": note, **details},
        "verification": {
            "rounds": rounds,
            "revised": len(rounds) > 1,
            "removed": removed,
            "summary_unresolved": rounds[-1]["summary_problems"],
            "draft_problems": len(rounds[0]["problems"]),
            "final_problems": sum(len(i["problems"]) for i in rounds[-1]["items"]
                                  if i["status"] != "failed") + len(rounds[-1]["summary_problems"]),
        },
        "draft": to_shared_format(draft),
        "evidence_log": log,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with-video", action="store_true", help="approach D (also send the video)")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rounds", type=int, default=1, help="revision rounds after checking (0 = none)")
    ap.add_argument("--tag", default="",
                    help="repeat run name (e.g. r2): writes to <output folder>_<tag> (stability)")
    ap.add_argument("--allow-avi", action="store_true",
                    help="also send AVI clips (transcript / video) to Gemini - only with written "
                         "approval from the AVI authors")
    args = ap.parse_args()

    out_dir = OUT_D if args.with_video else OUT_C
    if args.tag:
        out_dir = out_dir.with_name(f"{out_dir.name}_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = args.only or sorted(p.stem.removesuffix("_evidence")
                               for p in EVIDENCE_DIR.glob("*_evidence.json"))
    avi = [k for k in keys if parse_key(k) is not None]
    if avi and not args.allow_avi:
        print(f"[INFO] skipping {len(avi)} AVI clip(s): sending AVI data to Gemini needs the "
              f"authors' approval (use --allow-avi once you have it)")
        keys = [k for k in keys if parse_key(k) is None]
    name = "D (evidence + video)" if args.with_video else "C (evidence, no video)"
    print("=" * 66)
    print(f"  Grounded coaching {name}   model={MODEL_NAME}   clips={len(keys)}")
    print("=" * 66)

    writer = Writer(args.with_video)
    guard = BusyGuard()
    for key in keys:
        out = out_dir / f"{key}_feedback.json"
        if out.exists() and not args.force:
            print(f"[SKIP] {key} (exists, use --force)")
            continue
        if not (SCORES_DIR / f"{key}.json").exists():
            print(f"[SKIP] {key}: no Approach 1 scores (run run_pipeline.py)")
            continue
        print(f"--> {key}")
        try:
            res = run_clip(key, writer, args.with_video, args.rounds)
        except ModelBusy as e:
            if guard.failed(e):
                break
            continue
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue
        guard.ok()
        out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        v = res["verification"]
        print(f"    overall {res['ratings'].get('overall')}  | checker: draft problems "
              f"{v['draft_problems']}, revised {'yes' if v['revised'] else 'no'}, "
              f"removed {len(v['removed'])}, left {v['final_problems']}")
    print(f"\nSaved in {out_dir}")


if __name__ == "__main__":
    main()
