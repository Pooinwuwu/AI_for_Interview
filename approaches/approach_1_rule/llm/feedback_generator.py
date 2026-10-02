"""
llm/feedback_generator.py
Pipeline: scores + speech → prompt → Gemini → feedback JSON
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from llm_client import GeminiClient
from prompt_builder import build_prompt, FEEDBACK_SCHEMA
from base.llm_common import (MODEL_NAME, BANDS_TH, BANDS_EN, DIMENSIONS,
                             run_info, parse_run_args, should_skip, BusyGuard)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCORES_DIR   = PROJECT_ROOT / "output" / "scores"
FEATURES_DIR = PROJECT_ROOT / "output" / "features"
FEEDBACK_DIR = PROJECT_ROOT / "output" / "feedback"


def _load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def rule_ratings(scores: dict) -> dict:
    """Bands measured by the rule system (source of truth for Approach 1)."""
    en_to_th = dict(zip(BANDS_EN, BANDS_TH))
    def th(band):
        return band.get("label_th") or en_to_th.get(band.get("label_en"), "")
    dims = scores.get("dimensions", {})
    out = {d: th(dims.get(d, {}).get("band", {})) for d in DIMENSIONS}
    out["overall"] = th(scores.get("fusion", {}).get("overall_band", {}))
    return out


def generate_feedback(key: str,
                      client: GeminiClient) -> dict:
    """สร้าง feedback สำหรับ 1 ตัวอย่าง"""

    scores_path = SCORES_DIR / f"{key}.json"
    speech_path = FEATURES_DIR / f"{key}_speech.json"

    if not scores_path.exists():
        raise FileNotFoundError(f"ไม่พบ scores: {scores_path}")
    if not speech_path.exists():
        raise FileNotFoundError(f"ไม่พบ speech: {speech_path}")

    scores = _load_json(scores_path)
    speech = _load_json(speech_path)

    prompt = build_prompt(scores, speech)

    # เก็บ prompt ไว้ debug
    prompt_path = FEEDBACK_DIR / f"{key}_prompt.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    feedback = client.generate_json(prompt, FEEDBACK_SCHEMA)

    # ratings = measured bands, never the LLM's copy of them
    llm_ratings = feedback.get("ratings")
    feedback["ratings"] = rule_ratings(scores)
    if llm_ratings and llm_ratings != feedback["ratings"]:
        feedback["ratings_llm_mismatch"] = llm_ratings

    # ใส่ metadata
    feedback["key"] = key
    feedback.update(run_info())
    feedback["approach"] = "rule"
    feedback["overall_score"] = scores.get("fusion", {}).get("overall_score", 0)

    return feedback


def main():
    args = parse_run_args("Approach 1 — LLM feedback from rule-based scores")
    print("=" * 60)
    print("  LLM FEEDBACK GENERATOR  (Gemini)")
    print("=" * 60)

    if not SCORES_DIR.exists():
        print(f"[ERROR] ไม่พบ: {SCORES_DIR}")
        print("        รัน scoring/run_scoring.py ก่อน")
        sys.exit(1)

    score_files = sorted(SCORES_DIR.glob("*.json"))
    if args.only:
        score_files = [f for f in score_files if f.stem in set(args.only)]
    if not score_files:
        print(f"[ERROR] ไม่มีไฟล์ใน {SCORES_DIR}")
        sys.exit(1)

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Model   : {MODEL_NAME}")
    print(f"[INFO] Scores  : {SCORES_DIR.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] Output  : {FEEDBACK_DIR.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] พบ {len(score_files)} ตัวอย่าง\n")

    try:
        client = GeminiClient()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    success, failed = 0, 0
    guard = BusyGuard()

    for sf in score_files:
        key = sf.stem
        out_path = FEEDBACK_DIR / f"{key}_feedback.json"
        if should_skip(out_path, args.force):
            continue
        print(f"[RUN ] {key}")

        try:
            feedback = generate_feedback(key, client)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(feedback, f, ensure_ascii=False, indent=2)

            # print สรุปสั้น
            n_str = len(feedback.get("strengths_en", []))
            n_imp = len(feedback.get("improvements", []))
            print(f"       strengths={n_str}  improvements={n_imp}")
            print(f"       -> {out_path.relative_to(PROJECT_ROOT)}\n")
            success += 1
            guard.ok()

        except Exception as e:
            print(f"       [FAIL] {e}\n")
            failed += 1
            if guard.failed(e):
                break

    print("=" * 60)
    print(f"  เสร็จสิ้น: สำเร็จ {success}  |  ล้มเหลว {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()