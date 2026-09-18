"""
llm/feedback_generator.py
Pipeline: scores + speech → prompt → Gemini → feedback JSON
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import GeminiClient
from prompt_builder import build_prompt, FEEDBACK_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR   = PROJECT_ROOT / "output" / "scores"
FEATURES_DIR = PROJECT_ROOT / "output" / "features"
FEEDBACK_DIR = PROJECT_ROOT / "output" / "feedback"


def _load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_feedback(key: str,
                      client: GeminiClient,
                      model_name: str) -> dict:
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

    # ใส่ metadata
    feedback["key"] = key
    feedback["model"] = model_name
    feedback["overall_score"] = scores.get("fusion", {}).get("overall_score", 0)

    return feedback


def main():
    print("=" * 60)
    print("  LLM FEEDBACK GENERATOR  (Gemini)")
    print("=" * 60)

    # เลือกโมเดลจาก config — ถ้าไม่มีจะใช้ default
    MODEL_NAME = "gemini-3.6-flash"

    if not SCORES_DIR.exists():
        print(f"[ERROR] ไม่พบ: {SCORES_DIR}")
        print("        รัน scoring/run_scoring.py ก่อน")
        sys.exit(1)

    score_files = sorted(SCORES_DIR.glob("*.json"))
    if not score_files:
        print(f"[ERROR] ไม่มีไฟล์ใน {SCORES_DIR}")
        sys.exit(1)

    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Model   : {MODEL_NAME}")
    print(f"[INFO] Scores  : {SCORES_DIR.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] Output  : {FEEDBACK_DIR.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] พบ {len(score_files)} ตัวอย่าง\n")

    try:
        client = GeminiClient(model_name=MODEL_NAME)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    success, failed = 0, 0

    for sf in score_files:
        key = sf.stem
        print(f"[RUN ] {key}")

        try:
            feedback = generate_feedback(key, client, MODEL_NAME)

            out_path = FEEDBACK_DIR / f"{key}_feedback.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(feedback, f, ensure_ascii=False, indent=2)

            # print สรุปสั้น
            n_str = len(feedback.get("strengths_en", []))
            n_imp = len(feedback.get("improvements", []))
            print(f"       strengths={n_str}  improvements={n_imp}")
            print(f"       -> {out_path.relative_to(PROJECT_ROOT)}\n")
            success += 1

        except Exception as e:
            print(f"       [FAIL] {e}\n")
            failed += 1

    print("=" * 60)
    print(f"  เสร็จสิ้น: สำเร็จ {success}  |  ล้มเหลว {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()