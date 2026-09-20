"""
scoring/run_scoring.py

อ่าน features 2 ไฟล์ -> คำนวณคะแนน 5 ด้าน -> รวม -> บันทึก
Input : output/features/{key}.json + output/features/{key}_speech.json
Output: output/scores/{key}.json
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dimension_scores import (
    score_eye_contact,
    score_head_pose,
    score_hand_gesture,
    score_facial_expression,
    score_answer_quality,
)
from fusion import weighted_sum, DEFAULT_WEIGHTS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "output" / "features"
SCORES_DIR   = PROJECT_ROOT / "output" / "scores"


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_scores(key, visual, speech):
    dims = {
        "eye_contact":       score_eye_contact(visual.get("gaze", {})),
        "head_pose":         score_head_pose(visual.get("head_pose", {})),
        "hand_gesture":      score_hand_gesture(visual.get("hand_gesture", {})),
        "facial_expression": score_facial_expression(visual.get("facial_expression", {})),
        "answer_quality":    score_answer_quality(speech or {}),
    }
    fusion = weighted_sum(dims, DEFAULT_WEIGHTS)
    return {
        "key": key,
        "dimensions": dims,
        "fusion": fusion,
    }


def main():
    print("=" * 60)
    print("  SCORING")
    print("=" * 60)

    if not FEATURES_DIR.exists():
        print(f"[ERROR] ไม่พบ: {FEATURES_DIR}")
        sys.exit(1)

    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    # หา key ทั้งหมดจาก visual features
    visual_files = sorted(
        p for p in FEATURES_DIR.glob("*.json")
        if not p.stem.endswith("_speech")
    )

    if not visual_files:
        print(f"[ERROR] ไม่พบไฟล์ features ใน {FEATURES_DIR}")
        sys.exit(1)

    print(f"[INFO] Output: {SCORES_DIR.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] พบ {len(visual_files)} ตัวอย่าง\n")

    for vf in visual_files:
        key = vf.stem
        speech_file = FEATURES_DIR / f"{key}_speech.json"

        print(f"[RUN ] {key}")

        visual = load_json(vf)
        speech = load_json(speech_file) if speech_file.exists() else None
        if speech is None:
            print(f"       [WARN] ไม่พบ {speech_file.name} → answer_quality = 0")

        result = compute_scores(key, visual, speech)

        out = SCORES_DIR / f"{key}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        d = result["dimensions"]
        print(f"       eye_contact      = {d['eye_contact']['score']:5.1f}")
        print(f"       head_pose        = {d['head_pose']['score']:5.1f}")
        print(f"       hand_gesture     = {d['hand_gesture']['score']:5.1f}")
        print(f"       facial_expression= {d['facial_expression']['score']:5.1f}")
        print(f"       answer_quality   = {d['answer_quality']['score']:5.1f}")
        print(f"       -> OVERALL       = {result['fusion']['overall_score']:5.1f}\n")

    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()