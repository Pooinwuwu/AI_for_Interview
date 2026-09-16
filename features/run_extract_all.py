"""features/run_all.py — รันทุก feature extractor"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gaze
import head_pose
import hand_gesture
import facial_expression
from utils import get_video_aspect


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANDMARKS_DIR = PROJECT_ROOT / "output" / "landmarks"
FEATURES_OUT = PROJECT_ROOT / "output" / "features"


def extract_one(key, data):
    # ⭐ อ่าน aspect จาก metadata อัตโนมัติ
    aspect = get_video_aspect(key)

    return {
        "key": key,
        "aspect_used": round(aspect, 4),
        "total_frames": data.get("total_frames"),
        "detection_stats": data.get("detection_stats", {}),
        "gaze":              gaze.extract(data),
        "head_pose":         head_pose.extract(data, aspect=aspect, verbose=True),
        "hand_gesture":      hand_gesture.extract(data),
        "facial_expression": facial_expression.extract(data),
    }


def main():
    print("=" * 60)
    print("  FEATURE EXTRACTION")
    print("=" * 60)

    if not LANDMARKS_DIR.exists():
        print(f"[ERROR] ไม่พบโฟลเดอร์: {LANDMARKS_DIR}")
        sys.exit(1)

    files = sorted(LANDMARKS_DIR.glob("*.json"))
    if not files:
        print(f"[ERROR] ไม่พบไฟล์ landmarks ใน {LANDMARKS_DIR}")
        sys.exit(1)

    FEATURES_OUT.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Input  : {LANDMARKS_DIR.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] Output : {FEATURES_OUT.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] พบ {len(files)} ไฟล์\n")

    for fp in files:
        key = fp.stem
        print(f"[RUN ] {key}")

        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = extract_one(key, data)

        out_path = FEATURES_OUT / f"{key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(features, f, ensure_ascii=False, indent=2)

        g = features["gaze"]
        h = features["head_pose"]
        hd = features["hand_gesture"]
        fe = features["facial_expression"]
        print(f"       eye_contact={g.get('eye_contact_ratio')}  "
              f"pitch_std={h.get('pitch_std')}  "
              f"yaw_std={h.get('yaw_std')}  "
              f"hand_presence={hd.get('hand_presence_ratio')}  "
              f"smile={fe.get('smile_ratio')}")
        print(f"       -> {out_path.relative_to(PROJECT_ROOT)}\n")

    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()