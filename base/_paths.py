"""
base/_paths.py

Centralized paths — หา ROOT จากตำแหน่งไฟล์เอง
ไม่ต้องใช้ .env, ไม่ต้องนับ .parent ในไฟล์อื่น

Usage (ในไฟล์อื่น):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))
    from base._paths import AUDIO_DIR, EVIDENCE_DIR
"""

from pathlib import Path


# ============================================================
# ROOT
# ============================================================

# ไฟล์นี้อยู่ที่ root/base/_paths.py
#   parent        = base/
#   parent.parent = root ✅
ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# INPUT
# ============================================================

INPUT_DIR       = ROOT / "input"
VIDEOS_DIR      = INPUT_DIR / "videos"
METADATA_DIR    = INPUT_DIR / "metadata"
METADATA_FILE   = METADATA_DIR / "metadata.json"
QUESTIONS_DIR   = INPUT_DIR / "questions"
GROUND_TRUTH_DIR = INPUT_DIR / "ground_truth"

# AVI-Personality (main dataset) — videos + labels, see base/dataset/avi.py
AVI_DIR         = INPUT_DIR / "AVI-Personality"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR      = ROOT / "output"

AUDIO_DIR       = OUTPUT_DIR / "audio"
FRAMES_DIR      = OUTPUT_DIR / "frames"
LANDMARKS_DIR   = OUTPUT_DIR / "landmarks"
EVIDENCE_DIR    = OUTPUT_DIR / "evidence"
FEATURES_DIR    = OUTPUT_DIR / "features"
SCORES_DIR      = OUTPUT_DIR / "scores"
FEEDBACK_DIR    = OUTPUT_DIR / "feedback"


# ============================================================
# MODELS
# ============================================================

MODELS_DIR       = ROOT / "models"
FACE_MODEL       = MODELS_DIR / "face_landmarker.task"
HAND_MODEL       = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL       = MODELS_DIR / "pose_landmarker_lite.task"


# ============================================================
# APPROACHES (สำหรับอนาคต)
# ============================================================

APPROACHES_DIR   = ROOT / "approaches"
APPROACH_1_DIR   = APPROACHES_DIR / "approach_1_rule"
APPROACH_2_DIR   = APPROACHES_DIR / "approach_2_mllm_zero_shot"
APPROACH_3_DIR   = APPROACHES_DIR / "approach_3_hybrid"


# ============================================================
# VALIDATION
# ============================================================

VALIDATION_DIR   = ROOT / "validation"
REPORT_DIR       = ROOT / "report"


# ============================================================
# HELPERS
# ============================================================

def ensure_dirs(*dirs):
    """สร้างโฟลเดอร์ที่ระบุ ถ้ายังไม่มี"""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def relative(path):
    """คืน path แบบ relative กับ ROOT (สำหรับ print log)"""
    try:
        return Path(path).relative_to(ROOT)
    except ValueError:
        return path


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  BASE PATHS")
    print("=" * 60)
    print(f"ROOT           = {ROOT}")
    print()
    print(f"AUDIO_DIR      = {relative(AUDIO_DIR)}")
    print(f"FRAMES_DIR     = {relative(FRAMES_DIR)}")
    print(f"LANDMARKS_DIR  = {relative(LANDMARKS_DIR)}")
    print(f"EVIDENCE_DIR   = {relative(EVIDENCE_DIR)}")
    print(f"MODELS_DIR     = {relative(MODELS_DIR)}")
    print()
    print("=" * 60)
    print("  CHECK EXIST")
    print("=" * 60)
    checks = [
        ("input/videos",      VIDEOS_DIR),
        ("input/metadata",    METADATA_DIR),
        ("output/audio",      AUDIO_DIR),
        ("output/frames",     FRAMES_DIR),
        ("output/landmarks",  LANDMARKS_DIR),
        ("models",            MODELS_DIR),
    ]
    for label, p in checks:
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status:7s}]  {label}")
    print("=" * 60)