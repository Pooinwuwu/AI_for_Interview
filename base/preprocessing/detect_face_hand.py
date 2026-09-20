"""
detect_face_hand.py  (MediaPipe Tasks API version)

อ่านเฟรมจาก output/frames/{key}/*.jpg
ใช้ MediaPipe Tasks API สกัด landmark:
  - Face Landmarker : 478 จุด (รวม iris)
  - Hand Landmarker : 21 จุด/มือ (สูงสุด 2 มือ)
  - Pose Landmarker : 33 จุด
บันทึกผล -> output/landmarks/{key}.json
"""

import sys
import json
from pathlib import Path

import cv2
import mediapipe as mp

# ---------- UTF-8 fix สำหรับ Windows ----------
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CONFIG
# ============================================================

FACE_MIN_DETECTION = 0.5
FACE_MIN_TRACKING  = 0.5

HAND_MIN_DETECTION = 0.5
HAND_MIN_TRACKING  = 0.5
MAX_NUM_HANDS      = 2

POSE_MIN_DETECTION = 0.5
POSE_MIN_TRACKING  = 0.5

MAX_FRAMES     = None      # ทดสอบเร็วๆ: ตั้งเป็น 100
PROGRESS_EVERY = 100

# Paths
PROJECT_ROOT      = Path(__file__).resolve().parent.parent
FRAMES_DIR        = PROJECT_ROOT / "output" / "frames"
LANDMARKS_OUT_DIR = PROJECT_ROOT / "output" / "landmarks"
MODEL_DIR         = PROJECT_ROOT / "models"

FACE_MODEL = MODEL_DIR / "face_landmarker.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"


# ============================================================
# CHECK MODELS
# ============================================================

def check_models():
    missing = [p for p in (FACE_MODEL, HAND_MODEL, POSE_MODEL) if not p.exists()]
    if missing:
        print("[ERROR] ไม่พบโมเดล:")
        for p in missing:
            print(f"        {p}")
        print()
        print("ดาวน์โหลดได้จาก:")
        print("  face  : https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
        print("  hand  : https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        print("  pose  : https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
        sys.exit(1)


# ============================================================
# HELPERS
# ============================================================

def parse_timestamp(filename: str):
    """frame_00042_t8.40.jpg -> 8.40"""
    stem = Path(filename).stem
    if "_t" not in stem:
        return None
    try:
        return float(stem.rsplit("_t", 1)[-1])
    except ValueError:
        return None


def landmarks_to_list(landmarks):
    """list[NormalizedLandmark] -> list[dict]"""
    if not landmarks:
        return None
    return [
        {"x": round(lm.x, 4),
         "y": round(lm.y, 4),
         "z": round(lm.z, 4)}
        for lm in landmarks
    ]


# ============================================================
# PROCESS ONE FRAME
# ============================================================

def process_frame(mp_image, timestamp_ms, face_lmk, hand_lmk, pose_lmk):
    """รัน landmarkers บน mp.Image -> dict ของ landmark"""
    face_result = face_lmk.detect_for_video(mp_image, timestamp_ms)
    hand_result = hand_lmk.detect_for_video(mp_image, timestamp_ms)
    pose_result = pose_lmk.detect_for_video(mp_image, timestamp_ms)

    # ---- Face ----
    face_lms = None
    if face_result.face_landmarks:
        face_lms = landmarks_to_list(face_result.face_landmarks[0])

    # ---- Hands ----
    hand_lms = []
    if hand_result.hand_landmarks:
        for i, hand_pts in enumerate(hand_result.hand_landmarks):
            handedness = "Unknown"
            if hand_result.handedness and i < len(hand_result.handedness):
                cats = hand_result.handedness[i]
                if cats:
                    handedness = cats[0].category_name
            hand_lms.append({
                "handedness": handedness,
                "landmarks": landmarks_to_list(hand_pts),
            })

    # ---- Pose ----
    pose_lms = None
    if pose_result.pose_landmarks:
        pose_lms = landmarks_to_list(pose_result.pose_landmarks[0])

    return {"face": face_lms, "hands": hand_lms, "pose": pose_lms}


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(key: str, frame_dir: Path, out_path: Path) -> bool:
    frame_files = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
    if not frame_files:
        print(f"       [SKIP] ไม่พบเฟรมใน {frame_dir}")
        return False

    if MAX_FRAMES is not None:
        frame_files = frame_files[:MAX_FRAMES]

    # ---- Build landmarkers ----
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    face_opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=FACE_MIN_DETECTION,
        min_tracking_confidence=FACE_MIN_TRACKING,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    hand_opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=HAND_MIN_DETECTION,
        min_tracking_confidence=HAND_MIN_TRACKING,
    )
    pose_opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=POSE_MIN_DETECTION,
        min_tracking_confidence=POSE_MIN_TRACKING,
    )

    results = []
    detected_face = detected_hand = detected_pose = 0

    with mp.tasks.vision.FaceLandmarker.create_from_options(face_opts) as face_lmk, \
         mp.tasks.vision.HandLandmarker.create_from_options(hand_opts) as hand_lmk, \
         mp.tasks.vision.PoseLandmarker.create_from_options(pose_opts) as pose_lmk:

        for i, fp in enumerate(frame_files):
            img = cv2.imread(str(fp))
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # timestamp_ms ต้องเป็น int และเพิ่มขึ้นเรื่อยๆ
            ts = parse_timestamp(fp.name)
            if ts is None:
                ts_ms = int(i * 1000 / 5)   # fallback 5 fps
            else:
                ts_ms = int(round(ts * 1000))

            entry = process_frame(mp_image, ts_ms, face_lmk, hand_lmk, pose_lmk)

            results.append({
                "frame_file": fp.name,
                "timestamp":  ts,
                "timestamp_ms": ts_ms,
                **entry,
            })

            if entry["face"]:  detected_face += 1
            if entry["hands"]: detected_hand += 1
            if entry["pose"]:  detected_pose += 1

            if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == len(frame_files):
                pct = 100 * (i + 1) / len(frame_files)
                print(f"       ... {i+1}/{len(frame_files)} ({pct:.0f}%)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "key": key,
        "total_frames": len(frame_files),
        "detection_stats": {
            "face":  detected_face,
            "hands": detected_hand,
            "pose":  detected_pose,
        },
        "frames": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    n = len(frame_files)
    print(f"       [OK]  face {detected_face}/{n} | "
          f"hands {detected_hand}/{n} | pose {detected_pose}/{n}")
    return True


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  DETECT FACE / HANDS / POSE  (Tasks API)")
    print("=" * 60)

    check_models()

    if not FRAMES_DIR.exists():
        print(f"[ERROR] ไม่พบโฟลเดอร์เฟรม: {FRAMES_DIR}")
        print("        รัน extract_frames.py ก่อน")
        sys.exit(1)

    subdirs = sorted([d for d in FRAMES_DIR.iterdir() if d.is_dir()])
    if not subdirs:
        print(f"[ERROR] ไม่มีโฟลเดอร์ย่อยใน {FRAMES_DIR}")
        sys.exit(1)

    LANDMARKS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Input  : {FRAMES_DIR.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] Output : {LANDMARKS_OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] Models : {MODEL_DIR.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] พบ {len(subdirs)} วิดีโอ")
    if MAX_FRAMES is not None:
        print(f"[INFO] โหมดทดสอบ: {MAX_FRAMES} เฟรมแรก/วิดีโอ")
    print()

    success, failed = 0, 0
    for d in subdirs:
        key = d.name
        out_path = LANDMARKS_OUT_DIR / f"{key}.json"
        print(f"[RUN ] {key}")
        try:
            ok = process_video(key, d, out_path)
            if ok:
                size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"       file : {out_path.relative_to(PROJECT_ROOT)} "
                      f"({size_mb:.1f} MB)\n")
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"       [FAIL] {e}\n")
            failed += 1

    print("=" * 60)
    print(f"  เสร็จสิ้น: สำเร็จ {success}  |  ล้มเหลว {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()