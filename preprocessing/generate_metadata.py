"""
generate_metadata.py
อ่านไฟล์วิดีโอทั้งหมดใน input/videos/ แล้วสร้าง input/metadata/metadata.json
"""

import json
from pathlib import Path

import cv2

# ---------- Config ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "input" / "videos"
METADATA_DIR = PROJECT_ROOT / "input" / "metadata"
METADATA_FILE = METADATA_DIR / "metadata.json"

# ค่า default ที่ผู้ใช้ต้องกรอกเอง (script เดาไม่ได้)
DEFAULT_METADATA = {
    "language": "en",
    "interviewer_present": True,     # ← เปลี่ยนจาก False เป็น True
    "recorded_date": None,           # เช่น "2026-09-10"
    "consent_obtained": False,       # ต้องเซ็ตเป็น True ถ้าจะเผยแพร่
    "notes": ""
}


def probe_video(video_path: Path) -> dict:
    """อ่านค่า duration, fps, resolution จากไฟล์วิดีโอ"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"เปิดไฟล์ไม่ได้: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_sec = round(frame_count / fps, 2) if fps > 0 else None

    return {
        "file_path": f"videos/{video_path.name}",
        "duration_sec": duration_sec,
        "fps": round(fps, 2) if fps else None,
        "resolution": f"{width}x{height}",
        "frame_count": int(frame_count),
    }


def build_metadata() -> dict:
    """สร้าง metadata สำหรับทุกวิดีโอใน input/videos/"""
    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์: {VIDEO_DIR}")

    video_files = sorted(
        [p for p in VIDEO_DIR.iterdir()
         if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    )

    if not video_files:
        raise FileNotFoundError(f"ไม่พบไฟล์วิดีโอใน {VIDEO_DIR}")

    metadata = {}
    for video_path in video_files:
        key = video_path.stem  # เช่น "sample_01"
        try:
            info = probe_video(video_path)
            info.update(DEFAULT_METADATA)
            metadata[key] = info
            print(f"[OK] {key}  ->  {info['duration_sec']}s @ {info['fps']} fps")
        except Exception as e:
            print(f"[ERROR] {video_path.name}: {e}")

    return metadata


def save_metadata(metadata: dict) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\nบันทึกแล้ว: {METADATA_FILE}")


if __name__ == "__main__":
    metadata = build_metadata()
    save_metadata(metadata)