"""
extract_frames.py  (pipeline version)

อ่าน metadata.json -> สุ่มเฟรมภาพจากทุกวิดีโอใน input/videos/
output -> output/frames/{key}/frame_XXXXX_tSS.SS.jpg

หมายเหตุ:
- timestamp ในชื่อไฟล์ (วินาที) ใช้ sync กับ audio / transcript
- ค่า target fps และขนาดภาพ ปรับได้ที่ CONFIG ด้านล่าง
"""

import sys
import json
from pathlib import Path

import cv2

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

# สุ่มกี่เฟรมต่อวินาที (5 fps เพียงพอสำหรับ gaze/head/hand/face)
TARGET_FPS = 5

# ย่อขนาดภาพเพื่อประหยัดพื้นที่และเร็วขึ้น (เก็บ aspect ratio)
# ถ้าไม่ต้องการย่อ ตั้งเป็น None
RESIZE_WIDTH = 640

# รูปแบบภาพ
IMAGE_EXT = "jpg"
JPEG_QUALITY = 90        # 0-100 (ใช้เมื่อ IMAGE_EXT = "jpg")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_ROOT / "input" / "metadata" / "metadata.json"
FRAMES_OUT_DIR = PROJECT_ROOT / "output" / "frames"


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata() -> dict:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"ไม่พบ metadata: {METADATA_FILE}")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# HELPERS
# ============================================================

def resize_keep_ratio(frame, target_w):
    """ย่อภาพโดยคงสัดส่วน ถ้า target_w เป็น None หรือภาพเล็กกว่า ก็คืนค่าเดิม"""
    if target_w is None:
        return frame
    h, w = frame.shape[:2]
    if w <= target_w:
        return frame
    scale = target_w / w
    return cv2.resize(frame, (target_w, int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def save_image(path: Path, frame) -> bool:
    """บันทึกภาพตามนามสกุลที่กำหนด"""
    ext = path.suffix.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    elif ext == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        params = []
    return cv2.imwrite(str(path), frame, params)


# ============================================================
# EXTRACT FRAMES FROM ONE VIDEO
# ============================================================

def extract_frames(video_path: Path, out_dir: Path, target_fps: int) -> int:
    """
    สุ่มเฟรมจากวิดีโอ 1 ไฟล์
    คืนค่าจำนวนเฟรมที่บันทึกสำเร็จ
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"เปิดวิดีโอไม่ได้: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        cap.release()
        raise RuntimeError("อ่าน fps จากวิดีโอไม่ได้")

    # คำนวณว่าต้องข้ามกี่เฟรม
    step = max(1, int(round(src_fps / target_fps)))

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            frame = resize_keep_ratio(frame, RESIZE_WIDTH)
            ts = frame_idx / src_fps     # วินาที
            fname = f"frame_{saved:05d}_t{ts:.2f}.{IMAGE_EXT}"
            save_image(out_dir / fname, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  EXTRACT FRAMES  (pipeline)")
    print("=" * 60)

    metadata = load_metadata()
    if not metadata:
        print("[ERROR] metadata.json ว่างเปล่า")
        sys.exit(1)

    FRAMES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output folder : {FRAMES_OUT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] Target FPS    : {TARGET_FPS}")
    print(f"[INFO] Resize width  : {RESIZE_WIDTH}")
    print(f"[INFO] พบ {len(metadata)} รายการใน metadata")
    print()

    success, failed = 0, 0

    for key, info in metadata.items():
        rel_path = info.get("file_path")
        if not rel_path:
            print(f"[SKIP] {key}: ไม่มี 'file_path' ใน metadata")
            failed += 1
            continue

        video_path = PROJECT_ROOT / "input" / rel_path
        if not video_path.exists():
            print(f"[SKIP] {key}: ไม่พบไฟล์ {video_path}")
            failed += 1
            continue

        out_dir = FRAMES_OUT_DIR / key
        print(f"[RUN ] {key}")
        print(f"       video : {video_path.relative_to(PROJECT_ROOT)}")
        print(f"       out   : {out_dir.relative_to(PROJECT_ROOT)}/")

        try:
            n = extract_frames(video_path, out_dir, TARGET_FPS)
            print(f"       [OK]  {n} เฟรม\n")
            success += 1
        except Exception as e:
            print(f"       [FAIL] {e}\n")
            failed += 1

    print("=" * 60)
    print(f"  เสร็จสิ้น: สำเร็จ {success}  |  ล้มเหลว {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()