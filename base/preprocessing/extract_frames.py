"""
extract_frames.py

อ่าน metadata.json -> สุ่มเฟรมภาพจากทุกวิดีโอ
output -> output/frames/{key}/frame_XXXXX_tSS.SS.jpg
"""

import sys
import json
from pathlib import Path

# ไฟล์อยู่ที่ root/base/preprocessing/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base._paths import ROOT, METADATA_FILE, FRAMES_DIR

import cv2


# ---------- UTF-8 fix Windows ----------
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

# 30 fps = default for head pose / posture / nod / fidget / gaze stability
# (work plan v3 §2.2). Override with:  python extract_frames.py --fps 10
TARGET_FPS = 30
RESIZE_WIDTH = 640
IMAGE_EXT = "jpg"
JPEG_QUALITY = 90


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
    if target_w is None:
        return frame
    h, w = frame.shape[:2]
    if w <= target_w:
        return frame
    scale = target_w / w
    return cv2.resize(frame, (target_w, int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def save_image(path: Path, frame) -> bool:
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
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"เปิดวิดีโอไม่ได้: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        cap.release()
        raise RuntimeError("อ่าน fps จากวิดีโอไม่ได้")

    step = max(1, int(round(src_fps / target_fps)))
    out_dir.mkdir(parents=True, exist_ok=True)

    # remove frames from an earlier run (e.g. at another fps) so they don't mix
    for old in out_dir.glob(f"*.{IMAGE_EXT}"):
        old.unlink()

    saved = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            frame = resize_keep_ratio(frame, RESIZE_WIDTH)
            ts = frame_idx / src_fps
            fname = f"frame_{saved:05d}_t{ts:.2f}.{IMAGE_EXT}"
            save_image(out_dir / fname, frame)
            saved += 1

        frame_idx += 1

    cap.release()

    # record what was extracted (detect_face_hand.py only reads *.jpg)
    with open(out_dir / "_extraction.json", "w", encoding="utf-8") as f:
        json.dump({"target_fps": target_fps, "source_fps": src_fps, "step": step,
                   "effective_fps": round(src_fps / step, 3), "frames": saved}, f, indent=2)
    return saved


# ============================================================
# MAIN
# ============================================================

def main():
    import argparse
    global TARGET_FPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=TARGET_FPS,
                    help=f"frames per second to keep (default {TARGET_FPS})")
    TARGET_FPS = ap.parse_args().fps

    print("=" * 60)
    print("  EXTRACT FRAMES  (pipeline)")
    print("=" * 60)

    metadata = load_metadata()
    if not metadata:
        print("[ERROR] metadata.json ว่างเปล่า")
        sys.exit(1)

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output folder : {FRAMES_DIR}")
    print(f"[INFO] Target FPS    : {TARGET_FPS}")
    print(f"[INFO] Resize width  : {RESIZE_WIDTH}")
    print(f"[INFO] พบ {len(metadata)} รายการ\n")

    success, failed = 0, 0

    for key, info in metadata.items():
        rel_path = info.get("file_path")
        if not rel_path:
            print(f"[SKIP] {key}: ไม่มี 'file_path'")
            failed += 1
            continue

        video_path = ROOT / "input" / rel_path
        if not video_path.exists():
            print(f"[SKIP] {key}: ไม่พบ {video_path}")
            failed += 1
            continue

        out_dir = FRAMES_DIR / key
        print(f"[RUN ] {key}")
        print(f"       video : {video_path.name}")
        print(f"       out   : {out_dir.name}/")

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