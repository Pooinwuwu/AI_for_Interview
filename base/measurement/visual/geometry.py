"""
base/measurement/visual/geometry.py

MediaPipe returns landmarks normalised separately per axis: x / width, y / height.
Any distance or angle that mixes x and y (eye openness, mouth shape, head pose,
movement speed) is therefore distorted unless x is rescaled by the frame's aspect
ratio (width / height):

    720x1280 phone video (aspect 0.56): vertical distances look 1.8x too large
    640x480  AVI webcam  (aspect 1.33): vertical distances look 0.75x too small

Every Tier-1 visual module uses these helpers so all of them measure in the same,
undistorted units (1.0 = frame height).

video_aspect(key) looks for the frame size in this order:
  1. input/metadata/metadata.json  ("resolution": "720x1280")
  2. input/metadata/frame_sizes.json  (cache written by this module)
  3. the video file itself (input/videos, input/videos_archive, AVI dataset folders)
  4. otherwise 16:9 with a warning
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from base._paths import METADATA_FILE, METADATA_DIR, VIDEOS_DIR, INPUT_DIR

CACHE_FILE = METADATA_DIR / "frame_sizes.json"
DEFAULT_ASPECT = 16 / 9
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".MP4", ".MOV")
SEARCH_DIRS = [VIDEOS_DIR, INPUT_DIR / "videos_archive", INPUT_DIR / "videos_pilot",
               INPUT_DIR / "AVI-Personality" / "Training",
               INPUT_DIR / "AVI-Personality" / "Validation",
               INPUT_DIR / "AVI-Personality" / "Testing"]

_cache = None


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def _parse_res(res):
    try:
        w, h = str(res).lower().split("x")
        w, h = int(w), int(h)
        return (w, h) if w > 0 and h > 0 else None
    except Exception:
        return None


def _probe(key: str):
    try:
        import cv2
    except ImportError:
        return None
    for d in SEARCH_DIRS:
        for ext in VIDEO_EXT:
            p = d / f"{key}{ext}"
            if p.exists():
                cap = cv2.VideoCapture(str(p))
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if w > 0 and h > 0:
                    return w, h
    return None


def frame_size(key: str):
    """(width, height) in pixels, or None if unknown."""
    global _cache
    meta = _load(METADATA_FILE)
    entry = meta.get(key) if isinstance(meta, dict) else None
    if entry and _parse_res(entry.get("resolution")):
        return _parse_res(entry["resolution"])
    if _cache is None:
        _cache = _load(CACHE_FILE)
    if key in _cache:
        return tuple(_cache[key])
    size = _probe(key)
    if size:
        _cache[key] = list(size)
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(_cache, indent=1), encoding="utf-8")
        except Exception:
            pass
    return size


def video_aspect(key: str, default: float = DEFAULT_ASPECT) -> float:
    size = frame_size(key)
    if not size:
        logging.warning(f"{key}: frame size unknown (not in metadata, video not found) - "
                        f"assuming {default:.3f}. Distances mixing x and y may be off.")
        return default
    return size[0] / size[1]


def P(face, idx, aspect: float) -> dict:
    """Landmark in undistorted units (x scaled by aspect; 1.0 = frame height)."""
    p = face[idx]
    return {"x": p["x"] * aspect, "y": p["y"], "z": p.get("z", 0.0) * aspect}


def dist(a: dict, b: dict) -> float:
    return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5
