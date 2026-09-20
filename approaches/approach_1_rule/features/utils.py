"""features/utils.py — helper functions ร่วม"""

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_METADATA_FILE = _PROJECT_ROOT / "input" / "metadata" / "metadata.json"


def lm(face, idx):
    return face[idx]


def dist_2d(a, b):
    return ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5


def dist_3d(a, b):
    return ((a["x"] - b["x"]) ** 2
            + (a["y"] - b["y"]) ** 2
            + (a["z"] - b["z"]) ** 2) ** 0.5


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def valid_face_frames(landmarks_data, min_points=468):
    out = []
    for i, f in enumerate(landmarks_data["frames"]):
        face = f.get("face")
        if face and len(face) >= min_points:
            out.append((i, face))
    return out


def get_video_aspect(key, default=16/9):
    """
    อ่าน resolution จาก metadata.json แล้วคืน aspect ratio (w/h)
    คืน default ถ้าอ่านไม่ได้
    """
    try:
        if not _METADATA_FILE.exists():
            return default
        with open(_METADATA_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        info = meta.get(key)
        if not info:
            return default
        res = info.get("resolution")
        if not res or "x" not in res:
            return default
        w, h = res.lower().split("x")
        w, h = int(w.strip()), int(h.strip())
        if h == 0:
            return default
        return w / h
    except Exception:
        return default