"""features/utils.py — helper functions ร่วม"""

import json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # features -> approach_1_rule -> approaches -> root
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
    """Aspect ratio (w/h) of the clip. Delegates to base/measurement/visual/geometry.py,
    which also probes the video file when the clip is not in metadata.json (AVI clips)."""
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT))
    from base.measurement.visual.geometry import video_aspect
    return video_aspect(key, default)
