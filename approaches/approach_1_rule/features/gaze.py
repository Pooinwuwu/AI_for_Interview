"""features/gaze.py — eye contact ratio, gaze stability

v2 (2026-10-04): uses the SAME per-frame gaze states as the evidence log
(base/measurement/visual/gaze_events.py), so Approach A's eye-contact score and the
evidence log can no longer disagree. v1 had its own iris-only rule and silently
skipped frames whose eyes looked "closed" because of an aspect-ratio error, which
gave 90-100% eye contact on every pilot clip.
"""

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from base.measurement.visual.gaze_events import frame_measures, frame_states
from base.measurement.visual.geometry import video_aspect


def extract(landmarks_data, aspect=None):
    if aspect is None:
        aspect = video_aspect(landmarks_data.get("key", ""))
    states = [s for _, s in frame_states(landmarks_data, aspect)]
    contact = states.count("eye_contact")
    away = states.count("looking_away")
    valid = contact + away
    if valid == 0:
        return {"valid_frames": 0}
    m = [(h, v) for _, e, h, v in frame_measures(landmarks_data, aspect) if e is not None]
    hs, vs = [h for h, _ in m], [v for _, v in m]
    return {
        "eye_contact_ratio": round(contact / valid, 4),
        "gaze_stability_x":  round(st.pstdev(hs), 4) if len(hs) > 1 else 0.0,  # lower = steadier
        "gaze_stability_y":  round(st.pstdev(vs), 4) if len(vs) > 1 else 0.0,
        "mean_iris_x":       round(st.mean(hs), 4) if hs else 0.0,   # head yaw + iris (v2 units)
        "mean_iris_y":       round(st.mean(vs), 4) if vs else 0.0,   # head pitch proxy
        "no_face_ratio":     round(states.count("no_face") / max(1, len(states)), 4),
        "valid_frames":      valid,
    }
