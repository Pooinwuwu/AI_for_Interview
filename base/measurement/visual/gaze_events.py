"""
base/measurement/visual/gaze_events.py

Gaze events (eye_contact / looking_away / no_face) from face landmarks.
This is the ONLY gaze measure: Approach A's gaze feature (features/gaze.py) is
computed from the same per-frame states, so the evidence log and the scores agree.

Per frame
  no_face       face not detected
  low_lids      eyelids much lower than this person's open-eye level (blink, looking
                down at notes, eyes closed)
  looking_away  gaze direction far from this person's usual (camera-facing) direction:
                horizontal = head yaw + iris position, vertical = head pitch
  eye_contact   otherwise
The "usual direction" is the median over the clip, so it adapts to camera placement,
face shape and device. Assumption: the candidate looks at the camera most of the time.
Then
  low_lids shorter than BLINK_MAX_SEC  -> blink: keeps the surrounding state
  low_lids of BLINK_MAX_SEC or longer  -> looking_away (eyes lowered / closed)
  looking_away shorter than MIN_AWAY_SEC -> eye_contact (micro saccade)

v2 (2026-10-04): distances are computed in undistorted units (geometry.py).
v1 mixed x / y normalised coordinates, so eye openness depended on the video's
aspect ratio: on 720x1280 phone video most open-eye frames looked "closed" and
were counted as looking away.

Output: output/evidence/{key}_gaze_events.json
"""

import json
import logging
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from base._paths import LANDMARKS_DIR, EVIDENCE_DIR, ensure_dirs
from base.measurement.visual.geometry import P, dist, video_aspect

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

EYES = {
    "right": {"outer": 33,  "inner": 133, "top": 159, "bottom": 145, "iris": 468},
    "left":  {"outer": 263, "inner": 362, "top": 386, "bottom": 374, "iris": 473},
}

H_DEV_MAX = 0.08                # max |(yaw + iris) - median| still counted as eye contact
V_DEV_MAX = 0.10                # max |pitch - median|  (units: inter-ocular distance)
EAR_CLOSED_ABS = 0.12           # below this the eye is treated as (nearly) closed
EAR_LOW_REL = 0.65              # or below 65% of this person's open-eye level
BLINK_MAX_SEC = 0.4
MIN_AWAY_SEC = 0.2


def _eye(face, cfg, aspect):
    t, b = P(face, cfg["top"], aspect), P(face, cfg["bottom"], aspect)
    c1, c2 = P(face, cfg["outer"], aspect), P(face, cfg["inner"], aspect)
    left, right = (c1, c2) if c1["x"] <= c2["x"] else (c2, c1)   # image orientation
    w = dist(left, right)
    if w < 1e-6:
        return None
    iris = P(face, cfg["iris"], aspect)
    ux, uy = (right["x"] - left["x"]) / w, (right["y"] - left["y"]) / w
    x = ((iris["x"] - left["x"]) * ux + (iris["y"] - left["y"]) * uy) / w
    return {"ear": dist(t, b) / w, "x": x}


def _head(face, aspect):
    """Head direction proxies in inter-ocular units: nose tip vs. eye-corner midpoint."""
    r, l, n = P(face, 33, aspect), P(face, 263, aspect), P(face, 1, aspect)
    iod = dist(r, l)
    if iod < 1e-6:
        return None, None
    return ((n["x"] - (r["x"] + l["x"]) / 2) / iod, (n["y"] - (r["y"] + l["y"]) / 2) / iod)


def frame_measures(landmarks_data, aspect):
    """[(timestamp, ear, horizontal, vertical)] per frame (None values if no face)."""
    out = []
    for fr in landmarks_data.get("frames", []):
        face = fr.get("face")
        t = fr.get("timestamp", 0.0)
        if not face or len(face) < 478:
            out.append((t, None, None, None))
            continue
        eyes = [e for e in (_eye(face, c, aspect) for c in EYES.values()) if e]
        yaw, pitch = _head(face, aspect)
        if not eyes or yaw is None:
            out.append((t, None, None, None))
            continue
        out.append((t, st.mean(e["ear"] for e in eyes),
                    yaw + st.mean(e["x"] for e in eyes), pitch))
    return out


def frame_states(landmarks_data, aspect=None):
    """[(timestamp, state)] with blinks resolved. Shared with features/gaze.py."""
    if aspect is None:
        aspect = video_aspect(landmarks_data.get("key", ""))
    m = frame_measures(landmarks_data, aspect)
    ears = sorted(e for _, e, _, _ in m if e is not None)
    open_level = ears[int(len(ears) * 0.75)] if ears else 0.3     # typical open-eye EAR
    low = max(EAR_CLOSED_ABS, EAR_LOW_REL * open_level)
    opened = [(h, v) for _, e, h, v in m if e is not None and e >= low]
    h0 = st.median(h for h, _ in opened) if opened else 0.0
    v0 = st.median(v for _, v in opened) if opened else 0.0

    raw = []
    for t, ear, h, v in m:
        if ear is None:
            raw.append([t, "no_face"])
        elif ear < low:
            raw.append([t, "low_lids"])
        elif abs(h - h0) > H_DEV_MAX or abs(v - v0) > V_DEV_MAX:
            raw.append([t, "looking_away"])
        else:
            raw.append([t, "eye_contact"])

    # resolve low_lids runs: short = blink (take previous state), long = looking away
    i = 0
    while i < len(raw):
        if raw[i][1] != "low_lids":
            i += 1
            continue
        j = i
        while j < len(raw) and raw[j][1] == "low_lids":
            j += 1
        end_t = raw[j][0] if j < len(raw) else raw[-1][0]
        if end_t - raw[i][0] < BLINK_MAX_SEC:
            fill = raw[i - 1][1] if i > 0 else (raw[j][1] if j < len(raw) else "eye_contact")
            fill = "eye_contact" if fill == "low_lids" else fill
        else:
            fill = "looking_away"
        for k in range(i, j):
            raw[k][1] = fill
        i = j
    return [(t, s) for t, s in raw]


def _to_events(states):
    events = []
    for t, s in states:
        if events and events[-1]["event"] == s:
            events[-1]["end"] = t
        else:
            if events:
                events[-1]["end"] = t
            events.append({"event": s, "start": t, "end": t})
    for e in events:
        e["start"], e["end"] = round(e["start"], 2), round(e["end"], 2)
        e["duration"] = round(e["end"] - e["start"], 2)
    return events


def extract_gaze_events(landmarks_data, aspect=None):
    events = _to_events(frame_states(landmarks_data, aspect))
    # micro saccades -> eye contact, then merge neighbours with the same state
    for e in events:
        if e["event"] == "looking_away" and e["duration"] < MIN_AWAY_SEC:
            e["event"] = "eye_contact"
    merged = []
    for e in events:
        if merged and merged[-1]["event"] == e["event"]:
            merged[-1]["end"] = e["end"]
            merged[-1]["duration"] = round(merged[-1]["end"] - merged[-1]["start"], 2)
        else:
            merged.append(dict(e))
    return merged


def main():
    ensure_dirs(EVIDENCE_DIR)
    if not LANDMARKS_DIR.exists():
        logging.error(f"Landmarks directory not found: {LANDMARKS_DIR}")
        return
    landmark_files = list(LANDMARKS_DIR.glob("*.json"))
    if not landmark_files:
        logging.warning("No landmark files found.")
        return
    for lm_file in landmark_files:
        key = lm_file.stem
        out_file = EVIDENCE_DIR / f"{key}_gaze_events.json"
        logging.info(f"Processing: {key}")
        with open(lm_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("key", key)
        events = extract_gaze_events(data, video_aspect(key))
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=4, ensure_ascii=False)
        logging.info(f"  -> Saved {out_file.name} ({len(events)} events)")


if __name__ == "__main__":
    main()
