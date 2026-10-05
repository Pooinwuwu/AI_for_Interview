"""
base/measurement/visual/head_events.py

Extract discrete head pose events (Centered, Turned Left/Right/Up/Down) from landmarks.
Output: output/evidence/{key}_head_events.json
"""

import sys
import json
import logging
import cv2
import numpy as np
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from base._paths import LANDMARKS_DIR, EVIDENCE_DIR, ensure_dirs
from base.measurement.visual.geometry import video_aspect

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 3D face model for solvePnP (from physical mm)
MODEL_3D = np.array([
    (0.0,     0.0,    0.0),      # 1   nose tip
    (0.0,  -330.0,  -65.0),      # 152 chin
    (-225.0, 170.0, -135.0),     # 33  right eye outer
    (225.0,  170.0, -135.0),     # 263 left eye outer
    (-150.0, -150.0, -125.0),    # 61  right mouth corner
    (150.0,  -150.0, -125.0),    # 291 left mouth corner
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]
DEFAULT_ASPECT = 16 / 9

# Thresholds for events (degrees)
YAW_THRESH = 20.0
PITCH_THRESH = 20.0

def _build_camera_matrix(aspect: float):
    return np.array([
        [1.0, 0.0, aspect / 2.0],
        [0.0, 1.0, 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

def _euler_from_rvec(rvec):
    rmat, _ = cv2.Rodrigues(rvec)
    # Neutral pose correction (OpenCV coordinates to Face Mesh)
    rmat = rmat * np.array([
        [1, 1, 1],
        [-1, -1, -1],
        [-1, -1, -1],
    ], dtype=np.float64)

    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(rmat[2, 1], rmat[2, 2])
        yaw   = np.arctan2(-rmat[2, 0], sy)
        roll  = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = np.arctan2(-rmat[1, 2], rmat[1, 1])
        yaw   = np.arctan2(-rmat[2, 0], sy)
        roll  = 0.0

    return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))

def get_head_pose_state(pitch, yaw):
    if yaw > YAW_THRESH:
        return "turned_right"
    elif yaw < -YAW_THRESH:
        return "turned_left"
    elif pitch > PITCH_THRESH:
        return "turned_up"
    elif pitch < -PITCH_THRESH:
        return "turned_down"
    else:
        return "centered"

def extract_head_events(landmarks_data, aspect=None):
    """aspect = width / height of the video. v1 always used 16:9, which tilted the
    estimated pitch on 4:3 (AVI) and 9:16 (phone) video -> 'turned_up' for whole clips."""
    if aspect is None:
        aspect = video_aspect(landmarks_data.get("key", ""))
    events = []
    current_state = None
    start_time = 0.0

    frames = landmarks_data.get("frames", [])
    if not frames:
        return []
        
    cam_matrix = _build_camera_matrix(aspect)
    dist_coeffs = np.zeros((4, 1))

    for frame in frames:
        timestamp = frame.get("timestamp", 0.0)
        face = frame.get("face")
        
        state_name = "unknown"
        
        if face and len(face) >= 468:
            image_pts = np.array(
                [[face[i]["x"] * aspect, face[i]["y"]] for i in LANDMARK_IDS],
                dtype=np.float64,
            )
            ok, rvec, _ = cv2.solvePnP(
                MODEL_3D, image_pts, cam_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                p, y, r = _euler_from_rvec(rvec)
                state_name = get_head_pose_state(p, y)
                
        if state_name == "unknown":
            continue
            
        if current_state != state_name:
            if current_state is not None:
                events.append({
                    "event": current_state,
                    "start": round(start_time, 2),
                    "end": round(timestamp, 2),
                    "duration": round(timestamp - start_time, 2)
                })
            current_state = state_name
            start_time = timestamp
            
    # Add final event
    if current_state is not None:
        end_time = frames[-1].get("timestamp", start_time)
        events.append({
            "event": current_state,
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "duration": round(end_time - start_time, 2)
        })

    # Post-process: Filter out micro-movements (< 0.3s) and merge them into the surrounding state
    filtered = []
    for ev in events:
        if ev["duration"] < 0.3:
            # Skip micro movements, merge into previous state if exists
            if not filtered:
                filtered.append(ev)
            else:
                filtered[-1]["end"] = ev["end"]
                filtered[-1]["duration"] = round(filtered[-1]["end"] - filtered[-1]["start"], 2)
        else:
            if filtered and filtered[-1]["event"] == ev["event"]:
                # Merge consecutive identical states
                filtered[-1]["end"] = ev["end"]
                filtered[-1]["duration"] = round(filtered[-1]["end"] - filtered[-1]["start"], 2)
            else:
                filtered.append(ev)

    return filtered

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
        out_file = EVIDENCE_DIR / f"{key}_head_events.json"
        
        logging.info(f"Processing: {key}")
        with open(lm_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        data.setdefault("key", key)
        events = extract_head_events(data, video_aspect(key))
        
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=4, ensure_ascii=False)
        logging.info(f"  -> Saved {out_file.name} ({len(events)} events)")

if __name__ == "__main__":
    main()
