"""
base/measurement/visual/gaze_events.py

Extract discrete gaze events (Eye Contact, Looking Away) from landmarks.
Output: output/evidence/{key}_gaze_events.json
"""

import sys
import json
import logging
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from base._paths import LANDMARKS_DIR, EVIDENCE_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# MediaPipe Face Mesh indices
EYES = {
    "right": {"outer": 33,  "inner": 133, "top": 159, "bottom": 145, "iris": 468},
    "left":  {"outer": 263, "inner": 362, "top": 386, "bottom": 374, "iris": 473},
}

EYE_OPEN_MIN       = 0.15
X_CENTER_RANGE     = (0.35, 0.65)
Y_CENTER_RANGE     = (0.30, 0.70)

def dist_2d(p1, p2):
    return ((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)**0.5

def get_iris_position(face, cfg):
    top, bottom = face[cfg["top"]], face[cfg["bottom"]]
    left, right = face[cfg["outer"]], face[cfg["inner"]]

    eye_w = dist_2d(left, right)
    eye_h = dist_2d(top, bottom)
    if eye_w < 1e-6: return None

    ear = eye_h / eye_w
    if ear < EYE_OPEN_MIN: return None

    iris = face[cfg["iris"]]
    x_norm = (iris["x"] - left["x"]) / (right["x"] - left["x"] + 1e-6)
    y_norm = (iris["y"] - top["y"])  / (bottom["y"] - top["y"] + 1e-6)
    
    return {"x": x_norm, "y": y_norm}

def extract_gaze_events(landmarks_data):
    events = []
    current_state = None
    start_time = 0.0

    frames = landmarks_data.get("frames", [])
    if not frames:
        return []
        
    for frame in frames:
        timestamp = frame.get("timestamp", 0.0)
        face = frame.get("face")
        
        is_contact = False
        
        if face and len(face) >= 478:
            contact_eyes = 0
            valid_eyes = 0
            
            for cfg in EYES.values():
                pos = get_iris_position(face, cfg)
                if pos:
                    valid_eyes += 1
                    if (X_CENTER_RANGE[0] < pos["x"] < X_CENTER_RANGE[1] and 
                        Y_CENTER_RANGE[0] < pos["y"] < Y_CENTER_RANGE[1]):
                        contact_eyes += 1
                        
            # If at least one eye is open and making contact
            if valid_eyes > 0 and contact_eyes > 0:
                is_contact = True
                
        state_name = "eye_contact" if is_contact else "looking_away"
        
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
            
    # Add the final event
    if current_state is not None:
        end_time = frames[-1].get("timestamp", start_time)
        events.append({
            "event": current_state,
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "duration": round(end_time - start_time, 2)
        })
        
    # Post-process: Filter out micro-blinks (< 0.2s looking_away -> merge to eye_contact)
    filtered = []
    for ev in events:
        if ev["event"] == "looking_away" and ev["duration"] < 0.2:
            ev["event"] = "eye_contact"
            
        if filtered and filtered[-1]["event"] == ev["event"]:
            # Merge with previous
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
        out_file = EVIDENCE_DIR / f"{key}_gaze_events.json"
        
        logging.info(f"Processing: {key}")
        with open(lm_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        events = extract_gaze_events(data)
        
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=4, ensure_ascii=False)
        logging.info(f"  -> Saved {out_file.name} ({len(events)} events)")

if __name__ == "__main__":
    main()
