"""
base/measurement/visual/face_events.py

Extract discrete facial expression events (Smiling, Neutral) from landmarks.
Output: output/evidence/{key}_face_events.json
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
MOUTH_L = 61
MOUTH_R = 291
MOUTH_TOP = 13
MOUTH_BOT = 14
FACE_W_L = 33
FACE_W_R = 263

SMILE_MOUTH_W_RATIO = 0.45    # mouth_w / face_w
SMILE_MAR_MIN       = 0.05

def dist_2d(p1, p2):
    return ((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2)**0.5

def extract_face_events(landmarks_data):
    events = []
    current_state = None
    start_time = 0.0

    frames = landmarks_data.get("frames", [])
    if not frames:
        return []
        
    for frame in frames:
        timestamp = frame.get("timestamp", 0.0)
        face = frame.get("face")
        
        is_smiling = False
        
        if face and len(face) >= 468:
            ml = face[MOUTH_L]
            mr = face[MOUTH_R]
            mt = face[MOUTH_TOP]
            mb = face[MOUTH_BOT]
            fl = face[FACE_W_L]
            fr = face[FACE_W_R]
            
            mw = dist_2d(ml, mr)
            mh = dist_2d(mt, mb)
            fw = dist_2d(fl, fr)
            
            if mw > 1e-6 and fw > 1e-6:
                mar = mh / mw
                mwr = mw / fw
                if mwr > SMILE_MOUTH_W_RATIO and mar > SMILE_MAR_MIN:
                    is_smiling = True
                    
        state_name = "smiling" if is_smiling else "neutral"
        
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

    # Post-process: Filter out micro-expressions (< 0.2s)
    filtered = []
    for ev in events:
        if ev["duration"] < 0.2:
            if not filtered:
                filtered.append(ev)
            else:
                filtered[-1]["end"] = ev["end"]
                filtered[-1]["duration"] = round(filtered[-1]["end"] - filtered[-1]["start"], 2)
        else:
            if filtered and filtered[-1]["event"] == ev["event"]:
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
        out_file = EVIDENCE_DIR / f"{key}_face_events.json"
        
        logging.info(f"Processing: {key}")
        with open(lm_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        events = extract_face_events(data)
        
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=4, ensure_ascii=False)
        logging.info(f"  -> Saved {out_file.name} ({len(events)} events)")

if __name__ == "__main__":
    main()
