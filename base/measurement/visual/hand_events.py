"""
base/measurement/visual/hand_events.py

Extract discrete hand gesture events (Gesturing, Hands Still, Hands Hidden) from landmarks.
Output: output/evidence/{key}_hand_events.json
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

SPEED_THRESH = 0.5  # Normalized distance per second to be considered "gesturing"

def _hand_center(hand_landmarks):
    xs = [p["x"] for p in hand_landmarks]
    ys = [p["y"] for p in hand_landmarks]
    return sum(xs)/len(xs), sum(ys)/len(ys)

def extract_hand_events(landmarks_data):
    events = []
    current_state = None
    start_time = 0.0

    frames = landmarks_data.get("frames", [])
    if not frames:
        return []
        
    prev_cx, prev_cy, prev_t = None, None, None

    for frame in frames:
        timestamp = frame.get("timestamp", 0.0)
        hands = frame.get("hands") or []
        
        if not hands:
            state_name = "hands_hidden"
            prev_cx, prev_cy, prev_t = None, None, None
        else:
            cxs = [_hand_center(h["landmarks"]) for h in hands]
            cx = sum([c[0] for c in cxs]) / len(cxs)
            cy = sum([c[1] for c in cxs]) / len(cxs)
            
            if prev_t is not None:
                dt = timestamp - prev_t
                # Only calculate speed if dt is small enough (continuous frames)
                if 0 < dt < 1.0:
                    d = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
                    speed = d / dt
                    state_name = "gesturing" if speed >= SPEED_THRESH else "hands_still"
                else:
                    state_name = "hands_still"
            else:
                state_name = "hands_still"
                
            prev_cx, prev_cy, prev_t = cx, cy, timestamp

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
            
    if current_state is not None:
        end_time = frames[-1].get("timestamp", start_time)
        events.append({
            "event": current_state,
            "start": round(start_time, 2),
            "end": round(end_time, 2),
            "duration": round(end_time - start_time, 2)
        })

    # Post-process: Filter out micro-events (< 0.2s)
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
        out_file = EVIDENCE_DIR / f"{key}_hand_events.json"
        
        logging.info(f"Processing: {key}")
        with open(lm_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        events = extract_hand_events(data)
        
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(events, f, indent=4, ensure_ascii=False)
        logging.info(f"  -> Saved {out_file.name} ({len(events)} events)")

if __name__ == "__main__":
    main()
