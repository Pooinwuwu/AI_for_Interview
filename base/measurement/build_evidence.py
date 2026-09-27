"""
base/measurement/build_evidence.py

Aggregates all individual evidence files (transcript, prosody, gaze, head, hand, face)
into a single unified timeline Evidence Log for each video.
Output: output/evidence/{key}_evidence.json
"""

import sys
import json
import logging
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from base._paths import EVIDENCE_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_video_keys():
    # Discover keys from any existing evidence file
    keys = set()
    for f in EVIDENCE_DIR.glob("*.json"):
        # Ignore the aggregated evidence logs themselves
        if f.name.endswith("_evidence.json"):
            continue
            
        # Extract the key (e.g., 'vid_0021' from 'vid_0021_transcript.json')
        parts = f.stem.split('_')
        if len(parts) >= 2 and parts[0] == "vid":
            key = f"{parts[0]}_{parts[1]}"
            keys.add(key)
            
    return sorted(list(keys))

def load_json(filepath):
    if not filepath.exists():
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading {filepath.name}: {e}")
        return None

def build_evidence_log(key):
    unified_events = []

    # 1. Transcript
    transcript_file = EVIDENCE_DIR / f"{key}_transcript.json"
    transcript_data = load_json(transcript_file)
    if transcript_data:
        for segment in transcript_data.get("segments", []):
            unified_events.append({
                "category": "speech",
                "type": "transcript",
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "duration": round(segment.get("end", 0.0) - segment.get("start", 0.0), 2),
                "value": segment.get("text", "").strip()
            })
            
            # Words (optional, but good for fine-grained evidence)
            for word in segment.get("words", []):
                if "start" in word and "end" in word:
                    unified_events.append({
                        "category": "speech",
                        "type": "word",
                        "start": word["start"],
                        "end": word["end"],
                        "duration": round(word["end"] - word["start"], 2),
                        "value": word["word"]
                    })

    # 2. Prosody
    prosody_file = EVIDENCE_DIR / f"{key}_prosody.json"
    prosody_data = load_json(prosody_file)
    if prosody_data:
        for pause in prosody_data.get("pauses", []):
            unified_events.append({
                "category": "speech",
                "type": "pause",
                "start": pause.get("start", 0.0),
                "end": pause.get("end", 0.0),
                "duration": pause.get("duration", 0.0),
                "value": "silence"
            })
            
    # 3. Visual Events
    visual_types = ["gaze", "head", "hand", "face"]
    for vtype in visual_types:
        v_file = EVIDENCE_DIR / f"{key}_{vtype}_events.json"
        v_data = load_json(v_file)
        if v_data:
            for ev in v_data:
                unified_events.append({
                    "category": "visual",
                    "type": vtype,
                    "start": ev.get("start", 0.0),
                    "end": ev.get("end", 0.0),
                    "duration": ev.get("duration", 0.0),
                    "value": ev.get("event")
                })
                
    # Sort by start time, then by category, then by type
    unified_events.sort(key=lambda x: (x["start"], x["category"], x["type"]))
    
    return {
        "key": key,
        "total_events": len(unified_events),
        "evidence": unified_events
    }

def main():
    if not EVIDENCE_DIR.exists():
        logging.error(f"Evidence directory not found: {EVIDENCE_DIR}")
        return
        
    keys = get_video_keys()
    if not keys:
        logging.warning("No evidence files found to aggregate.")
        return
        
    for key in keys:
        logging.info(f"Building evidence log for: {key}")
        log_data = build_evidence_log(key)
        
        out_file = EVIDENCE_DIR / f"{key}_evidence.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=4, ensure_ascii=False)
            
        logging.info(f"  -> Saved {out_file.name} ({log_data['total_events']} events)")

if __name__ == "__main__":
    main()
