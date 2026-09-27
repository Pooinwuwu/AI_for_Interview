"""
approaches/approach_3_hybrid/build_evidence.py

Formats the unified JSON evidence log into a readable text format for the MLLM prompt.
"""
import json
from pathlib import Path

def format_evidence_text(evidence_json_path: Path) -> str:
    if not evidence_json_path.exists():
        return "No evidence log available."
        
    with open(evidence_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    events = data.get("evidence", [])
    if not events:
        return "No evidence events recorded."
        
    lines = []
    lines.append("TIMELINE OF EVENTS (Seconds):")
    lines.append("-" * 40)
    
    for ev in events:
        start = ev.get("start", 0)
        end = ev.get("end", 0)
        cat = ev.get("category", "")
        etype = ev.get("type", "")
        val = ev.get("value", "")
        
        # Truncate very long values to save context window (e.g. transcript text)
        val_str = str(val).replace('\n', ' ')
        if len(val_str) > 100:
            val_str = val_str[:97] + "..."
            
        lines.append(f"[{start:05.2f} - {end:05.2f}] [{cat}:{etype}] {val_str}")
        
    return "\n".join(lines)
