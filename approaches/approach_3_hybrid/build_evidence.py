"""
approaches/approach_3_hybrid/build_evidence.py

Formats the unified JSON evidence log into compact text for the MLLM prompt.

Single-word events (speech:word) are left out: the same words are already in
the transcript sentences, and they were about half of every log. The full log
(with words) stays in output/evidence/{key}_evidence.json for the
hallucination check.
"""
import json
from pathlib import Path

SKIP_TYPES = {"word"}


def format_evidence_text(evidence_json_path: Path) -> str:
    if not evidence_json_path.exists():
        return "No evidence log available."

    with open(evidence_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    events = [e for e in data.get("evidence", []) if e.get("type") not in SKIP_TYPES]
    if not events:
        return "No evidence events recorded."

    lines = []
    for ev in events:
        start, end = ev.get("start", 0), ev.get("end", 0)
        val = str(ev.get("value", "")).replace("\n", " ").strip()
        lines.append(f"[{start:.1f}-{end:.1f}] {ev.get('category', '')}:{ev.get('type', '')} {val}")
    return "\n".join(lines)
