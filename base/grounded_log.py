"""
base/grounded_log.py

Turns the Tier-1 outputs of one clip into a compact, citable evidence log.
Every item has an ID the LLM must cite, and the verifier checks claims against
exactly that item:

  S#  summary   whole-clip statistics per behaviour ("eye contact 94% of the time")
  E#  event     a visual episode of >= MIN_EVENT_SEC ("looking away 41.1-56.4 s")
  P#  pause     a silence of >= PAUSE_SEC between words (from WhisperX word times)
  T#  transcript sentence with its time

Raw Tier-1 events are very fine-grained (gaze switches every few frames, ~100+
events per clip). Here short flickers are folded into the summaries, and
neighbouring segments with the same value are merged, so the LLM sees episodes
a person would notice.

Inputs  : output/evidence/<key>_evidence.json, <key>_transcript.json, <key>_prosody.json
          output/features/<key>_speech.json  (optional, for rate / word count)
Output  : dict (see build); saved by the grounded approach next to its feedback.
"""

import json
from collections import defaultdict
from pathlib import Path

from base._paths import EVIDENCE_DIR, FEATURES_DIR

MIN_EVENT_SEC = 1.0      # visual episodes shorter than this only count in the summary
MERGE_GAP_SEC = 0.5      # same value separated by less than this -> one episode
PAUSE_SEC = 1.0          # silence between words counted as a pause

VISUAL_TYPES = ("gaze", "head", "hand", "face")

VALUE_TEXT = {
    "eye_contact": "looking at the camera", "looking_away": "looking away from the camera",
    "centered": "head straight", "turned_up": "head turned up", "turned_down": "head turned down",
    "turned_left": "head turned left", "turned_right": "head turned right",
    "gesturing": "hands gesturing", "hands_still": "hands visible and still",
    "hands_hidden": "hands not visible", "smiling": "smiling", "neutral": "neutral face",
}

# which evidence a dimension may cite
DIMENSION_SOURCES = {
    "eye_contact": {"gaze"},
    "head_pose": {"head"},
    "hand_gesture": {"hand"},
    "facial_expression": {"face"},
    "answer_quality": {"speech", "pause", "transcript"},
}


def _read(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _merge(segs, gap):
    out = []
    for s in sorted(segs, key=lambda x: x["start"]):
        if out and out[-1]["value"] == s["value"] and s["start"] - out[-1]["end"] <= gap:
            out[-1]["end"] = max(out[-1]["end"], s["end"])
        else:
            out.append(dict(s))
    return out


def _episodes(segs):
    """Merge flickers, keep episodes >= MIN_EVENT_SEC, merge again across removed flickers."""
    segs = _merge(segs, MERGE_GAP_SEC)
    segs = [s for s in segs if s["end"] - s["start"] >= MIN_EVENT_SEC]
    return _merge(segs, MIN_EVENT_SEC)


def build(key: str) -> dict:
    raw = (_read(EVIDENCE_DIR / f"{key}_evidence.json") or {}).get("evidence", [])
    transcript = _read(EVIDENCE_DIR / f"{key}_transcript.json") or {}
    speech = (_read(FEATURES_DIR / f"{key}_speech.json") or {}).get("features", {})
    return build_from(key, raw, transcript, speech)


def build_from(key: str, raw: list, transcript: dict, speech: dict) -> dict:
    """Same as build(), from data in memory (used for synthetic training clips)."""
    segments = transcript.get("segments", []) or []
    words = [w for s in segments for w in s.get("words", []) if "start" in w and "end" in w]
    duration = max([e.get("end", 0) for e in raw] + [s.get("end", 0) for s in segments]
                   + [float(speech.get("duration_sec") or 0)] + [0.0])

    by_type = defaultdict(list)
    for e in raw:
        if e.get("category") == "visual" and e.get("type") in VISUAL_TYPES:
            by_type[e["type"]].append({"start": float(e["start"]), "end": float(e["end"]),
                                       "value": e["value"]})

    summaries, events = [], []

    # ---- visual: summary per type + episodes
    for t in VISUAL_TYPES:
        segs = by_type.get(t)
        if not segs:
            summaries.append({"type": t, "text": f"{t}: not measured (no detections)", "share": {}})
            continue
        covered = sum(s["end"] - s["start"] for s in segs) or 1.0
        share = defaultdict(float)
        for s in segs:
            share[s["value"]] += (s["end"] - s["start"]) / covered
        eps = _episodes(segs)
        parts = ", ".join(f"{VALUE_TEXT.get(v, v)} {p:.0%}"
                          for v, p in sorted(share.items(), key=lambda x: -x[1]))
        extra = ""
        if t == "gaze":
            away = [e for e in eps if e["value"] == "looking_away"]
            if away:
                longest = max(away, key=lambda e: e["end"] - e["start"])
                extra = (f"; {len(away)} look-away episode(s) >= {MIN_EVENT_SEC:.0f}s, longest "
                         f"{longest['end'] - longest['start']:.1f}s")
        summaries.append({"type": t, "text": f"{t}: {parts} of measured time{extra}",
                          "share": {k: round(v, 3) for k, v in share.items()}})
        for e in eps:
            events.append({"type": t, **e, "text": VALUE_TEXT.get(e["value"], e["value"])})

    # ---- speech: summary, pauses from word gaps, transcript sentences
    if words:
        n_words = len(words)
        spoken = words[-1]["end"] - words[0]["start"]
        wpm = n_words / spoken * 60 if spoken > 0 else 0
        summaries.append({"type": "speech", "share": {},
                          "text": (f"speech: {n_words} words in {duration:.0f}s clip, "
                                   f"about {wpm:.0f} words/min while speaking, first word at "
                                   f"{words[0]['start']:.1f}s, last word ends {words[-1]['end']:.1f}s")})
        pauses = []
        if words[0]["start"] >= PAUSE_SEC:
            pauses.append({"start": 0.0, "end": words[0]["start"], "value": "silence_before_answer"})
        for a, b in zip(words, words[1:]):
            if b["start"] - a["end"] >= PAUSE_SEC:
                pauses.append({"start": a["end"], "end": b["start"], "value": "pause"})
        summaries.append({"type": "pause", "share": {},
                          "text": f"pauses >= {PAUSE_SEC:.0f}s: {len(pauses)}"
                                  + (f", longest {max(p['end'] - p['start'] for p in pauses):.1f}s"
                                     if pauses else "")})
        for p in pauses:
            events.append({"type": "pause", **p,
                           "text": "silence before the first word" if p["value"] != "pause"
                           else "pause in speech"})
    else:
        summaries.append({"type": "speech", "share": {},
                          "text": "speech: no transcript available (run the transcript step)"})
    summaries.append({"type": "filler", "share": {},
                      "text": "filler words (um, uh): NOT measured - the speech recogniser drops them"})

    # ---- assign IDs
    log = {"key": key, "duration": round(duration, 2), "items": []}
    for i, s in enumerate(summaries, 1):
        log["items"].append({"id": f"S{i}", "kind": "summary", "type": s["type"],
                             "start": 0.0, "end": round(duration, 2), "text": s["text"],
                             "share": s["share"]})
    events.sort(key=lambda e: (e["start"], e["type"]))
    ne = npz = 0
    for e in events:
        if e["type"] == "pause":
            npz += 1
            iid = f"P{npz}"
        else:
            ne += 1
            iid = f"E{ne}"
        log["items"].append({"id": iid, "kind": "event", "type": e["type"],
                             "start": round(e["start"], 2), "end": round(e["end"], 2),
                             "value": e["value"], "text": e["text"]})
    for i, s in enumerate(segments, 1):
        txt = (s.get("text") or "").strip()
        if txt:
            log["items"].append({"id": f"T{i}", "kind": "transcript", "type": "transcript",
                                 "start": round(float(s["start"]), 2),
                                 "end": round(float(s["end"]), 2), "text": txt})
    return log


def to_prompt_text(log: dict) -> str:
    lines = [f"Clip length: {log['duration']:.1f}s", "", "SUMMARY"]
    lines += [f"[{i['id']}] {i['text']}" for i in log["items"] if i["kind"] == "summary"]
    lines += ["", "EPISODES (visual >= 1s, pauses >= 1s)"]
    ev = [i for i in log["items"] if i["kind"] == "event"]
    lines += [f"[{i['id']}] {i['start']:.1f}-{i['end']:.1f}s {i['type']}: {i['text']}" for i in ev] \
        or ["(none)"]
    lines += ["", "TRANSCRIPT"]
    tr = [i for i in log["items"] if i["kind"] == "transcript"]
    lines += [f"[{i['id']}] {i['start']:.1f}-{i['end']:.1f}s \"{i['text']}\"" for i in tr] \
        or ["(no transcript)"]
    return "\n".join(lines)


def index(log: dict) -> dict:
    return {i["id"]: i for i in log["items"]}
