"""
approaches/approach_grounded/verify.py

Checks a grounded feedback draft against the evidence log it cites.
Reuses the claim parser of validation/hallucination_check.py (times, topics,
expected values), but checks against the *cited* items, not the whole log.

Problems found per strength / improvement:
  no_ids         no evidence cited
  unknown_id     cites an ID that is not in the log
  wrong_source   cites only evidence of another behaviour (e.g. eye contact citing E5 = face)
  time_mismatch  mentions a time not covered by any cited event (±1 s)
  contradicted   the cited evidence shows the opposite (says "looking away", log says camera)
  filler         talks about filler words, which are not measured
  quote          quotes words that are not in the cited transcript lines
Points citing only "VIDEO" (approach D) are not problems but are marked unverified.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.grounded_log import DIMENSION_SOURCES, index
from validation.hallucination_check import (TOL, expected_values, find_times, sentences,
                                            topic_for, _norm)

VISUAL = {"gaze", "head", "hand", "face"}
SHARE_CONTRADICTS = 0.25     # claim says X but X is < 25% of the clip in the cited summary


def _texts(item: dict):
    for f in ("text_en", "issue_en"):
        if item.get(f):
            yield item[f]
    yield from item.get("action_steps_en", []) or []


def check_item(item: dict, idx: dict, allow_video: bool) -> tuple:
    """Returns (problems: list[str], status: 'ok' | 'video_only' | 'failed')."""
    probs = []
    ids = [i.strip().strip("[]") for i in item.get("evidence_ids", []) or []]
    video = [i for i in ids if i.upper() == "VIDEO"]
    ids = [i for i in ids if i.upper() != "VIDEO"]
    dim = item.get("dimension")
    label = (item.get("text_en") or item.get("issue_en") or "")[:80]

    if not ids and not video:
        return [f'"{label}": cites no evidence'], "failed"
    unknown = [i for i in ids if i not in idx]
    if unknown:
        probs.append(f'"{label}": cites {", ".join(unknown)}, which are not in the log')
    cited = [idx[i] for i in ids if i in idx]
    if not cited:
        if video and allow_video:
            return probs, "video_only"
        if video:
            probs.append(f'"{label}": cites VIDEO, but no video was given')
        return probs, "failed"

    allowed = DIMENSION_SOURCES.get(dim, set())
    if allowed and not any(c["type"] in allowed for c in cited):
        probs.append(f'"{label}": is about {dim} but cites {", ".join(ids)} '
                     f'({", ".join(sorted({c["type"] for c in cited}))})')

    for text in _texts(item):
        for sent in sentences(text):
            topic_s = topic_for(sent, 0, dim)
            if topic_s == "filler":
                probs.append(f'"{sent[:80]}": filler words are not measured')
                continue
            # quoted words must be in the cited transcript lines
            for q in re.findall(r"[\"“]([^\"”]{6,})[\"”]", sent):
                said = " ".join(c["text"] for c in cited if c["kind"] == "transcript")
                qw = [_norm(w) for w in q.split() if _norm(w)]
                sw = {_norm(w) for w in said.split()}
                if qw and sum(w in sw for w in qw) / len(qw) < 0.6:
                    probs.append(f'"{sent[:80]}": the quote is not in the cited transcript lines')
            # claimed times must fall inside a cited event / transcript line
            for t0, t1, pos, mt in find_times(sent):
                spans = [c for c in cited if c["kind"] != "summary"]
                hit = [c for c in spans if c["start"] - TOL <= t1 and c["end"] + TOL >= t0]
                if not hit:
                    have = ", ".join(f'{c["id"]} {c["start"]:.1f}-{c["end"]:.1f}s' for c in spans) or "none"
                    probs.append(f'"{sent[:80]}": says {mt.strip()} but the cited items cover {have}')
                    continue
                topic = topic_for(sent, pos, dim)
                exp = expected_values(topic, sent) if topic in VISUAL else None
                same = [c for c in hit if c["type"] == topic]
                if exp and same and not any(c.get("value") in exp for c in same):
                    probs.append(f'"{sent[:80]}": cited {same[0]["id"]} shows '
                                 f'"{same[0]["text"]}", not {"/".join(sorted(exp))}')
            # whole-clip claims against cited summaries ("good eye contact" while 24%)
            topic = topic_for(sent, 0, dim)
            exp = expected_values(topic, sent) if topic in VISUAL else None
            if exp and not find_times(sent):
                for c in cited:
                    if c["kind"] == "summary" and c["type"] == topic and c.get("share"):
                        share = sum(c["share"].get(v, 0) for v in exp)
                        if share < SHARE_CONTRADICTS:
                            probs.append(f'"{sent[:80]}": {c["id"]} shows only {share:.0%} '
                                         f'{"/".join(sorted(exp))}')
    return probs, ("failed" if probs else ("ok" if not video else "ok+video"))


def check_summary(text: str, log: dict) -> list:
    """Times in the summary must match some log item of the same behaviour."""
    probs = []
    items = [i for i in log["items"] if i["kind"] != "summary"]
    for sent in sentences(text or ""):
        if topic_for(sent, 0, None) == "filler":
            probs.append(f'summary "{sent[:80]}": filler words are not measured')
        for t0, t1, pos, mt in find_times(sent):
            topic = topic_for(sent, pos, None)
            near = [i for i in items if i["start"] - TOL <= t1 and i["end"] + TOL >= t0
                    and (topic is None or i["type"] == topic or topic == "quote")]
            if not near:
                probs.append(f'summary "{sent[:80]}": nothing in the log at {mt.strip()}')
    return probs


def check(feedback: dict, log: dict, allow_video: bool = False) -> dict:
    """Full report: per-item status + list of problems (for the revision prompt)."""
    idx = index(log)
    report = {"items": [], "problems": []}
    for section in ("strengths", "improvements"):
        for n, item in enumerate(feedback.get(section, []) or []):
            probs, status = check_item(item, idx, allow_video)
            report["items"].append({"section": section, "index": n, "status": status,
                                    "problems": probs})
            report["problems"] += probs
    sp = check_summary(feedback.get("overall_summary_en"), log)
    report["summary_problems"] = sp
    report["problems"] += sp
    report["n_items"] = len(report["items"])
    report["n_failed"] = sum(i["status"] == "failed" for i in report["items"])
    return report


def drop_failed(feedback: dict, report: dict) -> tuple:
    """Remove items that still fail after revision. Returns (feedback, removed)."""
    bad = {(i["section"], i["index"]) for i in report["items"] if i["status"] == "failed"}
    removed = []
    out = dict(feedback)
    for section in ("strengths", "improvements"):
        keep = []
        for n, item in enumerate(feedback.get(section, []) or []):
            if (section, n) in bad:
                removed.append({"section": section, **item})
            else:
                keep.append(item)
        out[section] = keep
    return out, removed
