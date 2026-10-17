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
  too_many_ids   cites more than MAX_IDS items (citing everything is not evidence)
  ids_in_text    writes lists of evidence IDs into the text the candidate reads
  too_long       a text field longer than MAX_CHARS
  duplicate      an improvement that repeats an earlier one
Improved answer: numbers that are neither in the transcript nor in the resume (invented facts).
Points citing only "VIDEO" (approach D) are not problems but are marked unverified.
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base.grounded_log import DIMENSION_SOURCES, index
from validation.hallucination_check import (TOL, expected_values, find_times, sentences,
                                            topic_for, _norm)

VISUAL = {"gaze", "head", "hand", "face"}
SHARE_CONTRADICTS = 0.25     # claim says X but X is < 25% of the clip in the cited summary
MAX_IDS = 4                  # the 1-4 most specific items, not the whole log
MAX_CHARS = 320              # per text field (about two short sentences)
DUPLICATE_RATIO = 0.6        # issue texts this similar (same dimension) = the same point
EXTRA_FACTS = ""             # resume text; set by the app so resume numbers are allowed
ID_IN_TEXT = re.compile(r"\b[SEPT]\d{1,3}\b")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def _all_texts(item: dict):
    for f in ("text_en", "text_th", "issue_en", "issue_th"):
        if item.get(f):
            yield f, item[f]
    for f in ("action_steps_en", "action_steps_th"):
        for t in item.get(f, []) or []:
            yield f, t


def check_style(item: dict) -> list:
    """Problems that do not depend on the log: too many IDs, IDs in the text, length."""
    probs = []
    label = (item.get("text_en") or item.get("issue_en") or "")[:80]
    ids = [i for i in item.get("evidence_ids", []) or [] if str(i).upper() != "VIDEO"]
    if len(ids) > MAX_IDS:
        probs.append(f'"{label}": cites {len(ids)} items; cite only the {MAX_IDS} most specific')
    for f, t in _all_texts(item):
        if len(ID_IN_TEXT.findall(t)) > 2:
            probs.append(f'"{label}": {f} lists evidence IDs in the text; keep IDs only in evidence_ids')
            break
    for f, t in _all_texts(item):
        if len(t) > MAX_CHARS:
            probs.append(f'"{label}": {f} is {len(t)} characters; shorten to at most 2 short sentences')
            break
    return probs


def check_answer(feedback: dict, log: dict) -> list:
    """improved_answer may only use numbers the candidate said or the resume contains."""
    ia = feedback.get("improved_answer") or {}
    said = " ".join(i.get("text", "") for i in log["items"] if i.get("kind") == "transcript")
    known = set(NUMBER.findall(said + " " + (EXTRA_FACTS or "")))
    probs = []
    for f in ("rewritten_en", "rewritten_th"):
        text = re.sub(r"\[[^\]]*\]", " ", ia.get(f, "") or "")   # [skill 1] = placeholder
        new = [n for n in NUMBER.findall(text) if n not in known]
        if new:
            probs.append(f'improved_answer {f}: the numbers {", ".join(sorted(set(new)))} are not in '
                         f'the transcript or resume - do not invent facts; remove them')
    return probs


def _texts(item: dict):
    """(is_action_step, text). Action steps say what to DO next time ("keep your head
    centred"), so they are not claims about the clip and skip the whole-clip share check."""
    for f in ("text_en", "issue_en"):
        if item.get(f):
            yield False, item[f]
    for t in item.get("action_steps_en", []) or []:
        yield True, t


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

    for is_step, text in _texts(item):
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
            if exp and not find_times(sent) and not is_step:
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
        seen = []
        for n, item in enumerate(feedback.get(section, []) or []):
            probs, status = check_item(item, idx, allow_video)
            probs = probs + check_style(item)
            if section == "improvements":
                for m, prev in seen:
                    if prev.get("dimension") == item.get("dimension") and SequenceMatcher(
                            None, prev.get("issue_en", ""), item.get("issue_en", "")).ratio() > DUPLICATE_RATIO:
                        probs.append(f'"{item.get("issue_en", "")[:80]}": repeats improvement {m + 1}; '
                                     f'merge them or choose a different problem')
                        break
                seen.append((n, item))
            if probs:
                status = "failed"
            report["items"].append({"section": section, "index": n, "status": status,
                                    "problems": probs})
            report["problems"] += probs
    sp = check_summary(feedback.get("overall_summary_en"), log)
    report["summary_problems"] = sp
    report["problems"] += sp
    ap = check_answer(feedback, log)
    report["answer_problems"] = ap
    report["problems"] += ap
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
    if report.get("answer_problems"):          # invented facts: do not show the rewrite
        removed.append({"section": "improved_answer", **(feedback.get("improved_answer") or {})})
        out["improved_answer"] = {}
    return out, removed
