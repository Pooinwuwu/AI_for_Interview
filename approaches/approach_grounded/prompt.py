"""
approaches/approach_grounded/prompt.py

Prompt + JSON schema for the grounded approaches (C = text only, D = + video).

Division of roles
  judge   : scores come from Tier-1 measurement + the trained model (not from the LLM)
  writer  : the LLM explains those scores and coaches, citing evidence IDs
  checker : verify.py checks every cited claim; failures go back once for revision
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "approaches" / "approach_1_rule" / "llm"))

from base.llm_common import BANDS_TH, DIMENSIONS, question_block
from prompt_builder import DIMENSION_LABEL_EN

PROMPT_VERSION = "grounded-v1"

_IDS = {"type": "array", "items": {"type": "string"}, "minItems": 1,
        "description": "IDs from the evidence log that support this point, e.g. [\"S1\", \"E12\"]"}

FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_summary_en": {"type": "string", "description": "2 sentences, bands, no numbers"},
        "overall_summary_th": {"type": "string"},
        "strengths": {
            "type": "array", "maxItems": 3,
            "items": {"type": "object", "properties": {
                "dimension": {"type": "string", "enum": DIMENSIONS},
                "text_en": {"type": "string"}, "text_th": {"type": "string"},
                "evidence_ids": _IDS},
                "required": ["dimension", "text_en", "text_th", "evidence_ids"]}},
        "improvements": {
            "type": "array", "maxItems": 3,
            "items": {"type": "object", "properties": {
                "dimension": {"type": "string", "enum": DIMENSIONS},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                "issue_en": {"type": "string"}, "issue_th": {"type": "string"},
                "evidence_ids": _IDS,
                "action_steps_en": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
                "action_steps_th": {"type": "array", "items": {"type": "string"}, "maxItems": 2}},
                "required": ["dimension", "priority", "issue_en", "issue_th", "evidence_ids",
                             "action_steps_en", "action_steps_th"]}},
        "improved_answer": {
            "type": "object", "properties": {
                "original_snippet": {"type": "string"},
                "rewritten_en": {"type": "string"}, "rewritten_th": {"type": "string"},
                "why_better_en": {"type": "string"}},
            "required": ["original_snippet", "rewritten_en", "rewritten_th", "why_better_en"]},
    },
    "required": ["overall_summary_en", "overall_summary_th", "strengths", "improvements",
                 "improved_answer"],
}


def _scores_block(ratings: dict, overall_note: str) -> str:
    lines = [f"Overall: {ratings.get('overall', '-')}  ({overall_note})"]
    for d in DIMENSIONS:
        lines.append(f"• {DIMENSION_LABEL_EN[d]}: {ratings.get(d, '-')}")
    return "\n".join(lines)


def build_prompt(key: str, log_text: str, ratings: dict, overall_note: str,
                 with_video: bool) -> str:
    video_rule = (
        "• You also see the VIDEO. Use it to understand context, but every point must still "
        "cite log IDs. If something important is visible only in the video, you may cite "
        "\"VIDEO\" - such points are marked unverified."
        if with_video else
        "• You do NOT see the video. Use only the evidence log and transcript.")
    return f"""You are an interview coach. Explain the scores below and coach the candidate on this one answer.

{question_block(key)}

SCORES (already decided by measurement and a model trained on recruiter ratings - do not change them)
{_scores_block(ratings, overall_note)}
Bands: {" / ".join(BANDS_TH)}

EVIDENCE LOG (measured by computer vision / speech tools; cite the IDs in [ ])
{log_text}

RULES
• Every strength and improvement must list the evidence_ids that support it.
  Visual points cite S/E items of that behaviour; speech points cite S/P items;
  content points cite T items (what was said).
• Only mention a time if it is the time of an item you cite (e.g. "41-56s").
• Do not claim anything the log does not show. Filler words are NOT measured: do not
  count or quote fillers.
{video_rule}
• Content matters: judge what was said against "A STRONG ANSWER" using the transcript.
• Never quote numeric scores; use bands. English fields natural; _th fields same meaning in Thai.
• improved_answer: rewrite the first 1-2 sentences as a stronger opening for THIS question.
"""


def build_revision_prompt(original_prompt: str, draft_json: str, problems: list) -> str:
    probs = "\n".join(f"- {p}" for p in problems)
    return f"""{original_prompt}

YOUR PREVIOUS ANSWER
{draft_json}

A CHECKER COMPARED YOUR ANSWER WITH THE EVIDENCE LOG AND FOUND THESE PROBLEMS
{probs}

Return the full corrected answer in the same format. Fix each problem using only what the
log shows (correct the time, the cited IDs or the claim). If a point cannot be supported,
remove it. Keep everything else unchanged.
"""
