"""
llm/prompt_builder.py

Prompt + JSON schema for Gemini (Approach 1). FEEDBACK_SCHEMA is shared by
all three approaches (Approach 2 and 3 import it from here).

Compact version (prompt_version avi-v2):
  - short instructions; one line per dimension instead of raw numbers
  - smaller answer: max 3 strengths, max 3 improvements with 2 steps each,
    no 7-day plan (the web app can generate one on demand)
  - LLM sees bands; numbers are not quoted back to the user
"""

import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from base.llm_common import (RATINGS_SCHEMA, RATINGS_INSTRUCTIONS, BANDS_TH,
                             DIMENSIONS, question_block)


# ============================================================
# JSON SCHEMA (shared by all approaches)
# ============================================================

def _str_list(desc: str, max_items: int) -> dict:
    return {"type": "array", "items": {"type": "string"},
            "maxItems": max_items, "description": desc}


FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_summary_en": {"type": "string",
                               "description": "2 sentences. Bands, no numbers."},
        "overall_summary_th": {"type": "string",
                               "description": "2 ประโยค ใช้ระดับ ไม่ใช้ตัวเลข"},
        "strengths_en": _str_list("max 3 short strengths", 3),
        "strengths_th": _str_list("จุดแข็ง ไม่เกิน 3 ข้อ", 3),
        "improvements": {
            "type": "array",
            "maxItems": 3,
            "description": "the 3 most important improvements, highest priority first",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": DIMENSIONS},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    "issue_en": {"type": "string"},
                    "issue_th": {"type": "string"},
                    "action_steps_en": _str_list("2 concrete steps", 2),
                    "action_steps_th": _str_list("2 ขั้นตอนที่ทำได้จริง", 2),
                },
                "required": ["dimension", "priority", "issue_en", "issue_th",
                             "action_steps_en", "action_steps_th"],
            },
        },
        "improved_answer": {
            "type": "object",
            "properties": {
                "original_snippet": {"type": "string",
                                     "description": "verbatim first 1-2 sentences"},
                "rewritten_en": {"type": "string"},
                "rewritten_th": {"type": "string"},
                "why_better_en": {"type": "string", "description": "1 sentence"},
            },
            "required": ["original_snippet", "rewritten_en", "rewritten_th",
                         "why_better_en"],
        },
        # lets validation/evaluate_avi.py compare the approaches
        "ratings": RATINGS_SCHEMA,
    },
    "required": ["overall_summary_en", "overall_summary_th",
                 "strengths_en", "strengths_th",
                 "improvements", "improved_answer", "ratings"],
}


# ============================================================
# LABELS
# ============================================================

DIMENSION_LABEL_TH = {
    "eye_contact":       "การสบตา",
    "head_pose":         "การขยับศีรษะ",
    "hand_gesture":      "การขยับมือ",
    "facial_expression": "สีหน้า",
    "answer_quality":    "การตอบคำถาม",
}

DIMENSION_LABEL_EN = {
    "eye_contact":       "Eye contact",
    "head_pose":         "Head movement",
    "hand_gesture":      "Hand gestures",
    "facial_expression": "Facial expression",
    "answer_quality":    "Speech delivery",
}

# Shared rules for every approach's prompt
OUTPUT_RULES = f"""RULES
-----
• Be specific to this candidate and this question; give concrete, doable steps.
• Never quote numeric scores; speak in bands ({" / ".join(BANDS_TH)}).
• English fields: natural and professional. _th fields: same meaning in Thai.
• improved_answer: rewrite the first 1-2 sentences as a stronger opening for THIS question."""


# ============================================================
# FORMAT SCORES (compact: band + sub-component scores)
# ============================================================

def _format_scores(scores: Dict[str, Any]) -> str:
    dims = scores.get("dimensions", {})
    fusion = scores.get("fusion", {})
    lines = [f"Overall: {fusion.get('overall_band', {}).get('label_en', '-')}"]
    for key in DIMENSIONS:
        d = dims.get(key, {})
        band = d.get("band", {}).get("label_en", "-")
        comps = ", ".join(f"{c} {v:.0f}" for c, v in d.get("components", {}).items())
        lines.append(f"• {DIMENSION_LABEL_EN[key]}: {band}" + (f"  ({comps})" if comps else ""))
    return "\n".join(lines)


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(scores: Dict[str, Any],
                 speech: Dict[str, Any]) -> str:
    """Prompt from rule-based scores + transcript. Question context comes from the clip key."""
    key = scores.get("key", "unknown")
    transcript = speech.get("transcript", "(no transcript)")
    feats = speech.get("features", {})

    return f"""You are an interview coach. Give feedback on one recorded answer.

{question_block(key)}

MEASURED BEHAVIOUR (bands; sub-scores 0-100 in brackets, for your reasoning only)
---------------------------------------------------------------------------------
{_format_scores(scores)}
Duration {feats.get('duration_sec', 0)}s, {feats.get('word_count', 0)} words, \
{feats.get('speech_rate_wpm', 0)} wpm, {feats.get('filler_count', 0)} fillers, \
{feats.get('pause_count', 0)} pauses.

TRANSCRIPT
----------
"{transcript}"

{OUTPUT_RULES}

{RATINGS_INSTRUCTIONS}
Copy the measured bands above into 'ratings' (they are measured; do not change them).
"""
