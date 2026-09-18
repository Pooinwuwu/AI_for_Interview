"""
llm/prompt_builder.py
ประกอบ prompt + JSON schema สำหรับ Gemini
"""

import json
from typing import Dict, Any


# ============================================================
# JSON SCHEMA (บังคับ LLM ตอบตามโครงสร้าง)
# ============================================================

FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_summary_en": {
            "type": "string",
            "description": "2-3 sentence overall assessment in English",
        },
        "overall_summary_th": {
            "type": "string",
            "description": "2-3 ประโยคสรุปภาพรวมเป็นภาษาไทย",
        },
        "strengths_en": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-3 key strengths in English",
        },
        "strengths_th": {
            "type": "array",
            "items": {"type": "string"},
            "description": "จุดแข็ง 2-3 ข้อ เป็นภาษาไทย",
        },
        "improvements": {
            "type": "array",
            "description": "3-5 areas for improvement, ordered by priority",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "enum": [
                            "eye_contact",
                            "head_pose",
                            "hand_gesture",
                            "facial_expression",
                            "answer_quality",
                        ],
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "issue_en":  {"type": "string"},
                    "issue_th":  {"type": "string"},
                    "suggestion_en": {"type": "string"},
                    "suggestion_th": {"type": "string"},
                    "action_steps_en": {
                        "type": "array", "items": {"type": "string"},
                        "description": "3 concrete steps in English",
                    },
                    "action_steps_th": {
                        "type": "array", "items": {"type": "string"},
                        "description": "3 ขั้นตอนที่ทำได้จริง เป็นภาษาไทย",
                    },
                },
                "required": [
                    "dimension", "priority",
                    "issue_en", "issue_th",
                    "suggestion_en", "suggestion_th",
                    "action_steps_en", "action_steps_th",
                ],
            },
        },
        "improved_answer": {
            "type": "object",
            "properties": {
                "original_snippet": {
                    "type": "string",
                    "description": "Verbatim first 1-2 sentences from the transcript",
                },
                "rewritten_en": {"type": "string"},
                "rewritten_th": {"type": "string"},
                "why_better_en": {"type": "string"},
                "why_better_th": {"type": "string"},
            },
            "required": [
                "original_snippet",
                "rewritten_en", "rewritten_th",
                "why_better_en", "why_better_th",
            ],
        },
        "seven_day_plan_en": {
            "type": "array",
            "items": {"type": "string"},
            "description": "7 items: 'Day 1: ...' to 'Day 7: ...' in English",
        },
        "seven_day_plan_th": {
            "type": "array",
            "items": {"type": "string"},
            "description": "7 รายการ: 'วันที่ 1: ...' ถึง 'วันที่ 7: ...' เป็นภาษาไทย",
        },
    },
    "required": [
        "overall_summary_en", "overall_summary_th",
        "strengths_en", "strengths_th",
        "improvements",
        "improved_answer",
        "seven_day_plan_en", "seven_day_plan_th",
    ],
}


# ============================================================
# PROMPT BUILDER
# ============================================================

DIMENSION_LABEL = {
    "eye_contact":       "Eye Contact",
    "head_pose":         "Head Movement",
    "hand_gesture":      "Hand Gestures",
    "facial_expression": "Facial Expression",
    "answer_quality":    "Answer Quality",
}


def _format_scores(scores: Dict[str, Any]) -> str:
    """สรุปคะแนน + components เป็นข้อความสั้น อ่านง่าย"""
    lines = []
    dims = scores.get("dimensions", {})
    fusion = scores.get("fusion", {})

    lines.append(f"OVERALL: {fusion.get('overall_score', 0)}/100")
    lines.append("")

    for key in ["eye_contact", "head_pose", "hand_gesture",
                "facial_expression", "answer_quality"]:
        d = dims.get(key, {})
        score = d.get("score", 0)
        label = DIMENSION_LABEL[key]
        lines.append(f"• {label}: {score}/100")

        comps = d.get("components", {})
        for cname, cval in comps.items():
            lines.append(f"    - {cname}: {cval}")

        raw = d.get("raw", {})
        for rname, rval in raw.items():
            lines.append(f"    raw {rname}: {rval}")
        lines.append("")

    return "\n".join(lines)


def build_prompt(scores: Dict[str, Any],
                 speech: Dict[str, Any]) -> str:
    """
    สร้าง prompt จาก scores + speech features
    Scenario: single 'Tell me about yourself' question, < 1 minute
    """
    key = scores.get("key", "unknown")
    transcript = speech.get("transcript", "(no transcript)")
    words = speech.get("features", {}).get("word_count", 0)
    duration = speech.get("features", {}).get("duration_sec", 0)

    scores_text = _format_scores(scores)

    prompt = f"""You are an expert interview coach analyzing a candidate's response to a "Tell me about yourself" question in a job interview.

CONTEXT
-------
• Question: "Tell me about yourself" (opening question)
• Duration: {duration} seconds
• Word count: {words}
• Language: English

MULTIMODAL SCORES (rule-based, from video + audio analysis)
-----------------------------------------------------------
{scores_text}

CANDIDATE'S TRANSCRIPT
----------------------
"{transcript}"

YOUR TASK
---------
Provide structured, actionable feedback. The candidate is practicing for real job interviews, so your feedback must be:
1. Specific (reference actual scores and transcript content)
2. Actionable (concrete steps, not vague advice)
3. Encouraging (acknowledge strengths before critique)
4. Prioritized (focus on the 3-5 most impactful improvements)

LANGUAGE REQUIREMENTS
---------------------
• Primary content: English (detailed, natural, professional)
• Thai fields (ending in _th): concise but complete Thai translations
  NOT one-word — should convey the same meaning as the English field

FOCUS AREAS FOR "TELL ME ABOUT YOURSELF"
-----------------------------------------
This question is about FIRST IMPRESSION. Key things to evaluate:
• Opening hook — does it grab attention in the first 5 seconds?
• Structure — Present → Past → Future, or Hook → Background → Value?
• Relevance — does it connect to the target role?
• Conciseness — 60-90 seconds is ideal; too short = incomplete, too long = rambling
• Delivery — pace, filler words, eye contact, confident posture

For the 'improved_answer':
• Take the FIRST 1-2 sentences from the transcript as 'original_snippet'
• Rewrite as a stronger opening (with hook, structure, and clear value proposition)
• Keep it realistic for the candidate's apparent experience level

For 'seven_day_plan':
• Each day should be 15-30 minutes of focused practice
• Day 1-2: Foundation (structure, script)
• Day 3-4: Delivery (pace, filler, gestures)
• Day 5-6: Recording + self-review
• Day 7: Mock interview + final polish

Generate the JSON now following the schema exactly.
"""
    return prompt