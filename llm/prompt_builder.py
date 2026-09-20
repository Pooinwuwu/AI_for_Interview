"""
llm/prompt_builder.py

ประกอบ prompt + JSON schema สำหรับ Gemini

การเปลี่ยนแปลง:
  - LLM เห็น "band" (ระดับ) เป็นหลัก
  - ตัวเลขยังมีให้เห็น แต่ใช้เป็น reference (ไม่พูดออกมา)
  - Feedback ต้องพูดเป็น "ปานกลาง/ดี/ควรปรับ" ไม่ใช่ "68.4 คะแนน"
"""

import json
from typing import Dict, Any


# ============================================================
# JSON SCHEMA
# ============================================================

FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_summary_en": {
            "type": "string",
            "description": "2-3 sentence overall assessment in English. "
                           "Use qualitative bands (e.g., 'fair', 'good') "
                           "instead of numeric scores.",
        },
        "overall_summary_th": {
            "type": "string",
            "description": "2-3 ประโยคสรุปภาพรวมเป็นภาษาไทย ใช้ระดับ "
                           "(ดีมาก/ดี/ปานกลาง/ควรปรับ/ควรปรับมาก) "
                           "ไม่ต้องพูดตัวเลข",
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
                    "band_th": {
                        "type": "string",
                        "description": "ระดับคุณภาพของด้านนี้ (ภาษาไทย): "
                                       "ดีมาก/ดี/ปานกลาง/ควรปรับ/ควรปรับมาก",
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
                    "dimension", "priority", "band_th",
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
# DIMENSION LABELS
# ============================================================

DIMENSION_LABEL_TH = {
    "eye_contact":       "การสบตา",
    "head_pose":         "การขยับศีรษะ",
    "hand_gesture":      "การขยับมือ",
    "facial_expression": "สีหน้า",
    "answer_quality":    "การตอบคำถาม",
}

DIMENSION_LABEL_EN = {
    "eye_contact":       "Eye Contact",
    "head_pose":         "Head Movement",
    "hand_gesture":      "Hand Gestures",
    "facial_expression": "Facial Expression",
    "answer_quality":    "Answer Quality",
}

COMPONENT_LABEL_TH = {
    "eye_contact_ratio":   "สัดส่วนการสบตา",
    "gaze_stability":      "ความนิ่งของสายตา",
    "pitch_movement":      "การก้ม-เงย",
    "yaw_movement":        "การหันซ้าย-ขวา",
    "pitch_deviation":     "การก้ม-เงยเฉลี่ย",
    "yaw_deviation":       "การหันเฉลี่ย",
    "presence":            "การปรากฏของมือ",
    "movement_speed":      "ความเร็วมือ",
    "fidget":              "ความสั่นของมือ",
    "smile":               "การยิ้ม",
    "expressiveness":      "การแสดงออกทางสีหน้า",
    "eye_openness":        "การเปิดตา",
    "brow_movement":       "การขยับคิ้ว",
    "speech_rate":         "ความเร็วในการพูด",
    "filler_ratio":        "คำฟุ่มเฟือย",
    "pause_freq":          "ความถี่การหยุด",
    "long_pauses":         "การหยุดนาน",
}


# ============================================================
# FORMAT SCORES (แบบใหม่ — band เป็นหลัก)
# ============================================================

def _format_scores(scores: Dict[str, Any]) -> str:
    """
    สรุปคะแนนเป็น text โดยให้ band เป็นหลัก
    ตัวเลขยังมีให้เห็นเป็น (reference) แต่ระบุชัดว่า "ห้ามพูดออกมา"
    """
    lines = []
    dims = scores.get("dimensions", {})
    fusion = scores.get("fusion", {})

    # ---- Overall ----
    overall_score = fusion.get("overall_score", 0)
    overall_band  = fusion.get("overall_band", {})
    overall_th    = overall_band.get("label_th", "-")
    overall_en    = overall_band.get("label_en", "-")

    lines.append(f"OVERALL: {overall_th} / {overall_en}")
    lines.append(f"  (numeric reference: {overall_score}/100)")
    lines.append("")

    # ---- แต่ละ dimension ----
    for key in ["eye_contact", "head_pose", "hand_gesture",
                "facial_expression", "answer_quality"]:
        d = dims.get(key, {})
        score = d.get("score", 0)
        band  = d.get("band", {})
        band_th = band.get("label_th", "-")
        band_en = band.get("label_en", "-")

        label_th = DIMENSION_LABEL_TH[key]
        label_en = DIMENSION_LABEL_EN[key]

        lines.append(f"• {label_en} ({label_th}): {band_th} / {band_en}")
        lines.append(f"  (numeric reference: {score}/100)")

        # ---- components ----
        comps = d.get("components", {})
        if comps:
            lines.append("  Sub-components (0-100):")
            for cname, cval in comps.items():
                clabel = COMPONENT_LABEL_TH.get(cname, cname)
                lines.append(f"    - {clabel} ({cname}): {cval}")

        # ---- raw values ----
        raw = d.get("raw", {})
        if raw:
            lines.append("  Raw measurements:")
            for rname, rval in raw.items():
                lines.append(f"    - {rname}: {rval}")

        lines.append("")

    return "\n".join(lines)


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(scores: Dict[str, Any],
                 speech: Dict[str, Any]) -> str:
    """
    สร้าง prompt จาก scores + speech features
    Scenario: single 'Tell me about yourself' question, < 1 minute
    """
    key = scores.get("key", "unknown")
    transcript = speech.get("transcript", "(no transcript)")
    feats = speech.get("features", {})
    words = feats.get("word_count", 0)
    duration = feats.get("duration_sec", 0)

    scores_text = _format_scores(scores)

    prompt = f"""You are an expert interview coach analyzing a candidate's response to a "Tell me about yourself" question in a job interview.

CONTEXT
-------
• Question: "Tell me about yourself" (opening question)
• Duration: {duration} seconds
• Word count: {words}
• Language: English

MULTIMODAL ANALYSIS
-------------------
The candidate's behavior was measured across 5 dimensions.
Each dimension is reported as a QUALITATIVE BAND:

  85-100  Excellent    ดีมาก
  70-84   Good         ดี
  55-69   Fair         ปานกลาง
  40-54   Needs Work   ควรปรับ
  0-39    Priority     ควรปรับมาก

{scores_text}

CANDIDATE'S TRANSCRIPT
----------------------
"{transcript}"

YOUR TASK
---------
Provide structured, actionable feedback. The candidate is practicing for real job interviews, so your feedback must be:
1. Specific (reference actual behaviors and transcript content)
2. Actionable (concrete steps, not vague advice)
3. Encouraging (acknowledge strengths before critique)
4. Prioritized (focus on the 3-5 most impactful improvements)

⚠️ CRITICAL — HOW TO TALK ABOUT SCORES
---------------------------------------
• DO NOT quote numeric scores (e.g., "68.4", "72", "85") in your feedback.
• DO speak in QUALITATIVE TERMS using these Thai bands:
    - ดีมาก (excellent)
    - ดี (good)
    - ปานกลาง (fair)
    - ควรปรับ (needs work)
    - ควรปรับมาก (priority)
• Example GOOD phrasing:
    ✅ "การสบตาของคุณอยู่ในระดับปานกลาง"
    ✅ "Your eye contact was fair — you maintained focus most of the time."
• Example BAD phrasing (DO NOT USE):
    ❌ "คะแนนการสบตาคือ 68.4"
    ❌ "You scored 72/100 on head movement."
• Use the numeric scores and raw measurements internally to justify your reasoning,
  but express the conclusion as a band.

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

PRIORITIZATION RULES
--------------------
• Order improvements by priority: high → medium → low
• If overall band is "ดีมาก", focus on fine-tuning rather than major changes
• If overall band is "ควรปรับมาก", focus on the top 2-3 critical issues first
• Reference dimensions using the BAND labels (e.g., "your hand gestures were fair")

FOR THE 'improved_answer' FIELD
--------------------------------
• Take the FIRST 1-2 sentences from the transcript as 'original_snippet'
• Rewrite as a stronger opening (with hook, structure, and clear value proposition)
• Keep it realistic for the candidate's apparent experience level
• Explain WHY it's better (mention specific techniques like "signposting", "STAR", etc.)

FOR THE 'seven_day_plan' FIELD
------------------------------
• Each day should be 15-30 minutes of focused practice
• Day 1-2: Foundation (structure, script)
• Day 3-4: Delivery (pace, filler, gestures)
• Day 5-6: Recording + self-review
• Day 7: Mock interview + final polish

Generate the JSON now following the schema exactly. Remember: NO numeric scores in the output — only qualitative bands.
"""
    return prompt