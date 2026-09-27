"""
approaches/approach_3_hybrid/interpretation/prompt_builder.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "approaches" / "approach_1_rule" / "llm"))
from prompt_builder import FEEDBACK_SCHEMA

def build_hybrid_prompt(evidence_text: str) -> str:
    prompt = f"""You are an expert interview coach analyzing a candidate's response to a "Tell me about yourself" question.

CONTEXT
-------
• Question: "Tell me about yourself" (opening question)
• Language: English

YOUR TASK
---------
Watch the video AND review the detailed SYSTEM EVIDENCE LOG below. 
The log provides precise timestamps for speech, gaze, head movements, hand gestures, and facial expressions as extracted by computer vision and audio analysis tools.

Evaluate the candidate across 5 dimensions:
1. Eye Contact
2. Head Movement
3. Hand Gestures
4. Facial Expression
5. Answer Quality (Spoken content)

Use the evidence log to GROUND your visual observations (e.g., if the log says the candidate looked away frequently or had low hand presence, reflect that in your qualitative feedback).

SYSTEM EVIDENCE LOG
-------------------
{evidence_text}
-------------------

Evaluate each dimension as a QUALITATIVE BAND:
  - ดีมาก (excellent)
  - ดี (good)
  - ปานกลาง (fair)
  - ควรปรับ (needs work)
  - ควรปรับมาก (priority)

Provide structured, actionable feedback following the required JSON schema.
• Provide the primary content in English, and precise Thai translations in fields ending in `_th`.
• DO NOT invent numeric scores. Use qualitative bands only.
• Combine what you see in the video with the hard data from the evidence log to give highly specific feedback.
• Transcribe the first 1-2 sentences for the 'improved_answer' section and provide a better version.
"""
    return prompt
