"""
approaches/approach_student/student_prompt.py

ONE prompt used three times, so the student learns exactly the task it will get in the app:
  - the teacher (Gemini) answers it to make training data
  - the student is fine-tuned on (this prompt -> teacher answer)
  - the app sends it to the student (Ollama) at run time
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))

import questions   # app/questions.py: question text + "A STRONG ANSWER" + language rules

STUDENT_PROMPT_VERSION = "student-v1"
SYSTEM = "You are an interview practice coach. Answer with JSON only."

BANDS = "ดีมาก / ดี / ปานกลาง / ควรปรับ / ควรปรับมาก"
DIM_LABEL = {"eye_contact": "Eye contact", "head_pose": "Head pose", "hand_gesture": "Hand gesture",
             "facial_expression": "Facial expression", "answer_quality": "Answer quality"}


def build(qid: str, lang: str, ratings: dict, log_text: str) -> str:
    scores = "\n".join([f"Overall: {ratings.get('overall', '-')}"] +
                       [f"• {DIM_LABEL[d]}: {ratings.get(d, '-')}" for d in DIM_LABEL])
    return f"""Coach the candidate on this one practice answer.

{questions.question_block(qid, lang)}

SCORES (decided by measurement - do not change them; bands: {BANDS})
{scores}

EVIDENCE LOG (cite the IDs in [ ])
{log_text}

RULES
• Every strength and improvement lists the evidence_ids that support it.
  Visual points cite S/E items of that behaviour; speech points cite S/P items;
  content points cite T items (what was said).
• "dimension" must match the evidence: T, P and speech items -> answer_quality; gaze ->
  eye_contact; head -> head_pose; hand -> hand_gesture; face -> facial_expression.
• Only mention a time if it is the time of an item you cite, written like (41-56s).
• Do not claim anything the log does not show. A behaviour marked "not measured" must not be
  judged. Filler words are NOT measured: never mention them.
• Content matters: judge what was said against "A STRONG ANSWER" using the transcript. If
  something is missing, at least one improvement says what and where (cite the T item).
  Do not praise content the transcript does not show.
• At most 3 strengths and 3 improvements, most important first. Each point 1-2 short sentences.
  Action steps: concrete things to do in the next practice. No numeric scores.
• overall_summary: 2 sentences that only repeat points you listed.
• _en fields in plain English, _th fields the same meaning in natural Thai.
"""
