"""
app/question_gen.py — resume + job title -> interview questions made for this person.

  read_resume(file)                  text from a .pdf / .docx / .txt upload (or "" if unreadable)
  generate(resume, job, lang, n)     Gemini writes n questions; if Gemini is down, picks from
                                     the standard bank in questions.py (never fails)

Each question is a dict {"id", "focus", "en", "th", "why"}; "focus" is one of questions.FOCUS
and tells the coach what a strong answer contains.
Privacy: the resume text goes to Gemini only when the user ticks the consent box.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import questions

MAX_RESUME_CHARS = 6000

SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "focus": {"type": "string", "enum": list(questions.FOCUS)},
            "en": {"type": "string"}, "th": {"type": "string"},
            "why": {"type": "string", "description": "one short line in Thai: why this question for this candidate"},
        },
        "required": ["focus", "en", "th", "why"]}}},
    "required": ["questions"],
}


def read_resume(path) -> str:
    """Best effort; returns '' when the file type is not supported or a library is missing."""
    if not path:
        return ""
    p = Path(path)
    ext = p.suffix.lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages).strip()
        if ext == ".docx":
            import docx
            return "\n".join(par.text for par in docx.Document(str(p)).paragraphs).strip()
        if ext in (".txt", ".md"):
            return p.read_text(encoding="utf-8", errors="ignore").strip()
    except ImportError as e:
        print(f"[resume] cannot read {ext}: {e} (pip install pypdf python-docx)")
    except Exception as e:
        print(f"[resume] cannot read {p.name}: {e}")
    return ""


def _prompt(resume: str, job: str, n: int) -> str:
    focus = "\n".join(f"• {k}: {v[0]}" for k, v in questions.FOCUS.items())
    return f"""You are an experienced interviewer preparing a practice interview.

JOB APPLIED FOR: {job or "(not given - use the resume)"}
CANDIDATE RESUME (may be Thai or English):
{resume[:MAX_RESUME_CHARS] or "(no resume given)"}

Write {n} interview questions for THIS candidate and THIS job:
• Question 1 is an opening question (focus "intro") adapted to the job.
• At least half are behavioural questions (focus "story") that ask about a concrete past
  situation, ideally one that appears in the resume (name the project / activity / internship).
• Include one question about skills the job needs (focus "role") and one about motivation for
  this job or company type (focus "motivation") when n >= 4.
• Short, natural questions a real interviewer would ask, one question each (no multi-part lists).
• Do not ask about age, religion, marital status, health, or other personal/protected topics.
• en = English, th = the same question in natural spoken Thai.
• why = one short line in Thai saying why this question fits this candidate.

Focus types:
{focus}
"""


def _gemini(prompt: str) -> dict:
    from dotenv import load_dotenv
    from google import genai
    from base.llm_common import MODEL_NAME, gen_config, call_with_retry
    import base.llm_common as lc
    lc.RETRY_WAITS_BUSY[:] = [10, 20]             # app: do not wait minutes
    load_dotenv(ROOT / ".env")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return call_with_retry(lambda: json.loads(client.models.generate_content(
        model=MODEL_NAME, contents=prompt, config=gen_config(SCHEMA, 0.7)).text))


def fallback(n: int) -> list:
    """Standard bank (questions.py) when Gemini cannot be reached."""
    out = []
    for qid, focus, en, th in questions.QUESTIONS[:n]:
        out.append({"id": qid, "focus": focus, "en": en, "th": th, "why": "คำถามมาตรฐาน (AI สร้างคำถามไม่ได้ตอนนี้)"})
    return out


def generate(resume: str, job: str, n: int = 5) -> tuple:
    """Returns (questions, note). Never raises."""
    if not (resume or "").strip() and not (job or "").strip():
        return fallback(n), "ยังไม่ได้กรอก resume หรือตำแหน่งงาน จึงใช้คำถามมาตรฐาน"
    try:
        out = _gemini(_prompt(resume or "", job or "", n)).get("questions", [])
    except Exception as e:
        print(f"[questions] Gemini failed, using the standard bank: {str(e)[:200]}")
        return fallback(n), "Gemini ใช้งานไม่ได้ตอนนี้ จึงใช้คำถามมาตรฐานแทน"
    qs = []
    for i, q in enumerate(out[:n], 1):
        if q.get("en") and q.get("th"):
            qs.append({"id": f"g{i}", "focus": q.get("focus") if q.get("focus") in questions.FOCUS else "story",
                       "en": q["en"].strip(), "th": q["th"].strip(), "why": (q.get("why") or "").strip()})
    if not qs:
        return fallback(n), "AI สร้างคำถามไม่สำเร็จ จึงใช้คำถามมาตรฐาน"
    return qs, f"สร้างคำถามให้ {len(qs)} ข้อจาก resume และตำแหน่งงานของคุณ"
