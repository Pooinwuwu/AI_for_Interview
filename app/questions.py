"""
app/questions.py — question bank of the web app (English + Thai).

q1-q6 follow the AVI questions (so English answers use the same coaching focus as the research
part); "intro" is the classic opening question. Thai versions say the same thing in Thai.
The focus lists tell the coach what a strong answer contains.
"""

FOCUS = {
    "strengths": [
        "Strengths are specific and backed by a short example, not just adjectives",
        "The weakness is genuine and comes with what the candidate does to improve it",
        "Clear structure: strength -> evidence -> weakness -> improvement",
        "Delivery: pace, pauses, looking at the camera, steady posture",
    ],
    "friend": [
        "2-3 traits, each with a short example a friend would actually tell",
        "Links the traits to how the candidate works with other people",
        "Balanced and believable rather than a list of compliments",
        "Delivery: pace, pauses, looking at the camera, steady posture",
    ],
    "story": [
        "Describes ONE specific past situation rather than general claims",
        "STAR structure: Situation, Task, Action, Result",
        "Explains why they acted that way (self-reflection)",
        "Specific, honest and concise; 1-2 minutes",
        "Delivery: pace, pauses, looking at the camera, steady posture",
    ],
    "role": [
        "Names the 1-3 skills this job needs most and gives evidence for each (project, course, internship)",
        "Concrete: tools, tasks and results rather than adjectives",
        "Honest about what is still being learned, with a plan",
        "Delivery: pace, pauses, looking at the camera, steady posture",
    ],
    "motivation": [
        "A specific reason for THIS job / type of company, not a generic one",
        "Links the candidate's own experience or interests to the role",
        "Says what they want to contribute and learn",
        "Delivery: pace, pauses, looking at the camera, steady posture",
    ],
    "intro": [
        "Opening hook in the first 5 seconds",
        "Structure: present -> past -> future (or background -> skills -> why this job)",
        "Relevant to the job, with one concrete achievement",
        "Concise: 60-90 seconds",
        "Delivery: pace, pauses, looking at the camera, steady posture",
    ],
}

# id, focus, English text, Thai text
QUESTIONS = [
    ("intro", "intro",
     "Tell me about yourself.",
     "แนะนำตัวเองสั้น ๆ ให้ฟังหน่อย"),
    ("q1", "strengths",
     "What would you consider among your greatest strengths and weaknesses as an employee?",
     "คุณคิดว่าจุดแข็งและจุดอ่อนที่สำคัญที่สุดของคุณในฐานะพนักงานคืออะไร"),
    ("q2", "friend",
     "How would your best friend describe you?",
     "ถ้าให้เพื่อนสนิทอธิบายตัวคุณ เขาจะพูดว่าอย่างไร"),
    ("q3", "story",
     "Tell me about a time you had to make a difficult decision at work or school. What did you do and why?",
     "เล่าถึงครั้งที่คุณต้องตัดสินใจเรื่องยากในการเรียนหรือการทำงาน คุณทำอย่างไร และทำไม"),
    ("q4", "story",
     "Tell me about a time you worked in a team and something went wrong. What did you do?",
     "เล่าถึงครั้งที่คุณทำงานเป็นทีมแล้วเกิดปัญหา คุณทำอย่างไร"),
    ("q5", "story",
     "Tell me about a time you had a disagreement with someone. How did you handle it?",
     "เล่าถึงครั้งที่คุณเห็นต่างกับคนอื่น คุณจัดการอย่างไร"),
    ("q6", "story",
     "Tell me about a time you had to meet a tight deadline. How did you plan your work?",
     "เล่าถึงครั้งที่คุณต้องทำงานให้ทันกำหนดที่กระชั้นชิด คุณวางแผนอย่างไร"),
]

LANGS = {"en": "English", "th": "ภาษาไทย"}


def choices(lang: str):
    return [f"{i + 1}. {q[2] if lang == 'en' else q[3]}" for i, q in enumerate(QUESTIONS)]


def pick(label: str):
    """Dropdown label -> question tuple."""
    try:
        return QUESTIONS[int(label.split(".")[0]) - 1]
    except Exception:
        return QUESTIONS[0]


def as_question(q) -> dict:
    """qid (bank) or a generated question dict -> {"id", "focus", "en", "th"}."""
    if isinstance(q, dict):
        return q
    t = next((x for x in QUESTIONS if x[0] == q), QUESTIONS[0])
    return {"id": t[0], "focus": t[1], "en": t[2], "th": t[3]}


def _context(resume: str, job: str) -> str:
    out = ""
    if job:
        out += f"\nJOB APPLIED FOR: {job}"
    if resume:
        out += ("\nCANDIDATE RESUME (background only - it is NOT evidence of what was said; you may suggest "
                "resume experiences the answer could have used, and use resume facts in improved_answer):\n"
                + resume[:3000])
    return out


def question_block(q, lang: str, resume: str = "", job: str = "") -> str:
    """Replaces llm_common.question_block for web-app clips. q = bank id or generated question dict."""
    q = as_question(q)
    focus = "\n".join(f"• {f}" for f in FOCUS.get(q["focus"], FOCUS["story"]))
    ctx = _context(resume, job)
    facts = "facts the candidate said" + (" or facts in the resume" if resume else "")
    if lang == "th":
        return f"""SETTING: Practice job interview recorded alone on a phone or laptop, one take, 1-2 min.
Eye contact = looking at the camera. THE CANDIDATE ANSWERED IN THAI.{ctx}
QUESTION (asked in Thai): "{q['th']}"  (English: "{q['en']}")
A STRONG ANSWER:
{focus}
LANGUAGE RULES (Thai answer)
• The _th fields are what the candidate reads: write them in natural, friendly Thai.
  The _en fields say the same in English.
• Do not put the candidate's words in quotation marks; describe what was said instead.
• Write times as "41-56s".
• The transcript comes from automatic Thai speech recognition and may contain errors;
  speech rate is counted in Thai words, so do not compare it with English norms.
• improved_answer: rewrite the opening in Thai in rewritten_th; rewritten_en is its English meaning.
  Use only {facts}; NEVER invent jobs, numbers or events. Where a real example
  is needed write a placeholder such as [ตัวอย่างจริงของคุณ: สถานการณ์ - สิ่งที่ทำ - ผลลัพธ์]."""
    return f"""SETTING: Practice job interview recorded alone on a phone or laptop, one take, 1-2 min.
Eye contact = looking at the camera.{ctx}
QUESTION: "{q['en']}"
A STRONG ANSWER:
{focus}
• improved_answer: use only {facts}; NEVER invent jobs, numbers or events.
  Where a real example is needed write a placeholder such as
  [your own example: situation - what you did - result]."""
