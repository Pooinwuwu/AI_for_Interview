"""
base/llm_common.py

Settings and helpers shared by the LLM step of all three approaches, so the
comparison is fair: same model, same temperature, same question context, same
rating scale, and every output records which model/prompt produced it.

Change the model in ONE place:
  - .env:  GEMINI_MODEL=gemini-3.8-flash      (preferred, no code change)
  - or MODEL_NAME default below
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from base.dataset.avi import parse_key, question_for_key

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# ============================================================
# SHARED SETTINGS
# ============================================================

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
TEMPERATURE = 0.2           # low = more repeatable (stability test, RQ3)
PROMPT_VERSION = "avi-v2"   # bump whenever any prompt text changes (v2 = compact prompts/answers)

# Band scale used everywhere (system scores, LLM ratings, evaluation)
BANDS_TH = ["ดีมาก", "ดี", "ปานกลาง", "ควรปรับ", "ควรปรับมาก"]
BANDS_EN = ["Excellent", "Good", "Fair", "Needs Work", "Priority"]

DIMENSIONS = ["eye_contact", "head_pose", "hand_gesture",
              "facial_expression", "answer_quality"]

RATINGS_SCHEMA = {
    "type": "object",
    "description": "One band per dimension plus an overall band for this answer. "
                   "Use exactly one of: " + " / ".join(BANDS_TH),
    "properties": {
        name: {"type": "string", "enum": BANDS_TH}
        for name in DIMENSIONS + ["overall"]
    },
    "required": DIMENSIONS + ["overall"],
}


def run_info() -> dict:
    """Stored in every feedback JSON so results are traceable."""
    return {"model": MODEL_NAME, "temperature": TEMPERATURE,
            "prompt_version": PROMPT_VERSION}


# ============================================================
# GEMINI CALLS: config, retry, stop-when-down
# ============================================================

# waits (seconds) when the model is overloaded (503) or rate-limited (429)
RETRY_WAITS_BUSY = [15, 30, 60, 90, 120]      # ~5 min per clip before giving up
RETRY_WAITS_OTHER = [2, 5]                    # other errors: quick retries only
STOP_AFTER_BUSY_CLIPS = 2                     # consecutive clips failing with 503 -> stop run


class ModelBusy(RuntimeError):
    """Model stayed overloaded (503) or out of quota (429) after all retries."""

    def __init__(self, msg: str, kind: str):
        super().__init__(msg)
        self.kind = kind          # "overloaded" | "quota"


def _busy_kind(err):
    """'overloaded' (503, Google-side, clears in minutes), 'quota' (429, your
    key's limit; a daily limit only resets the next day), or None."""
    s = str(err)
    if any(t in s for t in ("429", "RESOURCE_EXHAUSTED", "quota")):
        # daily limit: retrying today is pointless
        if any(t in s for t in ("per day", "PerDay", "per_day", "RequestsPerDay")):
            return "daily_quota"
        return "quota"
    if any(t in s for t in ("503", "UNAVAILABLE", "overloaded", "high demand")):
        return "overloaded"
    return None


def gen_config(schema: dict, temperature: float = TEMPERATURE):
    """Same GenerateContentConfig for every approach."""
    from google.genai import types
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        # no tools are used; disabling AFC also removes the SDK's AFC notice
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def call_with_retry(fn):
    """
    Run fn() (one Gemini request). Retries with long waits on 503/429 and short
    waits on other errors. Never switches model (that would break the comparison).
    """
    import time
    busy_try = other_try = 0
    while True:
        try:
            return fn()
        except Exception as e:
            kind = _busy_kind(e)
            if kind == "daily_quota":
                raise ModelBusy(f"{MODEL_NAME} daily quota used up: {str(e)[:300]}", "quota")
            if kind:
                if busy_try >= len(RETRY_WAITS_BUSY):
                    raise ModelBusy(f"{MODEL_NAME} {kind} after {len(RETRY_WAITS_BUSY)} "
                                    f"retries: {str(e)[:300]}", kind)
                wait = RETRY_WAITS_BUSY[busy_try]
                busy_try += 1
                label = "503 overloaded" if kind == "overloaded" else "429 quota/rate limit"
                print(f"       [{label}] {MODEL_NAME} — wait {wait}s "
                      f"(retry {busy_try}/{len(RETRY_WAITS_BUSY)})")
            else:
                if other_try >= len(RETRY_WAITS_OTHER):
                    raise
                wait = RETRY_WAITS_OTHER[other_try]
                other_try += 1
                print(f"       [retry {other_try}] {str(e)[:100]} — wait {wait}s")
            time.sleep(wait)


class BusyGuard:
    """Stops a batch run when the model is clearly down instead of failing every clip."""

    def __init__(self, limit: int = STOP_AFTER_BUSY_CLIPS):
        self.limit, self.count = limit, 0

    def ok(self):
        self.count = 0

    def failed(self, err) -> bool:
        """Record a failed clip. Returns True if the run should stop now."""
        self.count = self.count + 1 if isinstance(err, ModelBusy) else 0
        if "daily quota" in str(err):
            self.count = self.limit          # no point trying more clips today
        if self.count >= self.limit:
            print()
            print("=" * 60)
            if err.kind == "quota":
                print(f"  STOPPED: 429 — your API key hit a quota / rate limit "
                      f"({self.count} clips in a row).")
                print("  This is YOUR key's limit, not Google being busy. If it is the")
                print("  daily free-tier limit, it resets the next day; check usage at")
                print("  https://aistudio.google.com (API keys -> usage / rate limits).")
            else:
                print(f"  STOPPED: 503 — {MODEL_NAME} was overloaded on Google's side "
                      f"for {self.count} clips in a row.")
            print("  Last error:", str(err)[:300])
            print("  Nothing is lost — finished clips are kept and skipped next time.")
            print("  Options:")
            print("   1) run the same command again later")
            print("   2) check which models respond:  python base\\check_models.py")
            print("   3) switch ALL approaches: set GEMINI_MODEL=... in .env,")
            print("      then rerun every approach with --force")
            print("=" * 60)
            return True
        return False


# ============================================================
# QUESTION CONTEXT
# ============================================================

SETTING_AVI = (
    "One-way video interview (management traineeship), recorded alone to a webcam, "
    "one take, 1-2 min. Eye contact = looking at the camera."
)

FOCUS = {
    "q1": [
        "Strengths are specific and backed by a short example, not just adjectives",
        "The weakness is genuine (not a disguised strength) and comes with what the "
        "candidate does to manage or improve it",
        "Relevance to a management traineeship (working with people, organising, learning)",
        "Clear structure: strengths -> evidence -> weakness -> improvement",
        "Delivery — pace, filler words, looking at the camera, steady posture",
    ],
    "q2": [
        "Concrete traits illustrated with a brief anecdote or example",
        "Answers from the friend's point of view, as the question asks",
        "Links the traits to how the candidate would work with colleagues",
        "Balanced and believable rather than a list of compliments",
        "Delivery — pace, filler words, looking at the camera, steady posture",
    ],
    "past_behaviour": [
        "Describes ONE specific past situation rather than general claims "
        "('I usually...')",
        "STAR structure: Situation, Task, Action, Result",
        "Answers the second part of the question: explains WHY they behave that way "
        "(self-reflection)",
        "Specific, honest and concise; stays within 1-2 minutes",
        "Delivery — pace, filler words, looking at the camera, steady posture",
    ],
    "fallback": [
        "Opening hook — does it grab attention in the first 5 seconds?",
        "Structure — Present -> Past -> Future, or Hook -> Background -> Value",
        "Relevance — does it connect to the target role?",
        "Conciseness — 60-90 seconds is ideal",
        "Delivery — pace, filler words, eye contact, confident posture",
    ],
}


def question_context(key: str) -> dict:
    """
    Question text + coaching focus for a clip.
    Note: for personality questions (q3-q6) we deliberately do NOT tell the LLM
    which trait the question targets — the coach gives feedback on observable
    answer quality and delivery, not on personality.
    """
    import re
    own = re.match(r"^own_\d+_q([1-6])$", key)      # researcher's own clips, e.g. own_03_q1
    if own:
        from base.dataset.avi import load_questions
        qno = f"q{own.group(1)}"
        q = load_questions()[qno]
        return {"text": q["text"], "type": q["type"],
                "setting": "One-way video interview, recorded alone on a phone, one take, 1-2 min. "
                           "Eye contact = looking at the camera.",
                "focus": FOCUS.get(qno) or FOCUS["past_behaviour"]}
    q = question_for_key(key)
    if q is None:   # non-AVI video (e.g. old vid_0021)
        return {"text": "Tell me about yourself", "type": "opening",
                "setting": "Live job interview, opening question.",
                "focus": FOCUS["fallback"]}
    qno = f"q{parse_key(key)['question_no']}"
    focus = FOCUS.get(qno) or FOCUS["past_behaviour"]
    return {"text": q["text"], "type": q["type"], "setting": SETTING_AVI,
            "focus": focus}


def question_block(key: str) -> str:
    """Ready-to-paste CONTEXT + FOCUS section for any prompt."""
    q = question_context(key)
    focus = "\n".join(f"• {f}" for f in q["focus"])
    return f"""SETTING: {q['setting']}
QUESTION: "{q['text']}"
A STRONG ANSWER:
{focus}"""


RATINGS_INSTRUCTIONS = (
    "RATINGS: one band per dimension + 'overall', judged as an experienced recruiter "
    f"would for this role. Bands: {' / '.join(BANDS_TH)} (= {' / '.join(BANDS_EN)})."
)


# ============================================================
# CLI
# ============================================================

def parse_run_args(description: str):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--force", action="store_true",
                    help="regenerate feedback even if the output file already exists")
    ap.add_argument("--only", nargs="+", default=None,
                    help="only these clip keys")
    ap.add_argument("--tag", default="",
                    help="repeat run name (e.g. r2): writes to <output folder>_<tag>, "
                         "so repeated runs can be compared for stability")
    ap.add_argument("--allow-avi", action="store_true",
                    help="also send AVI clips to Gemini - only with the AVI authors' approval")
    return ap.parse_args()


def should_skip(out_path: Path, force: bool) -> bool:
    if out_path.exists() and not force:
        print(f"[SKIP] {out_path.name} exists (use --force to regenerate)")
        return True
    return False
