"""
approaches/approach_student/synth.py

Synthetic practice clips for training the student coach. No real person is involved:

  1. answers   Gemini writes made-up candidate answers (strong / average / weak, with chosen
               flaws, English and Thai) for the app's questions
  2. clip      each answer gets a made-up "recording": speaking rate, pauses, and visual
               behaviour (gaze / head / hands / face) drawn at random from wide ranges
  3. log       the made-up recording goes through the SAME base/grounded_log.build_from as a
               real clip, so the student sees exactly the format it will get in the app
  4. ratings   dimension bands from simple thresholds on the made-up behaviour
"""

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from base.grounded_log import build_from

BANDS = ["ดีมาก", "ดี", "ปานกลาง", "ควรปรับ", "ควรปรับมาก"]

FLAWS = [
    "no concrete example", "vague general claims", "too short", "rambling / goes off-topic",
    "no result or outcome", "weak or missing ending", "repeats the same point",
    "does not answer the question asked", "sounds memorised and generic",
    "negative about others", "weakness is not a real weakness", "no link to the job",
]

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answers": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "quality": {"type": "string", "enum": ["strong", "average", "weak"]},
            "flaws": {"type": "array", "items": {"type": "string"}},
            "persona": {"type": "string"},
            "sentences": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["quality", "flaws", "persona", "sentences"]}}},
    "required": ["answers"],
}


def answers_prompt(question_en: str, question_th: str, lang: str, plan: list, rng: random.Random) -> str:
    """plan: list of (quality, flaws) - one answer each."""
    want = "\n".join(f"{i + 1}. quality={q}; flaws: {', '.join(f) if f else 'none'}"
                     for i, (q, f) in enumerate(plan))
    if lang == "th":
        lang_rule = ("Write every answer in natural SPOKEN Thai, as a Thai student or new graduate "
                     "would say it in a practice interview (polite particles ครับ/ค่ะ are fine).")
        q = question_th
    else:
        lang_rule = ("Write every answer in spoken English as a non-native speaker (a Thai student "
                     "or new graduate) would say it: simple, sometimes slightly ungrammatical.")
        q = question_en
    noise = ("In about one answer in three, add 1-2 speech-recognition mistakes (a wrong word that "
             "sounds similar), because real transcripts come from automatic speech recognition.")
    return f"""Write {len(plan)} different made-up answers to this job-interview practice question.
They are training data for an interview coach, so they must differ in quality and problems.

QUESTION: "{q}"

ANSWERS TO WRITE (one per line)
{want}

RULES
• {lang_rule}
• Each answer is what the candidate SAYS (a transcript), split into sentences.
  strong: 6-12 sentences; average: 4-10; weak: 2-9 ("too short" means 2-3 sentences).
• Make the flaws clearly visible in the words. A strong answer has a concrete example with a result.
• Use different made-up people: field of study / job, situation, details. Do not use real names
  of people or companies. No filler words (um, uh).
• {noise}
• persona: one short line, e.g. "4th-year engineering student applying for a data intern role".
"""


def answer_plan(n: int, rng: random.Random) -> list:
    out = []
    for _ in range(n):
        q = rng.choices(["strong", "average", "weak"], weights=[3, 4, 3])[0]
        k = {"strong": rng.choice([0, 0, 1]), "average": rng.choice([1, 2]), "weak": rng.choice([2, 3])}[q]
        out.append((q, rng.sample(FLAWS, k)))
    return out


# ------------------------------------------------------------
# made-up recording
# ------------------------------------------------------------

def _words(sentence: str, lang: str):
    if lang == "th":
        try:
            from pythainlp.tokenize import word_tokenize
            toks = word_tokenize(sentence, engine="newmm", keep_whitespace=False)
        except ImportError:
            toks = [sentence[i:i + 4] for i in range(0, len(sentence), 4)]
    else:
        toks = sentence.split()
    return [t for t in toks if t.strip() and any(c.isalnum() for c in t)] or [sentence]


def _alternate(duration: float, main: str, other: list, share_main: float, rng: random.Random,
               mean_main=6.0, flicker=0.25):
    """Segments alternating between `main` and values from `other`, so that `main` covers about
    share_main of the time; also short flickers like a real detector produces."""
    segs, t = [], 0.0
    mean_other = max(0.4, mean_main * (1 - share_main) / max(share_main, 0.05))
    cur_main = rng.random() < share_main
    while t < duration:
        if cur_main:
            d, v = rng.expovariate(1 / mean_main), main
        else:
            d, v = rng.expovariate(1 / mean_other), rng.choice(other)
        d = min(max(d, 0.1), duration - t)
        segs.append({"start": round(t, 2), "end": round(t + d, 2), "value": v})
        t += d
        if rng.random() < flicker and t < duration:          # detector flicker
            f = min(rng.uniform(0.1, 0.6), duration - t)
            segs.append({"start": round(t, 2), "end": round(t + f, 2),
                         "value": rng.choice(other) if cur_main else main})
            t += f
        cur_main = not cur_main
    return segs


def _band(x, cuts):
    """x high = good. cuts: 4 thresholds for ดีมาก/ดี/ปานกลาง/ควรปรับ."""
    for b, c in zip(BANDS, cuts):
        if x >= c:
            return b
    return BANDS[-1]


def make_clip(answer: dict, lang: str, rng: random.Random, key: str) -> dict:
    """Returns {"log", "ratings", "profile"} for one made-up recording of `answer`."""
    sents = [s.strip() for s in answer["sentences"] if s.strip()]
    # --- speech timing
    rate = rng.gauss(140, 25) if lang == "en" else rng.gauss(200, 40)     # words / min
    rate = min(max(rate, 80 if lang == "en" else 110), 200 if lang == "en" else 290)
    p_long = rng.choice([0.0, 0.05, 0.15, 0.3])                           # chance of a long pause
    t = rng.choices([rng.uniform(0.3, 1.0), rng.uniform(1.0, 3.0), rng.uniform(3.0, 6.0)],
                    weights=[7, 2, 1])[0]
    segments = []
    for s in sents:
        toks = _words(s, lang)
        dur = len(toks) / rate * 60 * rng.uniform(0.85, 1.15)
        step = dur / len(toks)
        segments.append({"start": round(t, 2), "end": round(t + dur, 2), "text": s,
                         "words": [{"word": w, "start": round(t + i * step, 2),
                                    "end": round(t + (i + 1) * step, 2)} for i, w in enumerate(toks)]})
        t += dur
        t += rng.uniform(1.2, 8.0) if rng.random() < p_long else rng.uniform(0.15, 0.8)
    duration = round(segments[-1]["end"] + rng.uniform(0.5, 2.5), 2) if segments else 10.0

    # --- visual behaviour (wide random profiles)
    eye = rng.betavariate(4, 1.6)                                    # mostly looking at camera
    head_c = rng.betavariate(2.5, 1.5)
    head_off = rng.choices(["turned_down", "turned_left", "turned_right", "turned_up"],
                           weights=[6, 2, 2, 1])[0]
    hand_mode = rng.choices(["hands_hidden", "hands_still", "gesturing"], weights=[4, 3, 3])[0]
    hand_main = rng.uniform(0.5, 1.0)
    smile = rng.betavariate(1.6, 2.0)
    raw = []
    measured = {t_: rng.random() > 0.04 for t_ in ("gaze", "head", "hand", "face")}  # rare: no detection
    if measured["gaze"]:
        raw += [{"type": "gaze", **s} for s in _alternate(duration, "eye_contact", ["looking_away"], eye, rng)]
    if measured["head"]:
        others = [head_off] * 4 + [v for v in ("turned_left", "turned_right", "turned_down") if v != head_off]
        raw += [{"type": "head", **s} for s in _alternate(duration, "centered", others, head_c, rng, mean_main=5)]
    if measured["hand"]:
        rest = [v for v in ("hands_hidden", "hands_still", "gesturing") if v != hand_mode]
        raw += [{"type": "hand", **s} for s in _alternate(duration, hand_mode, rest, hand_main, rng, mean_main=8)]
    if measured["face"]:
        raw += [{"type": "face", **s} for s in _alternate(duration, "smiling", ["neutral"], smile, rng)]
    for e in raw:
        e["category"] = "visual"

    log = build_from(key, raw, {"segments": segments}, {"duration_sec": duration})

    # --- ratings from the shares actually produced
    share = {i["type"]: i.get("share") or {} for i in log["items"] if i["kind"] == "summary"}
    ratings = {}
    if share.get("gaze"):
        ratings["eye_contact"] = _band(share["gaze"].get("eye_contact", 0), (0.85, 0.70, 0.50, 0.30))
    if share.get("head"):
        ratings["head_pose"] = _band(share["head"].get("centered", 0), (0.80, 0.60, 0.40, 0.20))
    if share.get("hand"):
        g, h = share["hand"].get("gesturing", 0), share["hand"].get("hands_hidden", 0)
        ratings["hand_gesture"] = ("ควรปรับ" if g > 0.75 else "ดี" if 0.2 <= g <= 0.6
                                   else "ปานกลาง" if h >= 0.6 or g < 0.2 else "ดี")
    if share.get("face"):
        ratings["facial_expression"] = _band(share["face"].get("smiling", 0), (0.60, 0.40, 0.25, 0.10))
    spoken = (segments[-1]["end"] - segments[0]["start"]) if segments else 0
    q = {"strong": 0.5, "average": 2.0, "weak": 3.4}[answer["quality"]]
    q += 0.6 if spoken < 30 else 0
    q += 0.4 * sum(1 for a, b in zip(segments, segments[1:]) if b["start"] - a["end"] >= 3.0)
    ratings["answer_quality"] = BANDS[min(4, int(round(q + rng.uniform(-0.4, 0.4))))]
    w = {"eye_contact": .2, "head_pose": .15, "hand_gesture": .1, "facial_expression": .15,
         "answer_quality": .4}
    tot = sum(w[d] for d in ratings)
    ratings["overall"] = BANDS[min(4, int(round(sum(w[d] * BANDS.index(r) for d, r in ratings.items()) / tot)))]
    profile = {"rate": round(rate), "p_long_pause": p_long, "eye": round(eye, 2),
               "head_centered": round(head_c, 2), "head_off": head_off, "hand_mode": hand_mode,
               "smile": round(smile, 2), "measured": measured}
    return {"log": log, "ratings": ratings, "profile": profile}
