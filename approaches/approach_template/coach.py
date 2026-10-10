"""
approaches/approach_template/coach.py

Approach T: rule-based "template coach" - no LLM.

Reads the same evidence log as the grounded coach (C) and writes feedback in the same
JSON format (FEEDBACK_SCHEMA), by filling sentence templates with measured numbers:

  1. judge      dimension bands from Approach 1 (same as C in the app)
  2. pick       strengths = dimensions rated good, improvements = rated fair / needs work
                (worst first; long pauses are always flagged)
  3. write      each point = a template sentence + the IDs it was filled from
  4. checker    the same verify.py as C (should find nothing; any failure is dropped)

Because every number comes from the log, it cannot invent facts. Its limits: fixed wording,
and it cannot judge WHAT was said (content) - only length, pace and pauses.

Used as (a) the offline fallback of the web app when Gemini is down and (b) the baseline
that a trained small model and Gemini are compared with.
"""

import re

TEMPLATE_VERSION = "template-v1"

GOOD = ("ดีมาก", "ดี")
SEVERITY = {"ควรปรับมาก": 0, "ควรปรับ": 1, "ปานกลาง": 2}
BAND_EN = {"ดีมาก": "Excellent", "ดี": "Good", "ปานกลาง": "Fair", "ควรปรับ": "Needs Work",
           "ควรปรับมาก": "Priority"}
DIM_EN = {"eye_contact": "eye contact", "head_pose": "head position", "hand_gesture": "hand use",
          "facial_expression": "facial expression", "answer_quality": "the spoken answer"}
DIM_TH = {"eye_contact": "การสบตากล้อง", "head_pose": "ท่าทางศีรษะ", "hand_gesture": "การใช้มือ",
          "facial_expression": "สีหน้า", "answer_quality": "การพูดตอบ"}
ORDER = ["answer_quality", "eye_contact", "head_pose", "facial_expression", "hand_gesture"]

SHARE_MIN = 0.30           # only state "X% of the time" when X is a real share of the clip
LONG_PAUSE = 3.0           # a pause this long is worth mentioning
SLOW_START = 3.0           # silence before the first word
PACE_EN = (100, 170)       # comfortable English pace (words/min while speaking)
LEN_SEC = (40, 150)        # comfortable answer length (seconds of speech)


# ------------------------------------------------------------
# reading the log
# ------------------------------------------------------------

class Log:
    def __init__(self, log: dict):
        self.items = log.get("items", [])
        self.summary = {i["type"]: i for i in self.items if i["kind"] == "summary"}
        self.events = [i for i in self.items if i["kind"] == "event"]
        self.transcript = [i for i in self.items if i["kind"] == "transcript"]

    def share(self, t, v):
        return (self.summary.get(t, {}).get("share") or {}).get(v, 0.0)

    def longest(self, t, values):
        ev = [e for e in self.events if e["type"] == t and e.get("value") in values]
        return max(ev, key=lambda e: e["end"] - e["start"]) if ev else None

    def speech(self):
        s = self.summary.get("speech")
        if not s:
            return None
        m = re.search(r"(\d+) words in .*?about (\d+) words/min.*?first word at ([\d.]+)s, "
                      r"last word ends ([\d.]+)s", s["text"])
        if not m:
            return None
        n, wpm, a, b = int(m[1]), int(m[2]), float(m[3]), float(m[4])
        return {"id": s["id"], "words": n, "wpm": wpm, "spoken": b - a}


def _span(e):
    return f"({e['start']:.0f}-{e['end']:.0f}s)", f"{e['start']:.0f}-{e['end']:.0f} วินาที"


def _pct(x):
    return f"{x:.0%}"


# ------------------------------------------------------------
# one function per dimension: returns (strength or None, improvement or None)
# ------------------------------------------------------------

def gaze(L: Log, good: bool):
    s = L.summary.get("gaze")
    if not s or not s.get("share"):
        return None, None
    ec, away = L.share("gaze", "eye_contact"), L.share("gaze", "looking_away")
    if good:
        if ec < SHARE_MIN:
            return None, None
        return {"text_en": f"Steady eye contact: you looked at the camera {_pct(ec)} of the measured time.",
                "text_th": f"สบตากล้องได้ดี มองกล้อง {_pct(ec)} ของเวลาที่วัดได้",
                "evidence_ids": [s["id"]]}, None
    ids, en, th = [], [], []
    if away >= SHARE_MIN:
        ids.append(s["id"])
        en.append(f"Looking away from the camera took {_pct(away)} of the measured time.")
        th.append(f"มองออกนอกกล้อง {_pct(away)} ของเวลาที่วัดได้")
    e = L.longest("gaze", {"looking_away"})
    if e:
        ids.append(e["id"])
        a, b = _span(e)
        en.append(f"The longest look-away was {a}.")
        th.append(f"ช่วงที่มองออกนานที่สุดคือ {b}")
    if not ids:
        return None, None
    return None, {"issue_en": " ".join(en), "issue_th": " ".join(th), "evidence_ids": ids,
                  "action_steps_en": ["Put a small sticker beside the recording light and say your key sentences to it.",
                                      "If you need to think, pause first, then come back to the note before you speak again."],
                  "action_steps_th": ["แปะกระดาษโน้ตเล็ก ๆ ข้างเลนส์กล้อง แล้วพูดประโยคสำคัญกับมัน",
                                      "ถ้าต้องหยุดคิด ให้หยุดก่อน แล้วกลับมามองกล้องก่อนเริ่มพูดต่อ"]}


HEAD_FIX = {
    "turned_down": (["Raise your laptop or phone so the webcam is at eye level.",
                     "Keep notes off the desk; if you need them, tape them beside the webcam."],
                    ["ยกโน้ตบุ๊กหรือมือถือให้กล้องอยู่ระดับสายตา",
                     "อย่าวางโน้ตไว้บนโต๊ะ ถ้าจำเป็นให้แปะไว้ข้างกล้อง"]),
    "turned_up": (["Lower the webcam to eye level so you face it straight.",
                   "When you think, try to keep your chin level."],
                  ["ปรับกล้องลงมาให้อยู่ระดับสายตา จะได้หันหน้าตรง",
                   "เวลาคิด พยายามให้คางอยู่ระดับเดิม ไม่เงยขึ้น"]),
    "turned_left": (["Place the webcam straight in front of you, not off to one side.",
                     "Sit centred in the frame with your shoulders facing the screen."],
                    ["วางกล้องไว้ตรงหน้า ไม่ใช่ด้านข้าง",
                     "นั่งให้อยู่กลางเฟรม หันไหล่เข้าหาจอ"]),
}
HEAD_FIX["turned_right"] = HEAD_FIX["turned_left"]
HEAD_TEXT = {"turned_down": ("turned down", "ก้มลง"), "turned_up": ("turned up", "เงยขึ้น"),
             "turned_left": ("turned left", "หันซ้าย"), "turned_right": ("turned right", "หันขวา")}


def head(L: Log, good: bool):
    s = L.summary.get("head")
    if not s or not s.get("share"):
        return None, None
    c = L.share("head", "centered")
    if good:
        if c < SHARE_MIN:
            return None, None
        return {"text_en": f"Your head stayed straight {_pct(c)} of the measured time, which looks confident.",
                "text_th": f"ศีรษะตั้งตรง {_pct(c)} ของเวลาที่วัดได้ ดูมั่นใจ",
                "evidence_ids": [s["id"]]}, None
    others = {v: p for v, p in s["share"].items() if v in HEAD_TEXT}
    if not others:
        return None, None
    v = max(others, key=others.get)
    en_v, th_v = HEAD_TEXT[v]
    ids, en, th = [], [], []
    if others[v] >= SHARE_MIN:
        ids.append(s["id"])
        en.append(f"Your head was {en_v} {_pct(others[v])} of the measured time.")
        th.append(f"ศีรษะ{th_v} {_pct(others[v])} ของเวลาที่วัดได้")
    e = L.longest("head", {v})
    if e:
        ids.append(e["id"])
        a, b = _span(e)
        en.append(f"The longest stretch was {a}.")
        th.append(f"ช่วงที่นานที่สุดคือ {b}")
    if not ids:
        return None, None
    steps = HEAD_FIX[v]
    return None, {"issue_en": " ".join(en), "issue_th": " ".join(th), "evidence_ids": ids,
                  "action_steps_en": steps[0], "action_steps_th": steps[1]}


def hand(L: Log, good: bool):
    s = L.summary.get("hand")
    if not s or not s.get("share"):
        return None, None
    g, still, hid = (L.share("hand", v) for v in ("gesturing", "hands_still", "hands_hidden"))
    if good:
        if g >= SHARE_MIN:
            return {"text_en": f"You used hand gestures {_pct(g)} of the measured time, which supports what you say.",
                    "text_th": f"ใช้มือประกอบการพูด {_pct(g)} ของเวลาที่วัดได้ ช่วยให้คำพูดหนักแน่นขึ้น",
                    "evidence_ids": [s["id"]]}, None
        if still >= SHARE_MIN:
            return {"text_en": f"Your hands stayed visible and still {_pct(still)} of the measured time, which looks calm.",
                    "text_th": f"มือนิ่งและอยู่ในเฟรม {_pct(still)} ของเวลาที่วัดได้ ดูสงบ",
                    "evidence_ids": [s["id"]]}, None
        return None, None
    if hid >= 0.5:
        return None, {"issue_en": f"Your hands were not visible {_pct(hid)} of the measured time.",
                      "issue_th": f"มือไม่อยู่ในเฟรม {_pct(hid)} ของเวลาที่วัดได้",
                      "evidence_ids": [s["id"]],
                      "action_steps_en": ["Sit a little further back so your hands are in the frame.",
                                          "Use your hands to count off your main ideas (first, second, third)."],
                      "action_steps_th": ["นั่งถอยหลังเล็กน้อยให้มือเข้ามาอยู่ในเฟรม",
                                          "ใช้นิ้วนับประเด็นหลักตอนพูด (หนึ่ง สอง สาม)"]}
    if g >= 0.6:
        return None, {"issue_en": f"Your hands were moving {_pct(g)} of the measured time, which can look restless.",
                      "issue_th": f"มือขยับ {_pct(g)} ของเวลาที่วัดได้ อาจดูไม่นิ่ง",
                      "evidence_ids": [s["id"]],
                      "action_steps_en": ["Put your hands on the desk between sentences.",
                                          "Save hand use for your main ideas only."],
                      "action_steps_th": ["วางมือบนโต๊ะระหว่างประโยค",
                                          "ใช้มือเฉพาะตอนพูดประเด็นสำคัญ"]}
    return None, None


def face(L: Log, good: bool):
    s = L.summary.get("face")
    if not s or not s.get("share"):
        return None, None
    sm, ne = L.share("face", "smiling"), L.share("face", "neutral")
    if good and sm >= SHARE_MIN:
        return {"text_en": f"You smiled {_pct(sm)} of the measured time, which looks warm and friendly.",
                "text_th": f"ยิ้ม {_pct(sm)} ของเวลาที่วัดได้ ดูเป็นมิตร",
                "evidence_ids": [s["id"]]}, None
    if not good and ne >= SHARE_MIN:
        return None, {"issue_en": f"Your face was neutral {_pct(ne)} of the measured time, which can look tense.",
                      "issue_th": f"หน้านิ่ง {_pct(ne)} ของเวลาที่วัดได้ อาจดูเกร็ง",
                      "evidence_ids": [s["id"]],
                      "action_steps_en": ["Begin and end the answer with a small, natural friendly expression.",
                                          "Before recording, think of one thing you enjoy about this job."],
                      "action_steps_th": ["เริ่มและจบคำตอบด้วยสีหน้าเป็นมิตรเล็กน้อย",
                                          "ก่อนอัด ลองนึกถึงสิ่งที่ชอบเกี่ยวกับงานนี้สักอย่าง"]}
    return None, None


def answer(L: Log, good: bool, lang: str):
    """Spoken answer: length, pace, pauses. Returns (strengths list, improvements list)."""
    sp = L.speech()
    if not sp:
        return [], []
    sid = sp["id"]
    pause_sum = L.summary.get("pause")
    pauses = [e for e in L.events if e["type"] == "pause"]
    start = next((p for p in pauses if p.get("value") == "silence_before_answer"), None)
    mid = [p for p in pauses if p.get("value") == "pause" and p["end"] - p["start"] >= LONG_PAUSE]
    strengths, issues = [], []

    # length
    if sp["spoken"] < LEN_SEC[0]:
        issues.append((1, {"issue_en": f"The answer was short: about {sp['spoken']:.0f} seconds of speech.",
                           "issue_th": f"คำตอบสั้นไป พูดประมาณ {sp['spoken']:.0f} วินาที",
                           "evidence_ids": [sid],
                           "action_steps_en": ["Add one real example: situation, what you did, and the result.",
                                               "Aim for about 60-90 seconds."],
                           "action_steps_th": ["เพิ่มตัวอย่างจริงหนึ่งเรื่อง: สถานการณ์ สิ่งที่ทำ และผลลัพธ์",
                                               "ตั้งเป้าพูดประมาณ 60-90 วินาที"]}))
    elif sp["spoken"] > LEN_SEC[1]:
        issues.append((2, {"issue_en": f"The answer was long: about {sp['spoken']:.0f} seconds of speech.",
                           "issue_th": f"คำตอบยาวไป พูดประมาณ {sp['spoken']:.0f} วินาที",
                           "evidence_ids": [sid],
                           "action_steps_en": ["Keep one main example and cut the rest.",
                                               "Finish with one sentence that links back to the job."],
                           "action_steps_th": ["เลือกตัวอย่างหลักเรื่องเดียว ตัดส่วนอื่นออก",
                                               "ปิดท้ายด้วยประโยคเดียวที่โยงกลับมาที่งาน"]}))
    else:
        strengths.append({"text_en": f"Good length: about {sp['spoken']:.0f} seconds of speech with {sp['words']} words.",
                          "text_th": f"ความยาวกำลังดี พูดประมาณ {sp['spoken']:.0f} วินาที",
                          "evidence_ids": [sid]})

    # pace (English only: Thai word counts depend on the tokeniser)
    if lang == "en":
        if sp["wpm"] > PACE_EN[1]:
            issues.append((2, {"issue_en": f"You spoke quickly, about {sp['wpm']} words per minute.",
                               "issue_th": f"พูดเร็วไป ประมาณ {sp['wpm']} คำต่อนาที",
                               "evidence_ids": [sid],
                               "action_steps_en": ["Breathe at every full stop.",
                                                   "Slow down on numbers, names and your key sentence."],
                               "action_steps_th": ["หายใจทุกครั้งที่จบประโยค",
                                                   "พูดช้าลงตอนพูดตัวเลข ชื่อ และประโยคสำคัญ"]}))
        elif sp["wpm"] < PACE_EN[0]:
            issues.append((2, {"issue_en": f"You spoke slowly, about {sp['wpm']} words per minute.",
                               "issue_th": f"พูดช้าไป ประมาณ {sp['wpm']} คำต่อนาที",
                               "evidence_ids": [sid],
                               "action_steps_en": ["Practise the answer aloud 2-3 times so the words come easily.",
                                                   "Prepare 3 keywords: beginning, middle and end."],
                               "action_steps_th": ["ซ้อมพูดออกเสียง 2-3 รอบให้คำพูดลื่นขึ้น",
                                                   "เตรียมคีย์เวิร์ด 3 คำ: ต้น กลาง จบ"]}))
        else:
            strengths.append({"text_en": f"Comfortable pace: about {sp['wpm']} words per minute.",
                              "text_th": f"ความเร็วพูดกำลังดี ประมาณ {sp['wpm']} คำต่อนาที",
                              "evidence_ids": [sid]})

    # pauses (always flagged when long, even if the answer was rated good)
    if mid:
        longest = max(mid, key=lambda p: p["end"] - p["start"])
        a, b = _span(longest)
        ids = ([pause_sum["id"]] if pause_sum else []) + [longest["id"]]
        en = (f"There {'was' if len(mid) == 1 else 'were'} {len(mid)} long pause"
              f"{'' if len(mid) == 1 else 's'} of {LONG_PAUSE:.0f}s or more. "
              f"The longest was {longest['end'] - longest['start']:.0f} seconds {a}.")
        th = (f"หยุดพูดนานเกิน {LONG_PAUSE:.0f} วินาที {len(mid)} ครั้ง "
              f"นานที่สุด {longest['end'] - longest['start']:.0f} วินาที ช่วง {b}")
        issues.append((1 if longest["end"] - longest["start"] >= 5 else 2,
                       {"issue_en": en, "issue_th": th, "evidence_ids": ids,
                        "action_steps_en": ["Plan 3 keywords before you start, so you know what comes next.",
                                            "If you need time, say a short bridge sentence such as 'Let me give you an example'."],
                        "action_steps_th": ["วางคีย์เวิร์ด 3 คำก่อนเริ่มพูด จะได้รู้ว่าต่อไปพูดอะไร",
                                            "ถ้าต้องการเวลาคิด ใช้ประโยคเชื่อมสั้น ๆ เช่น ขอยกตัวอย่างนะครับ แทนการเงียบ"]}))
    elif sp["spoken"] >= LEN_SEC[0] and pause_sum:
        strengths.append({"text_en": f"Fluent delivery: no pause of {LONG_PAUSE:.0f}s or more in the middle of the answer.",
                          "text_th": f"พูดได้ต่อเนื่อง ไม่มีช่วงหยุดนานเกิน {LONG_PAUSE:.0f} วินาทีระหว่างคำตอบ",
                          "evidence_ids": [pause_sum["id"]]})
    if start and start["end"] - start["start"] >= SLOW_START:
        a, b = _span(start)
        issues.append((3, {"issue_en": f"You started speaking after {start['end']:.0f} seconds of silence {a}.",
                           "issue_th": f"เริ่มพูดหลังเงียบไป {start['end']:.0f} วินาที ช่วง {b}",
                           "evidence_ids": [start["id"]],
                           "action_steps_en": ["Prepare your first sentence so you can start within 1-2 seconds."],
                           "action_steps_th": ["เตรียมประโยคแรกไว้ให้เริ่มพูดได้ภายใน 1-2 วินาที"]}))
    return strengths, issues


# ------------------------------------------------------------
# improved opening (structure only; no facts are invented)
# ------------------------------------------------------------

OPENING = {
    "intro": ("[Your name], [what you study or do now]. I'm good at [skill 1] and [skill 2]; for example, "
              "[one real achievement and its result]. I'm applying because [link to this job].",
              "สวัสดีครับ/ค่ะ ผม/ดิฉันชื่อ [ชื่อ] ตอนนี้ [กำลังเรียนหรือทำงานอะไร] จุดเด่นคือ [ทักษะ 1] และ [ทักษะ 2] "
              "เช่น [ผลงานจริงหนึ่งอย่างและผลลัพธ์] ที่สนใจตำแหน่งนี้เพราะ [เชื่อมกับงาน]",
              "present -> skills with one real example -> why this job"),
    "strengths": ("My greatest strength is [strength]; for example, [real situation - what you did - result]. "
                  "One weakness I'm working on is [weakness], so I [what you do about it].",
                  "จุดแข็งที่สุดของผม/ดิฉันคือ [จุดแข็ง] เช่น [สถานการณ์จริง - สิ่งที่ทำ - ผลลัพธ์] "
                  "ส่วนจุดที่กำลังพัฒนาคือ [จุดอ่อน] ตอนนี้แก้โดย [สิ่งที่ทำอยู่]",
                  "strength -> evidence -> weakness -> what you do about it"),
    "friend": ("My best friend would say I'm [trait 1] and [trait 2]. For example, [a short real story they "
               "would tell]. At work, that means I [how it helps a team].",
               "เพื่อนสนิทน่าจะบอกว่าผม/ดิฉันเป็นคน [นิสัย 1] และ [นิสัย 2] เช่น [เรื่องจริงสั้น ๆ ที่เพื่อนจะเล่า] "
               "ในที่ทำงานข้อนี้ช่วยให้ [ประโยชน์ต่อทีม]",
               "2 traits -> a short real story -> link to work"),
    "role": ("The skills this role needs most are [skill 1] and [skill 2]. For [skill 1], I [real project or "
             "course - what you did - result]. For [skill 2], I [real example]. I'm still improving [skill], so I [plan].",
             "ทักษะที่ตำแหน่งนี้ต้องใช้มากที่สุดคือ [ทักษะ 1] และ [ทักษะ 2] สำหรับ [ทักษะ 1] ผม/ดิฉันเคย "
             "[โปรเจกต์หรือวิชาจริง - สิ่งที่ทำ - ผลลัพธ์] ส่วน [ทักษะ 2] เคย [ตัวอย่างจริง] และกำลังพัฒนา [ทักษะ] โดย [แผน]",
             "key skills -> real evidence for each -> what you are still learning"),
    "motivation": ("I'm interested in this role because [specific reason about this job or company]. In [real "
                   "experience], I found that I enjoy [task]. Here I'd like to contribute [what you bring] and learn [what].",
                   "ที่สนใจตำแหน่งนี้เพราะ [เหตุผลเฉพาะของงานหรือบริษัทนี้] ตอนที่ [ประสบการณ์จริง] "
                   "ผม/ดิฉันพบว่าชอบ [งานแบบนี้] จึงอยากมาช่วย [สิ่งที่ทำได้] และเรียนรู้ [สิ่งที่อยากเรียน]",
                   "specific reason -> link to your experience -> what you will contribute"),
    "story": ("One example is when [situation]. My task was [task], so I [what you did]. As a result, "
              "[result], and I learned [lesson].",
              "ตัวอย่างหนึ่งคือตอนที่ [สถานการณ์] หน้าที่ของผม/ดิฉันคือ [งานที่ต้องทำ] จึง [สิ่งที่ทำ] "
              "ผลคือ [ผลลัพธ์] และได้เรียนรู้ว่า [บทเรียน]",
              "STAR: situation -> task -> action -> result"),
}


def improved_answer(L: Log, focus: str) -> dict:
    en, th, structure = OPENING.get(focus, OPENING["intro"])
    first = L.transcript[0]["text"] if L.transcript else ""
    return {"original_snippet": first, "rewritten_en": en, "rewritten_th": th,
            "why_better_en": (f"It follows the structure of a strong answer ({structure}). "
                              "Fill the brackets with your own real facts.")}


# ------------------------------------------------------------
# main entry
# ------------------------------------------------------------

DIM_FN = {"eye_contact": gaze, "head_pose": head, "hand_gesture": hand, "facial_expression": face}


def write(log: dict, ratings: dict, focus: str = "intro", lang: str = "en") -> dict:
    """Evidence log + bands -> feedback in FEEDBACK_SCHEMA."""
    L = Log(log)
    strengths, issues = [], []           # issues: (sort key, item)
    for dim in ORDER:
        band = ratings.get(dim)
        good = band in GOOD
        sev = SEVERITY.get(band, 3)
        if dim == "answer_quality":
            st, iss = answer(L, good, lang)
            strengths += [{"dimension": dim, **s} for s in st[:1]]
            for k, it in iss:
                issues.append(((min(sev, k), 0), {"dimension": dim, **it}))
            continue
        st, it = DIM_FN[dim](L, good)
        if st:
            strengths.append({"dimension": dim, **st})
        if it and band in SEVERITY:
            issues.append(((sev, ORDER.index(dim)), {"dimension": dim, **it}))

    issues.sort(key=lambda x: x[0])
    improvements = []
    for (sev, _), it in issues[:3]:
        improvements.append({**it, "priority": "high" if sev <= 1 else "medium" if sev == 2 else "low"})
    if not strengths and L.speech():
        sp = L.speech()
        strengths.append({"dimension": "answer_quality",
                          "text_en": f"You gave a complete answer of {sp['words']} words.",
                          "text_th": "ตอบคำถามได้ครบจนจบ", "evidence_ids": [sp["id"]]})
    strengths = strengths[:3]

    overall = ratings.get("overall", "ปานกลาง")
    best = next((s["dimension"] for s in strengths), None)
    first = improvements[0]["dimension"] if improvements else None
    en = [f"Overall: {BAND_EN.get(overall, overall)}."]
    th = [f"ภาพรวม: {overall}"]
    if best:
        en.append(f"Your strongest area was {DIM_EN[best]}.")
        th.append(f"จุดที่ทำได้ดีที่สุดคือ{DIM_TH[best]}")
    if first:
        en.append(f"Work on {DIM_EN[first]} first.")
        th.append(f"เริ่มฝึกที่{DIM_TH[first]}ก่อน")
    return {"overall_summary_en": " ".join(en), "overall_summary_th": " ".join(th),
            "strengths": strengths, "improvements": improvements,
            "improved_answer": improved_answer(L, focus)}
