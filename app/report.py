"""
app/report.py — turns one result (pipeline.load_result) into the HTML report page.
"""

import html
import json

DIM = {   # dimension -> (icon, Thai name)
    "eye_contact": ("👀", "การสบตา"),
    "head_pose": ("🙆", "ท่าทางศีรษะ"),
    "hand_gesture": ("👐", "การใช้มือ"),
    "facial_expression": ("😊", "สีหน้า"),
    "answer_quality": ("💬", "เนื้อหาและการพูด"),
}
PRIORITY = {"high": ("สำคัญมาก", "#d9480f"), "medium": ("ควรแก้", "#e8a317"),
            "low": ("เล็กน้อย", "#2f9e44")}
# timeline rows: evidence type -> (label, colour, which values count as "something to notice")
ROWS = [
    ("gaze", "สายตา", "#e8590c", {"looking_away"}),
    ("head", "ศีรษะ", "#7048e8", {"turned_up", "turned_down", "turned_left", "turned_right"}),
    ("hand", "มือ", "#1c7ed6", {"gesturing"}),
    ("face", "รอยยิ้ม", "#2f9e44", {"smiling"}),
    ("pause", "หยุดพูด", "#868e96", {"pause", "silence_before_answer"}),
]
VALUE_TH = {
    "looking_away": "ไม่ได้มองกล้อง", "turned_up": "เงยหน้า", "turned_down": "ก้มหน้า",
    "turned_left": "หันซ้าย", "turned_right": "หันขวา", "gesturing": "ใช้มือประกอบ",
    "smiling": "ยิ้ม", "pause": "หยุดพูด", "silence_before_answer": "เงียบก่อนเริ่มตอบ",
}

CSS = """
<style>
.ic{font-family:'Prompt','Sarabun',system-ui,sans-serif;color:#212529;max-width:860px;margin:0 auto}
.ic *{box-sizing:border-box}
.ic .hero{display:flex;gap:20px;align-items:center;background:linear-gradient(135deg,#fff4e6,#e7f5ff);
  border-radius:20px;padding:22px 24px;margin-bottom:18px}
.ic .hero h2{margin:0 0 6px;font-size:22px}
.ic .hero p{margin:0;line-height:1.6;font-size:15px}
.ic .chips{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0 4px}
.ic .chip{background:#fff;border-radius:14px;padding:8px 14px;box-shadow:0 1px 3px rgba(0,0,0,.08);font-size:14px}
.ic .chip b{font-size:18px;display:block}
.ic h3{font-size:18px;margin:26px 0 10px}
.ic .card{background:#fff;border:1px solid #e9ecef;border-radius:16px;padding:16px 18px;margin-bottom:12px;
  box-shadow:0 1px 2px rgba(0,0,0,.04)}
.ic .card .top{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.ic .icon{font-size:28px;line-height:1}
.ic .tag{font-size:12px;color:#fff;border-radius:999px;padding:2px 10px}
.ic .time{display:inline-block;background:#f1f3f5;border-radius:8px;padding:1px 8px;margin:2px 4px 2px 0;
  font-size:13px;font-variant-numeric:tabular-nums}
.ic ul{margin:6px 0 0 0;padding-left:20px;line-height:1.6}
.ic .good{border-left:5px solid #2f9e44}
.ic .fix{border-left:5px solid #e8590c}
.ic .quote{background:#f8f9fa;border-radius:12px;padding:12px 14px;line-height:1.6;margin:6px 0}
.ic .muted{color:#868e96;font-size:13px}
.ic details{margin-top:8px}
.ic summary{cursor:pointer;color:#1c7ed6}
.ic .tl{width:100%;height:auto;display:block}
</style>
"""


def _t(x) -> str:
    x = float(x)
    return f"{int(x // 60)}:{int(x % 60):02d}"


def _esc(s) -> str:
    return html.escape(str(s or ""))


def _times(ids, idx) -> str:
    out = []
    for i in ids or []:
        it = idx.get(i)
        if it and it.get("kind") != "summary":
            out.append(f'<span class="time">⏱ {_t(it["start"])}–{_t(it["end"])}</span>')
    return "".join(out)


def mascot(mood: str = "happy") -> str:
    """Original little coach character (inline SVG)."""
    mouth = ("M38 62 Q50 72 62 62" if mood == "happy" else "M38 66 Q50 62 62 66")
    return f"""<svg width="96" height="96" viewBox="0 0 100 100" aria-hidden="true">
<circle cx="50" cy="52" r="40" fill="#ffd8a8"/>
<path d="M14 44 Q50 2 86 44 Q70 30 50 30 Q30 30 14 44Z" fill="#495057"/>
<circle cx="37" cy="50" r="5" fill="#212529"/><circle cx="63" cy="50" r="5" fill="#212529"/>
<circle cx="39" cy="48" r="1.6" fill="#fff"/><circle cx="65" cy="48" r="1.6" fill="#fff"/>
<circle cx="28" cy="60" r="5" fill="#ffa8a8" opacity=".7"/><circle cx="72" cy="60" r="5" fill="#ffa8a8" opacity=".7"/>
<path d="{mouth}" stroke="#212529" stroke-width="3" fill="none" stroke-linecap="round"/>
<rect x="30" y="86" width="40" height="10" rx="5" fill="#1c7ed6"/></svg>"""


def timeline(log: dict, marks) -> str:
    dur = max(float(log.get("duration") or 0), 1.0)
    x0, w, rh = 90, 640, 30
    h = 40 + rh * len(ROWS) + 20
    X = lambda t: x0 + min(max(float(t), 0), dur) / dur * w
    parts = [f'<svg class="tl" viewBox="0 0 760 {h}" role="img" aria-label="ไทม์ไลน์พฤติกรรม">']
    step = 10 if dur <= 120 else 30
    t = 0
    while t <= dur + 0.01:
        parts.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="28" y2="{h - 18}" stroke="#e9ecef"/>'
                     f'<text x="{X(t):.1f}" y="20" font-size="11" text-anchor="middle" fill="#868e96">{_t(t)}</text>')
        t += step
    events = [i for i in log.get("items", []) if i.get("kind") == "event"]
    for r, (typ, label, colour, show) in enumerate(ROWS):
        y = 40 + r * rh
        parts.append(f'<text x="8" y="{y + 14}" font-size="13" fill="#495057">{label}</text>'
                     f'<rect x="{x0}" y="{y + 4}" width="{w}" height="14" rx="7" fill="#f8f9fa"/>')
        for e in events:
            if e["type"] == typ and e.get("value") in show:
                x1, x2 = X(e["start"]), X(e["end"])
                parts.append(f'<rect x="{x1:.1f}" y="{y + 4}" width="{max(x2 - x1, 3):.1f}" height="14" '
                             f'rx="7" fill="{colour}" opacity=".8"><title>{_t(e["start"])}–{_t(e["end"])} '
                             f'{VALUE_TH.get(e.get("value"), e.get("value"))}</title></rect>')
    for (t0, t1, text) in marks:
        parts.append(f'<line x1="{X(t0):.1f}" x2="{X(t0):.1f}" y1="30" y2="{h - 18}" stroke="#e8590c" '
                     f'stroke-width="2" stroke-dasharray="4 3"><title>{_esc(text)}</title></line>')
    parts.append("</svg>")
    return "".join(parts)


def render(res: dict) -> str:
    fb = res.get("feedback") or {}
    if not fb:
        return CSS + '<div class="ic"><p>ยังไม่มีผล</p></div>'
    log = fb.get("evidence_log", {})
    idx = {i["id"]: i for i in log.get("items", [])}
    sp = (res.get("speech") or {}).get("features", {})
    vis = res.get("visual") or {}
    eye = (vis.get("gaze") or {}).get("eye_contact_ratio")
    improvements = fb.get("improvements") or []
    strengths = fb.get("strengths") or []
    mood = "happy" if len(strengths) >= len(improvements) else "thinking"
    meta = res.get("meta") or {}
    thai = meta.get("language") == "th"
    q = meta.get("question") or {}
    qline = (("ตำแหน่ง: " + meta["job"] + " · ") if meta.get("job") else "") + \
        (("คำถาม: " + (q.get("th") if thai else q.get("en"))) if q else "")

    chips = []
    if sp.get("duration_sec"):
        chips.append(f'<div class="chip">⏳ ความยาว<b>{_t(sp["duration_sec"])} นาที</b></div>')
    if sp.get("speech_rate_wpm"):
        chips.append(f'<div class="chip">🗣️ ความเร็วพูด<b>{sp["speech_rate_wpm"]:.0f} คำ/นาที</b></div>')
    if eye is not None:
        chips.append(f'<div class="chip">👀 มองกล้อง<b>{eye * 100:.0f}% ของเวลา</b></div>')
    if sp.get("long_pause_count") is not None:
        chips.append(f'<div class="chip">⏸️ หยุดนาน (≥2 วิ)<b>{int(sp["long_pause_count"])} ครั้ง</b></div>')

    out = [CSS, '<div class="ic">',
           f'<div class="hero">{mascot(mood)}<div><h2>ผลการฝึกสัมภาษณ์</h2>'
           + (f'<p class="muted">{_esc(qline)}</p>' if qline else "") +
           f'<p>{_esc(fb.get("overall_summary_th") or fb.get("overall_summary_en"))}</p>'
           f'<div class="chips">{"".join(chips)}</div></div></div>']

    marks = []
    if improvements:
        out.append("<h3>🎯 เริ่มแก้ตรงนี้ก่อน</h3>")
        for it in improvements:
            icon, name = DIM.get(it.get("dimension"), ("📌", it.get("dimension")))
            plabel, pcol = PRIORITY.get(it.get("priority"), ("ควรแก้", "#e8a317"))
            steps = it.get("action_steps_th") or it.get("action_steps_en") or []
            out.append(f'<div class="card fix"><div class="top"><span class="icon">{icon}</span>'
                       f'<b>{_esc(name)}</b><span class="tag" style="background:{pcol}">{plabel}</span></div>'
                       f'<div>{_esc(it.get("issue_th") or it.get("issue_en"))}</div>'
                       f'<div>{_times(it.get("evidence_ids"), idx)}</div>'
                       + (f'<ul>{"".join(f"<li>{_esc(s)}</li>" for s in steps)}</ul>' if steps else "")
                       + '</div>')
            for i in it.get("evidence_ids") or []:
                e = idx.get(i)
                if e and e.get("kind") != "summary":
                    marks.append((e["start"], e["end"], it.get("issue_th") or it.get("issue_en")))

    if strengths:
        out.append("<h3>🌟 จุดเด่นของคุณ</h3>")
        for it in strengths:
            icon, name = DIM.get(it.get("dimension"), ("⭐", it.get("dimension")))
            out.append(f'<div class="card good"><div class="top"><span class="icon">{icon}</span>'
                       f'<b>{_esc(name)}</b></div><div>{_esc(it.get("text_th") or it.get("text_en"))}</div>'
                       f'<div>{_times(it.get("evidence_ids"), idx)}</div></div>')

    if log.get("items"):
        out.append('<h3>🕒 ไทม์ไลน์ระหว่างตอบ</h3><div class="card">' + timeline(log, marks) +
                   '<div class="muted">แถบสี = ช่วงเวลาที่ระบบวัดได้ · เส้นประสีส้ม = จุดที่โค้ชแนะนำให้ปรับ '
                   '(ชี้เมาส์ดูรายละเอียด)</div></div>')

    ia = fb.get("improved_answer") or {}
    if ia.get("rewritten_th") or ia.get("rewritten_en"):
        out.append('<h3>✍️ ลองเปิดคำตอบแบบนี้</h3><div class="card">'
                   + (f'<div class="muted">เดิม</div><div class="quote">{_esc(ia.get("original_snippet"))}</div>'
                      if ia.get("original_snippet") else "")
                   + '<div class="muted">ปรับใหม่</div>'
                   + "".join(f'<div class="quote">{_esc(ia.get(f))}</div>' for f in
                             (("rewritten_th", "rewritten_en") if thai else ("rewritten_en", "rewritten_th"))
                             if ia.get(f))
                   + '</div>')

    segs = (res.get("speech") or {}).get("segments") or []
    if segs:
        rows = "".join(f'<div><span class="time">{_t(s["start"])}</span>{_esc(s["text"])}</div>' for s in segs)
        out.append(f'<details><summary>ดูข้อความที่ถอดจากเสียง</summary><div class="card">{rows}</div></details>')

    ver = fb.get("verification") or {}
    removed = len(ver.get("removed") or [])
    who = {"gemini": "เขียนโดย Gemini", "local": "เขียนโดยโค้ชในเครื่อง (แม่แบบจากค่าที่วัดได้)"}.get(
        fb.get("coach_used"), "")
    if fb.get("fallback_reason"):
        who += " เพราะ Gemini ใช้งานไม่ได้ตอนนี้"
    out.append(f'<p class="muted">{who + " · " if who else ""}คำแนะนำทุกข้อถูกตรวจกับหลักฐานที่วัดได้ก่อนแสดง'
               f'{f" (ตัดข้อที่ไม่มีหลักฐานออก {removed} ข้อ)" if removed else ""} · '
               f'ระบบยังนับคำเติม (um, uh) ไม่ได้ · ใช้เพื่อฝึกซ้อมเท่านั้น ไม่ใช่การประเมินตัวบุคคล</p>')
    out.append("</div>")
    return "".join(out)
