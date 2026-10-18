"""
app/app.py — Interview practice coach (local research prototype)

  pip install gradio
  python app/app.py            -> open http://127.0.0.1:7860

Record (webcam) or upload a 1-2 minute answer, press the button, read the report.
Runs on this computer only (not shared on the internet). For your own clips and people who
agreed to take part. The transcript and measurements (not the video) are sent to Gemini.
"""

import json
import shutil
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import gradio as gr

import pipeline
import question_gen
import questions
import report

INTRO = """
# 🎤 ฝึกสัมภาษณ์งานกับ AI โค้ช
1. ใส่ตำแหน่งงานและ resume แล้วให้ AI สร้างคำถามเฉพาะคุณ (ข้ามได้ จะใช้คำถามมาตรฐาน)
2. **สัมภาษณ์ต่อเนื่อง**: ตอบทีละข้อเหมือนสัมภาษณ์จริง กดหยุดเมื่อไหร่ก็ได้ แล้วดูผลรวม
   หรือ **ฝึกทีละข้อ**: เลือกคำถามเดียว อัด แล้ววิเคราะห์

💡 นั่งหน้าตรง แสงพอ มองกล้อง · ต้นแบบงานวิจัย ใช้ในเครื่องนี้เท่านั้น
"""

CONSENT = ("คลิปและ resume เป็นของฉัน/ได้รับความยินยอมแล้ว และยอมให้ส่ง resume ข้อความถอดเสียง "
           "และผลวัด (ไม่ใช่วิดีโอ) ให้ Gemini เพื่อสร้างคำถามและเขียนคำแนะนำ")
ERR = '<p style="color:#d9480f">{}</p>'

COACH_CHOICES = {"🔒 Typhoon ในเครื่อง (ไม่ส่งข้อมูลออก)": "ollama",
                 "🔒 โค้ชแม่แบบ (เร็ว ไม่ใช้ LLM)": "local",
                 "☁️ อัตโนมัติ: Gemini ถ้าล่มใช้โค้ชในเครื่อง": "auto",
                 "☁️ Gemini เท่านั้น": "gemini",
                 "☁️ Groq API": "groq"}

BANK = [questions.as_question(q[0]) for q in questions.QUESTIONS]


def past_results():
    return sorted((p.stem for p in pipeline.FEEDBACK_DIR.glob("app_*.json")), reverse=True)


def lang_code(label):
    return "th" if label == questions.LANGS["th"] else "en"


def labels(qlist, lang):
    return [f"{i + 1}. {q['th'] if lang == 'th' else q['en']}" for i, q in enumerate(qlist)]


def pick(qlist, label):
    try:
        return qlist[int(label.split(".")[0]) - 1]
    except Exception:
        return qlist[0]


def why_text(qlist, label):
    q = pick(qlist, label) if label else None
    return f"💬 {q['why']}" if q and q.get("why") else ""


def change_lang(label, qlist):
    ch = labels(qlist, lang_code(label))
    return gr.update(choices=ch, value=ch[0])


def load_resume(file, current):
    text = question_gen.read_resume(file)
    if file and not text:
        return current, ERR.format("อ่านไฟล์นี้ไม่ได้ (รองรับ .pdf .docx .txt) ลองคัดลอกข้อความมาวางแทน")
    return (text or current), ("✅ อ่าน resume แล้ว ตรวจ/แก้ข้อความได้ด้านล่าง" if text else "")


def make_questions(job, resume, lang_label, consent, n, progress=gr.Progress()):
    if (resume or "").strip() and not consent:
        return (BANK, gr.update(), ERR.format("ติ๊กยืนยันใน “ตั้งค่าเพิ่มเติม” ก่อน เพราะต้องส่ง resume ให้ Gemini"), "")
    progress(0.2, desc="AI กำลังอ่าน resume และสร้างคำถาม ...")
    qlist, note = question_gen.generate(resume, job, int(n))
    ch = labels(qlist, lang_code(lang_label))
    return qlist, gr.update(choices=ch, value=ch[0]), f"✨ {note}", why_text(qlist, ch[0])


def analyse(video, lang_label, qlist, label, consent, fast, coach_label, job, resume,
            progress=gr.Progress()):
    mode = COACH_CHOICES.get(coach_label, "auto")
    if not consent and mode not in ("local", "ollama"):
        return ERR.format("โค้ช ☁️ ต้องติ๊กยืนยันใน “ตั้งค่าเพิ่มเติม” ก่อน (หรือเลือกโค้ช 🔒 ในเครื่อง)"), gr.update()
    if not video:
        return ERR.format("ยังไม่มีคลิป อัดหรืออัปโหลดก่อนนะ"), gr.update()
    try:
        key = pipeline.process(Path(video), pick(qlist, label), lang_code(lang_label),
                               progress=lambda f, t: progress(f, desc=t), fps=15 if fast else 30,
                               coach_mode=mode, resume=resume or "", job=job or "")
        return report.render(pipeline.load_result(key)), gr.update(choices=past_results(), value=key)
    except Exception as e:
        traceback.print_exc()
        return ERR.format(f"เกิดข้อผิดพลาด: {e}") + "<p>ดูรายละเอียดในหน้าต่าง command line</p>", gr.update()


def open_past(key):
    return report.render(pipeline.load_result(key)) if key else ""


# ------------------------------------------------------------
# interview session: questions one after another, stop any time
# each answer is analysed in the background while the next one is recorded
# ------------------------------------------------------------

SESSIONS = {}
RAW_DIR = pipeline.APP_VIDEOS / "raw"
SESSION_DIR = pipeline.FEEDBACK_DIR / "sessions"


def _q_md(s):
    q = s["qs"][s["idx"]]
    text = q["th"] if s["lang"] == "th" else q["en"]
    why = f"\n\n💬 {q['why']}" if q.get("why") else ""
    return f"#### ข้อ {s['idx'] + 1} / {len(s['qs'])}\n## {text}{why}"


def _submit(s, video):
    """Copy the browser recording (temp file) and queue its analysis."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw = RAW_DIR / f"{s['id']}_{s['idx'] + 1}{Path(video).suffix or '.webm'}"
    shutil.copy(video, raw)
    q = s["qs"][s["idx"]]
    fut = s["pool"].submit(pipeline.process, raw, q, s["lang"], None, 15 if s["fast"] else 30,
                           s["mode"], s["resume"], s["job"])
    s["jobs"].append((q, fut))
    s["idx"] += 1


# outputs of every session handler, in this order:
#   sid, question text, recorder, next button, stop button, start button, report, history
def _show(s, msg=""):
    last = s["idx"] == len(s["qs"]) - 1
    return (s["id"], _q_md(s) + (f"\n\n{msg}" if msg else ""), gr.update(value=None, visible=True),
            gr.update(visible=True, value="✅ ส่งคำตอบสุดท้ายและดูผล" if last else "ข้อต่อไป ➜"),
            gr.update(visible=True), gr.update(visible=False),
            gr.update(value="<p style='color:#868e96'>ตอบเสร็จแล้วกดปุ่มด้านล่างคลิป ระบบจะวิเคราะห์ข้อก่อนหน้าไปพร้อมกัน</p>"),
            gr.update())


def _idle(sid, md, html, history=gr.update()):
    return (sid, md, gr.update(value=None, visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=True), html, history)


def start_session(qlist, n, lang_label, consent, coach_label, fast, job, resume):
    mode = COACH_CHOICES.get(coach_label, "auto")
    if not consent and mode not in ("local", "ollama"):
        return _idle(None, "", ERR.format("โค้ช ☁️ ต้องติ๊กยืนยันใน “ตั้งค่าเพิ่มเติม” ก่อน (หรือเลือกโค้ช 🔒 ในเครื่อง)"))
    sid = time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4]
    SESSIONS[sid] = {"id": sid, "qs": list(qlist)[:max(1, int(n))], "idx": 0, "lang": lang_code(lang_label),
                     "mode": mode, "fast": fast, "job": job or "", "resume": resume or "",
                     "pool": ThreadPoolExecutor(max_workers=1), "jobs": []}
    return _show(SESSIONS[sid])


def _finish(s, progress):
    results, errors = [], []
    n = len(s["jobs"])
    for i, (q, fut) in enumerate(s["jobs"]):
        progress(i / max(n, 1), desc=f"กำลังวิเคราะห์ข้อ {i + 1}/{n} ...")
        try:
            results.append((q, pipeline.load_result(fut.result())))
        except Exception as e:
            traceback.print_exc()
            errors.append(f"ข้อ {i + 1} วิเคราะห์ไม่สำเร็จ: {e}")
    s["pool"].shutdown(wait=False)
    SESSIONS.pop(s["id"], None)
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    (SESSION_DIR / f"{s['id']}.json").write_text(json.dumps(
        {"id": s["id"], "planned": len(s["qs"]), "lang": s["lang"], "job": s["job"],
         "answers": [{"question": q, "key": r["key"]} for q, r in results], "errors": errors},
        ensure_ascii=False, indent=2), encoding="utf-8")
    html = report.render_session(results, len(s["qs"]), errors)
    done = f"✅ จบการสัมภาษณ์ ตอบ {len(results)} จาก {len(s['qs'])} ข้อ"
    return _idle(None, done, html, gr.update(choices=past_results()))


def next_answer(sid, video, progress=gr.Progress()):
    s = SESSIONS.get(sid)
    if not s:
        return _idle(None, "", ERR.format("ยังไม่ได้เริ่ม กด ▶ เริ่มสัมภาษณ์"))
    if not video:
        return _show(s, "⚠️ ยังไม่มีคลิปของข้อนี้ อัดคำตอบก่อน หรือกด ⏹ หยุดถ้าจะจบแค่นี้")
    _submit(s, video)
    if s["idx"] >= len(s["qs"]):
        return _finish(s, progress)
    return _show(s)


def stop_session(sid, video, progress=gr.Progress()):
    s = SESSIONS.get(sid)
    if not s:
        return _idle(None, "", gr.update())
    if video and s["idx"] < len(s["qs"]):          # the answer being recorded counts too
        _submit(s, video)
    if not s["jobs"]:
        s["pool"].shutdown(wait=False)
        SESSIONS.pop(sid, None)
        return _idle(None, "หยุดแล้ว ยังไม่ได้ตอบสักข้อ", "")
    return _finish(s, progress)


# Gradio 5 takes theme/css in Blocks(); Gradio 6 takes them in launch()
GR6 = int(gr.__version__.split(".")[0]) >= 6
THEME = gr.themes.Soft(
    primary_hue="orange", secondary_hue="blue", neutral_hue="slate", radius_size="lg",
    font=[gr.themes.GoogleFont("Prompt"), "Sarabun", "Tahoma", "sans-serif"],
)

HERO = """
<div class="hero">
  <div class="hero-text">
    <div class="eyebrow">ต้นแบบงานวิจัย · ทำงานในเครื่องนี้</div>
    <h1>ฝึกสัมภาษณ์งานกับ AI โค้ช</h1>
    <p>ไม่ให้คะแนนว่าคุณน่าจ้างไหม แต่ชี้ว่า <b>ควรแก้ตรงไหน</b> พร้อม <b>เวลาในคลิป</b> และหลักฐานที่ตรวจได้</p>
    <div class="steps">
      <span><i>1</i> ตำแหน่งงาน / resume</span>
      <span><i>2</i> อัดคำตอบ</span>
      <span><i>3</i> ดูคำแนะนำพร้อมเวลา</span>
    </div>
  </div>
  <div class="hero-badges">
    <div class="badge">🔒 วิดีโอไม่ออกนอกเครื่อง</div>
    <div class="badge">⏱ ทุกคำแนะนำมีเวลากำกับ</div>
    <div class="badge">✅ ตัวตรวจตัดข้อที่อ้างไม่ได้</div>
  </div>
</div>
"""
TIPS = "💡 นั่งหน้าตรง แสงด้านหน้าพอ วางกล้องระดับสายตา แล้วมองเลนส์"
FOOT = ('<div class="foot">ต้นแบบงานวิจัย · ใช้เฉพาะคลิปของตัวเองหรือที่ได้รับความยินยอม · '
        'ไม่ใช่เครื่องมือคัดเลือกพนักงาน</div>')

CSS = """
.gradio-container{max-width:1200px !important;margin:0 auto !important}
.hero{display:flex;gap:24px;align-items:center;justify-content:space-between;flex-wrap:wrap;
  background:linear-gradient(135deg,#fff4e6 0%,#ffe8cc 40%,#e7f5ff 100%);
  border-radius:24px;padding:28px 32px;margin:4px 0 8px}
.hero h1{font-size:30px;margin:4px 0 8px;color:#1b2436;font-weight:700}
.hero p{font-size:16px;margin:0;color:#343a40;line-height:1.6;max-width:560px}
.hero .eyebrow{font-size:13px;letter-spacing:1px;color:#d9480f;font-weight:600}
.hero .steps{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}
.hero .steps span{background:#fff;border-radius:999px;padding:6px 14px 6px 6px;font-size:14px;
  box-shadow:0 1px 3px rgba(0,0,0,.08)}
.hero .steps i{font-style:normal;display:inline-grid;place-items:center;width:24px;height:24px;
  border-radius:50%;background:#f76707;color:#fff;font-size:13px;margin-right:6px}
.hero-badges{display:flex;flex-direction:column;gap:8px}
.hero .badge{background:rgba(255,255,255,.85);border-radius:12px;padding:8px 14px;font-size:14px}
.panel{border-radius:20px !important;padding:6px !important}
.tips{color:#495057;font-size:14px}
.foot{text-align:center;color:#868e96;font-size:13px;margin:14px 0 4px}
button.lg{font-size:16px !important}
@media (prefers-color-scheme: dark){
  .hero{background:linear-gradient(135deg,#3b2a1a 0%,#2b2f3a 100%)}
  .hero h1{color:#f8f9fa} .hero p{color:#dee2e6}
  .hero .steps span,.hero .badge{background:rgba(0,0,0,.35);color:#f1f3f5}
}
"""

with gr.Blocks(title="AI Interview Coach",
               **({} if GR6 else {"theme": THEME, "css": CSS})) as demo:
    gr.HTML(HERO)
    qlist = gr.State(BANK)
    sid = gr.State(None)
    with gr.Row():
        with gr.Column(scale=1, elem_classes="panel"):
            with gr.Accordion("① ตำแหน่งงานและ resume (ไม่บังคับ)", open=True):
                job = gr.Textbox(label="ตำแหน่งที่สมัคร", placeholder="เช่น Data Analyst Intern, วิศวกรซอฟต์แวร์")
                resume_file = gr.File(label="อัปโหลด resume (.pdf .docx .txt)",
                                      file_types=[".pdf", ".docx", ".txt"], type="filepath")
                resume = gr.Textbox(label="หรือวางข้อความ resume", lines=6)
                n_q = gr.Slider(3, 8, value=5, step=1, label="จำนวนคำถามที่สร้าง")
                gen = gr.Button("✨ สร้างคำถามจาก resume")
                gen_note = gr.Markdown("")
            lang = gr.Radio(list(questions.LANGS.values()), value=questions.LANGS["en"],
                            label="ภาษาที่ใช้ตอบ")
            coach_pick = gr.Radio(list(COACH_CHOICES), value=list(COACH_CHOICES)[0],
                                  label="② เลือกโค้ช  (🔒 = ในเครื่อง · ☁️ = ส่งข้อความออกไป)")
            with gr.Accordion("ตั้งค่าเพิ่มเติม", open=False):
                consent = gr.Checkbox(label=CONSENT)
                fast = gr.Checkbox(label="โหมดเร็ว (วิเคราะห์ภาพ 15 เฟรม/วินาที)", value=True)
            history = gr.Dropdown(past_results(), label="ดูผลครั้งก่อน (รายข้อ)", allow_custom_value=False)
        with gr.Column(scale=2, elem_classes="panel"):
            gr.Markdown(TIPS, elem_classes="tips")
            with gr.Tabs():
                with gr.Tab("🎙️ สัมภาษณ์ต่อเนื่อง"):
                    n_session = gr.Slider(1, 8, value=3, step=1, label="จำนวนข้อในรอบนี้")
                    start_btn = gr.Button("▶ เริ่มสัมภาษณ์", variant="primary")
                    s_question = gr.Markdown("")
                    s_video = gr.Video(sources=["webcam", "upload"], label="คำตอบข้อนี้",
                                       include_audio=True, visible=False)
                    with gr.Row():
                        next_btn = gr.Button("ข้อต่อไป ➜", variant="primary", visible=False)
                        stop_btn = gr.Button("⏹ หยุดและดูผล", variant="stop", visible=False)
                    s_out = gr.HTML("")
                with gr.Tab("🎯 ฝึกทีละข้อ"):
                    first = labels(BANK, "en")
                    question = gr.Dropdown(first, value=first[0], label="คำถาม")
                    why = gr.Markdown("")
                    video = gr.Video(sources=["webcam", "upload"], label="คลิปคำตอบ", include_audio=True)
                    go = gr.Button("วิเคราะห์", variant="primary")
                    out = gr.HTML("<p style='color:#868e96'>ผลจะแสดงตรงนี้</p>")
    gr.HTML(FOOT)
    resume_file.change(load_resume, [resume_file, resume], [resume, gen_note])
    gen.click(make_questions, [job, resume, lang, consent, n_q], [qlist, question, gen_note, why])
    lang.change(change_lang, [lang, qlist], question)
    question.change(why_text, [qlist, question], why)
    go.click(analyse, [video, lang, qlist, question, consent, fast, coach_pick, job, resume],
             [out, history])
    history.change(open_past, history, out)
    s_outputs = [sid, s_question, s_video, next_btn, stop_btn, start_btn, s_out, history]
    start_btn.click(start_session, [qlist, n_session, lang, consent, coach_pick, fast, job, resume], s_outputs)
    next_btn.click(next_answer, [sid, s_video], s_outputs)
    stop_btn.click(stop_session, [sid, s_video], s_outputs)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", share=False, inbrowser=True,
                        **({"theme": THEME, "css": CSS} if GR6 else {}))
