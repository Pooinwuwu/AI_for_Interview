"""
app/app.py — Interview practice coach (local research prototype)

  pip install gradio
  python app/app.py            -> open http://127.0.0.1:7860

Record (webcam) or upload a 1-2 minute answer, press the button, read the report.
Runs on this computer only (not shared on the internet). For your own clips and people who
agreed to take part. The transcript and measurements (not the video) are sent to Gemini.
"""

import sys
import traceback
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
2. เลือกภาษาและคำถาม แล้วอัดคลิปตอบ 1–2 นาที (หรืออัปโหลดคลิปจากมือถือ)
3. กด **วิเคราะห์** แล้วรอ 3–6 นาที

💡 นั่งหน้าตรง แสงพอ มองกล้อง · ต้นแบบงานวิจัย ใช้ในเครื่องนี้เท่านั้น
"""

CONSENT = ("คลิปและ resume เป็นของฉัน/ได้รับความยินยอมแล้ว และยอมให้ส่ง resume ข้อความถอดเสียง "
           "และผลวัด (ไม่ใช่วิดีโอ) ให้ Gemini เพื่อสร้างคำถามและเขียนคำแนะนำ")
ERR = '<p style="color:#d9480f">{}</p>'

COACH_CHOICES = {"อัตโนมัติ: Gemini ถ้าล่มใช้โค้ชในเครื่อง": "auto",
                 "Gemini เท่านั้น": "gemini",
                 "โค้ชในเครื่อง (ไม่ใช้เน็ต ไม่ส่งข้อมูลออก)": "local"}

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
        return (BANK, gr.update(), ERR.format("ติ๊กยืนยันด้านบนก่อน เพราะต้องส่ง resume ให้ Gemini"), "")
    progress(0.2, desc="AI กำลังอ่าน resume และสร้างคำถาม ...")
    qlist, note = question_gen.generate(resume, job, int(n))
    ch = labels(qlist, lang_code(lang_label))
    return qlist, gr.update(choices=ch, value=ch[0]), f"✨ {note}", why_text(qlist, ch[0])


def analyse(video, lang_label, qlist, label, consent, fast, coach_label, job, resume,
            progress=gr.Progress()):
    mode = COACH_CHOICES.get(coach_label, "auto")
    if not consent and mode != "local":
        return ERR.format("กรุณาติ๊กยืนยันด้านบนก่อน (หรือเลือกโค้ชในเครื่องซึ่งไม่ส่งข้อมูลออก)"), gr.update()
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


# Gradio 5 takes theme in Blocks(); Gradio 6 takes it in launch()
GR6 = int(gr.__version__.split(".")[0]) >= 6
THEME = gr.themes.Soft()

with gr.Blocks(title="AI Interview Coach", **({} if GR6 else {"theme": THEME})) as demo:
    gr.Markdown(INTRO)
    qlist = gr.State(BANK)
    with gr.Row():
        with gr.Column(scale=1):
            consent = gr.Checkbox(label=CONSENT)
            with gr.Accordion("① ตำแหน่งงานและ resume (ไม่บังคับ)", open=True):
                job = gr.Textbox(label="ตำแหน่งที่สมัคร", placeholder="เช่น Data Analyst Intern, วิศวกรซอฟต์แวร์")
                resume_file = gr.File(label="อัปโหลด resume (.pdf .docx .txt)",
                                      file_types=[".pdf", ".docx", ".txt"], type="filepath")
                resume = gr.Textbox(label="หรือวางข้อความ resume", lines=6)
                n_q = gr.Slider(3, 8, value=5, step=1, label="จำนวนคำถาม")
                gen = gr.Button("✨ สร้างคำถามจาก resume")
                gen_note = gr.Markdown("")
            gr.Markdown("### ② เลือกคำถามแล้วอัดคลิป")
            lang = gr.Radio(list(questions.LANGS.values()), value=questions.LANGS["en"],
                            label="ภาษาที่ใช้ตอบ")
            first = labels(BANK, "en")
            question = gr.Dropdown(first, value=first[0], label="คำถาม")
            why = gr.Markdown("")
            video = gr.Video(sources=["webcam", "upload"], label="คลิปคำตอบ", include_audio=True)
            coach_pick = gr.Radio(list(COACH_CHOICES), value=list(COACH_CHOICES)[0], label="โค้ช")
            fast = gr.Checkbox(label="โหมดเร็ว (วิเคราะห์ภาพ 15 เฟรม/วินาที)", value=True)
            go = gr.Button("③ วิเคราะห์", variant="primary")
            history = gr.Dropdown(past_results(), label="ดูผลครั้งก่อน", allow_custom_value=False)
        with gr.Column(scale=2):
            out = gr.HTML("<p style='color:#868e96'>ผลจะแสดงตรงนี้</p>")
    resume_file.change(load_resume, [resume_file, resume], [resume, gen_note])
    gen.click(make_questions, [job, resume, lang, consent, n_q], [qlist, question, gen_note, why])
    lang.change(change_lang, [lang, qlist], question)
    question.change(why_text, [qlist, question], why)
    go.click(analyse, [video, lang, qlist, question, consent, fast, coach_pick, job, resume],
             [out, history])
    history.change(open_past, history, out)

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", share=False, inbrowser=True,
                        **({"theme": THEME} if GR6 else {}))
