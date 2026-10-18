"""
web/server.py — Interview practice coach, normal web version (replaces the Gradio screen only)

  pip install fastapi uvicorn python-multipart
  python web/server.py          -> opens http://127.0.0.1:8000

Same pipeline as app/app.py (measure -> evidence log -> coach -> verifier -> report). Runs on this
computer only (127.0.0.1). Videos never leave the computer. Cloud coaches (Gemini / Groq) and
question generation from a resume need the consent box, exactly as in the Gradio app.
"""

import json
import shutil
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import pipeline
import question_gen
import questions
import report

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HOST, PORT = "127.0.0.1", 8000
STATIC = Path(__file__).resolve().parent / "static"
RAW_DIR = pipeline.APP_VIDEOS / "raw"
SESSION_DIR = pipeline.FEEDBACK_DIR / "sessions"

COACHES = [  # (id, label, runs locally)
    ("ollama", "Typhoon ในเครื่อง", True),
    ("local", "โค้ชแม่แบบ (เร็ว ไม่ใช้ LLM)", True),
    ("auto", "Gemini (ถ้าล่มใช้โค้ชในเครื่อง)", False),
    ("gemini", "Gemini เท่านั้น", False),
    ("groq", "Groq API", False),
]
LOCAL = {c for c, _, loc in COACHES if loc}
BANK = [questions.as_question(q[0]) for q in questions.QUESTIONS]

app = FastAPI(title="AI Interview Coach", docs_url=None, redoc_url=None)
POOL = ThreadPoolExecutor(max_workers=1)          # one clip at a time: GPU / Ollama are shared
JOBS: dict = {}
LOCK = threading.Lock()


def _job_update(jid, **kw):
    with LOCK:
        JOBS[jid].update(kw)


DIMENSIONS = {"facial_expression", "eye_contact", "hand_gesture", "head_pose"}


def _save_skip(key: str, skip: list):
    """Remember which dimensions the user chose not to get advice on (report.py hides them)."""
    path = pipeline.FEEDBACK_DIR / f"{key}.json"
    if not skip or not path.exists():
        return
    fb = json.loads(path.read_text(encoding="utf-8"))
    fb["user_skip"] = skip
    path.write_text(json.dumps(fb, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(jid, raw: Path, question: dict, lang: str, fast: bool, coach: str, resume: str, job: str,
         skip: list = ()):
    _job_update(jid, status="running", text="เริ่มวิเคราะห์ ...")
    try:
        key = pipeline.process(
            raw, question, lang,
            progress=lambda f, t: _job_update(jid, progress=round(float(f), 3), text=t),
            fps=15 if fast else 30, coach_mode=coach, resume=resume, job=job)
        _save_skip(key, list(skip))
        _job_update(jid, status="done", progress=1.0, text="เสร็จแล้ว", key=key)
    except Exception as e:
        traceback.print_exc()
        _job_update(jid, status="error", error=str(e), text="วิเคราะห์ไม่สำเร็จ")


def _history():
    out = []
    for p in sorted(pipeline.FEEDBACK_DIR.glob("app_*.json"), reverse=True):
        key = p.stem
        meta = pipeline.load_result(key).get("meta") or {}
        q = meta.get("question") or {}
        th = meta.get("language") == "th"
        out.append({"key": key, "lang": meta.get("language", ""),
                    "question": (q.get("th") if th else q.get("en")) or "",
                    "when": f"{key[4:8]}-{key[8:10]}-{key[10:12]} {key[12:14]}:{key[14:16]}"
                            if len(key) > 16 else key})
    return out


# ---------------- pages ----------------

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------------- api ----------------

@app.get("/api/config")
def config():
    return {"coaches": [{"id": c, "label": l, "local": loc} for c, l, loc in COACHES],
            "questions": BANK, "langs": questions.LANGS}


@app.post("/api/resume")
async def read_resume(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".docx", ".txt"):
        raise HTTPException(400, "รองรับ .pdf .docx .txt")
    tmp = RAW_DIR / f"resume_{uuid.uuid4().hex[:8]}{suffix}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(await file.read())
    try:
        text = question_gen.read_resume(str(tmp)) or ""
    finally:
        tmp.unlink(missing_ok=True)
    if not text:
        raise HTTPException(400, "อ่านไฟล์นี้ไม่ได้ ลองคัดลอกข้อความมาวางแทน")
    return {"text": text}


@app.post("/api/generate")
def generate(payload: dict):
    resume = (payload.get("resume") or "").strip()
    if resume and not payload.get("consent"):
        raise HTTPException(400, "ต้องติ๊กยินยอมก่อน เพราะต้องส่ง resume ให้ Gemini")
    n = max(3, min(8, int(payload.get("n") or 5)))
    qlist, note = question_gen.generate(resume, payload.get("job") or "", n)
    return {"questions": qlist, "note": note}


@app.post("/api/analyse")
async def analyse(video: UploadFile = File(...), question: str = Form(...), lang: str = Form("en"),
                  coach: str = Form("ollama"), consent: bool = Form(False), fast: bool = Form(True),
                  job: str = Form(""), resume: str = Form(""), skip: str = Form("")):
    if coach not in {c for c, _, _ in COACHES}:
        raise HTTPException(400, "ไม่รู้จักโค้ชนี้")
    if coach not in LOCAL and not consent:
        raise HTTPException(400, "โค้ชออนไลน์ต้องติ๊กยินยอมก่อน หรือเลือกโค้ชในเครื่อง")
    lang = "th" if lang == "th" else "en"
    jid = time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(video.filename or "").suffix.lower() or ".webm"
    raw = RAW_DIR / f"web_{jid}{suffix}"
    with open(raw, "wb") as f:
        shutil.copyfileobj(video.file, f)
    q = json.loads(question)
    with LOCK:
        queued = sum(1 for j in JOBS.values() if j["status"] in ("queued", "running"))
        JOBS[jid] = {"id": jid, "status": "queued", "progress": 0.0,
                     "text": f"รอคิว ({queued} คลิปก่อนหน้า)" if queued else "รอคิว ...",
                     "key": None, "error": None, "question": q}
    skips = [d for d in skip.split(",") if d in DIMENSIONS]
    POOL.submit(_run, jid, raw, q, lang, fast, coach, resume, job, skips)
    return {"job": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    with LOCK:
        j = JOBS.get(jid)
        if not j:
            raise HTTPException(404, "ไม่พบงานนี้")
        return {k: v for k, v in j.items() if k != "question"}


@app.get("/api/report/{key}", response_class=HTMLResponse)
def report_html(key: str):
    if not key.startswith("app_") or "/" in key or "\\" in key:
        raise HTTPException(400, "bad key")
    return report.render(pipeline.load_result(key))


@app.post("/api/session", response_class=HTMLResponse)
def session_report(payload: dict):
    """payload: {"planned": n, "jobs": [job ids in answer order], "job": "...", "lang": "th"}"""
    results, errors = [], []
    for i, jid in enumerate(payload.get("jobs") or [], 1):
        j = JOBS.get(jid)
        if not j or j["status"] != "done":
            errors.append(f"ข้อ {i} วิเคราะห์ไม่สำเร็จ: {(j or {}).get('error') or 'ไม่พบผล'}")
            continue
        results.append((j["question"], pipeline.load_result(j["key"])))
    sid = time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:4]
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    (SESSION_DIR / f"{sid}.json").write_text(json.dumps(
        {"id": sid, "planned": payload.get("planned"), "lang": payload.get("lang"),
         "job": payload.get("job", ""),
         "answers": [{"question": q, "key": r["key"]} for q, r in results], "errors": errors},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return report.render_session(results, int(payload.get("planned") or len(results)), errors)


@app.get("/api/history")
def history():
    return _history()


@app.exception_handler(Exception)
def on_error(_, exc):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": f"เกิดข้อผิดพลาด: {exc}"})


if __name__ == "__main__":
    import uvicorn
    threading.Timer(1.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    print(f"เปิด http://{HOST}:{PORT}  (ปิดด้วย Ctrl+C)")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
