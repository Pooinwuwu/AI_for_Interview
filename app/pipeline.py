"""
app/pipeline.py — run the whole system on ONE clip recorded in the web app.

Same tools as run_pipeline.py, but per clip (run_pipeline.py processes every video in a folder):
  convert -> audio -> prosody -> transcript -> speech features -> frames -> landmarks
  -> gaze/head/hand/face events -> evidence log -> visual features -> rule scores
  -> coach (Gemini writes from the evidence log, the checker removes unsupported points)

Privacy: only for your own clips and people who agreed (consent form). The transcript and the
measured evidence (not the video) are sent to Gemini. No model trained on AVI or RecruitView is
used here: the judge is the rule-based Approach 1 only.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from base._paths import (AUDIO_DIR, EVIDENCE_DIR, FEATURES_DIR, FRAMES_DIR, INPUT_DIR,
                         LANDMARKS_DIR, METADATA_DIR, METADATA_FILE, OUTPUT_DIR, SCORES_DIR)

APP_VIDEOS = INPUT_DIR / "videos_app"
FEEDBACK_DIR = OUTPUT_DIR / "feedback_app"


_MODULES = {}


def _load(name: str, rel: str):
    """Import a project script by path (several folders have modules with the same names).
    Cached, so heavy models (WhisperX) load once per app session."""
    if name in _MODULES:
        return _MODULES[name]
    path = ROOT / rel
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MODULES[name] = mod
    return mod


def _save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------
# steps
# ------------------------------------------------------------

def convert(src: Path, key: str) -> Path:
    """Browser recordings are webm with odd frame rates: re-encode to a 30 fps mp4."""
    ea = _load("app_extract_audio", "base/preprocessing/extract_audio.py")
    ffmpeg = ea.get_ffmpeg_path() or "ffmpeg"
    APP_VIDEOS.mkdir(parents=True, exist_ok=True)
    out = APP_VIDEOS / f"{key}.mp4"
    cmd = [ffmpeg, "-y", "-i", str(src), "-vf", "fps=30", "-c:v", "libx264", "-preset", "veryfast",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-loglevel", "error", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg could not convert the video: {r.stderr[-300:]}")
    return out


def add_metadata(key: str, video: Path, question="intro", lang: str = "en",
                 resume: str = "", job: str = ""):
    gm = _load("app_generate_metadata", "base/preprocessing/generate_metadata.py")
    info = gm.probe_video(video)
    info.update(gm.DEFAULT_METADATA)
    info["file_path"] = f"videos_app/{video.name}"
    import questions
    q = questions.as_question(question)
    info["question_id"] = q["id"]
    info["question"] = q                  # full text (generated questions are not in the bank)
    info["language"] = lang
    info["job"] = job or ""
    info["resume"] = resume or ""         # stays on this computer (output/ is never committed)
    info["language"] = lang
    meta = _read(METADATA_FILE) if METADATA_FILE.exists() else {}
    meta[key] = info
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    _save(METADATA_FILE, meta)
    return info


def audio(key: str, video: Path) -> Path:
    ea = _load("app_extract_audio", "base/preprocessing/extract_audio.py")
    out = AUDIO_DIR / f"{key}.wav"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    if not ea.extract_audio(video, out):
        raise RuntimeError("could not extract the audio track")
    return out


def prosody(key: str, wav: Path):
    pm = _load("app_prosody", "base/measurement/speech/prosody.py")
    feats = pm.extract_prosody(wav)
    if feats:
        _save(EVIDENCE_DIR / f"{key}_prosody.json", feats)


_THAI = {}


def _thai_words(text: str):
    try:
        from pythainlp.tokenize import word_tokenize
        toks = word_tokenize(text, engine="newmm", keep_whitespace=False)
    except ImportError:                       # rough fallback: split on spaces
        toks = text.split()
    return [t for t in toks if t.strip() and any(ch.isalnum() for ch in t)]


def transcribe_thai(wav: Path, device: str, compute_type: str) -> dict:
    """Thai: Whisper 'small' (base is poor at Thai), no forced alignment (WhisperX has no Thai
    aligner). Word times are spread evenly inside each segment, so pauses are found between
    segments only. Words = PyThaiNLP tokens (Thai has no spaces between words)."""
    # faster-whisper directly (no transformers import, so no huggingface-hub version clash)
    from faster_whisper import WhisperModel
    if "asr" not in _THAI:
        _THAI["asr"] = WhisperModel("small", device=device, compute_type=compute_type)
    segs, _info = _THAI["asr"].transcribe(str(wav), language="th", beam_size=5, vad_filter=True)
    segments = []
    for s in segs:
        text = (s.text or "").strip()
        if not text:
            continue
        toks = _thai_words(text) or [text]
        t0, t1 = float(s.start), float(s.end)
        step = (t1 - t0) / len(toks)
        segments.append({"start": t0, "end": t1, "text": text,
                         "words": [{"word": w, "start": round(t0 + i * step, 3),
                                    "end": round(t0 + (i + 1) * step, 3)} for i, w in enumerate(toks)]})
    return {"language": "th", "word_timing": "interpolated within segments", "segments": segments}


def transcript(key: str, wav: Path, lang: str = "en"):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "float32"
    if lang == "th":
        res = transcribe_thai(wav, device, compute)
    else:
        tm = _load("app_transcribe", "base/measurement/speech/transcribe.py")
        res = tm.transcribe_audio(wav, device=device, compute_type=compute, language="en")
    if not res:
        raise RuntimeError("speech recognition failed (is there speech in the clip?)")
    _save(EVIDENCE_DIR / f"{key}_transcript.json", res)
    st = _load("app_speech_text", "approaches/approach_1_rule/features/speech_text.py")
    _save(FEATURES_DIR / f"{key}_speech.json", st.process_clip(key, res))


def landmarks(key: str, video: Path, fps: int):
    ef = _load("app_extract_frames", "base/preprocessing/extract_frames.py")
    df = _load("app_detect", "base/preprocessing/detect_face_hand.py")
    df.check_models()
    frame_dir = FRAMES_DIR / key
    ef.extract_frames(video, frame_dir, fps)
    try:
        if not df.process_video(key, frame_dir, LANDMARKS_DIR / f"{key}.json"):
            raise RuntimeError("no face / body found in the video")
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def visual(key: str):
    data = _read(LANDMARKS_DIR / f"{key}.json")
    data.setdefault("key", key)
    geo = _load("app_geometry", "base/measurement/visual/geometry.py")
    aspect = geo.video_aspect(key)
    for name, rel, fn in (
            ("gaze", "base/measurement/visual/gaze_events.py", "extract_gaze_events"),
            ("head", "base/measurement/visual/head_events.py", "extract_head_events"),
            ("hand", "base/measurement/visual/hand_events.py", "extract_hand_events"),
            ("face", "base/measurement/visual/face_events.py", "extract_face_events")):
        mod = _load(f"app_{name}_events", rel)
        _save(EVIDENCE_DIR / f"{key}_{name}_events.json", getattr(mod, fn)(data, aspect))
    be = _load("app_build_evidence", "base/measurement/build_evidence.py")
    _save(EVIDENCE_DIR / f"{key}_evidence.json", be.build_evidence_log(key))
    fx = _load("app_features", "approaches/approach_1_rule/features/run_extract_all.py")
    _save(FEATURES_DIR / f"{key}.json", fx.extract_one(key, data))


def scores(key: str):
    rs = _load("app_scoring", "approaches/approach_1_rule/scoring/run_scoring.py")
    vis = _read(FEATURES_DIR / f"{key}.json")
    sp_path = FEATURES_DIR / f"{key}_speech.json"
    sp = _read(sp_path) if sp_path.exists() else None
    _save(SCORES_DIR / f"{key}.json", rs.compute_scores(key, vis, sp))


COACH_MODES = ("auto", "gemini", "local")


def coach_gemini(key: str, quick: bool) -> dict:
    """Approach C: Gemini writes from the evidence log, verify.py checks it."""
    run = _load("app_grounded_run", "approaches/approach_grounded/run.py")
    import questions
    import base.llm_common as lc
    if quick:                            # app: give up after ~30 s and use the local coach
        lc.RETRY_WAITS_BUSY[:] = [10, 20]
    meta = (_read(METADATA_FILE) if METADATA_FILE.exists() else {}).get(key, {})
    q = meta.get("question") or meta.get("question_id", "intro")
    lang = meta.get("language", "en")
    run.build_prompt.__globals__["question_block"] = lambda k: questions.question_block(
        q, lang, meta.get("resume", ""), meta.get("job", ""))

    def judge_rule_only(k):              # never the AVI-trained 1b model in the app
        rule = _read(SCORES_DIR / f"{k}.json")
        ratings = {d: rule["dimensions"][d]["band"]["label_th"] for d in run.DIMENSIONS
                   if d in rule.get("dimensions", {})}
        band = rule.get("fusion", {}).get("overall_band", {}).get("label_th")
        ratings["overall"] = band if band in run.BANDS_TH else "ปานกลาง"
        return ratings, "rule-based measurement", {}

    run.judge = judge_rule_only
    writer = run.Writer(with_video=False)
    return run.run_clip(key, writer, with_video=False, max_rounds=1)


def coach_local(key: str) -> dict:
    """Approach T: template coach, no LLM, no internet."""
    tr = _load("app_template_run", "approaches/approach_template/run.py")
    import questions
    meta = (_read(METADATA_FILE) if METADATA_FILE.exists() else {}).get(key, {})
    q = questions.as_question(meta.get("question") or meta.get("question_id", "intro"))
    return tr.run_clip(key, focus=q["focus"], lang=meta.get("language", "en"))


def coach(key: str, mode: str = "auto") -> dict:
    """auto = Gemini, and the local coach if Gemini is down; gemini / local = only that one."""
    reason = None
    res = None
    if mode in ("auto", "gemini"):
        try:
            res = coach_gemini(key, quick=(mode == "auto"))
            res["coach_used"] = "gemini"
        except Exception as e:
            if mode == "gemini":
                raise
            reason = str(e)[:300]
            print(f"[coach] Gemini failed, using the local coach: {reason}")
    if res is None:
        res = coach_local(key)
        res["coach_used"] = "local"
        if reason:
            res["fallback_reason"] = reason
    _save(FEEDBACK_DIR / f"{key}.json", res)
    return res


# ------------------------------------------------------------
# all steps
# ------------------------------------------------------------

STEPS = [("แปลงไฟล์วิดีโอ", 0.03), ("แยกเสียง", 0.02), ("วัดน้ำเสียง", 0.03),
         ("ถอดเสียงเป็นข้อความ", 0.20), ("ตรวจจับใบหน้าและท่าทาง", 0.45),
         ("สรุปพฤติกรรม", 0.10), ("ให้คะแนนเบื้องต้น", 0.02), ("โค้ชเขียนคำแนะนำ", 0.15)]


def process(src_video: Path, question, lang: str = "en", progress=None, fps: int = 30,
            coach_mode: str = "auto", resume: str = "", job: str = "") -> str:
    """question = bank id ("q3") or a generated question dict."""
    """Runs everything; returns the clip key. progress(fraction, text) is optional."""
    key = f"app_{time.strftime('%Y%m%d%H%M%S')}{lang}_q1"      # question + language: metadata
    done = 0.0

    def step(i, fn, *a):
        nonlocal done
        if progress:
            progress(done, f"{i + 1}/{len(STEPS)} {STEPS[i][0]} ...")
        out = fn(*a)
        done += STEPS[i][1]
        return out

    video = step(0, convert, Path(src_video), key)
    add_metadata(key, video, question, lang, resume, job)
    wav = step(1, audio, key, video)
    step(2, prosody, key, wav)
    step(3, transcript, key, wav, lang)
    step(4, landmarks, key, video, fps)
    step(5, visual, key)
    step(6, scores, key)
    step(7, coach, key, coach_mode)
    if progress:
        progress(1.0, "เสร็จแล้ว")
    return key


def load_result(key: str) -> dict:
    """Everything the report page needs."""
    def opt(p):
        return _read(p) if p.exists() else {}
    return {
        "key": key,
        "feedback": opt(FEEDBACK_DIR / f"{key}.json"),
        "speech": opt(FEATURES_DIR / f"{key}_speech.json"),
        "visual": opt(FEATURES_DIR / f"{key}.json"),
        "meta": opt(METADATA_FILE).get(key, {}),
    }


if __name__ == "__main__":
    # re-run only the coach on a clip that was already measured, e.g. after Gemini was down:
    #   python app/pipeline.py app_20261009114704en_q1 local
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("mode", nargs="?", default="auto", choices=COACH_MODES)
    a = ap.parse_args()
    r = coach(a.key, a.mode)
    print(f"done: coach={r['coach_used']}  strengths={len(r.get('strengths', []))}  "
          f"improvements={len(r.get('improvements', []))}  -> {FEEDBACK_DIR / (a.key + '.json')}")
