"""
features/speech_text.py

ใช้ Whisper แปลง audio -> transcript + word timestamps
แล้วสกัด features:
  - speech_rate (words/min)
  - pause_count / duration
  - filler_word ratio
  - transcript เต็ม

Output: output/features/{key}_speech.json
"""

import sys
import json
import re
from pathlib import Path

# ---------- UTF-8 fix Windows ----------
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CONFIG
# ============================================================

# ---- Whisper ----
WHISPER_MODEL   = "medium"       # tiny / base / small / medium / large-v3
WHISPER_DEVICE  = "cpu"         # "auto" | "cpu" | "cuda"
WHISPER_COMPUTE = "int8"         # "auto" | "int8" (CPU) | "float16" (GPU)
WHISPER_LANG    = "en"           # None = auto-detect
INITIAL_PROMPT  = "บทสนทนาสัมภาษณ์งานภาษาไทย"  # ช่วยให้ Whisper แม่นขึ้น

# ---- Pause detection ----
PAUSE_THRESHOLD_SEC = 0.5        # ช่วงว่าง >= 0.5s ถือเป็น pause
LONG_PAUSE_SEC      = 2.0        # ช่วงว่าง >= 2s ถือเป็น long pause

# ---- Filler words ภาษาไทย ----
FILLER_WORDS = [
    "อืม", "อืมม", "เอ่อ", "เอ่ออ", "เอ้", "เอิ่ม", "อ่า",
    "คือ", "แบบ", "แบบว่า", "ประมาณว่า", "อะไรอย่างงี้",
    "นะ", "นะครับ", "นะคะ", "เนอะ",
]

# ---- Paths ----
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
AUDIO_DIR     = PROJECT_ROOT / "output" / "audio"
FEATURES_OUT  = PROJECT_ROOT / "output" / "features"


# ============================================================
# LOAD WHISPER (lazy — โหลดครั้งเดียวต่อ process)
# ============================================================

_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    print(f"[INFO] กำลังโหลด Whisper model: {WHISPER_MODEL} "
          f"(device={WHISPER_DEVICE}, compute={WHISPER_COMPUTE})")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[ERROR] ไม่พบ faster-whisper")
        print("        ติดตั้ง: pip install faster-whisper")
        sys.exit(1)

    _MODEL = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE,
    )
    print("[INFO] โหลด model สำเร็จ\n")
    return _MODEL


# ============================================================
# TRANSCRIBE
# ============================================================

def transcribe(audio_path: Path):
    """คืน (segments, info) จาก Whisper"""
    model = load_model()
    segments, info = model.transcribe(
        str(audio_path),
        language=WHISPER_LANG,
        word_timestamps=True,
        initial_prompt=INITIAL_PROMPT,
        vad_filter=True,                    # ตัดช่วงเงียบออก
        vad_parameters={"min_silence_duration_ms": 300},
    )

    # แปลง generator → list
    segments = list(segments)
    return segments, info


# ============================================================
# FEATURES
# ============================================================

def extract_words(segments):
    """รวม words จากทุก segment"""
    words = []
    for seg in segments:
        for w in (seg.words or []):
            words.append({
                "word": w.word.strip(),
                "start": round(w.start, 3),
                "end":   round(w.end,   3),
                "prob":  round(w.probability, 3),
            })
    return words


def extract_pauses(words, min_pause=PAUSE_THRESHOLD_SEC):
    """หาช่วงว่างระหว่างคำ"""
    pauses = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= min_pause:
            pauses.append({
                "start": round(words[i - 1]["end"], 3),
                "end":   round(words[i]["start"],   3),
                "dur":   round(gap, 3),
            })
    return pauses


def count_fillers(words):
    """นับ filler words + breakdown"""
    breakdown = {}
    total = 0
    for w in words:
        raw = w["word"].lower().strip(".,!?ๆฯ ")
        if raw in FILLER_WORDS:
            breakdown[raw] = breakdown.get(raw, 0) + 1
            total += 1
    return total, breakdown


def build_features(segments, words, duration_sec):
    n_words = len(words)

    # ---- speech rate ----
    speech_rate_wpm = 0.0
    if duration_sec > 0:
        speech_rate_wpm = n_words / duration_sec * 60.0

    # ---- articulation rate (ไม่นับ pause) ----
    if words:
        speaking_time = sum(w["end"] - w["start"] for w in words)
        articulation_wpm = (n_words / speaking_time * 60.0
                            if speaking_time > 0 else 0.0)
    else:
        speaking_time = 0.0
        articulation_wpm = 0.0

    # ---- pauses ----
    pauses = extract_pauses(words)
    pause_total = sum(p["dur"] for p in pauses)
    pause_mean = pause_total / len(pauses) if pauses else 0.0
    pause_max = max((p["dur"] for p in pauses), default=0.0)
    long_pauses = [p for p in pauses if p["dur"] >= LONG_PAUSE_SEC]

    # ---- fillers ----
    filler_count, filler_breakdown = count_fillers(words)
    filler_ratio = filler_count / n_words if n_words else 0.0

    return {
        "duration_sec":       round(duration_sec, 2),
        "speaking_sec":       round(speaking_time, 2),
        "word_count":         n_words,
        "segment_count":      len(segments),

        "speech_rate_wpm":    round(speech_rate_wpm, 2),
        "articulation_wpm":   round(articulation_wpm, 2),

        "pause_count":        len(pauses),
        "long_pause_count":   len(long_pauses),
        "pause_total_sec":    round(pause_total, 2),
        "pause_mean_sec":     round(pause_mean, 2),
        "pause_max_sec":      round(pause_max, 2),

        "filler_count":       filler_count,
        "filler_ratio":       round(filler_ratio, 4),
        "filler_breakdown":   filler_breakdown,
    }


# ============================================================
# PROCESS ONE AUDIO
# ============================================================

def process_audio(key: str, audio_path: Path) -> dict:
    print(f"[RUN ] {key}")
    print(f"       audio: {audio_path.name}")

    segments, info = transcribe(audio_path)

    words = extract_words(segments)
    duration = info.duration if hasattr(info, "duration") else 0.0

    transcript = " ".join(seg.text.strip() for seg in segments).strip()

    # segments แบบย่อ (ไม่เก็บ words ซ้ำ)
    seg_summary = [
        {
            "start": round(s.start, 3),
            "end":   round(s.end, 3),
            "text":  s.text.strip(),
        }
        for s in segments
    ]

    feats = build_features(segments, words, duration)

    result = {
        "key":         key,
        "language":    info.language,
        "language_prob": round(info.language_probability, 3),
        "model":       WHISPER_MODEL,
        "transcript":  transcript,
        "segments":    seg_summary,
        "words":       words,
        "features":    feats,
    }

    # print สรุป
    f = feats
    print(f"       lang={info.language} ({info.language_probability:.2f})  "
          f"words={f['word_count']}  wpm={f['speech_rate_wpm']}")
    print(f"       pauses={f['pause_count']} (long {f['long_pause_count']})  "
          f"fillers={f['filler_count']} ({f['filler_ratio']:.3f})")

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  SPEECH FEATURES  (Whisper)")
    print("=" * 60)

    if not AUDIO_DIR.exists():
        print(f"[ERROR] ไม่พบโฟลเดอร์: {AUDIO_DIR}")
        print("        รัน extract_audio.py ก่อน")
        sys.exit(1)

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))
    if not audio_files:
        print(f"[ERROR] ไม่พบไฟล์ .wav ใน {AUDIO_DIR}")
        sys.exit(1)

    FEATURES_OUT.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Input  : {AUDIO_DIR.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] Output : {FEATURES_OUT.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] พบ {len(audio_files)} ไฟล์\n")

    for audio_path in audio_files:
        key = audio_path.stem
        out_path = FEATURES_OUT / f"{key}_speech.json"

        try:
            result = process_audio(key, audio_path)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"       -> {out_path.relative_to(PROJECT_ROOT)}\n")
        except Exception as e:
            print(f"       [FAIL] {e}\n")

    print("=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()