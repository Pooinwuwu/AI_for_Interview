"""
features/speech_text.py

Speech features for Approach 1, computed from the WhisperX transcript that
Tier 1 already produced (base/measurement/speech/transcribe.py).

Why reuse WhisperX instead of running Whisper again
---------------------------------------------------
- One transcript for the whole project: Approach 1 features and the Evidence Log
  (Approach 3) now count the same words, fillers and pauses.
- Each clip is transcribed once instead of twice.
- On AVI clips the old faster-whisper "medium" + VAD + initial_prompt setup
  dropped "um"/"uh" and sometimes whole sentences near the end of an answer;
  WhisperX kept them. Filler and speech-rate features depend on that.

Features
--------
  - speech_rate (words/min), articulation rate
  - pause_count / duration (gaps between words)
  - filler_word ratio
  - full transcript

Input : output/evidence/{key}_transcript.json   (run transcribe.py first)
        output/audio/{key}.wav                   (for the clip duration)
Output: output/features/{key}_speech.json        (same format as before)
"""

import sys
import json
import wave
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

# ---- Pause detection ----
PAUSE_THRESHOLD_SEC = 0.5        # ช่วงว่าง >= 0.5s ถือเป็น pause
LONG_PAUSE_SEC      = 2.0        # ช่วงว่าง >= 2s ถือเป็น long pause

# ---- Filler words ภาษาอังกฤษ ----
# multi-word fillers ("you know") are matched on consecutive words
FILLER_WORDS = [
    "um", "uh", "er", "ah", "hmm",
    "like", "you know", "i mean",
    "kind of", "sort of", "basically",
    "actually", "literally",
]

# ---- Paths ----
PROJECT_ROOT  = Path(__file__).resolve().parents[3]  # features -> approach_1_rule -> approaches -> root
AUDIO_DIR     = PROJECT_ROOT / "output" / "audio"
EVIDENCE_DIR  = PROJECT_ROOT / "output" / "evidence"
FEATURES_OUT  = PROJECT_ROOT / "output" / "features"
METADATA_FILE = PROJECT_ROOT / "input" / "metadata" / "metadata.json"

PUNCT = ".,!?;:\"'()[]…-–—ๆฯ "


# ============================================================
# LOAD
# ============================================================

def load_transcript(key: str):
    path = EVIDENCE_DIR / f"{key}_transcript.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clip_duration(key: str) -> float:
    """Duration from the wav header; falls back to metadata.json."""
    wav = AUDIO_DIR / f"{key}.wav"
    if wav.exists():
        with wave.open(str(wav), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            info = json.load(f).get(key, {})
        return float(info.get("duration_sec") or 0.0)
    return 0.0


# ============================================================
# FEATURES
# ============================================================

def extract_words(segments):
    """
    รวม words จากทุก segment.
    WhisperX can leave a word without start/end when alignment fails
    (often numbers); those count as words but are skipped for timing.
    """
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            text = str(w.get("word", "")).strip()
            if not text:
                continue
            words.append({
                "word":  text,
                "start": round(w["start"], 3) if "start" in w else None,
                "end":   round(w["end"], 3) if "end" in w else None,
                "prob":  round(w["score"], 3) if "score" in w else None,
            })
    return words


def timed(words):
    return [w for w in words if w["start"] is not None and w["end"] is not None]


def extract_pauses(words, min_pause=PAUSE_THRESHOLD_SEC):
    """หาช่วงว่างระหว่างคำ (only words with timestamps)"""
    tw = timed(words)
    pauses = []
    for i in range(1, len(tw)):
        gap = tw[i]["start"] - tw[i - 1]["end"]
        if gap >= min_pause:
            pauses.append({
                "start": round(tw[i - 1]["end"], 3),
                "end":   round(tw[i]["start"],   3),
                "dur":   round(gap, 3),
            })
    return pauses


def count_fillers(words):
    """นับ filler words (single and multi-word) + breakdown"""
    tokens = [w["word"].lower().strip(PUNCT) for w in words]
    singles = {f for f in FILLER_WORDS if " " not in f}
    multis = [f.split() for f in FILLER_WORDS if " " in f]

    breakdown, total, i = {}, 0, 0
    while i < len(tokens):
        hit = None
        for m in multis:
            if tokens[i:i + len(m)] == m:
                hit = " ".join(m)
                i += len(m)
                break
        if hit is None and tokens[i] in singles:
            hit = tokens[i]
            i += 1
        elif hit is None:
            i += 1
            continue
        breakdown[hit] = breakdown.get(hit, 0) + 1
        total += 1
    return total, breakdown


def build_features(segments, words, duration_sec):
    n_words = len(words)

    # ---- speech rate ----
    speech_rate_wpm = n_words / duration_sec * 60.0 if duration_sec > 0 else 0.0

    # ---- articulation rate (ไม่นับ pause) ----
    tw = timed(words)
    speaking_time = sum(w["end"] - w["start"] for w in tw)
    articulation_wpm = len(tw) / speaking_time * 60.0 if speaking_time > 0 else 0.0

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
# PROCESS ONE CLIP
# ============================================================

def process_clip(key: str, data: dict) -> dict:
    segments = data.get("segments", [])
    words = extract_words(segments)
    duration = clip_duration(key)

    transcript = " ".join(str(s.get("text", "")).strip() for s in segments).strip()
    seg_summary = [
        {"start": round(s.get("start", 0.0), 3),
         "end":   round(s.get("end", 0.0), 3),
         "text":  str(s.get("text", "")).strip()}
        for s in segments
    ]

    feats = build_features(segments, words, duration)

    f = feats
    print(f"[OK  ] {key}")
    print(f"       words={f['word_count']}  wpm={f['speech_rate_wpm']}  "
          f"pauses={f['pause_count']} (long {f['long_pause_count']})  "
          f"fillers={f['filler_count']} ({f['filler_ratio']:.3f})")

    return {
        "key":        key,
        "language":   data.get("language"),
        "source":     "whisperx (output/evidence/{key}_transcript.json)",
        "transcript": transcript,
        "segments":   seg_summary,
        "words":      words,
        "features":   feats,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  SPEECH FEATURES  (from WhisperX transcript)")
    print("=" * 60)

    if not AUDIO_DIR.exists():
        print(f"[ERROR] ไม่พบโฟลเดอร์: {AUDIO_DIR}")
        print("        รัน base/preprocessing/extract_audio.py ก่อน")
        sys.exit(1)

    keys = sorted(p.stem for p in AUDIO_DIR.glob("*.wav"))
    if not keys:
        print(f"[ERROR] ไม่พบไฟล์ .wav ใน {AUDIO_DIR}")
        sys.exit(1)

    FEATURES_OUT.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Input  : {EVIDENCE_DIR.relative_to(PROJECT_ROOT)}/*_transcript.json")
    print(f"[INFO] Output : {FEATURES_OUT.relative_to(PROJECT_ROOT)}")
    print(f"[INFO] พบ {len(keys)} คลิป\n")

    done, missing = 0, []
    for key in keys:
        data = load_transcript(key)
        if data is None:
            missing.append(key)
            continue
        result = process_clip(key, data)
        out_path = FEATURES_OUT / f"{key}_speech.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        done += 1

    print("=" * 60)
    print(f"  DONE: {done} clip(s)")
    if missing:
        print(f"  [WARN] no WhisperX transcript for {len(missing)} clip(s) — "
              f"run base/measurement/speech/transcribe.py first:")
        for k in missing:
            print(f"         {k}")
    print("=" * 60)


if __name__ == "__main__":
    main()
