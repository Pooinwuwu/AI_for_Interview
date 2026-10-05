"""
approaches/approach_1b_learned/learned_model.py

Shared pieces of Approach 1b (learned overall score):
where the model is saved, how it is loaded, and how one clip is scored and explained.

Approach 1  = hand-set ideal bands + hand-set fusion weights   (no training)
Approach 1b = the same Tier-1 measurements, but the overall score comes from a ridge
              model trained on AVI recruiter ratings (report/model_decision.md:
              feature group 'audio_text'). Dimension bands are still Approach 1's,
              so per-dimension coaching is unchanged; only the overall score is learned.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd

from base._paths import MODELS_DIR, OUTPUT_DIR

MODEL_DIR = MODELS_DIR / "approach_1b"
MODEL_FILE = MODEL_DIR / "score_model.joblib"
CARD_FILE = MODEL_DIR / "model_card.json"
FIT_IDS_FILE = MODEL_DIR / "fit_participants.json"   # for the leakage check in evaluate_avi.py
SCORES_1B_DIR = OUTPUT_DIR / "scores_1b"

MODEL_VERSION = "1b-ridge-v1"

# percentile of the training participants' predicted scores -> band
PERCENTILE_BANDS = [(80, "ดีมาก", "Excellent"), (60, "ดี", "Good"), (40, "ปานกลาง", "Fair"),
                    (20, "ควรปรับ", "Needs work"), (0, "ควรปรับมาก", "Priority")]

OOD_Z = 3.0   # |z| above this = outside what the model saw in training

FEATURE_LABELS = {
    "sp.duration_sec":     ("ความยาวคลิป", "clip length"),
    "sp.speaking_sec":     ("เวลาที่พูดจริง", "time spent speaking"),
    "sp.word_count":       ("จำนวนคำ", "number of words"),
    "sp.speech_rate_wpm":  ("ความเร็วพูด (รวมช่วงหยุด)", "speech rate incl. pauses"),
    "sp.articulation_wpm": ("ความเร็วพูด (เฉพาะตอนพูด)", "articulation rate"),
    "sp.pause_count":      ("จำนวนครั้งที่หยุด", "number of pauses"),
    "sp.long_pause_count": ("จำนวนการหยุดนาน", "number of long pauses"),
    "sp.pause_total_sec":  ("เวลาหยุดรวม", "total pause time"),
    "sp.pause_mean_sec":   ("ความยาวเฉลี่ยของการหยุด", "mean pause length"),
    "sp.pause_max_sec":    ("การหยุดที่นานที่สุด", "longest pause"),
    "sp.filler_count":     ("จำนวนคำเติม (um, uh)", "filler count"),
    "sp.filler_ratio":     ("สัดส่วนคำเติม", "filler ratio"),
    "pr.pitch_std_hz":     ("ความขึ้นลงของน้ำเสียง", "pitch variation"),
    "pr.intensity_mean_db": ("ความดังของเสียง", "loudness"),
    "pr.jitter_local":     ("ความสั่นของเสียง", "voice jitter"),
    "pr.hnr_mean_db":      ("ความใสของเสียง", "voice clarity (HNR)"),
    "pr.silence_ratio":    ("สัดส่วนความเงียบ", "share of silence"),
}


# features are strongly correlated (e.g. pause count / pause time / silence), so single
# weights are not interpretable; explanations are given per group of related features
FEATURE_GROUPS = {
    "length":  ("ความยาวและปริมาณคำตอบ", "answer length",
                ["sp.duration_sec", "sp.speaking_sec", "sp.word_count"]),
    "pace":    ("จังหวะและการหยุดพูด", "pace and pauses",
                ["sp.speech_rate_wpm", "sp.articulation_wpm", "sp.pause_count", "sp.long_pause_count",
                 "sp.pause_total_sec", "sp.pause_mean_sec", "sp.pause_max_sec", "pr.silence_ratio"]),
    "fillers": ("คำเติม (um, uh)", "fillers", ["sp.filler_count", "sp.filler_ratio"]),
    "voice":   ("น้ำเสียง", "voice",
                ["pr.pitch_std_hz", "pr.intensity_mean_db", "pr.jitter_local", "pr.hnr_mean_db"]),
}


def label(feature: str, lang: str = "th") -> str:
    th, en = FEATURE_LABELS.get(feature, (feature, feature))
    return th if lang == "th" else en


def save(bundle: dict, card: dict, fit_ids):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)
    CARD_FILE.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    FIT_IDS_FILE.write_text(json.dumps(sorted(fit_ids), indent=1), encoding="utf-8")


def load():
    if not MODEL_FILE.exists():
        sys.exit(f"[ERROR] no trained model at {MODEL_FILE}\n"
                 f"        train it first:  python approaches/approach_1b_learned/train.py")
    return joblib.load(MODEL_FILE)


def fit_participants() -> set:
    if not FIT_IDS_FILE.exists():
        return set()
    return set(json.loads(FIT_IDS_FILE.read_text(encoding="utf-8")))


def band_for_percentile(pct: float):
    for cut, th, en in PERCENTILE_BANDS:
        if pct >= cut:
            return th, en
    return PERCENTILE_BANDS[-1][1:]


def score_row(bundle: dict, row: dict) -> dict:
    """Score one clip's feature row. Returns prediction, percentile, band, explanation.

    No score is given when features are missing (e.g. speech_text.py not run):
    a median-imputed answer would get a meaningless score.
    """
    feats = bundle["features"]
    pipe = bundle["pipeline"]
    X = pd.DataFrame([[row.get(f, np.nan) for f in feats]], columns=feats, dtype=float)
    missing = [f for f in feats if np.isnan(X.at[0, f])]
    if missing:
        return {"predicted_rating": None, "overall_score": None, "band": None,
                "raised_by": [], "lowered_by": [], "group_contributions": {},
                "missing_features": missing, "outside_training_range": [],
                "reliable": False,
                "note": "missing features - run the speech / prosody steps for this clip"}

    pred = float(pipe.predict(X)[0])
    ref = np.asarray(bundle["train_predictions"])
    pct = float((ref < pred).mean() * 100 + (ref == pred).mean() * 50)
    band_th, band_en = band_for_percentile(pct)

    # contribution = weight x standardised value, summed per feature group
    z = pipe[:-1].transform(X)[0]
    contrib = dict(zip(feats, pipe[-1].coef_ * z))
    groups = {}
    for gid, (th, en, members) in FEATURE_GROUPS.items():
        c = sum(contrib[f] for f in members if f in contrib)
        groups[gid] = {"label_th": th, "label_en": en, "contribution": round(float(c), 4)}
    ranked = sorted(groups.values(), key=lambda g: g["contribution"], reverse=True)
    ood = [{"feature": f, "label_th": label(f, "th"), "value": row.get(f), "z": round(float(zi), 2)}
           for f, zi in zip(feats, z) if abs(zi) > OOD_Z]

    return {
        "predicted_rating": round(pred, 3),
        "overall_score": round(pct, 1),
        "band": {"label_th": band_th, "label_en": band_en},
        "raised_by": [g for g in ranked if g["contribution"] > 0.01][:2],
        "lowered_by": [g for g in ranked[::-1] if g["contribution"] < -0.01][:2],
        "group_contributions": groups,
        "missing_features": [],
        "outside_training_range": ood,
        "reliable": not ood,
    }


def now():
    return datetime.now().isoformat(timespec="seconds")
