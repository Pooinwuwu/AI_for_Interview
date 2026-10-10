"""
approaches/doctor/doctor.py   (approach A: a model trained on recruiter ratings picks what to coach)

Three ridge models learn AVI hireability from Tier-1 measurements of AVI train + val people
(test people are never used):
  audio   length, pauses, pace                 (~516 people: speech pipeline)
  text    transcript embedding = content       (~516 people)
  visual  gaze, head, hands, face              (~130 people: only they have landmarks)
For one clip, each feature's contribution = weight x (standardised value); contributions are summed
per ISSUE. An issue's deficit = how far it pulls the predicted rating below average. Each model's
deficits are multiplied by its cross-validated Spearman with recruiters (0 if it does not predict),
so a model that does not predict recruiter ratings cannot pick the issue. The top issue is what
the coach should talk about first.

Issue codes are the same as report/issue_labels_GUIDE.md (content = no_example or off_topic).

Usage
  python approaches/doctor/doctor.py train --text-emb minilm
  python approaches/doctor/doctor.py diagnose --keys own_01_q1 own_02_q1
  python approaches/doctor/doctor.py diagnose --manifest avi_eval_subset.csv   # AVI test clips
Model: models/doctor/doctor.joblib (contains AVI participant-level statistics: keep it local,
never publish - AVI6 agreement item 7).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, cross_val_predict

from base._paths import GROUND_TRUTH_DIR, REPORT_DIR
from base.dataset.avi import load_labels, parse_key
from base.learned_features import (clip_features, clip_row, participant_table,
                                   text_embedding_columns)
from validation.train_score_model import ALPHAS, ALPHAS_EMB, has_modalities, make_model

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_PATH = ROOT / "models" / "doctor" / "doctor.joblib"
TARGET = "hireability"

# feature -> issue (fillers are not measured; voice/pitch features are left out on purpose:
# mean pitch encodes gender, and voice quality is not something the coach advises on)
ISSUE_FEATURES = {
    "too_short":   ["sp.duration_sec", "sp.speaking_sec", "sp.word_count"],
    # per minute / share of the clip, not counts: raw counts grow with answer length and the
    # model then learned "more pauses = longer answer = better" (doctor v1 bug)
    "long_pauses": ["d.long_pause_per_min", "d.pause_share", "sp.pause_mean_sec",
                    "sp.pause_max_sec", "pr.silence_ratio"],
    "pace":        ["sp.speech_rate_wpm", "sp.articulation_wpm"],
    "eye_contact": ["vis.gaze.eye_contact_ratio", "vis.gaze.gaze_stability_x",
                    "vis.gaze.gaze_stability_y", "vis.gaze.no_face_ratio"],
    "head_posture": ["vis.head_pose.mean_pitch", "vis.head_pose.yaw_std", "vis.head_pose.pitch_std",
                     "vis.head_pose.roll_std", "vis.head_pose.yaw_range", "vis.head_pose.pitch_range"],
    "hands":       ["vis.hand_gesture.hand_presence_ratio", "vis.hand_gesture.movement_speed_mean",
                    "vis.hand_gesture.movement_speed_std", "vis.hand_gesture.fidget_x",
                    "vis.hand_gesture.fidget_y"],
    "facial_expression": ["vis.facial_expression.smile_ratio", "vis.facial_expression.expressiveness",
                          "vis.facial_expression.mouth_aspect_ratio_std",
                          "vis.facial_expression.brow_height_std"],
}
MODELS = {"audio": ["too_short", "long_pauses", "pace"],
          "visual": ["eye_contact", "head_posture", "hands", "facial_expression"]}
CONTENT = "content"            # from the text model; matches no_example / off_topic labels
MIN_DEFICIT = 0.02             # below this (rating points x weight) -> "none"
DOCTOR_VERSION = "doctor-v2"


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Length-free pause measures (computed after averaging, on the same rows that are scored)."""
    df = df.copy()
    minutes = df["sp.duration_sec"].where(df["sp.duration_sec"] > 0) / 60
    df["d.long_pause_per_min"] = df.get("sp.long_pause_count") / minutes
    df["d.pause_share"] = df.get("sp.pause_total_sec") / (minutes * 60)
    return df


# ============================================================
# TRAIN
# ============================================================

def training_pool(text_emb):
    labels = load_labels()
    locked = set()
    p = GROUND_TRUTH_DIR / "avi_eval_subset.csv"
    if p.exists():
        locked = set(pd.read_csv(p, dtype=str).participant_id)
    pool_ids = set(labels.participant_id[labels.split.isin(["train", "val"])]) - locked
    per = participant_table(clip_features([1, 2], text_emb, None))
    lab = labels.set_index("participant_id")[[TARGET]]
    data = add_derived(per.join(lab, how="inner").dropna(subset=[TARGET]))
    return data[data.index.isin(pool_ids)]


def fit_one(data, cols, alphas, seed=0):
    d = data[has_modalities(data, cols)]
    y = d[TARGET].to_numpy()
    m = make_model(alphas)
    cv = cross_val_predict(m, d[cols], y, cv=KFold(5, shuffle=True, random_state=seed))
    rho = float(spearmanr(cv, y).statistic)
    m.fit(d[cols], y)
    return {"model": m, "cols": cols, "cv_rho": rho, "weight": max(rho, 0.0), "n": len(d),
            "mean_pred": float(np.mean(m.predict(d[cols])))}


def train(args):
    data = training_pool(args.text_emb)
    if len(data) < 40:
        sys.exit(f"[ERROR] only {len(data)} training people with features")
    parts = {}
    for name, issues in MODELS.items():
        cols = [c for i in issues for c in ISSUE_FEATURES[i] if c in data.columns]
        if cols:
            parts[name] = fit_one(data, cols, ALPHAS)
    txt = sorted((c for c in data.columns if c.startswith("txt.")), key=lambda c: int(c[4:]))
    if txt:
        parts["text"] = fit_one(data, txt, ALPHAS_EMB)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"parts": parts, "text_emb": args.text_emb, "target": TARGET,
                 "version": DOCTOR_VERSION}, MODEL_PATH)
    card = {n: {"n_people": p["n"], "cv_spearman": round(p["cv_rho"], 3),
                "weight": round(p["weight"], 3), "features": len(p["cols"])}
            for n, p in parts.items()}
    (MODEL_PATH.parent / "model_card.json").write_text(
        json.dumps({"version": DOCTOR_VERSION, "text_emb": args.text_emb, "target": TARGET,
                    "parts": card}, indent=2),
        encoding="utf-8")
    print("Doctor trained on AVI train + val people (test people excluded):")
    for n, c in card.items():
        print(f"  {n:7s} people={c['n_people']:4d}  CV Spearman={c['cv_spearman']:+.3f}  "
              f"weight={c['weight']:.3f}  features={c['features']}")
    print(f"Saved {MODEL_PATH}  (local only - do not publish)")


# ============================================================
# DIAGNOSE
# ============================================================

def _contrib(part, row: pd.DataFrame) -> pd.Series:
    """Per-feature contribution to the predicted rating (rating points, relative to average)."""
    imp, sc, ridge = part["model"][0], part["model"][1], part["model"][-1]
    z = sc.transform(imp.transform(row[part["cols"]]))[0]
    return pd.Series(z * ridge.coef_, index=part["cols"])


def diagnose_row(doc, row: pd.DataFrame) -> dict:
    deficits = {}
    for name, issues in MODELS.items():
        part = doc["parts"].get(name)
        if not part or not has_modalities(row, part["cols"]).iloc[0]:
            continue
        c = _contrib(part, row)
        for i in issues:
            cols = [f for f in ISSUE_FEATURES[i] if f in c.index]
            deficits[i] = -float(c[cols].sum()) * part["weight"]
    part = doc["parts"].get("text")
    if part and has_modalities(row, part["cols"]).iloc[0]:
        pred = float(part["model"].predict(row[part["cols"]])[0])
        deficits[CONTENT] = -(pred - part["mean_pred"]) * part["weight"]
    ranked = sorted(deficits.items(), key=lambda kv: -kv[1])
    top = [k for k, v in ranked if v > MIN_DEFICIT]
    return {"top_issue": top[0] if top else "none",
            "second_issue": top[1] if len(top) > 1 else "",
            "deficits": {k: round(v, 4) for k, v in ranked}}


def clip_frame(keys, text_emb):
    vecs = text_embedding_columns(text_emb) if text_emb else {}
    return add_derived(pd.DataFrame([{**clip_row(k), **vecs.get(k, {})} for k in keys]).set_index("key"))


# ---------- rule doctor (baseline: fixed thresholds, no training, cannot judge content)

RULES = [  # issue, feature, bad if below / above, threshold
    ("too_short", "sp.word_count", "below", 80),
    ("long_pauses", "sp.long_pause_count", "above", 2),
    ("pace", "sp.speech_rate_wpm", "below", 100),
    ("pace", "sp.speech_rate_wpm", "above", 180),
    ("eye_contact", "vis.gaze.eye_contact_ratio", "below", 0.6),
    ("head_posture", "vis.head_pose.yaw_std", "above", 15),
    ("facial_expression", "vis.facial_expression.smile_ratio", "below", 0.05),
]


def rule_diagnose(row: dict) -> dict:
    sev = {}
    for issue, feat, side, thr in RULES:
        v = row.get(feat)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        s = (thr - v) / abs(thr) if side == "below" else (v - thr) / abs(thr)
        if s > 0:
            sev[issue] = max(sev.get(issue, 0), s)
    ranked = sorted(sev, key=lambda k: -sev[k])
    return {"top_issue": ranked[0] if ranked else "none",
            "second_issue": ranked[1] if len(ranked) > 1 else ""}


def diagnose(args):
    if not MODEL_PATH.exists():
        sys.exit("[ERROR] train first:  python approaches/doctor/doctor.py train")
    doc = joblib.load(MODEL_PATH)
    if doc.get("version") != DOCTOR_VERSION:
        sys.exit("[ERROR] the saved doctor is an older version - run the train command again")
    keys = list(args.keys or [])
    if args.manifest:
        for r in csv.DictReader(open(GROUND_TRUTH_DIR / args.manifest, encoding="utf-8")):
            keys += r["clip_keys"].split()
    frame = clip_frame(keys, doc["text_emb"])
    rows = []
    for key in frame.index:
        row = frame.loc[[key]]
        d = diagnose_row(doc, row)
        r = rule_diagnose(frame.loc[key].to_dict())
        rows.append({"key": key, "doctor_top": d["top_issue"], "doctor_second": d["second_issue"],
                     "rule_top": r["top_issue"], "rule_second": r["second_issue"],
                     "deficits": json.dumps(d["deficits"])})
        if not args.quiet:
            print(f"  {key:40s} doctor: {d['top_issue']:18s} rule: {r['top_issue']}")
    out = REPORT_DIR / (args.out or "doctor_diagnoses.csv")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--text-emb", default="minilm")
    d = sub.add_parser("diagnose")
    d.add_argument("--keys", nargs="+")
    d.add_argument("--manifest")
    d.add_argument("--out")
    d.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    train(args) if args.cmd == "train" else diagnose(args)


if __name__ == "__main__":
    main()
