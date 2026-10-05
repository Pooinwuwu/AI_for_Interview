"""
validation/validate_gaze.py   (Tier-1 accuracy: is the gaze measure right?)

Compares output/evidence/<key>_gaze_events.json with labels a person gives while
watching the video. Use your own / consented clips (e.g. the pilot videos).

1. Make a labelling sheet (one row per WINDOW seconds):
     python validation/validate_gaze.py --make-template pilot_01 pilot_02 pilot_03 pilot_04 pilot_05
   -> report/gaze_labels.csv  (does not overwrite an existing sheet)
2. Watch each clip and fill the 'label' column for every window:
     camera   looking at the camera (lens) for most of the window
     away     looking elsewhere (screen, notes, around the room) for most of the window
     unclear  can't tell / face not visible  (ignored in the score)
   Label BEFORE looking at the system's output, so the labels stay independent.
3. Score:
     python validation/validate_gaze.py --score
   -> accuracy, Cohen's kappa, confusion table, per clip; report/gaze_validation.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base._paths import EVIDENCE_DIR, REPORT_DIR, METADATA_FILE

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SHEET = REPORT_DIR / "gaze_labels.csv"
WINDOW = 2.0
SYSTEM_TO_LABEL = {"eye_contact": "camera", "looking_away": "away"}


def _duration(key: str) -> float:
    ev = json.loads((EVIDENCE_DIR / f"{key}_gaze_events.json").read_text(encoding="utf-8"))
    d = max((e["end"] for e in ev), default=0.0)
    try:
        meta = json.loads(METADATA_FILE.read_text(encoding="utf-8")).get(key, {})
        d = max(d, float(meta.get("duration_sec") or 0))
    except Exception:
        pass
    return d


def make_template(keys):
    if SHEET.exists():
        sys.exit(f"[STOP] {SHEET} already exists - not overwriting your labels.")
    rows = []
    for key in keys:
        t, end = 0.0, _duration(key)
        while t < end - 0.5:
            rows.append({"key": key, "start": round(t, 1), "end": round(min(t + WINDOW, end), 1),
                         "label": "", "notes": ""})
            t += WINDOW
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SHEET, index=False, encoding="utf-8-sig")
    print(f"Saved {SHEET}  ({len(rows)} windows of {WINDOW:.0f}s). Fill 'label': camera / away / unclear")


def system_label(events, t0, t1):
    """Majority system state inside the window (None if no face most of the time)."""
    share = {}
    for e in events:
        ov = min(e["end"], t1) - max(e["start"], t0)
        if ov > 0:
            share[e["event"]] = share.get(e["event"], 0) + ov
    if not share:
        return None
    top = max(share, key=share.get)
    return SYSTEM_TO_LABEL.get(top)


def score():
    if not SHEET.exists():
        sys.exit(f"[ERROR] {SHEET} not found - run --make-template first")
    lab = pd.read_csv(SHEET, dtype={"label": str}).fillna("")
    lab["label"] = lab["label"].str.strip().str.lower()
    lab = lab[lab.label.isin(["camera", "away"])].copy()
    if lab.empty:
        sys.exit("[ERROR] no windows labelled camera / away yet")
    cache = {}
    sys_lab = []
    for r in lab.itertuples():
        if r.key not in cache:
            p = EVIDENCE_DIR / f"{r.key}_gaze_events.json"
            cache[r.key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        sys_lab.append(system_label(cache[r.key], r.start, r.end))
    lab["system"] = sys_lab
    lab = lab.dropna(subset=["system"])
    lab["correct"] = lab.label == lab.system

    print("=" * 60)
    print(f"  Gaze measure vs human labels   windows={len(lab)}")
    print("=" * 60)
    acc = lab.correct.mean()
    kappa = cohen_kappa_score(lab.label, lab.system) if lab.system.nunique() > 1 or lab.label.nunique() > 1 else float("nan")
    print(f"accuracy {acc:.1%}   Cohen's kappa {kappa:.2f}")
    cm = pd.DataFrame(confusion_matrix(lab.label, lab.system, labels=["camera", "away"]),
                      index=["human camera", "human away"], columns=["system camera", "system away"])
    print(cm.to_string())
    per = lab.groupby("key").agg(windows=("correct", "size"), accuracy=("correct", "mean"),
                                 human_away=("label", lambda s: (s == "away").mean()),
                                 system_away=("system", lambda s: (s == "away").mean()))
    print("\nPer clip:")
    print(per.round(2).to_string())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per.to_csv(REPORT_DIR / "gaze_validation.csv")
    lab.to_csv(REPORT_DIR / "gaze_validation_windows.csv", index=False)
    print(f"\nSaved {REPORT_DIR / 'gaze_validation.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-template", nargs="+", metavar="KEY")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.make_template:
        make_template(args.make_template)
    elif args.score:
        score()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
