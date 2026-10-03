"""
validation/train_score_model.py   (work plan v3 §3.2, RQ2)

Learn a hireability score from Tier-1 measurements instead of hand-set bands,
and test whether visual features add anything over audio + text (RQ2).

Data
----
Participant-level features = mean over that participant's clips of
  - output/features/<key>.json           visual (gaze, head, hands, face)
  - output/features/<key>_speech.json    speech rate, pauses, fillers, length
  - output/evidence/<key>_prosody.json   pitch variation, intensity, jitter, HNR, silence
Target = AVI recruiter rating (default: hireability).

Splits (AVI's official subject-level split, so no participant leakage)
  train : participants in input/ground_truth/avi_train*_subset.csv
  dev   : input/ground_truth/avi_dev_subset.csv        (model choice is made here)
  test  : input/ground_truth/avi_eval_subset.csv       (only with --final, run ONCE)

Models (ridge regression, standardised features, alpha chosen by CV on train)
  rule        Approach 1 overall score as is (no training) — reference
  length      duration + word count + speaking time      — "longer answers score higher" baseline
  audio_text  all speech + prosody features (incl. length)
  visual      visual features only
  all         audio_text + visual                         — RQ2: does vision add over audio_text?

Excluded on purpose
  - frame counts / detection counts (proxies for length and detection quality)
  - mean pitch (mostly encodes gender -> would score people by gender)

Usage
  python validation/train_score_model.py                 # train on train, evaluate on dev
  python validation/train_score_model.py --final         # train on train+dev, evaluate on test ONCE

Outputs (report/)
  score_model_<dev|test>.csv        Spearman + 95% CI per model
  score_model_coefficients.csv      standardised weights of the 'all' model (which features matter)
"""

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base._paths import OUTPUT_DIR, GROUND_TRUTH_DIR, REPORT_DIR, SCORES_DIR
from base.dataset.avi import load_labels, parse_key, TARGETS

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

FEATURES_DIR = OUTPUT_DIR / "features"
EVIDENCE_DIR = OUTPUT_DIR / "evidence"

LENGTH = ["sp.duration_sec", "sp.word_count", "sp.speaking_sec"]

EXCLUDE_SUBSTR = ("valid_frames", "total_frames", "detection_stats", "rejected_outliers",
                  "hand_frames", "aspect_used", "segment_count")
EXCLUDE_EXACT = {"pr.pitch_mean_hz"}          # gender proxy

ALPHAS = np.logspace(-2, 4, 25)


# ============================================================
# FEATURES
# ============================================================

def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


def clip_features(questions=None) -> pd.DataFrame:
    rows = []
    for f in FEATURES_DIR.glob("*.json"):
        if f.stem.endswith("_speech"):
            continue
        avi = parse_key(f.stem)
        if avi is None or (questions and avi["question_no"] not in questions):
            continue
        row = {"key": f.stem, "participant_id": avi["participant_id"]}
        row.update({f"vis.{k}": v for k, v in _flat(json.loads(f.read_text(encoding="utf-8"))).items()})
        sp = FEATURES_DIR / f"{f.stem}_speech.json"
        if sp.exists():
            feats = json.loads(sp.read_text(encoding="utf-8")).get("features", {})
            row.update({f"sp.{k}": float(v) for k, v in feats.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)})
        pr = EVIDENCE_DIR / f"{f.stem}_prosody.json"
        if pr.exists():
            row.update({f"pr.{k}": v for k, v in _flat(json.loads(pr.read_text(encoding="utf-8"))).items()})
        sc = SCORES_DIR / f"{f.stem}.json"
        if sc.exists():
            row["rule.overall"] = json.loads(sc.read_text(encoding="utf-8")).get(
                "fusion", {}).get("overall_score", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def feature_groups(columns):
    usable = [c for c in columns
              if c.split(".")[0] in ("vis", "sp", "pr")
              and not any(s in c for s in EXCLUDE_SUBSTR) and c not in EXCLUDE_EXACT]
    audio_text = [c for c in usable if c.startswith(("sp.", "pr."))]
    visual = [c for c in usable if c.startswith("vis.")]
    return {
        "length": [c for c in LENGTH if c in usable],
        "audio_text": audio_text,
        "visual": visual,
        "all": audio_text + visual,
    }


def manifest_ids(pattern: str):
    ids = set()
    for p in glob.glob(str(GROUND_TRUTH_DIR / pattern)):
        ids |= set(pd.read_csv(p, dtype=str).participant_id)
    return ids


# ============================================================
# MODEL + METRICS
# ============================================================

def make_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=ALPHAS))


def spearman_ci(x, y, n_boot=2000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    rho = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.unique(x[i]).size > 1 and np.unique(y[i]).size > 1:
            boots.append(spearmanr(x[i], y[i]).statistic)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return rho, lo, hi


def paired_diff_ci(pred_a, pred_b, y, n_boot=2000, seed=0):
    """Bootstrap CI of rho(a, y) - rho(b, y) on the same participants."""
    pa, pb, y = map(np.asarray, (pred_a, pred_b, y))
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if np.unique(y[i]).size > 1:
            d.append(spearmanr(pa[i], y[i]).statistic - spearmanr(pb[i], y[i]).statistic)
    obs = spearmanr(pa, y).statistic - spearmanr(pb, y).statistic
    lo, hi = np.percentile(d, [2.5, 97.5])
    return obs, lo, hi


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="hireability", choices=list(TARGETS))
    ap.add_argument("--questions", type=int, nargs="+", default=None)
    ap.add_argument("--train", default="avi_train*_subset.csv", help="manifest glob for training")
    ap.add_argument("--dev", default="avi_dev_subset.csv")
    ap.add_argument("--test", default="avi_eval_subset.csv")
    ap.add_argument("--final", action="store_true",
                    help="train on train+dev and evaluate on the test manifest (do this ONCE)")
    args = ap.parse_args()

    clips = clip_features(args.questions)
    if clips.empty:
        sys.exit("[ERROR] no AVI feature files in output/features/")
    per = clips.drop(columns=["key"]).groupby("participant_id").mean(numeric_only=True)
    per["n_clips"] = clips.groupby("participant_id").size()
    labels = load_labels().set_index("participant_id")[[args.target]]
    data = per.join(labels, how="inner").dropna(subset=[args.target])

    train_ids, dev_ids, test_ids = (manifest_ids(args.train), manifest_ids(args.dev),
                                    manifest_ids(args.test))
    if args.final:
        fit_ids, eval_ids, eval_name = train_ids | dev_ids, test_ids, "test"
    else:
        fit_ids, eval_ids, eval_name = train_ids, dev_ids, "dev"
    fit = data[data.index.isin(fit_ids)]
    ev = data[data.index.isin(eval_ids)]

    print("=" * 70)
    print(f"  Score model — target={args.target}   fit n={len(fit)}   {eval_name} n={len(ev)}")
    print("=" * 70)
    if len(fit) < 20 or len(ev) < 10:
        sys.exit(f"[ERROR] need >= 20 training and >= 10 {eval_name} participants with features "
                 f"(have {len(fit)} / {len(ev)}). Process more clips first.")
    if eval_ids & fit_ids:
        sys.exit("[ERROR] evaluation participants overlap with training participants")

    groups = feature_groups(data.columns)
    y_fit, y_ev = fit[args.target].to_numpy(), ev[args.target].to_numpy()
    cv = KFold(5, shuffle=True, random_state=0)

    rows, preds, coef = [], {}, None
    if "rule.overall" in ev:
        r, lo, hi = spearman_ci(ev["rule.overall"].fillna(ev["rule.overall"].median()), y_ev)
        preds["rule"] = ev["rule.overall"].fillna(ev["rule.overall"].median()).to_numpy()
        rows.append({"model": "rule", "features": 0, "cv_spearman_fit": np.nan,
                     "spearman": r, "ci_low": lo, "ci_high": hi})
    for name, cols in groups.items():
        if not cols:
            continue
        m = make_model()
        cv_pred = cross_val_predict(m, fit[cols], y_fit, cv=cv)
        m.fit(fit[cols], y_fit)
        p = m.predict(ev[cols])
        preds[name] = p
        r, lo, hi = spearman_ci(p, y_ev)
        rows.append({"model": name, "features": len(cols),
                     "cv_spearman_fit": spearmanr(cv_pred, y_fit).statistic,
                     "spearman": r, "ci_low": lo, "ci_high": hi,
                     "alpha": m[-1].alpha_})
        if name == "all":
            coef = pd.Series(m[-1].coef_, index=cols).sort_values(key=abs, ascending=False)

    res = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORT_DIR / f"score_model_{eval_name}.csv", index=False)
    pd.set_option("display.width", 140)
    print(res.round(3).to_string(index=False))
    print()

    for a, b, label in (("all", "audio_text", "RQ2: visual adds over audio+text?"),
                        ("audio_text", "length", "audio+text beats answer length alone?"),
                        ("all", "rule", "learned model beats hand-set bands?")):
        if a in preds and b in preds:
            d, lo, hi = paired_diff_ci(preds[a], preds[b], y_ev)
            verdict = "yes" if lo > 0 else ("no (worse)" if hi < 0 else "not shown (CI includes 0)")
            print(f"{label:42s} Δρ = {d:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}] -> {verdict}")

    if coef is not None:
        coef.rename("std_weight").to_csv(REPORT_DIR / "score_model_coefficients.csv")
        print("\nTop features in 'all' model (standardised weights):")
        print(coef.head(10).round(3).to_string())

    print(f"\nSaved: {REPORT_DIR / f'score_model_{eval_name}.csv'}")
    if not args.final:
        print("Choose the model on dev. Run --final once at the end to report on test.")


if __name__ == "__main__":
    main()
