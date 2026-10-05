"""
approaches/approach_1b_learned/train.py

Train the Approach 1b overall-score model and save it to models/approach_1b/.

The model is the one fixed in report/model_decision.md:
  ridge regression (median impute -> standardise -> RidgeCV) on the 'audio_text'
  feature group, target = AVI hireability, one row per participant (mean over q1/q2 clips).

Which participants it learns from (--fit)
  train+dev  (default)  avi_train*_subset + avi_dev_subset  -> the system model.
             Same model as `train_score_model.py --final`, so the AVI test set
             stays untouched until that single final run.
  train      avi_train*_subset only -> lets you look at dev results for 1b
             (evaluate_avi.py --split val --approaches learned).

Never trains on the test manifest (avi_eval_subset.csv): the script refuses.

Outputs
  models/approach_1b/score_model.joblib     pipeline + feature list + reference distribution
  models/approach_1b/model_card.json        what it was trained on, CV result, top weights
  models/approach_1b/fit_participants.json  participant ids used (leakage check)

Usage
  python approaches/approach_1b_learned/train.py
  python approaches/approach_1b_learned/train.py --fit train
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, cross_val_predict

from base.dataset.avi import load_labels, TARGETS
from base.learned_features import clip_features, feature_groups, participant_table
from validation.train_score_model import make_model, manifest_ids
import learned_model as m1b

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

GROUP = "audio_text"          # fixed in report/model_decision.md


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", choices=["train+dev", "train"], default="train+dev")
    ap.add_argument("--target", default="hireability", choices=list(TARGETS))
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--train", default="avi_train*_subset.csv")
    ap.add_argument("--dev", default="avi_dev_subset.csv")
    ap.add_argument("--test", default="avi_eval_subset.csv")
    args = ap.parse_args()

    fit_ids = manifest_ids(args.train) | (manifest_ids(args.dev) if args.fit == "train+dev" else set())
    test_ids = manifest_ids(args.test)
    if fit_ids & test_ids:
        sys.exit("[ERROR] training participants overlap with the test manifest — refusing.")

    clips = clip_features(args.questions)
    if clips.empty:
        sys.exit("[ERROR] no AVI feature files in output/features/")
    per = participant_table(clips)
    labels = load_labels().set_index("participant_id")[[args.target]]
    data = per.join(labels, how="inner").dropna(subset=[args.target])
    fit = data[data.index.isin(fit_ids)]
    if len(fit) < 20:
        sys.exit(f"[ERROR] only {len(fit)} training participants with features (need >= 20).")

    feats = feature_groups(data.columns)[GROUP]
    X, y = fit[feats], fit[args.target].to_numpy()

    pipe = make_model()
    cv_pred = cross_val_predict(pipe, X, y, cv=KFold(5, shuffle=True, random_state=0))
    cv_rho = float(spearmanr(cv_pred, y).statistic)
    pipe.fit(X, y)

    coef = pd.Series(pipe[-1].coef_, index=feats).sort_values(key=abs, ascending=False)
    bundle = {
        "version": m1b.MODEL_VERSION,
        "pipeline": pipe,
        "features": feats,
        "group": GROUP,
        "target": args.target,
        # out-of-fold predictions = how new people's scores are spread -> percentile reference
        "train_predictions": np.sort(cv_pred).tolist(),
    }
    card = {
        "version": m1b.MODEL_VERSION,
        "created": m1b.now(),
        "model": "ridge (median impute, standardise, RidgeCV; alpha by efficient leave-one-out)",
        "feature_group": GROUP,
        "features": feats,
        "target": f"AVI {args.target} (recruiter rating, participant mean)",
        "fit": args.fit,
        "questions": args.questions,
        "n_participants": int(len(fit)),
        "alpha": float(pipe[-1].alpha_),
        "cv5_spearman": round(cv_rho, 3),
        "prediction_range_train": [round(float(cv_pred.min()), 3), round(float(cv_pred.max()), 3)],
        "top_weights_standardised": {k: round(float(v), 4) for k, v in coef.head(8).items()},
        "sklearn_version": sklearn.__version__,
        "notes": [
            "Overall score only; dimension bands still come from Approach 1.",
            "Score = percentile among AVI training participants, not an absolute grade.",
            "Trained on English answers to AVI q1/q2; other questions, languages or "
            "phone-recorded clips are outside the training data.",
            "Signal is weak and mostly answer length (report/model_decision.md).",
        ],
    }
    m1b.save(bundle, card, fit.index)

    print("=" * 66)
    print(f"  Approach 1b model trained   fit={args.fit}   n={len(fit)} participants")
    print("=" * 66)
    print(f"  features : {GROUP} ({len(feats)})")
    print(f"  alpha    : {pipe[-1].alpha_:.3g}")
    print(f"  5-fold CV Spearman on the training people: {cv_rho:.3f}")
    print("  top weights (standardised):")
    for k, v in coef.head(6).items():
        print(f"    {v:+.3f}  {m1b.label(k)}  ({k})")
    print(f"\nSaved: {m1b.MODEL_FILE}")
    print(f"       {m1b.CARD_FILE}")
    print("Next : python approaches/approach_1b_learned/score.py")


if __name__ == "__main__":
    main()
