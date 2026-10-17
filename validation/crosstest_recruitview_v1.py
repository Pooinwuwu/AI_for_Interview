"""
validation/crosstest_recruitview.py   (cross-dataset test: AVI-trained models on RecruitView)

Fits each audio/text model on AVI train + val people (hireability), then scores the RecruitView
sample (run_recruitview.py) and compares with the psychologists' scores. No model is trained on
RecruitView; it is only used to measure. Visual models are left out (no landmarks for RecruitView).

Also reports how strongly answer length alone follows each RecruitView score.

Usage
  python validation/crosstest_recruitview.py --text-emb minilm bge-small e5-base
Output
  report/crosstest_recruitview.csv          model x RecruitView score: Spearman + 95% CI
  report/crosstest_recruitview_length.csv   raw length features vs each score
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from base._paths import GROUND_TRUTH_DIR, REPORT_DIR
from base.dataset.avi import load_labels
from base.learned_features import (LENGTH, clip_features, clip_row, feature_groups,
                                   participant_table, text_embedding_columns)
from validation.train_score_model import (ALPHAS, ALPHAS_EMB, has_modalities, make_model,
                                          spearman_ci)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SAMPLE = GROUND_TRUTH_DIR / "recruitview_sample.csv"
RV_TARGETS = ["interview_score", "overall_performance", "answer_score", "speaking_skills",
              "confidence_score", "facial_expression"]
GROUPS = ["length", "audio_text", "text_emb", "text_emb+audio_text"]


def avi_pool(emb, target):
    labels = load_labels()
    pool_ids = set(labels.participant_id[labels.split.isin(["train", "val"])])
    per = participant_table(clip_features([1, 2], emb, None))
    lab = labels.set_index("participant_id")[[target]]
    data = per.join(lab, how="inner").dropna(subset=[target])
    return data[data.index.isin(pool_ids)]


def rv_table(emb):
    sample = pd.read_csv(SAMPLE, dtype={"user_no": str})
    vecs = text_embedding_columns(emb) if emb else {}
    rows = [{**clip_row(k), **vecs.get(k, {})} for k in sample.key]
    feats = pd.DataFrame(rows).set_index("key")
    return feats.join(sample.set_index("key")[RV_TARGETS], how="inner")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text-emb", nargs="+", default=["minilm"])
    ap.add_argument("--target", default="hireability", help="AVI target the models learn")
    args = ap.parse_args()
    if not SAMPLE.exists():
        sys.exit("[ERROR] run  python run_recruitview.py  first")

    rows = []
    for k, emb in enumerate(args.text_emb or [None]):
        fit = avi_pool(emb, args.target)
        rv = rv_table(emb)
        groups = feature_groups(fit.columns)
        for name in GROUPS:
            if k > 0 and not name.startswith("text_emb"):
                continue                                   # length / audio once is enough
            cols = [c for c in groups.get(name, []) if c in rv.columns]
            if not cols:
                continue
            f = fit[has_modalities(fit, cols)]
            r = rv[has_modalities(rv, cols)]
            m = make_model(ALPHAS_EMB if name.startswith("text_emb") else ALPHAS)
            m.fit(f[cols], f[args.target])
            pred = m.predict(r[cols])
            label = name if (k == 0 or not name.startswith("text_emb")) \
                else name.replace("text_emb", f"text_emb[{emb}]")
            for t in RV_TARGETS:
                rho, lo, hi = spearman_ci(pred, r[t].to_numpy())
                rows.append({"model": label, "rv_score": t, "n_fit_avi": len(f), "n_rv": len(r),
                             "spearman": rho, "ci_low": lo, "ci_high": hi})
            print(f"  {label:32s} AVI n={len(f):3d}  RecruitView n={len(r):3d}  "
                  f"interview_score rho={rows[-len(RV_TARGETS)]['spearman']:.3f}")

    res = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORT_DIR / "crosstest_recruitview.csv", index=False)
    pd.set_option("display.width", 160)
    print("\nAVI-trained models on RecruitView (Spearman, rows = model, columns = RecruitView score)")
    print(res.pivot(index="model", columns="rv_score", values="spearman")[RV_TARGETS].round(3).to_string())

    rv = rv_table(None)
    lrows = []
    for c in [c for c in LENGTH if c in rv]:
        d = rv.dropna(subset=[c])
        for t in RV_TARGETS:
            rho, lo, hi = spearman_ci(d[c].to_numpy(), d[t].to_numpy())
            lrows.append({"feature": c, "rv_score": t, "n": len(d), "spearman": rho,
                          "ci_low": lo, "ci_high": hi})
    lres = pd.DataFrame(lrows)
    lres.to_csv(REPORT_DIR / "crosstest_recruitview_length.csv", index=False)
    print("\nAnswer length alone vs RecruitView scores (Spearman)")
    print(lres.pivot(index="feature", columns="rv_score", values="spearman")[RV_TARGETS].round(3).to_string())
    print("\nNote: RecruitView clips are ~30 s and cover 76 questions; AVI clips are 1-2 min on 2 "
          "questions, so lower numbers are expected. Report both datasets side by side.")


if __name__ == "__main__":
    main()
