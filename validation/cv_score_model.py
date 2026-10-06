"""
validation/cv_score_model.py   (RQ1, RQ2 — model choice without touching test)

Repeated k-fold cross-validation on the training pool = train + dev people (130 in the
subset track). The test people are never loaded for fitting or scoring here; they stay
locked for train_score_model.py --final.

Why: dev alone is 30 people, so one dev number moves by +-0.05 just from who is in it.
Cross-validation scores every pool person once per repeat (out-of-fold), and repeating
with different folds shows how much the number moves.

What it reports (report/cv_*.csv)
  cv_score_model<suffix>.csv          per model: mean / sd of Spearman over repeats,
                                      + bootstrap 95% CI on the repeat-averaged predictions
  cv_score_model<suffix>_comparisons.csv   paired delta-rho with 95% CI (same people)
  cv_learning_curve<suffix>.csv       RQ2: train on n random people of each training fold
  cv_m5_length<suffix>.csv            does the M5 rating mostly track answer length?

Models = the ones in train_score_model.py, plus
  llm+length          M5 rating + the 3 length features (does M5 add over length?)
  text_emb[<name>]    one M3 / M4 set per embedding given with --text-emb

Usage
  python validation/cv_score_model.py --text-emb minilm --llm qwen2.5:3b
  python validation/cv_score_model.py --text-emb minilm bge-small e5-base --llm qwen2.5:3b
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.model_selection import KFold

from base._paths import REPORT_DIR
from base.dataset.avi import load_labels, TARGETS
from base.learned_features import LENGTH, clip_features, feature_groups, participant_table
from validation.train_score_model import (ALPHAS, ALPHAS_EMB, has_modalities, make_model,
                                          paired_diff_ci, spearman_ci, split_ids)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LC_SIZES = [25, 50, 75, 100, 200, 300]   # sizes below the training fold, + the whole fold
LC_GROUPS = ["length", "audio_text", "visual", "text_emb", "llm+length"]


# ============================================================
# CROSS-VALIDATION
# ============================================================

def _alphas(cols):
    return ALPHAS_EMB if any(c.startswith("txt.") for c in cols) else ALPHAS


def cv_model(cols, pool, target, folds, repeats):
    """Out-of-fold predictions for every pool person with the needed modalities.
    Returns (repeat-averaged predictions as a Series, Spearman per repeat)."""
    sub = pool[has_modalities(pool, cols)]
    X, y = sub[cols], sub[target].to_numpy()
    all_p, rhos = [], []
    for r in range(repeats):
        p = np.empty(len(sub))
        for tr, te in KFold(folds, shuffle=True, random_state=r).split(sub):
            m = make_model(_alphas(cols))
            m.fit(X.iloc[tr], y[tr])
            p[te] = m.predict(X.iloc[te])
        all_p.append(p)
        rhos.append(spearmanr(p, y).statistic)
    return pd.Series(np.mean(all_p, axis=0), index=sub.index), np.array(rhos)


def cv_learning_curve(name, cols, pool, target, folds, repeats):
    """Each training fold is cut down to n random people; the held-out fold is unchanged,
    so every size is scored on the same people."""
    sub = pool[has_modalities(pool, cols)]
    X, y = sub[cols], sub[target].to_numpy()
    full = len(sub) - int(np.ceil(len(sub) / folds))          # smallest training fold
    rows = []
    for n in [s for s in LC_SIZES if s < full] + [None]:
        for r in range(repeats):
            rng = np.random.default_rng(1000 + r)
            p = np.empty(len(sub))
            for tr, te in KFold(folds, shuffle=True, random_state=r).split(sub):
                use = tr if n is None else rng.choice(tr, size=n, replace=False)
                m = make_model(_alphas(cols))
                m.fit(X.iloc[use], y[use])
                p[te] = m.predict(X.iloc[te])
            rows.append({"model": name, "n_train": n or full, "repeat": r,
                         "spearman": spearmanr(p, y).statistic})
    return rows


def partial_spearman(x, y, z):
    """Spearman of x and y after removing (rank-)linear effects of the columns in z."""
    rx, ry = rankdata(x), rankdata(y)
    Z = np.column_stack([np.ones(len(rx))] + [rankdata(c) for c in z])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return spearmanr(res(rx), res(ry)).statistic


def m5_length_check(pool, target):
    rows = []
    d = pool.dropna(subset=["llm.rating"] + [c for c in LENGTH if c in pool])
    if len(d) < 10:
        return rows
    for c in [c for c in LENGTH if c in d]:
        rows.append({"check": f"M5 rating vs {c}", "n": len(d),
                     "spearman": spearmanr(d["llm.rating"], d[c]).statistic})
        rows.append({"check": f"human rating vs {c}", "n": len(d),
                     "spearman": spearmanr(d[target], d[c]).statistic})
    lens = [d[c].to_numpy() for c in LENGTH if c in d]
    rows.append({"check": "M5 rating vs human rating", "n": len(d),
                 "spearman": spearmanr(d["llm.rating"], d[target]).statistic})
    rows.append({"check": "M5 vs human, length held constant (partial)", "n": len(d),
                 "spearman": partial_spearman(d["llm.rating"], d[target], lens)})
    return rows


# ============================================================
# MAIN
# ============================================================

def load_pool(args, text_emb, labels_all, pool_ids):
    clips = clip_features(args.questions, text_emb, args.llm)
    per = participant_table(clips)
    labels = labels_all.set_index("participant_id")[[args.target]]
    data = per.join(labels, how="inner").dropna(subset=[args.target])
    return data[data.index.isin(pool_ids)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="hireability", choices=list(TARGETS))
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--data", choices=["subset", "full"], default="subset")
    ap.add_argument("--train", default="avi_train*_subset.csv")
    ap.add_argument("--dev", default="avi_dev_subset.csv")
    ap.add_argument("--test", default="avi_eval_subset.csv")
    ap.add_argument("--text-emb", nargs="+", default=[],
                    help="one or more embedding names (output/embeddings/text_<name>.npz)")
    ap.add_argument("--llm", default=None, help="M5 ratings to add (e.g. qwen2.5:3b)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--no-learning-curve", action="store_true")
    args = ap.parse_args()

    labels_all = load_labels()
    train_ids, dev_ids, test_ids = split_ids(
        SimpleNamespace(data=args.data, train=args.train, dev=args.dev, test=args.test), labels_all)
    pool_ids = (train_ids | dev_ids) - test_ids
    if (train_ids | dev_ids) & test_ids:
        sys.exit("[ERROR] train/dev manifests overlap with test - fix the manifests first")

    base = load_pool(args, None, labels_all, pool_ids)
    if test_ids & set(base.index):
        sys.exit("[ERROR] a test participant reached the pool")
    if len(base) < 40:
        sys.exit(f"[ERROR] only {len(base)} pool participants with features")
    target = args.target

    # model name -> (feature columns, pool table)
    models = {}
    for name, cols in feature_groups(base.columns).items():
        if cols and not name.startswith("text_emb"):
            models[name] = (cols, base)
    if "llm.rating" in base:
        models["llm+length"] = (["llm.rating"] + [c for c in LENGTH if c in base], base)
    for k, emb in enumerate(args.text_emb):
        pool_e = load_pool(args, emb, labels_all, pool_ids)
        g = feature_groups(pool_e.columns)
        for name in ("text_emb", "text_emb+audio_text", "text_emb+all"):
            if g.get(name):
                key = name if k == 0 else name.replace("text_emb", f"text_emb[{emb}]")
                models[key] = (g[name], pool_e)
    first_emb = args.text_emb[0] if args.text_emb else None

    print("=" * 84)
    print(f"  Repeated CV on train+dev  pool n={len(base)}  folds={args.folds}  "
          f"repeats={args.repeats}  target={target}  (test people: not loaded)")
    if first_emb:
        print(f"  text_emb = {first_emb}; other embeddings are named text_emb[<name>]")
    print("=" * 84)

    rows, preds, y_all = [], {}, base[target]
    for col, name in (("rule.overall", "rule"), ("llm.rating", "llm_zero_shot")):
        if col in base:                                   # no training: the value is the score
            e = base[base[col].notna()]
            preds[name] = e[col]
            r, lo, hi = spearman_ci(e[col], e[target])
            rows.append({"model": name, "features": 0, "n": len(e), "cv_mean": r, "cv_sd": 0.0,
                         "ci_low": lo, "ci_high": hi})
    for name, (cols, pool) in models.items():
        p, rhos = cv_model(cols, pool, target, args.folds, args.repeats)
        preds[name] = p
        r, lo, hi = spearman_ci(p.to_numpy(), pool.loc[p.index, target].to_numpy())
        rows.append({"model": name, "features": len(cols), "n": len(p),
                     "cv_mean": rhos.mean(), "cv_sd": rhos.std(ddof=1),
                     "ci_low": lo, "ci_high": hi})
        print(f"  {name:32s} n={len(p):3d}  rho = {rhos.mean():.3f} (sd {rhos.std(ddof=1):.3f})")

    res = pd.DataFrame(rows).sort_values("cv_mean", ascending=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ((f"_{args.data}" if args.data != "subset" else "")
              + (f"_{'+'.join(args.text_emb)}" if args.text_emb else "")
              + (f"_llm-{args.llm.replace(':', '_')}" if args.llm else ""))
    res.to_csv(REPORT_DIR / f"cv_score_model{suffix}.csv", index=False)
    pd.set_option("display.width", 160)
    print("\n" + res.round(3).to_string(index=False) + "\n")

    comps = [("all", "audio_text", "RQ2: visual adds over audio+text?"),
             ("audio_text", "length", "M2 audio+text beats length?"),
             ("visual", "length", "M2 visual beats length?"),
             ("audio_text", "rule", "M2 learned beats hand-set rules?"),
             ("text_emb", "length", "M3 embedding beats length?"),
             ("text_emb+audio_text", "audio_text", "M4: embedding adds over M2?"),
             ("text_emb+all", "text_emb+audio_text", "RQ2 (M4): visual adds?"),
             ("llm_zero_shot", "length", "M5 zero-shot beats length?"),
             ("llm+length", "length", "M5 adds over length?"),
             ("llm+audio_text", "audio_text", "M5 adds over M2?"),
             ("text_emb", "llm_zero_shot", "M3 trained beats M5 zero-shot?")]
    comps += [(f"text_emb[{e}]", "text_emb", f"{e} beats {first_emb}?")
              for e in args.text_emb[1:]]
    comp = []
    for a, b, label in comps:
        if a in preds and b in preds:
            d, lo, hi, n = paired_diff_ci(preds[a], preds[b], y_all)
            verdict = "yes" if lo > 0 else ("no (worse)" if hi < 0 else "not shown (CI includes 0)")
            comp.append({"a": a, "b": b, "question": label, "n": n, "delta_rho": d,
                         "ci_low": lo, "ci_high": hi, "verdict": verdict})
            print(f"{label:36s} n={n:3d}  Δρ = {d:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}] -> {verdict}")
    pd.DataFrame(comp).to_csv(REPORT_DIR / f"cv_score_model{suffix}_comparisons.csv", index=False)

    m5 = m5_length_check(base, target) if "llm.rating" in base else []
    if m5:
        pd.DataFrame(m5).to_csv(REPORT_DIR / f"cv_m5_length{suffix}.csv", index=False)
        print("\nDoes M5 just follow answer length? (pool people, no training involved)")
        print(pd.DataFrame(m5).round(3).to_string(index=False))

    if not args.no_learning_curve:
        lc_rows = []
        for name in LC_GROUPS:
            if name in models:
                cols, pool = models[name]
                lc_rows += cv_learning_curve(name, cols, pool, target, args.folds, args.repeats)
        lc = pd.DataFrame(lc_rows)
        lc.to_csv(REPORT_DIR / f"cv_learning_curve{suffix}.csv", index=False)
        tab = lc.groupby(["model", "n_train"]).spearman.agg(["mean", "std"]).round(3)
        print("\nLearning curve (CV on train+dev; mean / sd over repeats):")
        print(tab.unstack("n_train").to_string())

    print(f"\nSaved: report/cv_score_model{suffix}.csv (+ _comparisons, cv_learning_curve, "
          f"cv_m5_length)")
    print("Test is still locked: report it once with train_score_model.py --final.")


if __name__ == "__main__":
    main()
