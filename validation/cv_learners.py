"""
validation/cv_learners.py   (robustness check: does the ranking of M0-M5 depend on Ridge?)

Same pool, same folds and same feature sets as cv_score_model.py, but each feature set is
fitted with 5 learners. Test people are never loaded. Learner settings are FIXED common
defaults (not tuned on these results), so the check cannot be gamed.

  ridge       RidgeCV (what the main results use)
  elasticnet  ElasticNetCV - linear, can drop features
  rf          Random forest (300 trees, min leaf 5)
  gbm         Histogram gradient boosting (300 iterations, lr 0.05, min leaf 20, L2 1.0)
  svr         RBF support-vector regression, C chosen by inner 3-fold CV from {0.1, 1, 10}

Output: report/cv_learners<suffix>.csv  (model, learner, n, mean / sd over repeats)

Usage
  python validation/cv_learners.py --data full --text-emb minilm --llm qwen2.5:3b
  python validation/cv_learners.py ... --repeats 2 --learners ridge rf     # quick try
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import sklearn

_SK = tuple(int(x) for x in sklearn.__version__.split(".")[:2])

from base._paths import REPORT_DIR
from base.dataset.avi import load_labels, TARGETS
from base.learned_features import feature_groups
from validation.train_score_model import ALPHAS, ALPHAS_EMB, has_modalities, make_model, split_ids
from validation.cv_score_model import load_pool

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODELS = ["length", "audio_text", "visual", "all", "text_emb", "text_emb+audio_text",
          "llm+length", "llm+audio_text"]


def learner(name, cols):
    emb = any(c.startswith("txt.") for c in cols)
    if name == "ridge":
        return make_model(ALPHAS_EMB if emb else ALPHAS)
    if name == "elasticnet":
        kw = dict(l1_ratio=[0.2, 0.5, 0.8, 1.0], cv=5, max_iter=5000, random_state=0)
        try:                                   # scikit-learn >= 1.7: alphas=<int>
            enet = ElasticNetCV(alphas=30, **kw)
            enet.get_params()
        except TypeError:
            enet = None
        if enet is None or _SK < (1, 7):       # older scikit-learn: n_alphas
            enet = ElasticNetCV(n_alphas=30, **kw)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), enet)
    if name == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                   max_features=0.33, n_jobs=-1, random_state=0))
    if name == "gbm":
        return make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                           min_samples_leaf=20, l2_regularization=1.0,
                                                           random_state=0))
    if name == "svr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             GridSearchCV(SVR(kernel="rbf", gamma="scale", epsilon=0.1),
                                          {"C": [0.1, 1.0, 10.0]}, cv=3))
    raise ValueError(name)


def cv_rho(make, X, y, folds, repeats):
    rhos = []
    for r in range(repeats):
        p = np.empty(len(y))
        for tr, te in KFold(folds, shuffle=True, random_state=r).split(X):
            m = make()
            m.fit(X.iloc[tr], y[tr])
            p[te] = m.predict(X.iloc[te])
        rhos.append(spearmanr(p, y).statistic)
    return np.array(rhos)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="hireability", choices=list(TARGETS))
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--data", choices=["subset", "full"], default="full")
    ap.add_argument("--train", default="avi_train*_subset.csv")
    ap.add_argument("--dev", default="avi_dev_subset.csv")
    ap.add_argument("--test", default="avi_eval_subset.csv")
    ap.add_argument("--text-emb", default="minilm")
    ap.add_argument("--llm", default="qwen2.5:3b")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--learners", nargs="+", default=["ridge", "elasticnet", "rf", "gbm", "svr"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()
    args.text_emb_list = [args.text_emb]

    labels_all = load_labels()
    tr, dev, test = split_ids(SimpleNamespace(data=args.data, train=args.train, dev=args.dev,
                                              test=args.test), labels_all)
    pool_ids = (tr | dev) - test
    pool = load_pool(args, args.text_emb, labels_all, pool_ids)
    if test & set(pool.index):
        sys.exit("[ERROR] a test participant reached the pool")
    groups = feature_groups(pool.columns)
    if "llm.rating" in pool:
        from base.learned_features import LENGTH
        groups["llm+length"] = ["llm.rating"] + [c for c in LENGTH if c in pool]
    target = args.target

    print("=" * 78)
    print(f"  Learner robustness  pool n={len(pool)}  folds={args.folds} repeats={args.repeats}"
          f"  (test people: not loaded)")
    print("=" * 78)
    rows = []
    for name in args.models:
        cols = groups.get(name)
        if not cols:
            print(f"  [skip] {name}: no features")
            continue
        sub = pool[has_modalities(pool, cols)]
        X, y = sub[cols], sub[target].to_numpy()
        for ln in args.learners:
            t0 = time.time()
            rhos = cv_rho(lambda: learner(ln, cols), X, y, args.folds, args.repeats)
            rows.append({"model": name, "learner": ln, "n": len(sub), "features": len(cols),
                         "cv_mean": rhos.mean(), "cv_sd": rhos.std(ddof=1) if len(rhos) > 1 else 0.0})
            print(f"  {name:22s} {ln:10s} n={len(sub):3d}  rho={rhos.mean():.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    res = pd.DataFrame(rows)
    suffix = f"_{args.data}_{args.text_emb}_llm-{args.llm.replace(':', '_')}"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORT_DIR / f"cv_learners{suffix}.csv", index=False)
    tab = res.pivot(index="model", columns="learner", values="cv_mean").reindex(
        [m for m in args.models if m in set(res.model)])
    print("\nSpearman rho (CV mean) - rows: feature set, columns: learner")
    print(tab.round(3).to_string())
    rank = tab.rank(ascending=False).round(0)
    print("\nRank of each feature set within each learner (1 = best):")
    print(rank.to_string())
    print(f"\nSaved: report/cv_learners{suffix}.csv   Test is still locked.")


if __name__ == "__main__":
    main()
