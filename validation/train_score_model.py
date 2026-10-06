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
  python validation/train_score_model.py                 # visual track (subset), train -> dev
  python validation/train_score_model.py --data full --text-emb minilm --learning-curve
                                                         # audio/text track, all AVI people
  python validation/train_score_model.py --final         # train+dev -> test, ONCE per track
  python validation/train_score_model.py --data full --text-emb minilm bge-small e5-base \
         --llm qwen2.5:3b --learning-curve --final
                         # audio/text track: all processed AVI train+val people -> the SAME
                         # locked 50-person test subset as the visual track
Models needing a modality (e.g. visual) only use participants where it was measured;
paired comparisons use the participants both models scored.

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

from base._paths import GROUND_TRUTH_DIR, REPORT_DIR
from base.dataset.avi import load_labels, TARGETS
from base.learned_features import clip_features, feature_groups, participant_table

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

ALPHAS = np.logspace(-2, 4, 25)


# ============================================================
# FEATURES  (shared with Approach 1b: base/learned_features.py)
# ============================================================

def manifest_ids(pattern: str):
    ids = set()
    for p in glob.glob(str(GROUND_TRUTH_DIR / pattern)):
        ids |= set(pd.read_csv(p, dtype=str).participant_id)
    return ids


# ============================================================
# MODEL + METRICS
# ============================================================

ALPHAS_EMB = np.logspace(-1, 6, 29)   # embeddings: hundreds of dims -> allow stronger shrinkage


def make_model(alphas=ALPHAS):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         RidgeCV(alphas=alphas))


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


def paired_diff_ci(pred_a: pd.Series, pred_b: pd.Series, y: pd.Series, n_boot=2000, seed=0):
    """Bootstrap CI of rho(a, y) - rho(b, y) on the participants both models scored."""
    common = pred_a.index.intersection(pred_b.index).intersection(y.index)
    pa, pb, yy = pred_a[common].to_numpy(), pred_b[common].to_numpy(), y[common].to_numpy()
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(yy), len(yy))
        if np.unique(yy[i]).size > 1:
            d.append(spearmanr(pa[i], yy[i]).statistic - spearmanr(pb[i], yy[i]).statistic)
    obs = spearmanr(pa, yy).statistic - spearmanr(pb, yy).statistic
    lo, hi = np.percentile(d, [2.5, 97.5])
    return obs, lo, hi, len(common)


# which column proves that a modality was measured for a participant
PRESENT = {"vis": "vis.gaze.eye_contact_ratio", "sp": "sp.word_count",
           "pr": "pr.silence_ratio", "txt": "txt.0", "llm": "llm.rating"}


def has_modalities(df: pd.DataFrame, cols) -> pd.Series:
    """Rows where every modality used by `cols` was actually measured (no imputing a
    whole missing modality: speech-only participants have no visual features)."""
    ok = pd.Series(True, index=df.index)
    for prefix in {c.split(".")[0] for c in cols}:
        col = PRESENT.get(prefix)
        if col in df:
            ok &= df[col].notna()
    return ok


# ============================================================
# MAIN
# ============================================================

COMPARISONS = (
    ("all", "audio_text", "RQ2: visual adds over audio+text?"),
    ("audio_text", "length", "M2 audio+text beats answer length?"),
    ("audio_text", "rule", "M2 learned beats hand-set rules?"),
    ("text_emb", "length", "M3 text embedding beats length?"),
    ("text_emb+audio_text", "audio_text", "M4: embedding adds over M2?"),
    ("text_emb+audio_text", "text_emb", "M4: M2 features add over embedding?"),
    ("text_emb+all", "text_emb+audio_text", "RQ2 (M4): visual adds?"),
    ("llm_zero_shot", "length", "M5 local LLM beats length?"),
    ("text_emb", "llm_zero_shot", "M3 trained embedding beats zero-shot LLM?"),
    ("llm+audio_text", "audio_text", "M5 rating adds over M2?"),
    ("llm+length", "length", "M5 rating adds over length?"),
)

LC_SIZES = [25, 50, 100, 200, 300]
LC_GROUPS = ["length", "audio_text", "text_emb", "text_emb+audio_text", "llm+length"]


def split_ids(args, labels):
    if args.data == "full":       # AVI official subject-level split, every processed participant
        by = labels.groupby("split").participant_id.apply(set)
        test = by.get("test", set())
        locked = manifest_ids(args.test) if args.test else set()
        # the test people stay the locked 50-person subset in every track, so the audio/text
        # track (more training people) and the visual track are scored on the same people
        return by.get("train", set()) - test, by.get("val", set()) - test, locked or test
    return manifest_ids(args.train), manifest_ids(args.dev), manifest_ids(args.test)


def fit_eval(name, cols, fit, ev, target):
    fit = fit[has_modalities(fit, cols)]
    ev = ev[has_modalities(ev, cols)]
    m = make_model(ALPHAS_EMB if name.startswith("text_emb") else ALPHAS)
    if len(fit) < 10 or len(ev) < 10:
        return m, pd.Series(dtype=float), np.nan, len(fit)
    y_fit = fit[target].to_numpy()
    cv_pred = cross_val_predict(m, fit[cols], y_fit, cv=KFold(5, shuffle=True, random_state=0))
    m.fit(fit[cols], y_fit)
    return m, pd.Series(m.predict(ev[cols]), index=ev.index), spearmanr(cv_pred, y_fit).statistic, len(fit)


def learning_curve(groups, fit, ev, target, repeats=5):
    rows = []
    for name in [g for g in LC_GROUPS if g in groups and groups[g]]:
        cols = groups[name]
        pool = fit[has_modalities(fit, cols)]
        evm = ev[has_modalities(ev, cols)]
        sizes = sorted({n for n in LC_SIZES if n < len(pool)} | {len(pool)})
        for n in sizes:
            for r in range(repeats if n < len(pool) else 1):
                sub = pool.sample(n, random_state=r)
                m = make_model(ALPHAS_EMB if name.startswith("text_emb") else ALPHAS)
                m.fit(sub[cols], sub[target])
                rows.append({"model": name, "n_train": n, "repeat": r,
                             "spearman": spearmanr(m.predict(evm[cols]), evm[target]).statistic})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="hireability", choices=list(TARGETS))
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--data", choices=["subset", "full"], default="subset",
                    help="subset = manifests (visual track); full = every processed AVI participant "
                         "by the official split (audio/text track, run_speech_pipeline.py)")
    ap.add_argument("--train", default="avi_train*_subset.csv", help="manifest glob (subset)")
    ap.add_argument("--dev", default="avi_dev_subset.csv")
    ap.add_argument("--test", default="avi_eval_subset.csv")
    ap.add_argument("--text-emb", nargs="+", default=[],
                    help="add M3/M4 groups from output/embeddings/text_<name>.npz (e.g. minilm); "
                         "more names add text_emb[<name>] models")
    ap.add_argument("--llm", default=None,
                    help="add M5 (local LLM zero-shot ratings, e.g. qwen2.5:3b)")
    ap.add_argument("--learning-curve", action="store_true",
                    help="also fit on growing random subsets of the training people")
    ap.add_argument("--final", action="store_true",
                    help="train on train+dev and evaluate on test (do this ONCE)")
    args = ap.parse_args()

    embs = args.text_emb
    clips = clip_features(args.questions, embs[0] if embs else None, args.llm)
    if clips.empty:
        sys.exit("[ERROR] no AVI feature files in output/features/")
    per = participant_table(clips)
    labels_all = load_labels()
    labels = labels_all.set_index("participant_id")[[args.target]]
    data = per.join(labels, how="inner").dropna(subset=[args.target])

    train_ids, dev_ids, test_ids = split_ids(args, labels_all)
    if args.final:
        fit_ids, eval_ids, eval_name = train_ids | dev_ids, test_ids, "test"
    else:
        fit_ids, eval_ids, eval_name = train_ids, dev_ids, "dev"
    if eval_ids & fit_ids:
        sys.exit("[ERROR] evaluation participants overlap with training participants")
    fit = data[data.index.isin(fit_ids)]
    ev = data[data.index.isin(eval_ids)]

    print("=" * 78)
    print(f"  Score model — data={args.data}  target={args.target}  fit n={len(fit)}  "
          f"{eval_name} n={len(ev)}")
    print("=" * 78)
    if len(fit) < 20 or len(ev) < 10:
        sys.exit(f"[ERROR] need >= 20 training and >= 10 {eval_name} participants with features "
                 f"(have {len(fit)} / {len(ev)}). Process more clips first.")

    groups = feature_groups(data.columns)
    y_ev = ev[args.target]
    # model name -> (columns, fit table, eval table); extra embeddings get their own tables
    runs = {name: (cols, fit, ev) for name, cols in groups.items()}
    for e in embs[1:]:
        d2 = participant_table(clip_features(args.questions, e, None)).join(labels, how="inner")
        d2 = d2.dropna(subset=[args.target])
        g2 = feature_groups(d2.columns)
        for n in ("text_emb", "text_emb+audio_text", "text_emb+all"):
            if g2.get(n):
                runs[n.replace("text_emb", f"text_emb[{e}]")] = (
                    g2[n], d2[d2.index.isin(fit_ids)], d2[d2.index.isin(eval_ids)])

    rows, preds, coef = [], {}, None
    if "rule.overall" in ev:
        e = ev[ev["rule.overall"].notna()]
        if len(e) >= 10:
            preds["rule"] = e["rule.overall"]
            r, lo, hi = spearman_ci(e["rule.overall"], e[args.target])
            rows.append({"model": "rule", "features": 0, "n_fit": 0, "n_eval": len(e),
                         "cv_spearman_fit": np.nan, "spearman": r, "ci_low": lo, "ci_high": hi})
    if "llm.rating" in ev:                      # M5 zero-shot: the rating itself, no training
        e = ev[ev["llm.rating"].notna()]
        if len(e) >= 10:
            preds["llm_zero_shot"] = e["llm.rating"]
            r, lo, hi = spearman_ci(e["llm.rating"], e[args.target])
            rows.append({"model": "llm_zero_shot", "features": 0, "n_fit": 0, "n_eval": len(e),
                         "cv_spearman_fit": np.nan, "spearman": r, "ci_low": lo, "ci_high": hi})
    for name, (cols, fit_t, ev_t) in runs.items():
        if not cols:
            continue
        m, p, cv_rho, n_fit = fit_eval(name, cols, fit_t, ev_t, args.target)
        if len(p) < 10:
            continue
        preds[name] = p
        r, lo, hi = spearman_ci(p.to_numpy(), y_ev[p.index].to_numpy())
        rows.append({"model": name, "features": len(cols), "n_fit": n_fit, "n_eval": len(p),
                     "cv_spearman_fit": cv_rho, "spearman": r, "ci_low": lo, "ci_high": hi,
                     "alpha": m[-1].alpha_})
        if name == "all":
            coef = pd.Series(m[-1].coef_, index=cols).sort_values(key=abs, ascending=False)

    res = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ((f"_{args.data}" if args.data != "subset" else "")
              + (f"_{'+'.join(embs)}" if embs else "")
              + (f"_llm-{args.llm.replace(':', '_')}" if args.llm else ""))
    res.to_csv(REPORT_DIR / f"score_model_{eval_name}{suffix}.csv", index=False)
    pd.set_option("display.width", 160)
    print(res.round(3).to_string(index=False))
    print()

    comp = []
    for a, b, label in COMPARISONS + tuple((f"text_emb[{e}]", "text_emb", f"{e} beats {embs[0]}?")
                                          for e in embs[1:]):
        if a in preds and b in preds:
            d, lo, hi, n = paired_diff_ci(preds[a], preds[b], y_ev)
            verdict = "yes" if lo > 0 else ("no (worse)" if hi < 0 else "not shown (CI includes 0)")
            comp.append({"a": a, "b": b, "question": label, "n": n, "delta_rho": d,
                         "ci_low": lo, "ci_high": hi, "verdict": verdict})
            print(f"{label:40s} n={n:3d}  Δρ = {d:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}] -> {verdict}")
    if comp:
        pd.DataFrame(comp).to_csv(REPORT_DIR / f"score_model_{eval_name}{suffix}_comparisons.csv",
                                  index=False)

    if coef is not None:
        coef.rename("std_weight").to_csv(REPORT_DIR / "score_model_coefficients.csv")
        print("\nTop features in 'all' model (standardised weights):")
        print(coef.head(10).round(3).to_string())

    if args.learning_curve:
        lc = learning_curve(groups, fit, ev, args.target)
        lc.to_csv(REPORT_DIR / f"learning_curve_{eval_name}{suffix}.csv", index=False)
        tab = lc.groupby(["model", "n_train"]).spearman.agg(["mean", "std", "count"]).round(3)
        print("\nLearning curve (Spearman on", eval_name, "; mean/sd over random training subsets):")
        print(tab.to_string())

    print(f"\nSaved: {REPORT_DIR / f'score_model_{eval_name}{suffix}.csv'}")
    if not args.final:
        print("Dev is for checking only (protocol v2: every method is reported on test).")


if __name__ == "__main__":
    main()
