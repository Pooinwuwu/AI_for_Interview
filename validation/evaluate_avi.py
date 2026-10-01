"""
validation/evaluate_avi.py

Compare system scores with the AVI-Personality ground truth
(recruiter ratings, 1-5, one value per participant).

How it works
------------
1. Collect clip-level scores from each approach
     rule    output/scores/<key>.json            fusion.overall_score (0-100) + 5 dimension scores
     mllm    output/feedback_mllm/<key>_feedback.json    "ratings" field (bands)
     hybrid  output/feedback_hybrid/<key>_feedback.json  "ratings" field (bands)
   Only AVI clip keys (<participant>_q<n>_<type>) are used; old vid_xxxx files are ignored.
2. Average clip scores per participant (labels are per participant).
3. Join with ground truth and compute, per approach x measure x target:
     - Spearman rho + 95% bootstrap CI (resampling participants)
     - QWK on 1-5 bands (overall measure vs each target, target rounded to 1-5)
4. Bias check: does the system track accent / English proficiency / gender
   more than the human raters do?

Note for Approach 2 & 3
-----------------------
Their current feedback schema has no overall rating (only band_th for the
dimensions listed under "improvements"). To evaluate them, add a required
"ratings" object to the schema, e.g.
    "ratings": {"eye_contact": "ดี", "head_pose": "ปานกลาง", "hand_gesture": "ดี",
                "facial_expression": "ดี", "answer_quality": "ดีมาก", "overall": "ดี"}
Thai or English band labels are both accepted.

Usage
-----
    python validation/evaluate_avi.py                       # test split, all approaches
    python validation/evaluate_avi.py --manifest input/ground_truth/avi_eval_subset.csv
    python validation/evaluate_avi.py --split val --questions 1 2
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base._paths import OUTPUT_DIR, SCORES_DIR, REPORT_DIR
from base.dataset.avi import load_labels, parse_key, TARGETS, PRIMARY_TARGET

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CONFIG
# ============================================================

DIMENSIONS = ["eye_contact", "head_pose", "hand_gesture", "facial_expression", "answer_quality"]

BAND_TO_NUM = {
    "excellent": 5, "ดีมาก": 5,
    "good": 4, "ดี": 4,
    "fair": 3, "ปานกลาง": 3,
    "needs work": 2, "ควรปรับ": 2,
    "priority": 1, "poor": 1, "ควรปรับมาก": 1,
}

# same cut-offs as report/validation_plan.md
SCORE_BAND_CUTS = [(85, 5), (70, 4), (55, 3), (40, 2), (-np.inf, 1)]

APPROACH_DIRS = {
    "mllm":   OUTPUT_DIR / "feedback_mllm",
    "hybrid": OUTPUT_DIR / "feedback_hybrid",
}

BIAS_COLS = ["accent_strength", "english_proficiency"]


# ============================================================
# HELPERS
# ============================================================

def score_to_band(score: float) -> int:
    for cut, band in SCORE_BAND_CUTS:
        if score >= cut:
            return band
    return 1


def band_to_num(label):
    if label is None:
        return np.nan
    return BAND_TO_NUM.get(str(label).strip().lower(), np.nan)


def target_to_band(x: float) -> int:
    return int(np.clip(np.round(x), 1, 5))


def bootstrap_spearman(x, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        stats.append(spearmanr(x[idx], y[idx]).statistic)
    if not stats:
        return np.nan, np.nan
    return tuple(np.percentile(stats, [2.5, 97.5]))


# ============================================================
# LOAD CLIP SCORES
# ============================================================

def load_rule_clips() -> pd.DataFrame:
    rows = []
    for f in SCORES_DIR.glob("*.json"):
        avi = parse_key(f.stem)
        if avi is None:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        row = {"key": f.stem, **avi, "approach": "rule"}
        row["overall"] = d.get("fusion", {}).get("overall_score", np.nan)
        for dim in DIMENSIONS:
            row[dim] = d.get("dimensions", {}).get(dim, {}).get("score", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def load_llm_clips(approach: str) -> pd.DataFrame:
    rows, no_ratings = [], 0
    folder = APPROACH_DIRS[approach]
    if not folder.exists():
        return pd.DataFrame()
    for f in folder.glob("*_feedback.json"):
        key = f.stem.removesuffix("_feedback")
        avi = parse_key(key)
        if avi is None:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        ratings = d.get("ratings")
        if not isinstance(ratings, dict):
            no_ratings += 1
            continue
        row = {"key": key, **avi, "approach": approach,
               "overall": band_to_num(ratings.get("overall"))}
        for dim in DIMENSIONS:
            row[dim] = band_to_num(ratings.get(dim))
        rows.append(row)
    if no_ratings:
        print(f"[WARN] {approach}: {no_ratings} AVI feedback file(s) have no 'ratings' "
              f"field — skipped (see docstring).")
    return pd.DataFrame(rows)


# ============================================================
# EVALUATE
# ============================================================

def aggregate(clips: pd.DataFrame) -> pd.DataFrame:
    measures = ["overall"] + DIMENSIONS
    agg = clips.groupby(["approach", "participant_id"])[measures].mean()
    agg["n_clips"] = clips.groupby(["approach", "participant_id"]).size()
    return agg.reset_index()


def evaluate(per_pid: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    rows = []
    for approach, g in per_pid.groupby("approach"):
        for measure in ["overall"] + DIMENSIONS:
            for target in TARGETS:
                d = g[[measure, target]].dropna()
                n = len(d)
                row = {"approach": approach, "measure": measure, "target": target, "n": n}
                if n >= 5 and d[measure].nunique() > 1:
                    res = spearmanr(d[measure], d[target])
                    lo, hi = bootstrap_spearman(d[measure], d[target], n_boot)
                    row.update(spearman=res.statistic, ci_low=lo, ci_high=hi, p=res.pvalue)
                    if measure == "overall":
                        # rule scores are 0-100; LLM scores are already 1-5 band means
                        sys_band = (d[measure].map(score_to_band) if approach == "rule"
                                    else d[measure].round().clip(1, 5).astype(int))
                        row["qwk"] = cohen_kappa_score(d[target].map(target_to_band), sys_band,
                                                       weights="quadratic", labels=[1, 2, 3, 4, 5])
                rows.append(row)
    return pd.DataFrame(rows)


def bias_check(per_pid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for approach, g in per_pid.groupby("approach"):
        for col in BIAS_COLS:
            d = g[["overall", PRIMARY_TARGET, col]].dropna()
            if len(d) < 5:
                continue
            rows.append({
                "approach": approach, "attribute": col, "n": len(d),
                "stat": "spearman",
                "system": spearmanr(d["overall"], d[col]).statistic,
                "human": spearmanr(d[PRIMARY_TARGET], d[col]).statistic,
            })
        # gender: standardised mean difference (male=1 vs female=2), system vs human
        d = g[g.gender.isin([1, 2])][["overall", PRIMARY_TARGET, "gender"]].dropna()
        if d.gender.nunique() == 2 and len(d) >= 10:
            def smd(col):
                a, b = d[d.gender == 1][col], d[d.gender == 2][col]
                return (a.mean() - b.mean()) / d[col].std(ddof=1)
            rows.append({"approach": approach, "attribute": "gender", "n": len(d),
                         "stat": "SMD male-female", "system": smd("overall"),
                         "human": smd(PRIMARY_TARGET)})
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--manifest", type=Path, default=None,
                    help="only evaluate participants listed in this subset CSV")
    ap.add_argument("--questions", type=int, nargs="+", default=None,
                    help="only aggregate these question numbers (default: all found)")
    ap.add_argument("--approaches", nargs="+", default=["rule", "mllm", "hybrid"])
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    frames = []
    if "rule" in args.approaches:
        frames.append(load_rule_clips())
    for a in ("mllm", "hybrid"):
        if a in args.approaches:
            frames.append(load_llm_clips(a))
    clips = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()

    if clips.empty:
        print("[ERROR] No AVI clip results found in output/. Run the pipeline on AVI clips first "
              "(python base/dataset/build_subset.py).")
        sys.exit(1)

    if args.questions:
        clips = clips[clips.question_no.isin(args.questions)]

    labels = load_labels()
    if args.split != "all":
        labels = labels[labels.split == args.split]
    if args.manifest:
        keep = set(pd.read_csv(args.manifest, dtype=str).participant_id)
        labels = labels[labels.participant_id.isin(keep)]

    per_pid = aggregate(clips).merge(labels, on="participant_id", how="inner")
    per_pid = per_pid.dropna(subset=[PRIMARY_TARGET])

    if per_pid.empty:
        print(f"[ERROR] None of the scored participants are in split='{args.split}'"
              + (f" / manifest {args.manifest}" if args.manifest else ""))
        sys.exit(1)

    summary = evaluate(per_pid, args.bootstrap)
    bias = bias_check(per_pid)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    per_pid.to_csv(REPORT_DIR / "avi_eval_participants.csv", index=False)
    summary.to_csv(REPORT_DIR / "avi_eval_summary.csv", index=False)
    bias.to_csv(REPORT_DIR / "avi_eval_bias.csv", index=False)

    # ---------- print ----------
    pd.set_option("display.width", 140)
    print("=" * 70)
    print(f"  AVI-Personality evaluation — split={args.split}")
    print("=" * 70)
    print(per_pid.groupby("approach").agg(participants=("participant_id", "nunique"),
                                          clips_per_participant=("n_clips", "mean")).to_string())
    print()
    print(f"Overall score vs {PRIMARY_TARGET} (primary result):")
    main_rows = summary[(summary.measure == "overall") & (summary.target == PRIMARY_TARGET)]
    print(main_rows[["approach", "n", "spearman", "ci_low", "ci_high", "p", "qwk"]]
          .round(3).to_string(index=False))
    print()
    print("Spearman: each measure vs each target (exploratory):")
    print(summary.pivot_table(index=["approach", "measure"], columns="target",
                              values="spearman").round(2).to_string())
    print()
    if not bias.empty:
        print("Bias check (system should not track these more than humans do):")
        print(bias.round(3).to_string(index=False))
        print()
    print(f"Saved: {REPORT_DIR / 'avi_eval_summary.csv'}")
    print(f"       {REPORT_DIR / 'avi_eval_participants.csv'}")
    print(f"       {REPORT_DIR / 'avi_eval_bias.csv'}")


if __name__ == "__main__":
    main()
