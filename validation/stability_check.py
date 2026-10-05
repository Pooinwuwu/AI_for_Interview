"""
validation/stability_check.py   (RQ3: is the AI feedback stable across repeated runs?)

Run an approach several times on the same clips, each with its own --tag:
    python approaches/approach_2_mllm_zero_shot/run.py --only own_01_q1 ...              (run 1)
    python approaches/approach_2_mllm_zero_shot/run.py --only own_01_q1 ... --tag r2     (run 2)
    python approaches/approach_2_mllm_zero_shot/run.py --only own_01_q1 ... --tag r3     (run 3)
    (same for approaches/approach_grounded/run.py)
then:
    python validation/stability_check.py

Per approach and clip, across runs:
  dims_jaccard      overlap of the dimensions flagged as "improvements" (1 = same every run)
  overall_agree     share of run pairs giving the same overall band (Gemini-judged approaches)
  halluc_rate_sd    spread of the automatic hallucination rate between runs
Output: report/stability.csv (per clip) and a summary per approach.
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base._paths import OUTPUT_DIR, REPORT_DIR
from validation.hallucination_check import (APPROACH_DIRS, HALLUCINATED, extract_claims,
                                            load_evidence, verify)

import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

APPROACHES = ["mllm", "grounded", "grounded_video"]


def run_dirs(approach):
    base = APPROACH_DIRS[approach]
    return [d for d in [base] + sorted(base.parent.glob(base.name + "_*")) if d.is_dir()]


def halluc_rate(fb, key):
    events, duration = load_evidence(key)
    if events is None:
        return np.nan
    v = [verify(c, events, duration) for c in extract_claims(fb)]
    v = [x for x in v if x != "unverifiable"]
    return sum(x in HALLUCINATED for x in v) / len(v) if v else np.nan


def main():
    rows = []
    for approach in APPROACHES:
        dirs = run_dirs(approach)
        if len(dirs) < 2:
            continue
        keys = set.intersection(*[{f.stem.removesuffix("_feedback") for f in d.glob("*_feedback.json")}
                                  for d in dirs])
        for key in sorted(keys):
            fbs = [json.loads((d / f"{key}_feedback.json").read_text(encoding="utf-8")) for d in dirs]
            dims = [{i.get("dimension") for i in fb.get("improvements", [])} for fb in fbs]
            jac = [len(a & b) / len(a | b) if a | b else 1.0 for a, b in itertools.combinations(dims, 2)]
            overall = [(fb.get("ratings") or {}).get("overall") for fb in fbs]
            agree = [a == b for a, b in itertools.combinations(overall, 2)]
            rates = [halluc_rate(fb, key) for fb in fbs]
            rows.append({"approach": approach, "key": key, "runs": len(fbs),
                         "dims_jaccard": np.mean(jac), "overall_agree": np.mean(agree),
                         "overall_bands": " / ".join(str(o) for o in overall),
                         "halluc_rate_mean": np.nanmean(rates) if not all(np.isnan(rates)) else np.nan,
                         "halluc_rate_sd": np.nanstd(rates) if not all(np.isnan(rates)) else np.nan})
    if not rows:
        sys.exit("[ERROR] need at least 2 runs of an approach (run it again with --tag r2)")
    df = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "stability.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 160)
    print(df.round(2).to_string(index=False))
    print("\nSummary per approach (mean over clips):")
    print(df.groupby("approach")[["runs", "dims_jaccard", "overall_agree", "halluc_rate_mean",
                                  "halluc_rate_sd"]].mean().round(2).to_string())
    print(f"\nSaved {REPORT_DIR / 'stability.csv'}")


if __name__ == "__main__":
    main()
