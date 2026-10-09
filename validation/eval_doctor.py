"""
validation/eval_doctor.py   (RQ2: does the trained doctor pick the right thing to fix?)

Needs
  report/doctor_diagnoses_test.csv   python approaches/doctor/doctor.py diagnose --manifest avi_eval_subset.csv --out doctor_diagnoses_test.csv --quiet
  report/issue_labels.csv            filled by the researcher (blind)          [check 1]
  report/issue_labels_friend.csv     filled by a second rater (optional)        [rater agreement]

Checks
  1. top issue vs the researcher's label   doctor vs rule doctor vs "always the most common label"
  2. people the doctor flags for an issue: is their recruiter rating lower than everyone else's?
Output: report/eval_doctor.csv, report/eval_doctor_groups.csv

Check 2 reads the recruiter ratings of the locked AVI test people: run it ONCE, after the doctor
and the rule thresholds are fixed. Never tune the doctor on these results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from base._paths import REPORT_DIR
from base.dataset.avi import load_labels, parse_key

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CONTENT_LABELS = {"no_example", "off_topic"}


def same(pred: str, label: str) -> bool:
    if not isinstance(pred, str) or not isinstance(label, str) or not label.strip():
        return False
    label = label.strip()
    return pred == label or (pred == "content" and label in CONTENT_LABELS)


def kappa(a, b):
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def read_labels(path):
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df = df[df["main_issue"].fillna("").str.strip() != ""]
    df["main_issue"] = df["main_issue"].str.strip()
    df["second_issue"] = df["second_issue"].fillna("").str.strip()
    return df


def check_labels(diag):
    p = REPORT_DIR / "issue_labels.csv"
    if not p.exists():
        print("[skip 1] report/issue_labels.csv not found")
        return []
    lab = read_labels(p)
    if lab.empty:
        print("[skip 1] issue_labels.csv has no filled rows yet")
        return []
    m = lab.merge(diag, on="key", how="inner")
    majority = m["main_issue"].mode().iat[0]
    rows = []
    for name, top in (("doctor", "doctor_top"), ("rule", "rule_top"), ("always_" + majority, None)):
        pred = m[top] if top else pd.Series([majority] * len(m), index=m.index)
        hit1 = np.mean([same(p, l) for p, l in zip(pred, m["main_issue"])])
        hit2 = np.mean([same(p, l) or same(p, s) for p, l, s in zip(pred, m["main_issue"], m["second_issue"])])
        rows.append({"check": "1 label", "method": name, "n": len(m),
                     "top_matches_main": round(hit1, 3), "top_matches_main_or_second": round(hit2, 3)})
    print(f"\n1. Top issue vs your labels ({len(m)} clips)")
    print(pd.DataFrame(rows).drop(columns="check").to_string(index=False))

    pf = REPORT_DIR / "issue_labels_friend.csv"
    if pf.exists():
        fr = read_labels(pf)
        both = lab.merge(fr, on="key", suffixes=("_you", "_friend"))
        if len(both) >= 5:
            k = kappa(both["main_issue_you"], both["main_issue_friend"])
            agree = float(np.mean(both["main_issue_you"] == both["main_issue_friend"]))
            print(f"   rater agreement on {len(both)} clips: same main issue {agree:.0%}, Cohen's kappa {k:.2f}")
            rows.append({"check": "raters", "method": "you vs friend", "n": len(both),
                         "top_matches_main": round(agree, 3), "kappa": round(k, 3)})
    return rows


def check_groups(diag):
    labels = load_labels().set_index("participant_id")["hireability"]
    d = diag.copy()
    d["pid"] = [parse_key(k)["participant_id"] if parse_key(k) else None for k in d["key"]]
    d = d.dropna(subset=["pid"])
    d["rating"] = d["pid"].map(labels)
    d = d.dropna(subset=["rating"])
    rng = np.random.default_rng(0)
    rows = []
    for who, col in (("doctor", "doctor_top"), ("rule", "rule_top")):
        for issue in sorted(d[col].unique()):
            flag = d[col] == issue
            if flag.sum() < 3 or (~flag).sum() < 3:
                continue
            a, b = d.loc[flag, "rating"].to_numpy(), d.loc[~flag, "rating"].to_numpy()
            diff = a.mean() - b.mean()
            boots = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(2000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"method": who, "issue": issue, "n_flagged": int(flag.sum()),
                         "mean_rating_flagged": round(a.mean(), 3), "mean_rating_others": round(b.mean(), 3),
                         "diff": round(diff, 3), "ci_low": round(lo, 3), "ci_high": round(hi, 3),
                         "flagged_rated_lower": "yes" if hi < 0 else ("no" if lo > 0 else "not shown")})
    res = pd.DataFrame(rows)
    print(f"\n2. Recruiter rating of clips flagged for each issue vs the rest ({len(d)} AVI test clips)")
    print(res.to_string(index=False) if len(res) else "   (not enough flagged clips)")
    return res


def main():
    p = REPORT_DIR / "doctor_diagnoses_test.csv"
    if not p.exists():
        sys.exit("[ERROR] run: python approaches/doctor/doctor.py diagnose --manifest "
                 "avi_eval_subset.csv --out doctor_diagnoses_test.csv --quiet")
    diag = pd.read_csv(p)
    print("Doctor top issues on AVI test:", diag["doctor_top"].value_counts().to_dict())
    print("Rule   top issues on AVI test:", diag["rule_top"].value_counts().to_dict())
    rows = check_labels(diag)
    groups = check_groups(diag)
    pd.DataFrame(rows).to_csv(REPORT_DIR / "eval_doctor.csv", index=False)
    groups.to_csv(REPORT_DIR / "eval_doctor_groups.csv", index=False)
    print(f"\nSaved report/eval_doctor.csv, report/eval_doctor_groups.csv")


if __name__ == "__main__":
    main()
