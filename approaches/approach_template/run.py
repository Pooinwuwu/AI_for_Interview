"""
approaches/approach_template/run.py

Runs Approach T (template coach, no LLM) and checks it with the same verify.py as C.

Output: output/feedback_template/<key>_feedback.json   (same format as C)

Usage
  python approaches/approach_template/run.py                    # all non-AVI clips with evidence
  python approaches/approach_template/run.py --only own_03_q2
Runs offline and sends nothing anywhere, so AVI clips are allowed with --allow-avi
(their outputs stay in output/, never committed).
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "approaches" / "approach_grounded"))

from base._paths import EVIDENCE_DIR, METADATA_FILE, OUTPUT_DIR, SCORES_DIR
from base.dataset.avi import parse_key
from base.grounded_log import build as build_log
from base.llm_common import BANDS_TH, DIMENSIONS
import coach
import verify

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUT = OUTPUT_DIR / "feedback_template"
# question number -> focus (same as app/questions.py)
FOCUS_BY_Q = {"1": "strengths", "2": "friend", "3": "story", "4": "story", "5": "story", "6": "story"}


def judge_rule(key: str) -> dict:
    rule = json.loads((SCORES_DIR / f"{key}.json").read_text(encoding="utf-8"))
    ratings = {d: rule["dimensions"][d]["band"]["label_th"] for d in DIMENSIONS
               if d in rule.get("dimensions", {})}
    band = rule.get("fusion", {}).get("overall_band", {}).get("label_th")
    ratings["overall"] = band if band in BANDS_TH else "ปานกลาง"
    return ratings


def clip_info(key: str):
    """(focus, lang) from metadata (web-app clips) or from the key (own_/friend_ clips)."""
    meta = {}
    if METADATA_FILE.exists():
        meta = json.loads(METADATA_FILE.read_text(encoding="utf-8")).get(key, {})
    lang = meta.get("language", "en")
    qid = meta.get("question_id")
    if qid is None:
        q = key.rsplit("_q", 1)[-1] if "_q" in key else ""
        qid = f"q{q}" if q in FOCUS_BY_Q else "intro"
    focus = "intro" if qid == "intro" else FOCUS_BY_Q.get(qid.lstrip("q"), "story")
    return focus, lang


def run_clip(key: str, focus: str = None, lang: str = None) -> dict:
    f0, l0 = clip_info(key)
    focus, lang = focus or f0, lang or l0
    ratings = judge_rule(key)
    log = build_log(key)
    draft = coach.write(log, ratings, focus, lang)
    rep = verify.check(draft, log)
    final, removed = verify.drop_failed(draft, rep)
    out = dict(final)
    out["strengths_en"] = [s["text_en"] for s in final.get("strengths", [])]
    out["strengths_th"] = [s["text_th"] for s in final.get("strengths", [])]
    return {
        **out,
        "ratings": ratings, "key": key, "approach": "T_template",
        "model": coach.TEMPLATE_VERSION, "temperature": None, "prompt_version": None,
        "scoring": {"overall_note": "rule-based measurement"},
        "verification": {"rounds": [rep], "revised": False, "removed": removed,
                         "summary_unresolved": rep["summary_problems"],
                         "draft_problems": len(rep["problems"]),
                         "final_problems": len(rep["summary_problems"])},
        "draft": draft,
        "evidence_log": log,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--allow-avi", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    keys = args.only or sorted(p.stem.removesuffix("_evidence")
                               for p in EVIDENCE_DIR.glob("*_evidence.json"))
    if not args.allow_avi:
        keys = [k for k in keys if parse_key(k) is None]
    n_prob = 0
    for key in keys:
        if not (SCORES_DIR / f"{key}.json").exists():
            print(f"[SKIP] {key}: no Approach 1 scores")
            continue
        res = run_clip(key)
        (OUT / f"{key}_feedback.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
        v = res["verification"]
        n_prob += v["draft_problems"]
        print(f"{key:28s} strengths {len(res['strengths'])}  improvements {len(res['improvements'])}"
              f"  checker problems {v['draft_problems']}")
        for p in v["rounds"][0]["problems"]:
            print("    !", p)
    print(f"\n{len(keys)} clips, checker problems in total: {n_prob}.  Saved in {OUT}")


if __name__ == "__main__":
    main()
