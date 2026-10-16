"""
scripts/compare_coaches.py - run several coaches on the same clips and compare them.

Every coach gets the same evidence log, prompt (LLM coaches) and checker (verify.py).
Results per coach are kept in output/feedback_app/by_coach/<clip>__<coach>.json.

Usage
  # run the coaches, then compare (clips must already be measured by the app)
  python scripts/compare_coaches.py --run --coaches gemini groq ollama local --keys app_2026...q1 app_...
  # only compare what is already in by_coach/
  python scripts/compare_coaches.py

Output
  report/coach_compare.csv            one row per clip x coach (automatic measures)
  report/coach_compare_summary.csv    mean per coach
  report/coach_compare/<clip>.md      the advice of every coach side by side, for reading
Automatic measures do NOT say which advice is better; the human audit (true / useful) and
the blind preference test still decide RQ3.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd

from base._paths import OUTPUT_DIR, REPORT_DIR

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BY_COACH = OUTPUT_DIR / "feedback_app" / "by_coach"
ORDER = ["gemini", "groq", "ollama", "local"]
NAME = {"gemini": "Gemini", "groq": "Groq", "ollama": "LLM ในเครื่อง (Ollama)", "local": "แม่แบบ (ไม่มี LLM)"}


def points(fb):
    return [("จุดเด่น", s) for s in fb.get("strengths", [])] + \
           [("ควรแก้", s) for s in fb.get("improvements", [])]


def th_chars(item):
    t = (item.get("text_th") or item.get("issue_th") or "")
    return len(t) + sum(len(x) for x in item.get("action_steps_th", []) or [])


def measures(fb):
    v = fb.get("verification", {})
    pts = [p for _, p in points(fb)]
    removed = v.get("removed", []) or []
    return {
        "model": fb.get("model"),
        "seconds": fb.get("seconds_coach"),
        "strengths": len(fb.get("strengths", [])),
        "improvements": len(fb.get("improvements", [])),
        "draft_problems": v.get("draft_problems"),
        "revised": v.get("revised"),
        "removed_points": sum(r.get("section") != "improved_answer" for r in removed),
        "improved_answer_removed": any(r.get("section") == "improved_answer" for r in removed),
        "ids_per_point": round(sum(len(p.get("evidence_ids", [])) for p in pts) / len(pts), 2) if pts else None,
        "thai_chars_per_point": round(sum(th_chars(p) for p in pts) / len(pts)) if pts else None,
        "summary_th_chars": len(fb.get("overall_summary_th", "") or ""),
    }


def side_by_side(key, results):
    out = [f"# {key}", ""]
    for c in [c for c in ORDER if c in results] + [c for c in results if c not in ORDER]:
        fb = results[c]
        m = measures(fb)
        out += [f"## {NAME.get(c, c)}  ·  {m['model']}  ·  {m['seconds']} s", "",
                f"**สรุป:** {fb.get('overall_summary_th', '')}", ""]
        for sec, p in points(fb):
            txt = p.get("text_th") or p.get("issue_th") or ""
            out.append(f"- **{sec}** ({p.get('dimension')}, {', '.join(p.get('evidence_ids', []))}): {txt}")
            for st in p.get("action_steps_th", []) or []:
                out.append(f"  - {st}")
        ia = fb.get("improved_answer") or {}
        if ia.get("rewritten_th"):
            out += ["", f"**ตัวอย่างคำตอบที่ปรับแล้ว:** {ia['rewritten_th']}"]
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="run the coaches first")
    ap.add_argument("--coaches", nargs="+", default=ORDER)
    ap.add_argument("--keys", nargs="+", default=None, help="clips (default: all in by_coach/)")
    a = ap.parse_args()

    if a.run:
        import pipeline as P
        if not a.keys:
            sys.exit("--run needs --keys")
        for key in a.keys:
            for c in a.coaches:
                t0 = time.time()
                try:
                    r = P.coach(key, c)
                    print(f"[ok]   {key} {c:7s} {time.time() - t0:6.0f} s  "
                          f"strengths={len(r.get('strengths', []))} improvements={len(r.get('improvements', []))}",
                          flush=True)
                except Exception as e:
                    print(f"[FAIL] {key} {c}: {str(e)[:200]}", flush=True)

    files = sorted(BY_COACH.glob("*__*.json"))
    rows, by_key = [], {}
    for f in files:
        key, coach = f.stem.rsplit("__", 1)
        if a.keys and key not in a.keys:
            continue
        fb = json.loads(f.read_text(encoding="utf-8"))
        by_key.setdefault(key, {})[coach] = fb
        rows.append({"clip": key, "coach": coach, **measures(fb)})
    if not rows:
        sys.exit(f"nothing in {BY_COACH} yet - run with --run first")

    df = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "coach_compare.csv", index=False, encoding="utf-8-sig")
    num = ["seconds", "strengths", "improvements", "draft_problems", "removed_points",
           "ids_per_point", "thai_chars_per_point"]
    summ = df.groupby("coach")[num].mean(numeric_only=True).round(1)
    summ.insert(0, "clips", df.groupby("coach").size())
    summ = summ.reindex([c for c in ORDER if c in summ.index] + [c for c in summ.index if c not in ORDER])
    summ.to_csv(REPORT_DIR / "coach_compare_summary.csv", encoding="utf-8-sig")
    md = REPORT_DIR / "coach_compare"
    md.mkdir(exist_ok=True)
    for key, res in by_key.items():
        (md / f"{key}.md").write_text(side_by_side(key, res), encoding="utf-8")
    print("\nMean per coach (automatic measures only - the human audit decides which is better):")
    print(summ.to_string())
    print(f"\nSaved: report/coach_compare.csv, report/coach_compare_summary.csv, report/coach_compare/*.md")


if __name__ == "__main__":
    main()
