"""
approaches/approach_1b_learned/score.py

Score clips with the trained Approach 1b model (no API, runs in milliseconds).

Input : output/features/<key>.json, <key>_speech.json, output/evidence/<key>_prosody.json
        output/scores/<key>.json   (Approach 1 dimension bands, copied through if present)
        models/approach_1b/score_model.joblib   (python approaches/approach_1b_learned/train.py)
Output: output/scores_1b/<key>.json
  overall.predicted_rating   model output on the AVI 1-5 recruiter scale (bunches near 3)
  overall.overall_score      0-100 = percentile among AVI training participants
  overall.band               ดีมาก / ดี / ปานกลาง / ควรปรับ / ควรปรับมาก from that percentile
  overall.raised_by / lowered_by   feature groups (length, pace, fillers, voice) that
                             pushed the score up / down most
  overall.reliable           false if a value is far outside the training data
                             (e.g. not an AVI-style clip). No score at all if features are missing.
  dimensions                 Approach 1 bands, unchanged

Usage
  python approaches/approach_1b_learned/score.py                 # all clips with features
  python approaches/approach_1b_learned/score.py --only pilot_01 pilot_02
  python approaches/approach_1b_learned/score.py --force          # overwrite existing results
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base._paths import SCORES_DIR
from base.learned_features import all_keys, clip_row
import learned_model as m1b

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=None, help="score only these clip keys")
    ap.add_argument("--force", action="store_true", help="overwrite existing results")
    args = ap.parse_args()

    bundle = m1b.load()
    keys = args.only or all_keys()
    m1b.SCORES_1B_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"  Approach 1b scoring   model={bundle['version']}   clips={len(keys)}")
    print("=" * 66)
    done = skipped = unscored = 0
    for key in keys:
        out = m1b.SCORES_1B_DIR / f"{key}.json"
        if out.exists() and not args.force:
            skipped += 1
            continue
        row = clip_row(key)
        if len(row) <= 1:
            print(f"  [skip] {key}: no feature files")
            continue
        overall = m1b.score_row(bundle, row)

        rule_path = SCORES_DIR / f"{key}.json"
        rule = json.loads(rule_path.read_text(encoding="utf-8")) if rule_path.exists() else {}
        result = {
            "key": key,
            "approach": "1b_learned",
            "model_version": bundle["version"],
            "scored_at": m1b.now(),
            "overall": overall,
            "rule_overall_score": rule.get("fusion", {}).get("overall_score"),
            "dimensions": rule.get("dimensions", {}),
            "dimension_source": "approach_1_rule bands",
        }
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1

        if overall["overall_score"] is None:
            print(f"  {key:40s}   -   no score: {len(overall['missing_features'])} feature(s) missing "
                  f"(run speech_text.py / prosody.py)")
            unscored += 1
            continue
        flag = ""
        if overall["outside_training_range"]:
            flag = "  ⚠ outside training range: " + ", ".join(
                o["label_th"] for o in overall["outside_training_range"])
        up = ", ".join(g["label_th"] for g in overall["raised_by"]) or "-"
        down = ", ".join(g["label_th"] for g in overall["lowered_by"]) or "-"
        print(f"  {key:40s} {overall['overall_score']:5.1f}  {overall['band']['label_th']:10s} "
              f"(rule {result['rule_overall_score']})  ↑ {up}  ↓ {down}{flag}")

    print(f"\nscored {done - unscored}, no score {unscored}, "
          f"skipped {skipped} existing (use --force to redo)")
    print(f"Saved in {m1b.SCORES_1B_DIR}")


if __name__ == "__main__":
    main()
