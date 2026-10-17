"""
scripts/eval_test_clips.py - RQ2 / RQ3 on the 28 staged test clips.

Ground truth: input/video_test_web/label_*.xlsx (first file = primary annotator, second = agreement).
System output: output/feedback_app/by_coach/test_<clip>__<coach>.json (evidence log + advice).

RQ2 (measurement layer): is each labelled problem present in the evidence log at the right time?
     match = same behaviour, time overlap with the label (+-2 s); whole-clip labels use the rules below.
     Alerts = "major" problem episodes (>= 3 s) and whole-clip flags; an alert that matches no label
     counts as a false alarm (annotators were asked to add any extra problem they saw).
RQ3 (advice): per coach, which labelled problems the final report names (by dimension), how many
     points the checker removed, and advice given on the control clips. Computed for the checker as
     run (v1) and with the action-step bug fixed (v2, re-checking the saved answers - no new LLM run).
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "approaches" / "approach_grounded"))

import openpyxl
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

LABELS = sorted((ROOT / "input" / "video_test_web").glob("label_*.xlsx"))
BY = ROOT / "output" / "feedback_app" / "by_coach"
OUT = ROOT / "report"
TOL = 2.0
MAJOR = 3.0

TYPE = {  # label text (startswith) -> internal type
    "ไม่มีปัญหา": "none", "ก้มหน้า": "head_down", "เริ่มตอบช้า": "slow_start", "หยุดพูดนาน": "long_pause",
    "หันหน้าซ้าย": "head_left", "หันหน้าขวา": "head_right", "มือไม่อยู่ในเฟรม": "hands_hidden",
    "หน้านิ่ง": "neutral_face", "คำตอบสั้นไป": "too_short", "คำตอบยาวไป": "too_long",
    "เนื้อหาขาดตัวอย่าง": "no_example", "พูดเร็วเกินไป": "too_fast", "คำฟุ่มเฟือย": "fillers",
    "มองออกนอกกล้อง": "look_away", "มือขยับมาก": "hands_busy"}
NOT_MEASURED = {"no_example": "content - judged by the coach, not by measurement",
                "fillers": "filler words are not measured (by design)"}
DIM = {"look_away": "eye_contact", "head_down": "head_pose", "head_left": "head_pose",
       "head_right": "head_pose", "hands_hidden": "hand_gesture", "hands_busy": "hand_gesture",
       "neutral_face": "facial_expression", "too_short": "answer_quality", "too_long": "answer_quality",
       "no_example": "answer_quality", "too_fast": "answer_quality", "long_pause": "answer_quality",
       "slow_start": "answer_quality", "fillers": "answer_quality"}
EVENT = {"head_down": ("head", "turned_down"), "head_left": ("head", "turned_left"),
         "head_right": ("head", "turned_right"), "look_away": ("gaze", "looking_away")}


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_labels(path):
    ws = openpyxl.load_workbook(path, data_only=True)["label"]
    rows = []
    for r in ws.iter_rows(min_row=3, values_only=True):
        if not r[1] or not str(r[0]).replace(".0", "").isdigit():
            continue
        t = next((v for k, v in TYPE.items() if str(r[5]).startswith(k)), None)
        rows.append({"clip": Path(str(r[1])).stem, "lang": r[3], "type": t, "label": r[5],
                     "occurred": r[7] in ("เกิด", "เกิดบางส่วน"), "start": num(r[8]), "end": num(r[9])})
    return pd.DataFrame(rows)


def agreement(a, b):
    m = a.merge(b, left_index=True, right_index=True, suffixes=("_1", "_2"))
    same = (m.occurred_1 == m.occurred_2).mean()
    timed = m.dropna(subset=["start_1", "end_1", "start_2", "end_2"])
    overlap = ((timed.start_1 <= timed.end_2 + TOL) & (timed.start_2 <= timed.end_1 + TOL)).mean()
    bdiff = pd.concat([(timed.start_1 - timed.start_2).abs(), (timed.end_1 - timed.end_2).abs()]).mean()
    return {"items": len(m), "occurred_agree": same, "timed": len(timed), "time_overlap": overlap,
            "mean_boundary_diff_s": bdiff}


def summ(log, typ):
    return next((i for i in log["items"] if i["kind"] == "summary" and i["type"] == typ), None)


def speech(log):
    s = next((i["text"] for i in log["items"] if i["kind"] == "summary" and i["text"].startswith("speech")), "")
    g = lambda pat: float(m.group(1)) if (m := re.search(pat, s)) else None
    first, last = g(r"first word at ([\d.]+)s"), g(r"last word ends ([\d.]+)s")
    return {"wpm": g(r"about ([\d.]+) words/min"),
            "dur": (last - first) if first is not None and last is not None else None}


def alerts(log, lang):
    """Major problems the measurement layer reports: (type, start, end) - whole clip = (None, None)."""
    out = []
    for i in log["items"]:
        if i["kind"] != "event":
            continue
        d = i["end"] - i["start"]
        for t, (typ, val) in EVENT.items():
            if i["type"] == typ and i.get("value") == val and d >= MAJOR:
                out.append((t, i["start"], i["end"]))
        if i["type"] == "head" and i.get("value") == "turned_up" and d >= MAJOR:
            out.append(("head_up", i["start"], i["end"]))
        if i["type"] == "pause" and d >= MAJOR:
            out.append(("slow_start" if i.get("value") == "silence_before_answer" else "long_pause",
                        i["start"], i["end"]))
    h, f = summ(log, "hand"), summ(log, "face")
    if h and h.get("share", {}).get("hands_hidden", 0) >= 0.5:
        out.append(("hands_hidden", None, None))
    if h and h.get("share", {}).get("gesturing", 0) >= 0.3:
        out.append(("hands_busy", None, None))
    if f and f.get("share", {}).get("neutral", 0) >= 0.5:
        out.append(("neutral_face", None, None))
    sp = speech(log)
    if sp["dur"] is not None and sp["dur"] < 30:
        out.append(("too_short", None, None))
    if sp["dur"] is not None and sp["dur"] > 150:
        out.append(("too_long", None, None))
    if lang == "en" and sp["wpm"] and sp["wpm"] > 160:
        out.append(("too_fast", None, None))
    return out


def matches(label, alert):
    t, s, e = alert
    if t != label["type"]:
        return False
    if s is None or label["start"] is None:
        return True
    return s <= label["end"] + TOL and label["start"] <= e + TOL


def reverify(fb):
    """v2: re-check the saved final answer + the points v1 removed with the fixed checker."""
    import importlib
    import verify
    importlib.reload(verify)
    pre = {k: fb.get(k) for k in ("overall_summary_en", "overall_summary_th", "improved_answer")}
    for sec in ("strengths", "improvements"):
        pre[sec] = list(fb.get(sec, [])) + [{k: v for k, v in r.items() if k != "section"}
                                            for r in fb["verification"].get("removed", [])
                                            if r.get("section") == sec]
    if not pre["improved_answer"]:
        ia = [r for r in fb["verification"].get("removed", []) if r.get("section") == "improved_answer"]
        pre["improved_answer"] = {k: v for k, v in ia[0].items() if k != "section"} if ia else {}
    rep = verify.check(pre, fb["evidence_log"])
    final, removed = verify.drop_failed(pre, rep)
    return final, removed


def main():
    lab = read_labels(LABELS[0])
    agree = agreement(lab, read_labels(LABELS[1])) if len(LABELS) > 1 else {}
    clips = sorted(lab["clip"].unique())
    logs = {}
    for c in clips:
        f = BY / f"test_{c}__ollama.json"
        f = f if f.exists() else BY / f"test_{c}__local.json"
        logs[c] = json.loads(f.read_text(encoding="utf-8"))["evidence_log"]

    # ---------------- RQ2
    rows, alert_rows = [], []
    for c in clips:
        L = lab[lab["clip"] == c]
        lang = "th" if L.lang.iloc[0] == "ไทย" else "en"
        al = alerts(logs[c], lang)
        used = set()
        for _, r in L.iterrows():
            if r.type in ("none",) or r.type in NOT_MEASURED or (r.type == "too_fast" and lang == "th"):
                why = NOT_MEASURED.get(r.type, "Thai speech rate has no norm (not flagged by design)") \
                    if r.type != "none" else "control"
                rows.append({**r, "detected": None, "note": why})
                continue
            hit = [k for k, a in enumerate(al) if matches(r, a)]
            used.update(hit)
            rows.append({**r, "detected": bool(hit) if r.occurred else None,
                         "false_alarm": bool(hit) and not r.occurred, "note": ""})
        for k, a in enumerate(al):
            alert_rows.append({"clip": c, "type": a[0], "start": a[1], "end": a[2], "matched": k in used})
    det = pd.DataFrame(rows)
    al = pd.DataFrame(alert_rows)
    scored = det[det.detected.notna()].copy()
    scored["detected"] = scored["detected"].astype(float)
    recall = scored.detected.mean()
    recall_by = scored.groupby("type").detected.agg(["sum", "count"])
    precision = al.matched.mean()
    control = al[al["clip"].str.endswith(("C01", "C08"))]

    # ---------------- RQ3
    adv = []
    for coach in ("ollama", "local"):
        for c in clips:
            f = BY / f"test_{c}__{coach}.json"
            if not f.exists():
                continue
            fb = json.loads(f.read_text(encoding="utf-8"))
            L = lab[(lab["clip"] == c) & lab.occurred & ~lab.type.isin(["none"])]
            for ver in ("v1", "v2"):
                if ver == "v1":
                    final, removed = fb, fb["verification"].get("removed", []) or []
                else:
                    final, removed = reverify(fb)
                dims = [x["dimension"] for x in final.get("improvements", [])]
                covered = [DIM.get(t) in dims for t in L.type]
                adv.append({"coach": coach, "version": ver, "clip": c,
                            "control": c.endswith(("C01", "C08")),
                            "labelled": len(L), "covered": sum(covered),
                            "improvements": len(dims), "strengths": len(final.get("strengths", [])),
                            "removed_points": sum(r.get("section") != "improved_answer" for r in removed),
                            "improvements_on_unlabelled_dim": sum(d not in {DIM.get(t) for t in L.type} for d in dims),
                            "seconds": fb.get("seconds_coach")})
    adv = pd.DataFrame(adv)
    a = adv[~adv.control]
    rq3 = a.groupby(["coach", "version"]).agg(labelled=("labelled", "sum"), covered=("covered", "sum"),
                                               improvements=("improvements", "sum"),
                                               removed=("removed_points", "sum"),
                                               off_label=("improvements_on_unlabelled_dim", "sum"))
    rq3["coverage"] = rq3.covered / rq3.labelled
    ctrl = adv[adv.control].groupby(["coach", "version"]).improvements.sum()
    secs = adv[(adv.coach == "ollama") & (adv.version == "v1")].seconds

    OUT.mkdir(exist_ok=True)
    det.to_csv(OUT / "test_rq2_labels.csv", index=False, encoding="utf-8-sig")
    al.to_csv(OUT / "test_rq2_alerts.csv", index=False, encoding="utf-8-sig")
    adv.to_csv(OUT / "test_rq3_advice.csv", index=False, encoding="utf-8-sig")
    res = {"agreement": agree,
           "rq2": {"recall": recall, "detected": int(scored.detected.sum()), "scored": len(scored),
                   "not_scored": det[det.detected.isna() & (det.type != "none")].groupby("note").size().to_dict(),
                   "by_type": {t: f"{int(s)}/{int(n)}" for t, (s, n) in recall_by.iterrows()},
                   "by_lang": scored.groupby("lang").detected.mean().round(3).to_dict(),
                   "by_person": scored.groupby(scored["clip"].str[:2]).detected.mean().round(3).to_dict(),
                   "alerts": len(al), "alerts_matched": int(al.matched.sum()), "precision": precision,
                   "unmatched_by_type": al[~al.matched].type.value_counts().to_dict(),
                   "control_clip_alerts": control.type.value_counts().to_dict(),
                   "false_alarm_on_negative_labels": int(det.get("false_alarm", pd.Series(dtype=bool)).fillna(False).sum())},
           "rq3": {f"{k[0]}_{k[1]}": {kk: (round(vv, 3) if isinstance(vv, float) else int(vv))
                                      for kk, vv in row.items()} for k, row in rq3.iterrows()},
           "rq3_control_improvements": {f"{k[0]}_{k[1]}": int(v) for k, v in ctrl.items()},
           "typhoon_seconds": {"mean": round(secs.mean(), 1), "min": secs.min(), "max": secs.max()}}
    (OUT / "test_eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
