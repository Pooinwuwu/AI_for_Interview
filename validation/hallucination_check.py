"""
validation/hallucination_check.py   (RQ1)

Does the Tier-1 evidence log reduce hallucination in LLM coaching feedback?

For every feedback file (Approach 2 = video only, Approach 3 = video + evidence log,
and any other approach that cites times):
  1. Extract claims that point at a moment in the clip
       "at 12.4s", "around 0:45", "between 30 and 35 seconds", "(12.4s)", "the 20-second mark"
     Durations such as "60-90 seconds is ideal" or "for 3 seconds" are NOT claims.
  2. Decide what the claim is about from keywords in the same sentence:
       gaze, head, hand, face, pause, filler, quote ("..." the candidate said)
     and, where possible, what the log should show (e.g. "looked away" -> looking_away).
  3. Check the FULL evidence log (output/evidence/<key>_evidence.json, incl. words)
     within +/- TOL seconds of the claimed time:
       supported     matching event is there
       contradicted  events of that kind are there, but with a different value
       unsupported   no event of that kind near that time
       out_of_range  claimed time is after the end of the clip  (certain hallucination)
       unverifiable  topic not recognised (not counted in the rate)
  4. Report per approach: claims per clip, % per verdict, and
       hallucination rate = (contradicted + unsupported + out_of_range) / verifiable claims
     plus a paired comparison on clips that both approaches processed
     (mean difference + 95% bootstrap CI over clips).
  5. Export a BLIND audit sample for human raters (no approach names, no auto verdict).

The automatic check is a proxy. Report it together with the human audit:
    python validation/hallucination_check.py                    # check + export audit sample
    python validation/hallucination_check.py --score-audit       # after raters filled the sample

Outputs (report/):
    hallucination_claims.csv          every claim with its verdict
    hallucination_summary.csv         per approach
    hallucination_audit_sample.csv    give to raters (blind)
    hallucination_audit_key.csv       keep private: approach + auto verdict per audit_id
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base._paths import OUTPUT_DIR, EVIDENCE_DIR, REPORT_DIR

import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CONFIG
# ============================================================

TOL = 1.0   # seconds of slack around a claimed time

APPROACH_DIRS = {
    "rule":   OUTPUT_DIR / "feedback",
    "mllm":   OUTPUT_DIR / "feedback_mllm",
    "hybrid": OUTPUT_DIR / "feedback_hybrid",
    # grounded approaches C / D (approaches/approach_grounded/run.py)
    "grounded":       OUTPUT_DIR / "feedback_grounded",
    "grounded_video": OUTPUT_DIR / "feedback_grounded_video",
}
# "<approach>_draft" = the same files, but the draft written BEFORE the checker
# (grounded vs grounded_draft = how much the checker removes)
DRAFT_SUFFIX = "_draft"

# paired comparisons printed when both sides have results
PAIRS = [("mllm", "hybrid"), ("mllm", "grounded_video"), ("grounded_video", "grounded"),
         ("grounded_draft", "grounded"), ("grounded_video_draft", "grounded_video")]

FILLERS = {"um", "uh", "er", "ah", "hmm", "erm", "like"}

# keyword -> topic. Order matters only for ties (nearest keyword to the time wins).
TOPIC_PATTERNS = {
    "gaze":   r"eye contact|eyes?\b|gaze|glanc\w*|camera|lens|look(?:s|ed|ing)? away|stare\w*",
    "head":   r"\bhead\b|nod\w*|tilt\w*|chin|look(?:s|ed|ing)? (?:down|up)\b|turn(?:s|ed|ing)? (?:left|right|away)",
    "hand":   r"\bhands?\b|gestur\w*|fidget\w*|fingers?|arms?\b|touch\w* (?:your |his |her )?(?:face|hair|nose|chin)",
    "face":   r"smil\w*|grin\w*|laugh\w*|frown\w*|facial|expression\w*|\bface\b",
    "pause":  r"paus\w*|silen\w*|hesitat\w*",
    "filler": r"filler\w*|\bums?\b|\buhs?\b|\"(?:um|uh|like)\"",
}

TIME_UNIT = r"(?:(?:s|sec|secs|seconds?)\b)"
NUM = r"(\d{1,3}(?:\.\d+)?)"
MMSS = r"(\d{1,2}):([0-5]\d)"

# time POINTS / SPANS we treat as claims
RANGE_RE = re.compile(
    rf"(?:between|from)\s+{NUM}\s*{TIME_UNIT}?\s*(?:and|to|-|–)\s*{NUM}\s*{TIME_UNIT}"
    rf"|\(\s*{NUM}\s*{TIME_UNIT}?\s*[-–]\s*{NUM}\s*{TIME_UNIT}\s*\)"
    rf"|\b(?:at|around|near|by|until|till)\s+{NUM}\s*[-–]\s*{NUM}\s*{TIME_UNIT}",
    re.I)
POINT_RE = re.compile(
    rf"\b(?:at|around|near|by|until|till|after|before|starting at|from)\s+(?:the\s+)?(?:~\s*)?{NUM}\s*{TIME_UNIT}"
    rf"|\(\s*(?:at\s+)?{NUM}\s*{TIME_UNIT}\s*\)"
    rf"|\b{NUM}\s*-?\s*(?:s|second)\s+mark\b",
    re.I)
MMSS_RE = re.compile(rf"\b(?:at|around|near|by|from|until|\()?\s*{MMSS}\b", re.I)


# ============================================================
# EXTRACT CLAIMS
# ============================================================

def english_texts(fb: dict):
    """(field, text, dimension) for every English free-text field."""
    out = []
    for f in ("overall_summary_en",):
        if fb.get(f):
            out.append((f, fb[f], None))
    for s in fb.get("strengths_en", []) or []:
        out.append(("strengths_en", s, None))
    for imp in fb.get("improvements", []) or []:
        dim = imp.get("dimension")
        for f in ("issue_en", "suggestion_en"):
            if imp.get(f):
                out.append((f"improvements.{f}", imp[f], dim))
        for s in imp.get("action_steps_en", []) or []:
            out.append(("improvements.action_steps_en", s, dim))
    ia = fb.get("improved_answer") or {}
    if ia.get("why_better_en"):
        out.append(("improved_answer.why_better_en", ia["why_better_en"], None))
    return out


def sentences(text: str):
    # keep decimals like 12.4 together
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text) if s.strip()]


def find_times(sent: str):
    """List of (t0, t1, char_pos, matched_text)."""
    found, taken = [], []

    def free(a, b):
        return all(b <= x or a >= y for x, y in taken)

    for m in RANGE_RE.finditer(sent):
        nums = [float(g) for g in m.groups() if g is not None]
        if len(nums) >= 2 and free(*m.span()):
            found.append((min(nums[:2]), max(nums[:2]), m.start(), m.group(0)))
            taken.append(m.span())
    for m in POINT_RE.finditer(sent):
        nums = [float(g) for g in m.groups() if g is not None]
        if nums and free(*m.span()):
            found.append((nums[0], nums[0], m.start(), m.group(0)))
            taken.append(m.span())
    for m in MMSS_RE.finditer(sent):
        g = [x for x in m.groups() if x is not None]
        if len(g) >= 2 and free(*m.span()):
            t = int(g[-2]) * 60 + int(g[-1])
            found.append((t, t, m.start(), m.group(0).strip()))
            taken.append(m.span())
    return found


def topic_for(sent: str, pos: int, dimension):
    """Nearest topic keyword to the time mention; quoted speech wins."""
    if re.search(r"[\"“][^\"”]{3,}[\"”]", sent) and not re.search(r"[\"“](?:um|uh|like)[\"”]", sent, re.I):
        return "quote"
    # specific phrases beat single keywords ("touch your face" is about hands)
    if re.search(r"touch\w* (?:your |his |her |their )?(?:face|hair|nose|chin)", sent, re.I):
        return "hand"
    best, best_d = None, 10 ** 9
    for topic, pat in TOPIC_PATTERNS.items():
        for m in re.finditer(pat, sent, re.I):
            d = abs(m.start() - pos)
            if d < best_d:
                best, best_d = topic, d
    if best is None:
        best = {"eye_contact": "gaze", "head_pose": "head", "hand_gesture": "hand",
                "facial_expression": "face"}.get(dimension)
    return best


def expected_values(topic: str, sent: str):
    """Values the log should show for the claim to hold (None = any value of that topic)."""
    s = sent.lower()
    if topic == "gaze":
        if re.search(r"away|glanc|off[- ]camera|break|broke|lost|avoid|down|aside|elsewhere|wander", s):
            return {"looking_away"}
        if re.search(r"eye contact|camera|lens|steady|maintain|direct", s):
            return {"eye_contact"}
    if topic == "head":
        for word, val in (("down", "turned_down"), ("lower", "turned_down"), ("up", "turned_up"),
                          ("left", "turned_left"), ("right", "turned_right")):
            if re.search(rf"\b{word}\w*", s):
                return {val}
        if re.search(r"tilt|turn|nod|mov|shift", s):
            return {"turned_down", "turned_up", "turned_left", "turned_right"}
        if re.search(r"steady|still|centered|centred|straight", s):
            return {"centered"}
    if topic == "hand":
        if re.search(r"hidden|not visible|out of (?:the )?frame|no gestur|absent|invisible", s):
            return {"hands_hidden"}
        if re.search(r"fidget|gestur|mov|wav|point|animated", s):
            return {"gesturing"}
        if re.search(r"still|rest|static|clasp|folded", s):
            return {"hands_still"}
    if topic == "face":
        if re.search(r"smil|grin|laugh|warm", s) and not re.search(r"no smil|without (?:a )?smil|not smil|stopped smil", s):
            return {"smiling"}
        if re.search(r"neutral|serious|flat|blank|no smil|without (?:a )?smil|not smil|stern", s):
            return {"neutral"}
    return None


def extract_claims(fb: dict):
    claims = []
    for field, text, dim in english_texts(fb):
        for sent in sentences(text):
            for t0, t1, pos, mt in find_times(sent):
                topic = topic_for(sent, pos, dim)
                claims.append({
                    "field": field, "dimension": dim, "sentence": sent,
                    "time_text": mt, "t0": t0, "t1": t1, "topic": topic,
                    "expected": sorted(expected_values(topic, sent) or []) if topic else [],
                    "quote": (re.search(r"[\"“]([^\"”]{3,})[\"”]", sent).group(1)
                              if topic == "quote" else None),
                })
    return claims


# ============================================================
# VERIFY AGAINST THE EVIDENCE LOG
# ============================================================

EVENT_TYPE = {"gaze": "gaze", "head": "head", "hand": "hand", "face": "face", "pause": "pause"}


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def verify(claim: dict, events: list, duration: float, tol: float = TOL) -> str:
    t0, t1, topic = claim["t0"], claim["t1"], claim["topic"]
    if t0 > duration + tol:
        return "out_of_range"
    if topic is None:
        return "unverifiable"
    w0, w1 = t0 - tol, t1 + tol
    near = [e for e in events if e["start"] <= w1 and e["end"] >= w0]

    if topic in EVENT_TYPE:
        kind = [e for e in near if e["type"] == EVENT_TYPE[topic]]
        if not kind:
            return "unsupported"
        exp = set(claim["expected"])
        if not exp or any(e["value"] in exp for e in kind):
            return "supported"
        return "contradicted"

    words = [_norm(e["value"]) for e in near if e["type"] == "word"]
    if topic == "filler":
        return "supported" if any(w in FILLERS for w in words) else "unsupported"
    if topic == "quote":
        q = [_norm(w) for w in claim["quote"].split() if _norm(w)]
        if not q:
            return "unverifiable"
        # the quote's words must be spoken near that time (allow a wider window: speech is long)
        wide = [_norm(e["value"]) for e in events
                if e["type"] == "word" and e["start"] <= t1 + 3 * tol and e["end"] >= t0 - 3 * tol]
        hit = sum(1 for w in q if w in wide) / len(q)
        return "supported" if hit >= 0.6 else "unsupported"
    return "unverifiable"


def load_evidence(key: str):
    p = EVIDENCE_DIR / f"{key}_evidence.json"
    if not p.exists():
        return None, None
    ev = json.loads(p.read_text(encoding="utf-8")).get("evidence", [])
    duration = max((e.get("end", 0) for e in ev), default=0.0)
    return ev, duration


# ============================================================
# RUN
# ============================================================

HALLUCINATED = ("contradicted", "unsupported", "out_of_range")


def collect(approaches, tol):
    rows, clips = [], []
    for approach in approaches:
        base = approach.removesuffix(DRAFT_SUFFIX)
        folder = APPROACH_DIRS[base]
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*_feedback.json")):
            key = f.stem.removesuffix("_feedback")
            events, duration = load_evidence(key)
            if events is None:
                print(f"[SKIP] {approach}/{key}: no evidence log")
                continue
            fb = json.loads(f.read_text(encoding="utf-8"))
            if approach.endswith(DRAFT_SUFFIX):
                if "draft" not in fb:
                    continue
                fb = {**fb["draft"], "model": fb.get("model"),
                      "prompt_version": fb.get("prompt_version")}
            claims = extract_claims(fb)
            clips.append({"approach": approach, "key": key, "n_claims": len(claims),
                          "model": fb.get("model"), "prompt_version": fb.get("prompt_version")})
            for c in claims:
                c.update(approach=approach, key=key, clip_duration=round(duration, 2),
                         verdict=verify(c, events, duration, tol),
                         expected=" ".join(c["expected"]))
                rows.append(c)
    return pd.DataFrame(rows), pd.DataFrame(clips)


def per_clip_rate(claims: pd.DataFrame, clips: pd.DataFrame) -> pd.DataFrame:
    v = claims[claims.verdict != "unverifiable"] if not claims.empty else claims
    if v.empty:
        out = clips[["approach", "key"]].copy()
        out["verifiable"], out["hallucinated"] = 0, 0
    else:
        g = v.groupby(["approach", "key"]).verdict
        out = pd.DataFrame({"verifiable": g.size(),
                            "hallucinated": g.apply(lambda s: s.isin(HALLUCINATED).sum())}).reset_index()
        out = clips[["approach", "key"]].merge(out, how="left").fillna(0)
    out["rate"] = np.where(out.verifiable > 0, out.hallucinated / out.verifiable.clip(lower=1), np.nan)
    return out


def summarize(claims: pd.DataFrame, clips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for approach, cg in clips.groupby("approach"):
        cl = claims[claims.approach == approach] if not claims.empty else claims
        n = len(cl)
        verifiable = cl[cl.verdict != "unverifiable"] if n else cl
        row = {"approach": approach, "clips": len(cg), "claims": n,
               "claims_per_clip": round(n / len(cg), 2) if len(cg) else 0,
               "verifiable": len(verifiable)}
        for v in ("supported", "contradicted", "unsupported", "out_of_range", "unverifiable"):
            row[v] = int((cl.verdict == v).sum()) if n else 0
        row["hallucination_rate"] = (round(verifiable.verdict.isin(HALLUCINATED).mean(), 3)
                                     if len(verifiable) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_compare(rates: pd.DataFrame, a: str, b: str, n_boot=2000, seed=0):
    """Clips processed by both approaches with >=1 verifiable claim in each."""
    pa = rates[(rates.approach == a) & (rates.verifiable > 0)].set_index("key").rate
    pb = rates[(rates.approach == b) & (rates.verifiable > 0)].set_index("key").rate
    common = pa.index.intersection(pb.index)
    if len(common) < 3:
        return None
    d = (pa[common] - pb[common]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_clips": len(common), "mean_diff": d.mean(), "ci_low": lo, "ci_high": hi}


def export_audit(claims: pd.DataFrame, n: int, seed: int):
    """Blind sample for raters: stratified per approach, shuffled, no approach/verdict."""
    if claims.empty:
        return 0, None, None
    v = claims[claims.verdict != "unverifiable"]
    per = max(1, n // max(1, v.approach.nunique()))
    parts = [g.sample(min(len(g), per), random_state=seed) for _, g in v.groupby("approach")]
    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample.insert(0, "audit_id", [f"A{i:03d}" for i in range(1, len(sample) + 1)])
    blind = sample[["audit_id", "key", "time_text", "t0", "t1", "sentence"]].copy()
    blind["rater1_true_false"] = ""
    blind["rater2_true_false"] = ""
    blind["notes"] = ""
    key = sample[["audit_id", "approach", "verdict", "topic", "expected"]]
    sample_p = REPORT_DIR / "hallucination_audit_sample.csv"
    key_p = REPORT_DIR / "hallucination_audit_key.csv"
    if sample_p.exists():
        old = pd.read_csv(sample_p, dtype=str).fillna("")
        filled = [c for c in ("rater1_true_false", "rater2_true_false") if c in old]
        if any((old[c].str.strip() != "").any() for c in filled):
            # never overwrite an audit that raters already filled in
            sample_p = REPORT_DIR / "hallucination_audit_sample_new.csv"
            key_p = REPORT_DIR / "hallucination_audit_key_new.csv"
            print("[INFO] existing audit sample is already rated - new sample written to *_new.csv")
    blind.to_csv(sample_p, index=False, encoding="utf-8-sig")
    key.to_csv(key_p, index=False, encoding="utf-8-sig")
    return len(sample), sample_p, key_p


def score_audit():
    """Agreement of auto verdicts with human raters (after the sample was filled in)."""
    from sklearn.metrics import cohen_kappa_score
    s = pd.read_csv(REPORT_DIR / "hallucination_audit_sample.csv", dtype=str).fillna("")
    k = pd.read_csv(REPORT_DIR / "hallucination_audit_key.csv", dtype=str)
    df = s.merge(k, on="audit_id")

    def to_bool(x):
        x = str(x).strip().lower()
        return {"true": 1, "t": 1, "1": 1, "yes": 1, "y": 1,
                "false": 0, "f": 0, "0": 0, "no": 0, "n": 0}.get(x, np.nan)

    df["r1"], df["r2"] = df.rater1_true_false.map(to_bool), df.rater2_true_false.map(to_bool)
    done = df.dropna(subset=["r1"])
    if done.empty:
        sys.exit("No ratings yet: fill rater1_true_false (true/false) in "
                 "report/hallucination_audit_sample.csv")
    both = done.dropna(subset=["r2"])
    # two raters: agree -> that value; disagree -> tie, resolve by discussion (excluded here)
    mean = (done.r1 + done.r2.fillna(done.r1)) / 2
    ties = done[mean == 0.5]
    done = done[mean != 0.5].assign(human=mean[mean != 0.5])
    done = done.assign(auto=(done.verdict == "supported").astype(int))

    print("=" * 60)
    print("  Hallucination audit — auto check vs human raters")
    print("=" * 60)
    if len(both) >= 5:
        print(f"Rater agreement (Cohen's kappa, n={len(both)}): "
              f"{cohen_kappa_score(both.r1, both.r2):.2f}")
    if len(ties):
        print(f"Rater disagreements (excluded, discuss and fill the same value): "
              f"{len(ties)}  -> {', '.join(ties.audit_id)}")
    print(f"Auto vs human agreement (n={len(done)}): {(done.auto == done.human).mean():.1%}")
    print()
    print("Human-judged hallucination rate per approach (share of claims rated false):")
    print(done.groupby("approach").human.apply(lambda s: 1 - s.mean()).round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    names = list(APPROACH_DIRS) + [a + DRAFT_SUFFIX for a in ("grounded", "grounded_video")]
    ap.add_argument("--approaches", nargs="+", choices=names,
                    default=["rule", "mllm", "hybrid", "grounded", "grounded_draft",
                             "grounded_video", "grounded_video_draft"])
    ap.add_argument("--tol", type=float, default=TOL, help="seconds of slack (default 1.0)")
    ap.add_argument("--audit-n", type=int, default=40, help="claims in the human audit sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score-audit", action="store_true",
                    help="score the filled-in audit sample instead of running the check")
    args = ap.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.score_audit:
        score_audit()
        return

    claims, clips = collect(args.approaches, args.tol)
    if clips.empty:
        sys.exit("[ERROR] No feedback files with matching evidence logs found in output/.")

    summary = summarize(claims, clips)
    rates = per_clip_rate(claims, clips)
    claims.to_csv(REPORT_DIR / "hallucination_claims.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORT_DIR / "hallucination_summary.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 160)
    print("=" * 70)
    print(f"  RQ1 hallucination check  (tolerance ±{args.tol}s)")
    print("=" * 70)
    print(summary.to_string(index=False))
    print()
    mixed = clips.groupby("approach").agg(models=("model", "nunique"),
                                          prompts=("prompt_version", "nunique"))
    if (mixed > 1).any().any():
        print("[WARN] some approach has outputs from more than one model / prompt version:")
        print(mixed.to_string())
        print()

    shown = 0
    for a, b in PAIRS:
        res = paired_compare(rates, a, b)
        if res:
            shown += 1
            print(f"Paired, clips with claims from both (n={res['n_clips']}): "
                  f"rate({a}) - rate({b}) = {res['mean_diff']:+.3f} "
                  f"[95% CI {res['ci_low']:+.3f}, {res['ci_high']:+.3f}]  (> 0: {b} hallucinates less)")
    if not shown:
        print("Paired comparisons need >= 3 clips with timed claims from both approaches.")
    print()

    n, sample_p, key_p = export_audit(claims, args.audit_n, args.seed)
    print(f"Saved: {REPORT_DIR / 'hallucination_claims.csv'}")
    print(f"       {REPORT_DIR / 'hallucination_summary.csv'}")
    if n:
        print(f"Audit: {sample_p}  ({n} claims, blind)")
        print(f"       {key_p}  (keep away from raters)")


if __name__ == "__main__":
    main()
