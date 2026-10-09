"""
approaches/coach_distill/make_data.py   (training data for the fine-tuned coach, RQ2)

For each AVI clip (train + val people only, never test):
  1. evidence log from Tier 1           base/grounded_log.py   (S/E/P/T items)
  2. a LOCAL teacher LLM writes coaching feedback that cites those IDs (Ollama, e.g. qwen2.5:7b)
  3. the checker (approach_grounded/verify.py) finds unsupported points -> one revision round
  4. points that still fail are dropped
The (prompt, final feedback) pair is one supervised training example for the small coach.

Privacy: AVI data stays on this computer (AVI6 user agreement, items 2 and 7). The script refuses
any Ollama host that is not localhost. Outputs live in output/ (gitignored); never publish them.

Usage
  ollama pull qwen2.5:7b
  python approaches/coach_distill/make_data.py --limit 300           # clips with video evidence first
  python approaches/coach_distill/make_data.py --export              # -> output/coach_data/sft_*.jsonl
  python approaches/coach_distill/make_data.py --audit 50            # -> report/coach_data_audit.csv
Re-running skips clips already done (--force to redo).
"""

import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "approaches" / "approach_grounded"))

from base._paths import EVIDENCE_DIR, OUTPUT_DIR, REPORT_DIR
from base.dataset.avi import load_labels, parse_key
from base.grounded_log import build, to_prompt_text
from base.llm_common import question_block
import copy

from prompt import FEEDBACK_SCHEMA, build_revision_prompt
from verify import check, drop_failed

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

COACH_VERSION = "coach-v2"


def _strip_th(node):
    """Same schema without the *_th fields: Thai text costs many tokens on a local model;
    the English feedback is translated for the user later (own clips only)."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "properties":
                out[k] = {pk: _strip_th(pv) for pk, pv in v.items() if not pk.endswith("_th")}
            elif k == "required":
                out[k] = [r for r in v if not r.endswith("_th")]
            else:
                out[k] = _strip_th(v)
        return out
    return node


SCHEMAS = {"both": FEEDBACK_SCHEMA, "en": _strip_th(copy.deepcopy(FEEDBACK_SCHEMA))}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
DATA_ROOT = OUTPUT_DIR / "coach_data"


# ============================================================
# PROMPT  (the same text is the student's input at training and at use time)
# ============================================================

def coach_prompt(key: str, log_text: str, lang: str = "en") -> str:
    return f"""You are an interview coach. Coach the candidate on this one answer.

{question_block(key)}

EVIDENCE LOG (measured by computer vision / speech tools; cite the IDs in [ ])
{log_text}

RULES
• Every strength and improvement must list the evidence_ids that support it.
  Visual points cite S/E items of that behaviour; speech points cite S/P items;
  content points cite T items (what was said).
• Only mention a time if it is the time of an item you cite (e.g. "41-56s").
• Do not claim anything the log does not show. A behaviour marked "not measured" must not
  be judged. Filler words are NOT measured: do not count or quote fillers.
• Content matters: judge what was said against "A STRONG ANSWER" using the transcript.
  If the answer lacks a concrete example, say where one should go (cite the T item).
• "dimension" must match the evidence: points citing T items are answer_quality; gaze ->
  eye_contact; head -> head_pose; hand -> hand_gesture; face -> facial_expression.
• If the answer misses something in "A STRONG ANSWER", at least one improvement must say what
  is missing and where (cite the T item). Do not praise content the transcript does not show.
• overall_summary: 2 sentences that only repeat the strengths and improvements you listed.
• Give at most 3 strengths and 3 improvements, most important first. No numeric scores.
{"• Write short, plain English. Keep each point to 1-2 sentences." if lang == "en" else "• English fields natural; _th fields same meaning in Thai."}
• improved_answer: rewrite the first 1-2 sentences as a stronger opening for THIS question.
  Use only facts the candidate said. NEVER invent jobs, numbers or events; where a real example
  is needed write a placeholder such as [your own example: situation - what you did - result].
"""


# ============================================================
# LOCAL LLM
# ============================================================

class Ollama:
    def __init__(self, host: str, model: str, num_ctx: int, schema: dict):
        if urlparse(host).hostname not in LOCAL_HOSTS:
            sys.exit(f"[STOP] {host} is not local. AVI data must stay on this computer.")
        self.base, self.model, self.num_ctx, self.schema = host.rstrip("/"), model, num_ctx, schema

    def check(self):
        try:
            with urllib.request.urlopen(self.base + "/api/tags", timeout=5) as r:
                names = [m["name"] for m in json.load(r).get("models", [])]
        except urllib.error.URLError:
            sys.exit("[ERROR] Ollama is not running. Start the Ollama app (or `ollama serve`).")
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            sys.exit(f"[ERROR] model '{self.model}' not found. Run:  ollama pull {self.model}")

    def ask(self, prompt: str) -> dict:
        body = json.dumps({
            "model": self.model, "stream": False, "format": self.schema,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.2, "seed": 42, "num_ctx": self.num_ctx,
                        "num_predict": 1200},
        }).encode("utf-8")
        req = urllib.request.Request(self.base + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        last = ""
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    return json.loads(json.load(r)["message"]["content"])
            except (urllib.error.URLError, ValueError, KeyError, TypeError) as e:
                last = str(e)[:200]
                time.sleep(3)
        raise RuntimeError(f"no valid JSON after 3 tries: {last}")


# ============================================================
# WHICH CLIPS
# ============================================================

def pool_keys(splits, people_split):
    """AVI clips of train/val people that have a transcript; clips with video evidence first."""
    keys = []
    for p in EVIDENCE_DIR.glob("*_transcript.json"):
        key = p.stem.removesuffix("_transcript")
        avi = parse_key(key)
        if avi and people_split.get(avi["participant_id"]) in splits:
            keys.append(key)
    has_video = lambda k: (EVIDENCE_DIR / f"{k}_evidence.json").exists()
    return sorted(keys, key=lambda k: (not has_video(k), k))


SOURCE_DIM = {"transcript": "answer_quality", "pause": "answer_quality", "speech": "answer_quality",
              "gaze": "eye_contact", "head": "head_pose", "hand": "hand_gesture",
              "face": "facial_expression"}


def repair_dimensions(feedback: dict, log: dict) -> int:
    """Small models often put a correct point under the wrong dimension label (e.g. a content
    point citing T2 labelled eye_contact). If every cited item points to one other dimension,
    relabel it instead of letting the checker drop a correct point. Returns how many changed."""
    types = {i["id"]: i["type"] for i in log["items"]}
    changed = 0
    for section in ("strengths", "improvements"):
        for item in feedback.get(section, []) or []:
            dims = {SOURCE_DIM.get(types.get(i)) for i in item.get("evidence_ids", []) or []
                    if types.get(i) in SOURCE_DIM}
            if len(dims) == 1:
                d = dims.pop()
                if item.get("dimension") != d:
                    item["dimension"] = d
                    changed += 1
    return changed


def is_done(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("coach_version") == COACH_VERSION
    except ValueError:
        return False


def generate(args, llm, keys, out_dir):
    done = skipped = failed = 0
    t0 = time.time()
    todo = [k for k in keys if args.force or not is_done(out_dir / f"{k}.json")]
    skipped = len(keys) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"[coach-data] teacher={args.model}  to do={len(todo)}  already done={skipped}  -> {out_dir}")
    for n, key in enumerate(todo, 1):
        t_clip = time.time()
        log = build(key)
        if not any(i["kind"] == "transcript" for i in log["items"]):
            continue
        prompt = coach_prompt(key, to_prompt_text(log), args.lang)
        try:
            draft = llm.ask(prompt)
            relabeled = repair_dimensions(draft, log)
            rep1 = check(draft, log)
            final, rep2 = draft, rep1
            if rep1["problems"]:
                final = llm.ask(build_revision_prompt(prompt, json.dumps(draft, ensure_ascii=False),
                                                      rep1["problems"]))
                relabeled += repair_dimensions(final, log)
                rep2 = check(final, log)
            final, removed = drop_failed(final, rep2)
        except Exception as e:
            print(f"  [FAIL] {key}: {e}")
            failed += 1
            continue
        (out_dir / f"{key}.json").write_text(json.dumps({
            "key": key, "teacher": args.model, "coach_version": COACH_VERSION, "lang": args.lang,
            "seconds": round(time.time() - t_clip, 1),
            "has_video_evidence": (EVIDENCE_DIR / f"{key}_evidence.json").exists(),
            "prompt": prompt, "draft": draft, "final": final, "removed": removed,
            "draft_failed": rep1["n_failed"], "draft_items": rep1["n_items"], "relabeled": relabeled,
            "final_items": len(final.get("strengths", [])) + len(final.get("improvements", [])),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1
        if done % 10 == 0 or n == len(todo):
            per = (time.time() - t0) / done
            print(f"  {n}/{len(todo)}  {per:.0f} s/clip  ~{per * (len(todo) - n) / 3600:.1f} h left")
    print(f"[coach-data] made {done}, skipped {skipped} existing, failed {failed}")


# ============================================================
# EXPORT + AUDIT
# ============================================================

def load_done(out_dir):
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*.json"))]
    return [d for d in docs if d.get("coach_version") == COACH_VERSION]


def export(out_dir, people_split, min_items):
    """Chat-format JSONL; AVI train people -> train file, AVI val people -> val file."""
    files = {"train": [], "val": []}
    dropped = 0
    for d in load_done(out_dir):
        if d["final_items"] < min_items:
            dropped += 1
            continue
        split = people_split.get(parse_key(d["key"])["participant_id"])
        if split not in files:
            continue
        files[split].append({"key": d["key"], "messages": [
            {"role": "user", "content": d["prompt"]},
            {"role": "assistant", "content": json.dumps(d["final"], ensure_ascii=False)}]})
    for split, rows in files.items():
        p = out_dir.parent / f"sft_{split}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {p}  {len(rows)} examples")
    print(f"  left out {dropped} clips with fewer than {min_items} supported points")


def audit(out_dir, n, seed=0):
    """CSV for a human check of training labels: one row per feedback point."""
    docs = load_done(out_dir)
    random.Random(seed).shuffle(docs)
    rows = []
    for d in docs[:n]:
        for section in ("strengths", "improvements"):
            for item in d["final"].get(section, []) or []:
                rows.append({"key": d["key"], "section": section, "dimension": item.get("dimension"),
                             "text_en": item.get("text_en") or item.get("issue_en"),
                             "evidence_ids": " ".join(item.get("evidence_ids", [])),
                             "correct (y/n/unsure)": "", "useful (y/n)": "", "note": ""})
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    p = REPORT_DIR / "coach_data_audit.csv"
    if p.exists():
        p = REPORT_DIR / "coach_data_audit_new.csv"      # never overwrite a rated sheet
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["key"])
        w.writeheader()
        w.writerows(rows)
    print(f"  {p}  {len(rows)} points from {min(n, len(docs))} clips - open it next to the log "
          f"(output/coach_data/<teacher>/<key>.json, field 'prompt') and fill the empty columns")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:7b", help="local teacher model (Ollama)")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=None, help="at most this many new clips this run")
    ap.add_argument("--num-ctx", type=int, default=6144)
    ap.add_argument("--lang", choices=["en", "both"], default="en",
                    help="en = English only (about half the tokens, much faster on a local model)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--export", action="store_true", help="write sft_train / sft_val JSONL")
    ap.add_argument("--min-items", type=int, default=2, help="export: minimum supported points")
    ap.add_argument("--audit", type=int, default=0, help="write a human-check CSV for N clips")
    args = ap.parse_args()

    labels = load_labels()
    people_split = dict(zip(labels.participant_id, labels.split))     # train / val / test
    out_dir = DATA_ROOT / args.model.replace(":", "_").replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.export or args.audit:
        if args.export:
            export(out_dir, people_split, args.min_items)
        if args.audit:
            audit(out_dir, args.audit)
        return

    keys = pool_keys(set(args.splits), people_split)
    if any(people_split.get(parse_key(k)["participant_id"]) == "test" for k in keys):
        sys.exit("[ERROR] a test participant reached the pool")
    llm = Ollama(args.host, args.model, args.num_ctx, SCHEMAS[args.lang])
    llm.check()
    generate(args, llm, keys, out_dir)


if __name__ == "__main__":
    main()
