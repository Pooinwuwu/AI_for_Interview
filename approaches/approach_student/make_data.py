"""
approaches/approach_student/make_data.py   (training data for OUR coach model, step 2)

Everything here is synthetic - made-up answers and made-up recordings - so it can be sent to
Gemini and to Colab. No AVI / RecruitView / real participant data is used.

Stages (run in order; each one resumes where it stopped, so just run it again after a 503/429)
  answers   Gemini writes made-up candidate answers          -> output/student_data/answers.jsonl
  teacher   made-up recording + evidence log for each answer, Gemini (teacher) writes the
            feedback, verify.py checks it, 1 revision, failed points dropped
                                                             -> output/student_data/teacher/<id>.json
  export    (prompt -> checked feedback) pairs for fine-tuning, 90% train / 10% val
                                                             -> output/student_data/sft_train.jsonl, sft_val.jsonl
  stats     how much is done / kept

Usage
  python approaches/approach_student/make_data.py answers --per-question 60     # 7 questions x 2 languages
  python approaches/approach_student/make_data.py teacher --limit 5             # try a few first
  python approaches/approach_student/make_data.py teacher
  python approaches/approach_student/make_data.py export
  python approaches/approach_student/make_data.py stats
  add --teacher claude to use Claude instead of Gemini (ANTHROPIC_API_KEY in .env, paid)
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "approaches" / "approach_grounded"))

from base._paths import OUTPUT_DIR
from base.grounded_log import to_prompt_text
from base.llm_common import MODEL_NAME, BusyGuard, ModelBusy, call_with_retry, gen_config
from prompt import FEEDBACK_SCHEMA, build_revision_prompt       # approach_grounded/prompt.py
import questions
import student_prompt
import synth
import verify

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA = OUTPUT_DIR / "student_data"
ANSWERS = DATA / "answers.jsonl"
TEACHER = DATA / "teacher"
LANGS = ("en", "th")
PER_CALL = 5


class Gemini:
    def __init__(self):
        from dotenv import load_dotenv
        from google import genai
        load_dotenv(ROOT / ".env")
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("[ERROR] GEMINI_API_KEY missing in .env")
        self.client = genai.Client(api_key=key)
        self.name = MODEL_NAME

    def json(self, prompt: str, schema: dict, temperature: float) -> dict:
        return call_with_retry(lambda: json.loads(self.client.models.generate_content(
            model=MODEL_NAME, contents=prompt, config=gen_config(schema, temperature)).text))


class Claude:
    """Anthropic API (paid, no free tier). JSON is forced through a tool with the schema.
    .env: ANTHROPIC_API_KEY=...   optional CLAUDE_MODEL=... (default below)"""

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        try:
            import anthropic
        except ImportError:
            sys.exit("[ERROR] pip install anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("[ERROR] ANTHROPIC_API_KEY missing in .env")
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=6)      # SDK waits and retries 429/529
        self.name = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

    def json(self, prompt: str, schema: dict, temperature: float) -> dict:
        a = self.anthropic
        try:
            r = self.client.messages.create(
                model=self.name, max_tokens=8000, temperature=temperature,
                tools=[{"name": "submit", "description": "Return the result.", "input_schema": schema}],
                tool_choice={"type": "tool", "name": "submit"},
                messages=[{"role": "user", "content": prompt}])
        except a.RateLimitError as e:
            raise ModelBusy(f"{self.name} rate limit: {str(e)[:200]}", "quota")
        except a.APIStatusError as e:
            if e.status_code in (500, 503, 529):
                raise ModelBusy(f"{self.name} overloaded: {str(e)[:200]}", "overloaded")
            raise
        return next(b.input for b in r.content if b.type == "tool_use")


TEACHERS = {"gemini": Gemini, "claude": Claude}


def read_answers() -> list:
    if not ANSWERS.exists():
        return []
    return [json.loads(l) for l in ANSWERS.read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------------------------
# stage 1: answers
# ------------------------------------------------------------

def stage_answers(args):
    DATA.mkdir(parents=True, exist_ok=True)
    have = Counter((a["qid"], a["lang"]) for a in read_answers())
    todo = [(q, lang) for q in questions.QUESTIONS for lang in LANGS
            if have[(q[0], lang)] < args.per_question]
    print(f"answers: target {args.per_question} per question x language, "
          f"{len(todo)} combinations still to fill")
    llm, guard = TEACHERS[args.teacher](), BusyGuard()
    rng = random.Random(args.seed + sum(have.values()))
    with ANSWERS.open("a", encoding="utf-8") as f:
        for q, lang in todo:
            qid, focus, en, th = q
            while have[(qid, lang)] < args.per_question:
                plan = synth.answer_plan(min(PER_CALL, args.per_question - have[(qid, lang)]), rng)
                try:
                    out = llm.json(synth.answers_prompt(en, th, lang, plan, rng),
                                   synth.ANSWER_SCHEMA, temperature=1.0)
                except ModelBusy as e:
                    if guard.failed(e):
                        return
                    continue
                except Exception as e:
                    print(f"   [ERROR] {qid} {lang}: {str(e)[:150]}")
                    continue
                guard.ok()
                for a in out.get("answers", []):
                    if len([s for s in a.get("sentences", []) if s.strip()]) < 2:
                        continue
                    have[(qid, lang)] += 1
                    a.update(id=f"{qid}_{lang}_{have[(qid, lang)]:04d}", qid=qid, lang=lang)
                    f.write(json.dumps(a, ensure_ascii=False) + "\n")
                f.flush()
                print(f"   {qid:5s} {lang}  {have[(qid, lang)]}/{args.per_question}")
    print(f"done: {sum(have.values())} answers in {ANSWERS}")


# ------------------------------------------------------------
# stage 2: teacher feedback
# ------------------------------------------------------------

FEEDBACK_KEYS = ("overall_summary_en", "overall_summary_th", "strengths", "improvements",
                 "improved_answer")


def make_example(a: dict) -> dict:
    """Made-up recording + log + prompt for one answer (deterministic per answer id)."""
    seed = int(hashlib.md5(a["id"].encode()).hexdigest()[:8], 16)
    clip = synth.make_clip(a, a["lang"], random.Random(seed), key=f"syn_{a['id']}")
    prompt = student_prompt.build(a["qid"], a["lang"], clip["ratings"], to_prompt_text(clip["log"]))
    return {**clip, "prompt": prompt}


def stage_teacher(args):
    TEACHER.mkdir(parents=True, exist_ok=True)
    answers = [a for a in read_answers() if not (TEACHER / f"{a['id']}.json").exists()]
    random.Random(args.seed).shuffle(answers)          # mix questions / languages if stopped early
    if args.limit:
        answers = answers[:args.limit]
    llm, guard = TEACHERS[args.teacher](), BusyGuard()
    print(f"teacher: model={llm.name}  {len(answers)} answers to do")
    for n, a in enumerate(answers, 1):
        ex = make_example(a)
        full = f"{student_prompt.SYSTEM}\n\n{ex['prompt']}"
        try:
            draft = llm.json(full, FEEDBACK_SCHEMA, temperature=0.2)
            rounds = [verify.check(draft, ex["log"])]
            current = draft
            if rounds[0]["problems"]:
                current = llm.json(build_revision_prompt(full, json.dumps(draft, ensure_ascii=False),
                                                         rounds[0]["problems"]),
                                   FEEDBACK_SCHEMA, temperature=0.2)
                rounds.append(verify.check(current, ex["log"]))
        except ModelBusy as e:
            if guard.failed(e):
                return
            continue
        except Exception as e:
            print(f"   [ERROR] {a['id']}: {str(e)[:150]}")
            continue
        guard.ok()
        final, removed = verify.drop_failed(current, rounds[-1])
        final = {k: final[k] for k in FEEDBACK_KEYS if k in final}
        keep = bool(final.get("improvements")) and not rounds[-1]["summary_problems"]
        rec = {"id": a["id"], "qid": a["qid"], "lang": a["lang"], "quality": a["quality"],
               "flaws": a["flaws"], "ratings": ex["ratings"], "profile": ex["profile"],
               "prompt": ex["prompt"], "final": final, "keep": keep,
               "draft_problems": len(rounds[0]["problems"]), "revised": len(rounds) > 1,
               "removed": len(removed), "teacher": llm.name,
               "prompt_version": student_prompt.STUDENT_PROMPT_VERSION, "log": ex["log"]}
        (TEACHER / f"{a['id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
        print(f"   [{n}/{len(answers)}] {a['id']:16s} {a['quality']:7s} draft problems "
              f"{rec['draft_problems']}, removed {rec['removed']}, {'kept' if keep else 'NOT kept'}")


# ------------------------------------------------------------
# stage 3: export
# ------------------------------------------------------------

def is_val(aid: str) -> bool:
    return int(hashlib.md5(aid.encode()).hexdigest()[:8], 16) % 10 == 0


def stage_export(args):
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(TEACHER.glob("*.json"))]
    recs = [r for r in recs if r["keep"]]
    split = {"train": [], "val": []}
    for r in recs:
        msg = {"id": r["id"], "messages": [
            {"role": "system", "content": student_prompt.SYSTEM},
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": json.dumps(r["final"], ensure_ascii=False,
                                                        separators=(",", ":"))}]}
        split["val" if is_val(r["id"]) else "train"].append(msg)
    for name, rows in split.items():
        p = DATA / f"sft_{name}.jsonl"
        p.write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in rows), encoding="utf-8")
        print(f"{name}: {len(rows)} examples -> {p}")
    print("Upload both files to Colab (they are synthetic, no participant data).")


def stage_stats(args):
    answers = read_answers()
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in TEACHER.glob("*.json")]
    print(f"answers: {len(answers)}  " + str(dict(Counter((a['lang'], a['quality']) for a in answers))))
    if recs:
        kept = [r for r in recs if r["keep"]]
        print(f"teacher: {len(recs)} done, {len(kept)} kept ({len(kept) / len(recs):.0%})")
        print(f"  drafts with checker problems: {sum(r['draft_problems'] > 0 for r in recs)}"
              f"   points removed after revision: {sum(r['removed'] for r in recs)}")
        print("  kept by language: " + str(dict(Counter(r['lang'] for r in kept))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["answers", "teacher", "export", "stats"])
    ap.add_argument("--per-question", type=int, default=60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--teacher", choices=list(TEACHERS), default="gemini",
                    help="who writes the answers / feedback (data is synthetic, any provider is fine)")
    args = ap.parse_args()
    {"answers": stage_answers, "teacher": stage_teacher, "export": stage_export,
     "stats": stage_stats}[args.stage](args)


if __name__ == "__main__":
    main()
