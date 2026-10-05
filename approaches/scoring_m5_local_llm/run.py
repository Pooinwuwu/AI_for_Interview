"""
approaches/scoring_m5_local_llm/run.py   (Part 1, method M5: local LLM, zero-shot)

A small LLM running on YOUR machine (Ollama) reads the question + transcript and rates
the answer the way a recruiter would. No training: this tests whether an off-the-shelf
LLM's judgement of the CONTENT agrees with AVI recruiters, compared with the trained
methods M0-M4.

Privacy: AVI transcripts must not leave the computer, so only a local Ollama server is
allowed (localhost / 127.0.0.1). The script refuses any other host.

Setup (once)
  1. Install Ollama (https://ollama.com) and start it
  2. ollama pull qwen2.5:3b          (~2 GB, fits 4 GB VRAM; or llama3.2:3b / gemma3:4b)

Usage
  python approaches/scoring_m5_local_llm/run.py                       # all AVI clips with a transcript
  python approaches/scoring_m5_local_llm/run.py --model llama3.2:3b
  python approaches/scoring_m5_local_llm/run.py --only <key> --force
Output: output/scores_m5/<model>/<key>.json  {rating 1-10, reason, ...}
Then:   python validation/train_score_model.py --text-emb minilm --llm qwen2.5:3b
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from base._paths import EVIDENCE_DIR, OUTPUT_DIR
from base.dataset.avi import parse_key
from base.llm_common import question_context

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUT_ROOT = OUTPUT_DIR / "scores_m5"
PROMPT_VERSION = "m5-v1"
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "rating": {"type": "integer", "minimum": 1, "maximum": 10},
    },
    "required": ["reason", "rating"],
}


def slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def out_dir(model: str) -> Path:
    return OUT_ROOT / slug(model)


def transcript_text(key: str) -> str:
    p = EVIDENCE_DIR / f"{key}_transcript.json"
    if not p.exists():
        return ""
    segs = json.loads(p.read_text(encoding="utf-8")).get("segments", [])
    return " ".join((s.get("text") or "").strip() for s in segs).strip()


def build_prompt(key: str, text: str) -> str:
    q = question_context(key)
    focus = "\n".join(f"- {f}" for f in q["focus"])
    return f"""You are an experienced recruiter screening candidates for a management traineeship.
Setting: {q['setting']}
Interview question: "{q['text']}"

What a strong answer does:
{focus}

Transcript of the candidate's answer (automatic speech recognition, may contain errors;
filler words are not transcribed):
\"\"\"{text}\"\"\"

Judge ONLY the content and structure of what was said (you cannot see or hear the candidate).
First give a one-sentence reason, then a rating from 1 to 10:
1-2 = very weak, 3-4 = weak, 5-6 = average, 7-8 = good, 9-10 = excellent.
How likely would you be to invite this candidate to the next round, based on this answer?"""


class Ollama:
    def __init__(self, host: str, model: str):
        h = urlparse(host).hostname
        if h not in LOCAL_HOSTS:
            sys.exit(f"[STOP] {host} is not a local Ollama server. AVI transcripts must stay on "
                     f"this computer (AVI user agreement).")
        self.url = host.rstrip("/") + "/api/chat"
        self.tags = host.rstrip("/") + "/api/tags"
        self.model = model

    def check(self):
        try:
            with urllib.request.urlopen(self.tags, timeout=5) as r:
                names = [m["name"] for m in json.load(r).get("models", [])]
        except urllib.error.URLError:
            sys.exit("[ERROR] Ollama is not running. Start the Ollama app (or `ollama serve`).")
        if not any(n == self.model or n.split(":")[0] == self.model for n in names):
            sys.exit(f"[ERROR] model '{self.model}' not found. Run:  ollama pull {self.model}\n"
                     f"        installed: {', '.join(names) or 'none'}")

    def rate(self, prompt: str) -> dict:
        body = json.dumps({
            "model": self.model, "stream": False, "format": SCHEMA,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096},
        }).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        last = "rating outside 1-10"
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    content = json.load(r)["message"]["content"]
                d = json.loads(content)
                rating = int(d["rating"])
                if 1 <= rating <= 10:
                    return {"rating": rating, "reason": str(d.get("reason", ""))[:500]}
            except (urllib.error.URLError, ValueError, KeyError, TypeError) as e:
                last = str(e)[:200]
                time.sleep(2)
        raise RuntimeError(f"no valid rating after 3 tries: {last}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--include-non-avi", action="store_true", help="also rate pilot / own clips")
    args = ap.parse_args()

    llm = Ollama(args.host, args.model)
    llm.check()
    keys = args.only or sorted(p.stem.removesuffix("_transcript")
                               for p in EVIDENCE_DIR.glob("*_transcript.json"))
    if not args.only and not args.include_non_avi:
        keys = [k for k in keys if parse_key(k) is not None]
    od = out_dir(args.model)
    od.mkdir(parents=True, exist_ok=True)
    print(f"[M5] model={args.model}  clips={len(keys)}  -> {od}")

    done = skipped = failed = 0
    t0 = time.time()
    for n, key in enumerate(keys, 1):
        out = od / f"{key}.json"
        if out.exists() and not args.force:
            skipped += 1
            continue
        text = transcript_text(key)
        if not text:
            print(f"  [skip] {key}: no transcript")
            continue
        try:
            res = llm.rate(build_prompt(key, text))
        except Exception as e:
            print(f"  [FAIL] {key}: {e}")
            failed += 1
            continue
        out.write_text(json.dumps({"key": key, "model": args.model, "prompt_version": PROMPT_VERSION,
                                   "temperature": 0, **res}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        done += 1
        if done % 20 == 0:
            rate = (time.time() - t0) / done
            print(f"  {n}/{len(keys)}  ({rate:.1f} s/clip)")
    print(f"[M5] rated {done}, skipped {skipped} existing, failed {failed}")


if __name__ == "__main__":
    main()
