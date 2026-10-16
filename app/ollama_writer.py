"""
app/ollama_writer.py - coach writer that runs a local LLM through Ollama (approach C-local).

Same prompt, same FEEDBACK_SCHEMA and same verify.py as the Gemini coach; only the model
changes. Nothing leaves this computer: the host must be localhost.

Settings (environment variables or .env):
  LOCAL_LLM_MODEL   Ollama model name   (default qwen3:4b)
  OLLAMA_HOST       default http://localhost:11434
  OLLAMA_NUM_CTX    context window     (default 8192; lower it if the GPU runs out of memory)
"""

import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

DEFAULT_MODEL = "qwen3:4b"


def settings():
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except Exception:
        pass
    return (os.environ.get("LOCAL_LLM_MODEL", DEFAULT_MODEL),
            os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            int(os.environ.get("OLLAMA_NUM_CTX", "8192")))


class OllamaWriter:
    """Drop-in for approach_grounded.run.Writer (text only, no video)."""

    def __init__(self, schema: dict, temperature: float = 0.2, model: str = None):
        m, host, ctx = settings()
        if not host.startswith("http"):
            host = "http://" + host
        if urlparse(host).hostname not in ("localhost", "127.0.0.1", "::1"):
            raise RuntimeError(f"{host} is not a local Ollama server - refusing to send data out")
        self.model = model or m
        self.url = host.rstrip("/") + "/api/chat"
        self.tags = host.rstrip("/") + "/api/tags"
        self.schema, self.temperature, self.num_ctx = schema, temperature, ctx
        self.with_video, self.video = False, None
        self.seconds = []                      # time per call, for the report
        self.tokens = []                       # prompt / output tokens per call (to size num_ctx)
        self._check()

    def _check(self):
        try:
            with urllib.request.urlopen(self.tags, timeout=5) as r:
                names = [x["name"] for x in json.load(r).get("models", [])]
        except urllib.error.URLError:
            raise RuntimeError("Ollama is not running - start the Ollama app")
        if not any(n == self.model or n == self.model + ":latest" for n in names):
            raise RuntimeError(f"model '{self.model}' not installed - run: ollama pull {self.model}")

    def _post(self, think_off: bool):
        """Streams the answer, so progress shows and Ctrl+C works between pieces."""
        body = {"model": self.model, "stream": True, "format": self.schema,
                "messages": [{"role": "user", "content": self._prompt}],
                "options": {"temperature": self.temperature, "seed": 42, "num_ctx": self.num_ctx}}
        if think_off:
            body["think"] = False              # qwen3 & co: skip the long hidden reasoning
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        parts, t0, shown = [], time.time(), 0.0
        print(f"[local LLM] {self.model}: reading the prompt (no output for a while is normal) ...",
              flush=True)
        with urllib.request.urlopen(req, timeout=300) as r:
            for line in r:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("error"):
                    raise RuntimeError(d["error"])
                parts.append(d.get("message", {}).get("content", ""))
                if time.time() - shown > 2 or d.get("done"):
                    shown = time.time()
                    n = sum(len(x) for x in parts)
                    print(f"\r[local LLM] writing ... {n} characters, {time.time() - t0:.0f} s",
                          end="", flush=True)
                if d.get("done"):
                    self.tokens.append({"prompt": d.get("prompt_eval_count"), "output": d.get("eval_count")})
                    break
        print(flush=True)
        return "".join(parts)

    def write(self, prompt: str) -> dict:
        self._prompt = prompt
        last = ""
        for attempt in range(2):
            t0 = time.time()
            try:
                try:
                    content = self._post(think_off=True)
                except urllib.error.HTTPError as e:   # model without a thinking switch
                    if e.code != 400:
                        raise
                    content = self._post(think_off=False)
                self.seconds.append(round(time.time() - t0, 1))
                return json.loads(content)
            except (urllib.error.URLError, ValueError, KeyError, TimeoutError, OSError) as e:
                last = str(e)[:200]
                time.sleep(2)
        raise RuntimeError(f"local LLM gave no valid JSON after 2 tries: {last}")

    def upload(self, path):
        raise RuntimeError("the local LLM coach does not read video")

    def cleanup(self):
        pass
