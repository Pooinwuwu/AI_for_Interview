"""
app/groq_writer.py - coach writer through the Groq API (cloud, OpenAI-compatible).

Same prompt, FEEDBACK_SCHEMA and verify.py as the Gemini coach. Data DOES leave the computer
(like Gemini): use only with consenting users' own clips, never with AVI / RecruitView data.

Settings (.env):
  GROQ_API_KEY   required
  GROQ_MODEL     default openai/gpt-oss-120b  (structured outputs: gpt-oss-20b/120b, qwen/qwen3.8-27b)
"""

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqWriter:
    """Drop-in for approach_grounded.run.Writer (text only, no video)."""

    def __init__(self, schema: dict, temperature: float = 0.2, model: str = None):
        try:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        except Exception:
            pass
        self.key = os.environ.get("GROQ_API_KEY")
        if not self.key:
            raise RuntimeError("GROQ_API_KEY missing in .env (get one at console.groq.com)")
        self.model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
        self.schema, self.temperature = schema, temperature
        self.with_video, self.video = False, None
        self.seconds, self.usage = [], []

    def _post(self, prompt: str) -> dict:
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "interview_feedback", "strict": False, "schema": self.schema}}}
        if "gpt-oss" in self.model:
            body["reasoning_effort"] = "low"      # faster; the checker catches mistakes
        req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}",
            "User-Agent": "ai-for-interview/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)

    def write(self, prompt: str) -> dict:
        last = ""
        for attempt in range(5):
            t0 = time.time()
            try:
                d = self._post(prompt)
                self.seconds.append(round(time.time() - t0, 1))
                self.usage.append(d.get("usage", {}))
                return json.loads(d["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:300]
                last = f"HTTP {e.code}: {msg}"
                if e.code == 429:                  # rate limit: wait as told, then retry
                    wait = float(e.headers.get("retry-after") or 20)
                    print(f"[groq] rate limit, waiting {wait:.0f} s", flush=True)
                    time.sleep(min(wait, 90))
                elif e.code == 400 and "schema" in msg.lower():
                    time.sleep(1)                  # best-effort JSON missed the schema: retry
                elif e.code >= 500:
                    time.sleep(5 * (attempt + 1))
                else:
                    break
            except (urllib.error.URLError, ValueError, KeyError, TimeoutError, OSError) as e:
                last = str(e)[:300]
                time.sleep(3)
        raise RuntimeError(f"Groq failed: {last}")

    def upload(self, path):
        raise RuntimeError("the Groq coach does not read video")

    def cleanup(self):
        pass
