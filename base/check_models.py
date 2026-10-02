"""
base/check_models.py

Quick check of which Gemini models answer right now (one tiny request each).
Run before a long batch:

    python base\\check_models.py
    python base\\check_models.py gemini-3.6-flash gemini-2.5-flash
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from base.llm_common import MODEL_NAME

from google import genai
from google.genai import types

DEFAULT_CANDIDATES = [MODEL_NAME, "gemini-3.6-flash", "gemini-2.5-flash"]


def ping(client, model: str) -> str:
    t0 = time.time()
    try:
        r = client.models.generate_content(
            model=model, contents="Reply with the single word OK.",
            config=types.GenerateContentConfig(
                temperature=0, max_output_tokens=5,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
        return f"OK      {time.time() - t0:5.1f}s  -> {(r.text or '').strip()[:20]!r}"
    except Exception as e:
        s = str(e)
        tag = ("QUOTA" if ("429" in s or "RESOURCE_EXHAUSTED" in s) else
               "BUSY" if ("503" in s or "UNAVAILABLE" in s) else
               "NOTFOUND" if ("404" in s or "NOT_FOUND" in s) else "ERROR")
        return f"{tag:7s} {time.time() - t0:5.1f}s  {s[:200]}"


def main():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not found in .env")
    client = genai.Client(api_key=key)
    models = sys.argv[1:] or list(dict.fromkeys(DEFAULT_CANDIDATES))
    print(f"Current setting: GEMINI_MODEL = {MODEL_NAME}\n")
    for m in models:
        print(f"{m:28s} {ping(client, m)}")
    print("\nOnly switch if needed, and then for ALL approaches (.env GEMINI_MODEL=...).")


if __name__ == "__main__":
    main()
