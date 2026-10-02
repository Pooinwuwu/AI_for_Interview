"""
llm/llm_client.py
Wrapper สำหรับ Gemini — throttle + retry (ปรับปรุงสำหรับ 503)
"""

import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from base.llm_common import MODEL_NAME, TEMPERATURE, gen_config, call_with_retry

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types


class GeminiClient:
    def __init__(self, model_name: str = MODEL_NAME, temperature: float = TEMPERATURE):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ไม่พบ GEMINI_API_KEY — สร้างไฟล์ .env ที่ root โปรเจกต์"
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self._last_call_ts = 0.0
        self._min_interval = 4.5   # free tier ≈ 15 RPM

    # ------------------------------------------------------------
    def _throttle(self):
        elapsed = time.time() - self._last_call_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_ts = time.time()

    # ------------------------------------------------------------
    def generate_json(self, prompt: str, schema: dict) -> dict:
        """เรียก Gemini → คืน JSON ตาม schema (retry/wait: base.llm_common.call_with_retry)"""
        def once():
            self._throttle()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=gen_config(schema, self.temperature),
            )
            return json.loads(response.text)
        return call_with_retry(once)

    # ------------------------------------------------------------
    def generate_json_with_fallback(self, prompt: str, schema: dict,
                                     models: list, retries_per_model: int = 3):
        """ลองโมเดลทีละตัว จนกว่าจะสำเร็จ

        NOT used in experiments: switching model silently makes approaches
        incomparable. If a clip fails, rerun it later with the same model.
        """
        last_err = None
        original = self.model_name
        for m in models:
            self.model_name = m
            try:
                result = self.generate_json(prompt, schema)
                print(f"       [OK] โมเดลที่ใช้: {m}")
                return result
            except Exception as e:
                last_err = e
                print(f"       [FAIL] {m} — {str(e)[:60]}")
                continue
        self.model_name = original
        raise RuntimeError(f"ทุกโมเดลล้มเหลว: {last_err}")

    # ------------------------------------------------------------
    def list_models(self):
        out = []
        for m in self.client.models.list():
            if "generateContent" in (m.supported_actions or []):
                out.append(m.name)
        return out