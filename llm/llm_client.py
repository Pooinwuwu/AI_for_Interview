"""
llm/llm_client.py
Wrapper สำหรับ Gemini — throttle + retry + JSON output
"""

import os
import time
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types


class GeminiClient:
    def __init__(self, model_name: str = "gemini-3.6-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ไม่พบ GEMINI_API_KEY — สร้างไฟล์ .env ที่ root โปรเจกต์"
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self._last_call_ts = 0.0
        # Free tier ≈ 15 RPM → เว้น 4.5s ต่อ request เพื่อความปลอดภัย
        self._min_interval = 4.5

    # ------------------------------------------------------------
    def _throttle(self):
        elapsed = time.time() - self._last_call_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_ts = time.time()

    # ------------------------------------------------------------
    def generate_json(self, prompt: str, schema: dict, retries: int = 3) -> dict:
        """เรียก Gemini → คืน JSON ตาม schema (พร้อม retry)"""
        last_err = None
        for attempt in range(retries):
            try:
                self._throttle()
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                return json.loads(response.text)

            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    print(f"       [retry {attempt+1}/{retries}] {e} — รอ {wait}s")
                    time.sleep(wait)

        raise RuntimeError(f"เรียก Gemini ล้มเหลวหลัง {retries} ครั้ง: {last_err}")

    # ------------------------------------------------------------
    def list_models(self):
        """list โมเดลที่ใช้ generateContent ได้"""
        out = []
        for m in self.client.models.list():
            if "generateContent" in (m.supported_actions or []):
                out.append(m.name)
        return out