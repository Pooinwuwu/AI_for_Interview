"""
llm/llm_client.py
Wrapper สำหรับ Gemini — throttle + retry (ปรับปรุงสำหรับ 503)
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
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ไม่พบ GEMINI_API_KEY — สร้างไฟล์ .env ที่ root โปรเจกต์"
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self._last_call_ts = 0.0
        self._min_interval = 4.5   # free tier ≈ 15 RPM

    # ------------------------------------------------------------
    def _throttle(self):
        elapsed = time.time() - self._last_call_ts
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_ts = time.time()

    # ------------------------------------------------------------
    def generate_json(self, prompt: str, schema: dict,
                      retries: int = 5) -> dict:
        """เรียก Gemini → คืน JSON ตาม schema พร้อม retry"""
        last_err = None
        backoff_503 = [10, 20, 40, 60, 90]
        backoff_other = [2, 4, 8, 16, 32]

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
                err_str = str(e)
                is_503 = "503" in err_str or "UNAVAILABLE" in err_str
                is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if attempt < retries - 1:
                    if is_503:
                        wait = backoff_503[attempt]
                        print(f"       [503] server busy — รอ {wait}s "
                              f"(retry {attempt+1}/{retries})")
                    elif is_429:
                        wait = 30
                        print(f"       [429] rate limit — รอ {wait}s "
                              f"(retry {attempt+1}/{retries})")
                    else:
                        wait = backoff_other[attempt]
                        print(f"       [retry {attempt+1}/{retries}] "
                              f"{err_str[:80]} — รอ {wait}s")
                    time.sleep(wait)
                else:
                    print(f"       [fail] ล้มเหลวหลัง {retries} ครั้ง")

        raise RuntimeError(
            f"เรียก Gemini ล้มเหลวหลัง {retries} ครั้ง: {last_err}"
        )

    # ------------------------------------------------------------
    def generate_json_with_fallback(self, prompt: str, schema: dict,
                                     models: list, retries_per_model: int = 3):
        """ลองโมเดลทีละตัว จนกว่าจะสำเร็จ"""
        last_err = None
        original = self.model_name
        for m in models:
            self.model_name = m
            try:
                result = self.generate_json(prompt, schema, retries=retries_per_model)
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