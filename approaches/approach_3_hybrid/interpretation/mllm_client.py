"""
approaches/approach_3_hybrid/interpretation/mllm_client.py
"""
import os
import time
import json
from pathlib import Path
from google import genai
from google.genai import types

class HybridGeminiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        
    def upload_video(self, video_path: Path):
        print("       Uploading video to Gemini...")
        video_file = self.client.files.upload(file=str(video_path))
        while video_file.state.name == "PROCESSING":
            print("       Processing video on server...")
            time.sleep(3)
            video_file = self.client.files.get(name=video_file.name)
        if video_file.state.name == "FAILED":
            raise RuntimeError("Video processing failed.")
        return video_file
        
    def generate_feedback(self, video_file, prompt: str, schema: dict, retries: int = 5):
        backoff = [10, 20, 40, 60, 90]
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[video_file, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.2,
                    )
                )
                return json.loads(response.text)
            except Exception as e:
                err_str = str(e)
                if attempt < retries - 1:
                    wait = backoff[attempt] if attempt < len(backoff) else 60
                    print(f"       [RETRY] API error: {err_str[:80]} - waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Failed after {retries} retries: {e}")
        
    def delete_video(self, video_file):
        print("       Cleaning up video from server...")
        self.client.files.delete(name=video_file.name)
