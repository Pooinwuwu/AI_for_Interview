"""
approaches/approach_3_hybrid/interpretation/mllm_client.py
"""
import os
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from base.llm_common import MODEL_NAME, TEMPERATURE, gen_config, call_with_retry
from google import genai
from google.genai import types

class HybridGeminiClient:
    def __init__(self, api_key: str, model_name: str = MODEL_NAME,
                 temperature: float = TEMPERATURE):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        
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
        
    def generate_feedback(self, video_file, prompt: str, schema: dict):
        return call_with_retry(lambda: json.loads(self.client.models.generate_content(
            model=self.model_name,
            contents=[video_file, prompt],
            config=gen_config(schema, self.temperature),
        ).text))

    def delete_video(self, video_file):
        print("       Cleaning up video from server...")
        self.client.files.delete(name=video_file.name)
