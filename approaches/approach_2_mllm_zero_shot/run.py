"""
approaches/approach_2_mllm_zero_shot/run.py
Zero-shot MLLM approach: Upload video directly to Gemini and get feedback.
"""

import os
import sys
import time
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEOS_DIR = PROJECT_ROOT / "input" / "videos"
OUTPUT_DIR = PROJECT_ROOT / "output" / "feedback_mllm"

# Import schema from approach 1
sys.path.insert(0, str(PROJECT_ROOT / "approaches" / "approach_1_rule" / "llm"))
from prompt_builder import FEEDBACK_SCHEMA

PROMPT = """You are an expert interview coach analyzing a candidate's response to a "Tell me about yourself" question in a job interview.

CONTEXT
-------
• Question: "Tell me about yourself" (opening question)
• Language: English

YOUR TASK
---------
Watch the video and evaluate the candidate across 5 dimensions:
1. Eye Contact
2. Head Movement
3. Hand Gestures
4. Facial Expression
5. Answer Quality (Spoken content)

Evaluate each dimension as a QUALITATIVE BAND:
  - ดีมาก (excellent)
  - ดี (good)
  - ปานกลาง (fair)
  - ควรปรับ (needs work)
  - ควรปรับมาก (priority)

Provide structured, actionable feedback following the required JSON schema exactly.

CRITICAL INSTRUCTIONS:
• Provide the primary content in English, and provide precise Thai translations in the fields ending in `_th`.
• DO NOT invent numeric scores. Use qualitative bands only.
• Be specific to what you see and hear in the video.
• Transcribe the first 1-2 sentences for the 'improved_answer' section and provide a better version.
"""

def process_video(client, video_path: Path):
    key = video_path.stem
    print(f"\n[RUN ] {key}")
    
    print("       Uploading video to Gemini...")
    video_file = client.files.upload(file=str(video_path))
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        print("       Processing video on server...")
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)
        
    if video_file.state.name == "FAILED":
        raise RuntimeError("Video processing failed on server.")
        
    print("       Generating feedback (Zero-shot MLLM)...")
    # Using 2.5-flash as default, can be modified
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[video_file, PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FEEDBACK_SCHEMA,
            temperature=0.2,
        )
    )
    
    feedback = json.loads(response.text)
    feedback["key"] = key
    feedback["model"] = "gemini-3.8-flash (zero-shot)"
    
    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{key}_feedback.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)
        
    print("       Cleaning up video from server...")
    client.files.delete(name=video_file.name)
    
    print(f"       -> {out_path.relative_to(PROJECT_ROOT)}")
    return feedback

def main():
    print("=" * 60)
    print("  APPROACH 2: MLLM ZERO-SHOT")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found in environment (.env).")
        sys.exit(1)
        
    if not VIDEOS_DIR.exists():
        print(f"[ERROR] Videos directory not found: {VIDEOS_DIR}")
        sys.exit(1)
        
    video_files = sorted(list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.mov")))
    if not video_files:
        print(f"[ERROR] No videos found in {VIDEOS_DIR}")
        sys.exit(1)
        
    print(f"[INFO] Found {len(video_files)} videos.")
    print(f"[INFO] Output: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}\n")
    
    client = genai.Client(api_key=api_key)
    
    success = 0
    for vf in video_files:
        try:
            process_video(client, vf)
            success += 1
        except Exception as e:
            print(f"       [FAIL] {e}")
            
    print("=" * 60)
    print(f"  Done: {success}/{len(video_files)} succeeded.")
    print("=" * 60)

if __name__ == "__main__":
    main()
