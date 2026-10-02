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
from prompt_builder import FEEDBACK_SCHEMA, OUTPUT_RULES

sys.path.insert(0, str(PROJECT_ROOT))
from base.llm_common import (MODEL_NAME, RATINGS_INSTRUCTIONS, question_block,
                             run_info, parse_run_args, should_skip,
                             gen_config, call_with_retry, BusyGuard)

def build_prompt(key: str) -> str:
    return f"""You are an interview coach. Watch the video and give feedback on this one answer.

{question_block(key)}

Judge eye contact, head movement, hand gestures, facial expression, and speech delivery
from what you see and hear.

{OUTPUT_RULES}

{RATINGS_INSTRUCTIONS}
"""


def process_video(client, video_path: Path):
    key = video_path.stem
    print(f"\n[RUN ] {key}")
    
    print("       Uploading video to Gemini...")
    video_file = client.files.upload(file=str(video_path))
    try:
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            print("       Processing video on server...")
            time.sleep(3)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise RuntimeError("Video processing failed on server.")

        print(f"       Generating feedback (Zero-shot MLLM, {MODEL_NAME})...")
        prompt = build_prompt(key)
        response = call_with_retry(lambda: client.models.generate_content(
            model=MODEL_NAME,
            contents=[video_file, prompt],
            config=gen_config(FEEDBACK_SCHEMA),
        ))
    finally:
        # always remove the upload, also when generation fails
        print("       Cleaning up video from server...")
        try:
            client.files.delete(name=video_file.name)
        except Exception as e:
            print(f"       [WARN] could not delete upload: {e}")

    feedback = json.loads(response.text)
    feedback["key"] = key
    feedback.update(run_info())
    feedback["approach"] = "mllm"
    
    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{key}_feedback.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)

    
    print(f"       -> {out_path.relative_to(PROJECT_ROOT)}")
    return feedback

def main():
    args = parse_run_args("Approach 2 — zero-shot MLLM feedback from video")
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
    if args.only:
        video_files = [v for v in video_files if v.stem in set(args.only)]
    if not video_files:
        print(f"[ERROR] No videos found in {VIDEOS_DIR}")
        sys.exit(1)
        
    print(f"[INFO] Found {len(video_files)} videos.")
    print(f"[INFO] Output: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}\n")
    
    client = genai.Client(api_key=api_key)
    
    success = 0
    guard = BusyGuard()
    for vf in video_files:
        if should_skip(OUTPUT_DIR / f"{vf.stem}_feedback.json", args.force):
            continue
        try:
            process_video(client, vf)
            success += 1
            guard.ok()
        except Exception as e:
            print(f"       [FAIL] {e}")
            if guard.failed(e):
                break
            
    print("=" * 60)
    print(f"  Done: {success}/{len(video_files)} succeeded.")
    print("=" * 60)

if __name__ == "__main__":
    main()
