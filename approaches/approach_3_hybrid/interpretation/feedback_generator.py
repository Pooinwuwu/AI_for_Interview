"""
approaches/approach_3_hybrid/interpretation/feedback_generator.py
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
VIDEOS_DIR = PROJECT_ROOT / "input" / "videos"
EVIDENCE_DIR = PROJECT_ROOT / "output" / "evidence"
OUTPUT_DIR = PROJECT_ROOT / "output" / "feedback_hybrid"

# Import build_evidence from parent dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_evidence import format_evidence_text

from mllm_client import HybridGeminiClient
from hybrid_prompt import build_hybrid_prompt, FEEDBACK_SCHEMA

def main():
    print("=" * 60)
    print("  APPROACH 3: HYBRID (Video + Evidence Log)")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found in environment (.env).")
        sys.exit(1)
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = HybridGeminiClient(api_key=api_key)
    
    video_files = sorted(list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.mov")))
    success = 0
    
    for vf in video_files:
        key = vf.stem
        print(f"\n[RUN ] {key}")
        
        evidence_file = EVIDENCE_DIR / f"{key}_evidence.json"
        if not evidence_file.exists():
            print(f"       [WARN] Evidence log not found for {key}, continuing without it.")
        evidence_text = format_evidence_text(evidence_file)
        
        prompt = build_hybrid_prompt(evidence_text)
        
        try:
            video_file = client.upload_video(vf)
            print("       Generating feedback (Hybrid)...")
            feedback = client.generate_feedback(video_file, prompt, FEEDBACK_SCHEMA)
            
            feedback["key"] = key
            feedback["model"] = "gemini-2.5-flash (hybrid)"
            
            out_path = OUTPUT_DIR / f"{key}_feedback.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(feedback, f, ensure_ascii=False, indent=2)
                
            client.delete_video(video_file)
            print(f"       -> {out_path.relative_to(PROJECT_ROOT)}")
            success += 1
        except Exception as e:
            print(f"       [FAIL] {e}")
            
    print("=" * 60)
    print(f"  Done: {success}/{len(video_files)} succeeded.")
    print("=" * 60)

if __name__ == "__main__":
    main()
