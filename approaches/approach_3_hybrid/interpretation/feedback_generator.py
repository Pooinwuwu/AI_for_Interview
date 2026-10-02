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

sys.path.insert(0, str(PROJECT_ROOT))
from base.llm_common import run_info, parse_run_args, should_skip, BusyGuard

def main():
    args = parse_run_args("Approach 3 — hybrid feedback (video + evidence log)")
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
    if args.only:
        video_files = [v for v in video_files if v.stem in set(args.only)]
    print(f"[INFO] Model: {client.model_name}  temperature={client.temperature}")
    success = 0
    guard = BusyGuard()
    
    for vf in video_files:
        key = vf.stem
        out_path = OUTPUT_DIR / f"{key}_feedback.json"
        if should_skip(out_path, args.force):
            continue
        print(f"\n[RUN ] {key}")
        
        evidence_file = EVIDENCE_DIR / f"{key}_evidence.json"
        if not evidence_file.exists():
            print(f"       [WARN] Evidence log not found for {key}, continuing without it.")
        evidence_text = format_evidence_text(evidence_file)
        
        prompt = build_hybrid_prompt(evidence_text, key)

        video_file = None
        try:
            video_file = client.upload_video(vf)
            print("       Generating feedback (Hybrid)...")
            feedback = client.generate_feedback(video_file, prompt, FEEDBACK_SCHEMA)

            feedback["key"] = key
            feedback.update(run_info())
            feedback["approach"] = "hybrid"
            feedback["evidence_log_used"] = evidence_file.exists()

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(feedback, f, ensure_ascii=False, indent=2)

            print(f"       -> {out_path.relative_to(PROJECT_ROOT)}")
            success += 1
            guard.ok()
        except Exception as e:
            print(f"       [FAIL] {e}")
            if guard.failed(e):
                break
        finally:
            # always remove the upload, also when generation fails
            if video_file is not None:
                try:
                    client.delete_video(video_file)
                except Exception as e:
                    print(f"       [WARN] could not delete upload: {e}")
            
    print("=" * 60)
    print(f"  Done: {success}/{len(video_files)} succeeded.")
    print("=" * 60)

if __name__ == "__main__":
    main()
