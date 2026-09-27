"""
approaches/approach_1_rule/run.py

Master script to run the entire Approach 1 pipeline:
1. Feature Extraction
2. Scoring
3. LLM Feedback Generation
"""

import sys
import subprocess
from pathlib import Path

def run_stage(name, script_path, cwd):
    print("=" * 60)
    print(f"  [STAGE] {name}")
    print("=" * 60)
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(cwd)
    )
    
    if result.returncode != 0:
        print(f"\n[ERROR] Stage '{name}' failed with return code {result.returncode}")
        sys.exit(result.returncode)
    
    print("\n")

def main():
    approach_dir = Path(__file__).resolve().parent
    project_root = approach_dir.parent.parent
    
    stages = [
        ("Feature Extraction", approach_dir / "features" / "run_extract_all.py"),
        ("Scoring", approach_dir / "scoring" / "run_scoring.py"),
        ("LLM Feedback", approach_dir / "llm" / "run_feedback.py"),
    ]
    
    print("Starting Approach 1 Pipeline...\n")
    
    for name, script_path in stages:
        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_path}")
            sys.exit(1)
            
        run_stage(name, script_path, project_root)
        
    print("=" * 60)
    print("  ALL STAGES COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
