"""
llm/run_feedback.py
Entrypoint เดียว — รันจาก root ของโปรเจกต์
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    script = Path(__file__).parent / "feedback_generator.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()