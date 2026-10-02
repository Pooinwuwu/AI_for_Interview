"""
llm/run_feedback.py
Entrypoint เดียว — รันจาก root ของโปรเจกต์
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main():
    script = Path(__file__).parent / "feedback_generator.py"
    result = subprocess.run(
        [sys.executable, str(script), *sys.argv[1:]],  # forward --force / --only
        cwd=str(PROJECT_ROOT),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()