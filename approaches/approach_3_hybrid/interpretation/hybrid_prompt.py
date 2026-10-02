"""
approaches/approach_3_hybrid/interpretation/hybrid_prompt.py

Prompt for Approach 3: video + Tier-1 evidence log.
Question context, rules and rating scale are shared with Approaches 1 & 2
(base/llm_common.py, approach_1_rule/llm/prompt_builder.py), so the three
approaches differ only in their inputs.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "approaches" / "approach_1_rule" / "llm"))
sys.path.insert(0, str(PROJECT_ROOT))
from prompt_builder import FEEDBACK_SCHEMA, OUTPUT_RULES  # noqa: F401  (schema re-exported)
from base.llm_common import RATINGS_INSTRUCTIONS, question_block


def build_hybrid_prompt(evidence_text: str, key: str) -> str:
    return f"""You are an interview coach. Watch the video and give feedback on this one answer.

{question_block(key)}

Judge eye contact, head movement, hand gestures, facial expression, and speech delivery.
Ground your observations in the EVIDENCE LOG (measured by computer vision / audio tools).
When you mention a specific moment, cite its time from the log (e.g. "at 12.4s").

EVIDENCE LOG [start - end seconds]
{evidence_text}

{OUTPUT_RULES}

{RATINGS_INSTRUCTIONS}
"""
