"""
base/dataset/avi.py

AVI-Personality as the main dataset.

What this module gives you
--------------------------
- load_labels()        participant-level ground truth (one row per participant)
- parse_key(key)       "5484821e..._q1_generic" -> participant id, question no, type
- question_for_key()   the exact interview question text for a clip
- clip_path()          where a clip lives inside input/AVI-Personality/<Split>/

Important facts about the labels (see input/AVI-Personality/CODEBOOK.md)
------------------------------------------------------------------------
- Labels are PER PARTICIPANT, not per clip. Professional recruiters rated the
  competencies after watching all six answers. Clip-level system scores must be
  aggregated per participant before comparing (validation/evaluate_avi.py does this).
- Scale is a 1-5 BARS mean (continuous, e.g. 3.1).
- We use ONLY the recruiter-rated competency columns as ground truth.
  Personality (HEXACO) and cognitive ability columns are deliberately NOT used:
  this project measures observable interview behaviour and does not predict
  personality (work plan v3, section 1 & 5.3).
- The official split is subject-level (70/10/20), so it already prevents
  participant leakage. Tune thresholds/weights on Training/Validation only and
  report final numbers on Testing.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base._paths import AVI_DIR, QUESTIONS_DIR

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

# split name used in code -> folder / csv name on disk
SPLITS = {
    "train": "Training",
    "val":   "Validation",
    "test":  "Testing",
}

# Ground-truth targets (recruiter ratings, 1-5). Short name -> CSV column.
TARGETS = {
    "hireability":             "mean_rating_hirea",   # overall interview performance
    "integrity":               "mean_rating_Integr",
    "collegiality":            "mean_rating_Colleg",
    "social_versatility":      "mean_rating_Soc_vers",
    "development_orientation": "mean_rating_dev_orient",
}
PRIMARY_TARGET = "hireability"

# Kept only for bias / subgroup analysis — never used as model input or target.
DEMOGRAPHIC_COLS = [
    "gender", "age", "ethnicity", "english_proficiency",
    "accent", "accent_strength", "education", "work_experience",
]

# Generic questions are ordinary selection questions; q3-q6 are
# trait-activating personality questions.
GENERIC_QUESTIONS = (1, 2)
ALL_QUESTIONS = (1, 2, 3, 4, 5, 6)

CLIP_RE = re.compile(
    r"^(?P<pid>[0-9a-f]{24})_q(?P<q>[1-6])_(?P<qtype>generic|personality)$"
)


# ============================================================
# KEYS & PATHS
# ============================================================

def parse_key(key: str):
    """'5484821efdf99b07b28f2300_q1_generic' -> dict, or None if not an AVI clip."""
    m = CLIP_RE.match(Path(key).stem)
    if not m:
        return None
    return {
        "participant_id": m.group("pid"),
        "question_no": int(m.group("q")),
        "question_type": m.group("qtype"),
    }


def make_key(participant_id: str, question_no: int) -> str:
    qtype = "generic" if question_no in GENERIC_QUESTIONS else "personality"
    return f"{participant_id}_q{question_no}_{qtype}"


def clip_path(participant_id: str, question_no: int, split: str) -> Path:
    return AVI_DIR / SPLITS[split] / f"{make_key(participant_id, question_no)}.mp4"


# ============================================================
# QUESTIONS
# ============================================================

_QUESTIONS = None


def load_questions() -> dict:
    global _QUESTIONS
    if _QUESTIONS is None:
        with open(QUESTIONS_DIR / "avi_questions.json", "r", encoding="utf-8") as f:
            _QUESTIONS = json.load(f)
    return _QUESTIONS


def question_for_key(key: str):
    """Return {'type', 'trait', 'text'} for an AVI clip key, or None."""
    info = parse_key(key)
    if info is None:
        return None
    return load_questions()[f"q{info['question_no']}"]


# ============================================================
# LABELS
# ============================================================

def load_labels(splits=("train", "val", "test")) -> pd.DataFrame:
    """
    Participant-level ground truth.
    Columns: participant_id, split, <target short names>, <demographics>
    Rows with a missing primary target are kept (NaN); evaluators drop them.
    """
    frames = []
    for split in splits:
        csv_path = AVI_DIR / f"{SPLITS[split]}.csv"
        df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"id": str})
        keep = ["id"] + list(TARGETS.values()) + [c for c in DEMOGRAPHIC_COLS if c in df.columns]
        df = df[keep].rename(columns={"id": "participant_id"})
        df = df.rename(columns={v: k for k, v in TARGETS.items()})
        df.insert(1, "split", split)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    labels = load_labels()
    print("=" * 60)
    print("  AVI-Personality — ground truth summary")
    print("=" * 60)
    print(labels.groupby("split").size().rename("participants").to_string())
    print()
    print(labels[list(TARGETS)].describe().round(2).to_string())
    print()
    missing = labels[PRIMARY_TARGET].isna().sum()
    print(f"Missing {PRIMARY_TARGET}: {missing}")
    print()
    print("Correlation between targets (Spearman):")
    print(labels[list(TARGETS)].corr(method="spearman").round(2).to_string())
    print()
    k = make_key(labels.participant_id.iloc[0], 1)
    print("Example key :", k)
    print("Parsed      :", parse_key(k))
    print("Question    :", question_for_key(k)["text"])
