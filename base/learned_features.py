"""
base/learned_features.py

Feature table used by the learned score model — shared by
  validation/train_score_model.py          (experiments: which feature group works?)
  approaches/approach_1b_learned/train.py  (train + save the system model)
  approaches/approach_1b_learned/score.py  (score any clip with the saved model)

One row per clip, built from Tier-1 outputs:
  output/features/<key>.json           vis.*  gaze, head pose, hands, face
  output/features/<key>_speech.json    sp.*   speech rate, pauses, fillers, length
  output/evidence/<key>_prosody.json   pr.*   pitch variation, intensity, jitter, HNR, silence
  output/scores/<key>.json             rule.overall (Approach 1, reference only)
  output/embeddings/text_<name>.npz    txt.*  pretrained transcript embedding (M3, optional)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from base._paths import FEATURES_DIR, EVIDENCE_DIR, SCORES_DIR
from base.dataset.avi import parse_key

LENGTH = ["sp.duration_sec", "sp.word_count", "sp.speaking_sec"]

# proxies for clip length / detection quality, not behaviour
EXCLUDE_SUBSTR = ("valid_frames", "total_frames", "detection_stats", "rejected_outliers",
                  "hand_frames", "aspect_used", "segment_count")
EXCLUDE_EXACT = {"pr.pitch_mean_hz"}          # mostly encodes gender


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


def _read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def clip_row(key: str) -> dict:
    """All features available for one clip (any key, AVI or not)."""
    row = {"key": key}
    vis = FEATURES_DIR / f"{key}.json"
    if vis.exists():
        row.update({f"vis.{k}": v for k, v in _flat(_read(vis)).items()})
    sp = FEATURES_DIR / f"{key}_speech.json"
    if sp.exists():
        feats = _read(sp).get("features", {})
        row.update({f"sp.{k}": float(v) for k, v in feats.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)})
    pr = EVIDENCE_DIR / f"{key}_prosody.json"
    if pr.exists():
        row.update({f"pr.{k}": v for k, v in _flat(_read(pr)).items()})
    sc = SCORES_DIR / f"{key}.json"
    if sc.exists():
        row["rule.overall"] = _read(sc).get("fusion", {}).get("overall_score", np.nan)
    return row


def all_keys():
    """Every clip with visual features and/or speech features (speech-only clips come
    from run_speech_pipeline.py, which processes the full AVI without video frames)."""
    keys = {f.stem.removesuffix("_speech") for f in FEATURES_DIR.glob("*.json")}
    return sorted(keys)


def text_embedding_columns(name: str) -> dict:
    """{key: {"txt.0": v0, ...}} from output/embeddings/text_<name>.npz."""
    from base.embeddings.text_embed import load
    return {k: {f"txt.{i}": float(x) for i, x in enumerate(v)} for k, v in load(name).items()}


def llm_columns(model: str) -> dict:
    """{key: {"llm.rating": r}} from output/scores_m5/<model>/ (M5, local LLM zero-shot)."""
    folder = Path(__file__).resolve().parents[1] / "output" / "scores_m5" / \
        model.replace(":", "_").replace("/", "_")
    out = {}
    for f in folder.glob("*.json") if folder.exists() else []:
        d = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(d.get("rating"), (int, float)):
            out[f.stem] = {"llm.rating": float(d["rating"])}
    return out


def clip_features(questions=None, text_emb: str = None, llm: str = None) -> pd.DataFrame:
    """AVI clips only, with participant_id (for training / evaluation).
    text_emb: name of a text-embedding file to add as txt.* columns (M3).
    llm: Ollama model whose zero-shot ratings to add as llm.rating (M5)."""
    emb = text_embedding_columns(text_emb) if text_emb else {}
    llm_r = llm_columns(llm) if llm else {}
    if llm and not llm_r:
        raise SystemExit(f"[ERROR] no M5 ratings for '{llm}' - run "
                         f"python approaches/scoring_m5_local_llm/run.py --model {llm}")
    if text_emb and not emb:
        raise SystemExit(f"[ERROR] no embeddings for '{text_emb}' - run "
                         f"python base/embeddings/text_embed.py --model {text_emb}")
    rows = []
    for key in all_keys():
        avi = parse_key(key)
        if avi is None or (questions and avi["question_no"] not in questions):
            continue
        rows.append({**clip_row(key), **emb.get(key, {}), **llm_r.get(key, {}),
                     "participant_id": avi["participant_id"]})
    return pd.DataFrame(rows)


def feature_groups(columns):
    usable = [c for c in columns
              if c.split(".")[0] in ("vis", "sp", "pr")
              and not any(s in c for s in EXCLUDE_SUBSTR) and c not in EXCLUDE_EXACT]
    audio_text = [c for c in usable if c.startswith(("sp.", "pr."))]
    visual = [c for c in usable if c.startswith("vis.")]
    groups = {
        "length": [c for c in LENGTH if c in usable],
        "audio_text": audio_text,
        "visual": visual,
        "all": audio_text + visual,
    }
    text = sorted((c for c in columns if c.startswith("txt.")), key=lambda c: int(c[4:]))
    if text:                                    # M3 / M4 (only when embeddings were loaded)
        groups["text_emb"] = text
        groups["text_emb+audio_text"] = text + audio_text
        groups["text_emb+all"] = text + audio_text + visual
    if "llm.rating" in columns:                 # M5 rating used as one extra feature
        groups["llm+audio_text"] = ["llm.rating"] + audio_text
        groups["llm+length"] = ["llm.rating"] + groups["length"]
    return groups


def participant_table(clips: pd.DataFrame) -> pd.DataFrame:
    """Mean of each feature over a participant's clips (labels are per participant)."""
    per = clips.drop(columns=["key"]).groupby("participant_id").mean(numeric_only=True)
    per["n_clips"] = clips.groupby("participant_id").size()
    return per
