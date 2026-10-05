"""
base/embeddings/text_embed.py   (Part 1, method M3: pretrained text embeddings)

Turns each clip's transcript (output/evidence/<key>_transcript.json, WhisperX) into
one fixed-size vector with a pretrained sentence encoder. A small regressor is then
trained on these vectors (validation/train_score_model.py --text-emb <name>).

Why: the AVI Challenge 2026 entries found text the most useful modality; a frozen
encoder + low-capacity regressor suits small data (110 training people).

Long answers are split into chunks of <= CHUNK_WORDS words (encoders read ~256-512
tokens); chunk vectors are averaged, weighted by word count.

Output: output/embeddings/text_<name>.npz   (keys, vectors, model id) - updated
        incrementally: clips already embedded are skipped unless --force.
Runs on GPU if available (4 GB VRAM is plenty), otherwise CPU (~1-2 s per clip).

Usage
  python base/embeddings/text_embed.py                   # default model: minilm
  python base/embeddings/text_embed.py --model bge-small
  python base/embeddings/text_embed.py --model e5-base --force
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from base._paths import EVIDENCE_DIR, OUTPUT_DIR

import numpy as np

EMB_DIR = OUTPUT_DIR / "embeddings"
CHUNK_WORDS = 180

# name -> (Hugging Face id, text prefix the model expects, licence)
MODELS = {
    "minilm":    ("sentence-transformers/all-MiniLM-L6-v2", "", "Apache-2.0"),
    "bge-small": ("BAAI/bge-small-en-v1.5", "", "MIT"),
    "e5-base":   ("intfloat/e5-base-v2", "query: ", "MIT"),
}


def transcript_text(key: str) -> str:
    p = EVIDENCE_DIR / f"{key}_transcript.json"
    if not p.exists():
        return ""
    segs = json.loads(p.read_text(encoding="utf-8")).get("segments", [])
    return " ".join((s.get("text") or "").strip() for s in segs).strip()


def chunks(text: str, n: int = CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)] or [""]


def emb_path(name: str) -> Path:
    return EMB_DIR / f"text_{name}.npz"


def load(name: str) -> dict:
    """{key: vector} for an embedding file (empty dict if missing)."""
    p = emb_path(name)
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=False)
    return {k: v for k, v in zip(z["keys"].tolist(), z["vectors"])}


class Encoder:
    def __init__(self, name: str, device: str = "auto"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        model_id, self.prefix, _ = MODELS[name]
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.model_id = model_id

    def encode(self, texts, batch: int = 16) -> np.ndarray:
        out = []
        torch = self.torch
        for i in range(0, len(texts), batch):
            b = [self.prefix + t for t in texts[i:i + batch]]
            enc = self.tok(b, padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)       # mean pooling
            v = torch.nn.functional.normalize(v, dim=-1)
            out.append(v.cpu().numpy())
        return np.vstack(out)


def embed_clip(enc: Encoder, text: str) -> np.ndarray:
    parts = chunks(text)
    vecs = enc.encode(parts)
    w = np.array([max(1, len(p.split())) for p in parts], dtype=float)
    v = (vecs * w[:, None]).sum(0) / w.sum()
    return v / (np.linalg.norm(v) + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="minilm", choices=list(MODELS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    keys = sorted(p.stem.removesuffix("_transcript") for p in EVIDENCE_DIR.glob("*_transcript.json"))
    have = {} if args.force else load(args.model)
    todo = [k for k in keys if k not in have]
    print(f"[text_embed] model={args.model} ({MODELS[args.model][0]})  clips={len(keys)}  "
          f"already={len(have)}  to do={len(todo)}")
    if not todo:
        return
    enc = Encoder(args.model, args.device)
    print(f"[text_embed] device={enc.device}")
    empty = 0
    for n, key in enumerate(todo, 1):
        text = transcript_text(key)
        if not text:
            empty += 1
            continue
        have[key] = embed_clip(enc, text)
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}")
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    ks = sorted(have)
    np.savez_compressed(emb_path(args.model), keys=np.array(ks),
                        vectors=np.vstack([have[k] for k in ks]).astype(np.float32),
                        model=np.array(enc.model_id))
    print(f"[text_embed] saved {len(ks)} vectors -> {emb_path(args.model)}"
          + (f"  ({empty} empty transcripts skipped)" if empty else ""))


if __name__ == "__main__":
    main()
