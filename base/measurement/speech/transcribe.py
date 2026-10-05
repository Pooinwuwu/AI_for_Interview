"""
base/measurement/speech/transcribe.py

Transcribe audio using WhisperX to get word-level timestamps.
Diarization is optional (requires HF_TOKEN in .env).

Output: output/evidence/{key}_transcript.json
"""

import sys
import os
import json
import logging
from pathlib import Path
import traceback
import warnings

# Suppress annoying FutureWarnings from torch/whisper
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import whisperx
from dotenv import load_dotenv

# Setup paths
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from base._paths import AUDIO_DIR, EVIDENCE_DIR, ensure_dirs

# Load environment variables
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_MODELS = {}


def _get_model(device, compute_type):
    if "asr" not in _MODELS:
        logging.info("Loading WhisperX model (base) once...")
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id="Systran/faster-whisper-base")
        _MODELS["asr"] = whisperx.load_model(model_dir, device, compute_type=compute_type)
    return _MODELS["asr"]


def _get_align(language, device):
    if ("align", language) not in _MODELS:
        _MODELS[("align", language)] = whisperx.load_align_model(language_code=language, device=device)
    return _MODELS[("align", language)]


def transcribe_audio(audio_path: Path, device: str = "cpu", compute_type: str = "float32",
                     language: str = "en") -> dict:
    """
    Transcribes audio using WhisperX and aligns it to get word-level timestamps.
    """
    try:
        # models are loaded ONCE and reused for every clip (loading per clip was the slow part)
        model = _get_model(device, compute_type)

        logging.info(f"Transcribing {audio_path.name}...")
        audio = whisperx.load_audio(str(audio_path))
        # language is fixed (default "en"): auto-detection on short / accented clips can pick
        # another language, and then alignment fails
        result = model.transcribe(audio, batch_size=4, language=language)

        # Align timestamps
        logging.info("Aligning timestamps...")
        model_a, metadata = _get_align(result["language"], device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # Optional Diarization
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            logging.info("HF_TOKEN found, running diarization...")
            try:
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
                diarize_segments = diarize_model(audio)
                result_aligned = whisperx.assign_word_speakers(diarize_segments, result_aligned)
            except Exception as e:
                logging.error(f"Diarization failed: {e}")
        else:
            logging.info("HF_TOKEN not found in .env, skipping diarization.")

        return {
            "language": result["language"],
            "segments": result_aligned["segments"]
        }

    except Exception as e:
        logging.error(f"Error transcribing {audio_path.name}: {e}")
        logging.error(traceback.format_exc())
        return None
    finally:
        # Clear CUDA cache to prevent memory leaks if running multiple files
        if device == "cuda":
            torch.cuda.empty_cache()

def main():
    ensure_dirs(EVIDENCE_DIR)
    
    if not AUDIO_DIR.exists():
        logging.error(f"Audio directory not found: {AUDIO_DIR}")
        return
        
    audio_files = list(AUDIO_DIR.glob("*.wav"))
    if not audio_files:
        logging.warning(f"No .wav files found in {AUDIO_DIR}")
        return
        
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # float16 is much faster and saves VRAM on GPU. CPU usually requires float32 or int8
    compute_type = "float16" if device == "cuda" else "float32"
    logging.info(f"Using device: {device}, compute_type: {compute_type}")
    
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default=os.environ.get("TRANSCRIBE_LANGUAGE", "en"),
                    help="spoken language (default en); 'auto' = let Whisper detect it")
    args = ap.parse_args()
    language = None if args.language == "auto" else args.language

    failed = []
    for audio_path in audio_files:
        key = audio_path.stem
        out_file = EVIDENCE_DIR / f"{key}_transcript.json"
        
        # Skip if already exists to save time during tests
        if out_file.exists():
            logging.info(f"Skipping {key}, transcript already exists.")
            continue
            
        logging.info(f"Processing: {key}")
        
        features = transcribe_audio(audio_path, device=device, compute_type=compute_type,
                                    language=language)
        if not features:
            failed.append(key)
        if features:
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=4, ensure_ascii=False)
            logging.info(f"  -> Saved {out_file.name}")

    if failed:
        logging.error(f"Transcription FAILED for {len(failed)} file(s): {', '.join(failed)} "
                      f"(see errors above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
