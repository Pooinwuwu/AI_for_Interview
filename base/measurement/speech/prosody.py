"""
base/measurement/speech/prosody.py

Extract prosody features using parselmouth:
- Pitch (F0)
- Intensity (dB)
- HNR (Harmonics-to-Noise Ratio)
- Jitter
- Pause metrics (silence vs speaking time)

Output: output/evidence/{key}_prosody.json
"""

import sys
import json
import logging
from pathlib import Path
import traceback

import parselmouth
from parselmouth.praat import call
import numpy as np

# Setup paths
# root/base/measurement/speech/prosody.py -> parent.parent.parent.parent is root
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from base._paths import AUDIO_DIR, EVIDENCE_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def extract_prosody(audio_path: Path) -> dict:
    """
    Extracts prosody features using praat-parselmouth.
    """
    try:
        snd = parselmouth.Sound(str(audio_path))
        
        # 1. Pitch
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        # Filter out unvoiced frames (0 Hz)
        voiced_pitch = pitch_values[pitch_values > 0]
        
        if len(voiced_pitch) > 0:
            mean_pitch = np.mean(voiced_pitch)
            std_pitch = np.std(voiced_pitch)
        else:
            mean_pitch = 0.0
            std_pitch = 0.0

        # 2. Intensity
        intensity = snd.to_intensity()
        intensity_values = intensity.values.T
        valid_intensity = intensity_values[intensity_values > 0]
        if len(valid_intensity) > 0:
            mean_intensity = np.mean(valid_intensity)
        else:
            mean_intensity = 0.0

        # 3. Voice Report (Jitter, HNR)
        point_process = call(snd, "To PointProcess (periodic, cc)", 75.0, 500.0)
        
        # Jitter (local)
        try:
            jitter = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
        except parselmouth.PraatError:
            jitter = 0.0
            
        # HNR (Harmonics-to-Noise Ratio)
        harmonicity = snd.to_harmonicity()
        hnr_values = harmonicity.values[harmonicity.values > 0]
        if len(hnr_values) > 0:
            mean_hnr = np.mean(hnr_values)
        else:
            mean_hnr = 0.0

        # 4. Pause / Speaking Rate estimation
        # We can estimate pauses by looking at intensity drops below a threshold.
        # Let's say threshold is mean_intensity - 15 dB or a fixed threshold like 50 dB
        pauses = []
        if mean_intensity > 0:
            threshold = max(50.0, mean_intensity - 15)
            # Find continuous segments of silence
            is_silence = (intensity_values < threshold).flatten()
            silence_ratio = float(np.mean(is_silence))
            
            # Extract pause intervals
            times = intensity.xs()
            padded = np.concatenate(([False], is_silence, [False]))
            diff = np.diff(padded.astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            for s_idx, e_idx in zip(starts, ends):
                if s_idx < len(times):
                    p_start = times[s_idx]
                    p_end = times[min(e_idx, len(times)-1)]
                    p_duration = p_end - p_start
                    # Consider silence > 0.5s as a distinct pause event
                    if p_duration >= 0.5:
                        pauses.append({
                            "start": round(float(p_start), 2),
                            "end": round(float(p_end), 2),
                            "duration": round(float(p_duration), 2)
                        })
        else:
            silence_ratio = 1.0
            pauses.append({
                "start": 0.0,
                "end": round(float(snd.duration), 2),
                "duration": round(float(snd.duration), 2)
            })

        return {
            "pitch_mean_hz": float(mean_pitch),
            "pitch_std_hz": float(std_pitch),
            "intensity_mean_db": float(mean_intensity),
            "jitter_local": float(jitter) if not np.isnan(jitter) else 0.0,
            "hnr_mean_db": float(mean_hnr),
            "silence_ratio": float(silence_ratio),
            "pauses": pauses,
        }
    except Exception as e:
        logging.error(f"Error extracting prosody from {audio_path.name}: {e}")
        logging.error(traceback.format_exc())
        return None

def main():
    ensure_dirs(EVIDENCE_DIR)
    
    if not AUDIO_DIR.exists():
        logging.error(f"Audio directory not found: {AUDIO_DIR}")
        return
        
    audio_files = list(AUDIO_DIR.glob("*.wav"))
    if not audio_files:
        logging.warning(f"No .wav files found in {AUDIO_DIR}")
        return
        
    logging.info(f"Found {len(audio_files)} audio files.")
    force = "--force" in sys.argv
    skipped = 0

    for audio_path in audio_files:
        key = audio_path.stem
        if (EVIDENCE_DIR / f"{key}_prosody.json").exists() and not force:
            skipped += 1          # already done (use --force to recompute)
            continue
        logging.info(f"Processing: {key}")
        
        features = extract_prosody(audio_path)
        if features:
            out_file = EVIDENCE_DIR / f"{key}_prosody.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(features, f, indent=4, ensure_ascii=False)
            logging.info(f"  -> Saved {out_file.name}")
    if skipped:
        logging.info(f"Skipped {skipped} file(s) that already had prosody (--force to redo)")

if __name__ == "__main__":
    main()
