# AI_for_Interview

This project is an AI-powered interview analysis tool. Currently, its primary feature is a **Video to Audio Extractor** that processes interview video recordings and extracts the audio tracks for further analysis.

## Features
- **Video to Audio Extraction**: Reliably extracts audio from video files.
- **Portable FFmpeg**: Utilizes `imageio-ffmpeg` to ensure audio extraction works without requiring a system-wide FFmpeg installation.

## Project Structure
- `input/`: Place source video files here.
- `output/`: Extracted audio files will be saved here.
- `models/`: Directory for storing machine learning models (e.g., transcription or NLP models).
- `report/`: Directory for generating interview analysis reports.
- `ai_for_interview/`, `approaches/`, `base/`: Source code modules.

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your video files in the `input/` directory.
3. Run the main script (coming soon):
   ```bash
   python main.py
   ```