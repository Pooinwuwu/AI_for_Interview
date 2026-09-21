"""
extract_audio.py

อ่าน metadata.json -> แปลงทุกวิดีโอใน input/videos/ เป็น wav
output -> output/audio/{key}.wav

FFmpeg detection มี fallback:
  1. PATH ของระบบ
  2. Windows Registry PATH
  3. imageio-ffmpeg
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# ไฟล์อยู่ที่ root/base/preprocessing/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from base._paths import METADATA_FILE, AUDIO_DIR


# ---------- UTF-8 fix Windows ----------
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CONFIG
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1
CODEC = "pcm_s16le"


# ============================================================
# FFMPEG DETECTION
# ============================================================

FFMPEG_CMD = None


def get_ffmpeg_path():
    global FFMPEG_CMD
    if FFMPEG_CMD:
        return FFMPEG_CMD

    # 1. PATH
    exe = shutil.which("ffmpeg")
    if exe:
        FFMPEG_CMD = exe
        return FFMPEG_CMD

    # 2. Windows Registry
    if sys.platform == "win32":
        try:
            import winreg
            for root, subkey in [
                (winreg.HKEY_LOCAL_MACHINE,
                 r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                (winreg.HKEY_CURRENT_USER, r"Environment"),
            ]:
                try:
                    with winreg.OpenKey(root, subkey) as key:
                        val, _ = winreg.QueryValueEx(key, "Path")
                        for p in val.split(";"):
                            expanded = os.path.expandvars(p.strip())
                            if expanded and expanded not in os.environ.get("PATH", "").split(";"):
                                os.environ["PATH"] += ";" + expanded
                except Exception:
                    pass

            exe = shutil.which("ffmpeg")
            if exe:
                FFMPEG_CMD = exe
                return FFMPEG_CMD
        except Exception:
            pass

    # 3. imageio-ffmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            FFMPEG_CMD = exe
            return FFMPEG_CMD
    except ImportError:
        pass

    return None


def check_ffmpeg() -> bool:
    exe = get_ffmpeg_path()
    if not exe:
        return False
    try:
        result = subprocess.run(
            [exe, "-version"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata() -> dict:
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"ไม่พบ metadata: {METADATA_FILE}")
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# EXTRACT ONE
# ============================================================

def extract_audio(video_path: Path, out_path: Path) -> bool:
    ffmpeg_exe = get_ffmpeg_path() or "ffmpeg"

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", CODEC,
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-loglevel", "error",
        str(out_path),
    ]

    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] ffmpeg: {e}")
        return False


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  EXTRACT AUDIO  (pipeline)")
    print("=" * 60)

    if not check_ffmpeg():
        print("[ERROR] ไม่พบ FFmpeg ในเครื่อง")
        print("        ติดตั้ง: https://ffmpeg.org/download.html")
        print("        หรือ: pip install imageio-ffmpeg")
        sys.exit(1)

    metadata = load_metadata()
    if not metadata:
        print("[ERROR] metadata.json ว่างเปล่า")
        sys.exit(1)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output folder: {AUDIO_DIR}")
    print(f"[INFO] พบ {len(metadata)} รายการใน metadata\n")

    success, failed = 0, 0
    for key, info in metadata.items():
        rel_path = info.get("file_path")
        if not rel_path:
            print(f"[SKIP] {key}: ไม่มี 'file_path'")
            failed += 1
            continue

        # video path — ใช้ relative กับ base._paths.ROOT
        from base._paths import ROOT
        video_path = ROOT / "input" / rel_path

        if not video_path.exists():
            print(f"[SKIP] {key}: ไม่พบ {video_path}")
            failed += 1
            continue

        out_path = AUDIO_DIR / f"{key}.wav"
        print(f"[RUN ] {key}")
        print(f"       video : {video_path.name}")
        print(f"       audio : {out_path.name}")

        if extract_audio(video_path, out_path):
            size_kb = out_path.stat().st_size / 1024
            print(f"       [OK]  {size_kb:.1f} KB\n")
            success += 1
        else:
            print(f"       [FAIL]\n")
            failed += 1

    print("=" * 60)
    print(f"  เสร็จสิ้น: สำเร็จ {success}  |  ล้มเหลว {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()