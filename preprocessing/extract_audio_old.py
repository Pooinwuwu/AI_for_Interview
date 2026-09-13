import os
import sys
import subprocess
import shutil
from pathlib import Path

# Ensure UTF-8 output encoding on Windows to prevent charmap/UnicodeEncodeError
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ============================================================
# CONFIGURATION
# ============================================================

# Video extensions that the program will recognize
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v"
}

# Audio output format
AUDIO_EXTENSION = ".wav"

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Default directories for videos and extracted audio
DEFAULT_VIDEO_DIR = (
    BASE_DIR / "Videos" if (BASE_DIR / "Videos").is_dir()
    else (BASE_DIR / "videos" if (BASE_DIR / "videos").is_dir() else BASE_DIR / "Videos")
)

DEFAULT_AUDIO_DIR = (
    BASE_DIR / "Audio" if (BASE_DIR / "Audio").is_dir()
    else (BASE_DIR / "audio" if (BASE_DIR / "audio").is_dir() else BASE_DIR / "Audio")
)


# ============================================================
# CHECK FFMPEG
# ============================================================

FFMPEG_CMD = None


def get_ffmpeg_path():
    """
    Locate ffmpeg binary by checking:
    1. Standard system PATH
    2. Windows Registry PATH (for freshly installed tools before shell restart)
    3. imageio-ffmpeg Python fallback
    """
    global FFMPEG_CMD
    if FFMPEG_CMD:
        return FFMPEG_CMD

    # 1. Direct which check
    exe = shutil.which("ffmpeg")
    if exe:
        FFMPEG_CMD = exe
        return FFMPEG_CMD

    # 2. Refresh PATH from Windows registry (handles freshly installed packages)
    if sys.platform == "win32":
        try:
            import winreg
            for root, subkey in [
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                (winreg.HKEY_CURRENT_USER, r"Environment")
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

    # 3. Fallback to imageio-ffmpeg package if available
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            FFMPEG_CMD = exe
            return FFMPEG_CMD
    except ImportError:
        pass

    return None


def check_ffmpeg():
    """
    Check whether FFmpeg is installed and accessible.
    """
    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe:
        return False

    try:
        result = subprocess.run(
            [ffmpeg_exe, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode == 0:
            return True

    except Exception:
        pass

    return False


# ============================================================
# EXTRACT AUDIO FROM ONE VIDEO
# ============================================================

def extract_audio(video_path, output_folder):
    """
    Extract audio from one video file.

    Example:
        video:
            interview_01.mp4

        output:
            Audio_interview_01.wav
    """

    video_path = Path(video_path)
    output_folder = Path(output_folder)

    # Check video exists
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return False

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create output filename
    output_filename = f"Audio_{video_path.stem}{AUDIO_EXTENSION}"
    output_path = output_folder / output_filename

    print()
    print("=" * 60)
    print(f"Video : {video_path.name}")
    print(f"Audio : {output_filename}")
    print("=" * 60)

    # FFmpeg command
    ffmpeg_exe = get_ffmpeg_path() or "ffmpeg"
    command = [
        ffmpeg_exe,

        # Overwrite existing file automatically
        "-y",

        # Input video
        "-i", str(video_path),

        # Audio codec
        "-vn",

        # WAV PCM 16-bit
        "-acodec", "pcm_s16le",

        # Mono
        "-ac", str(CHANNELS),

        # Sample rate
        "-ar", str(SAMPLE_RATE),

        # Output
        str(output_path)
    ]

    try:
        result = subprocess.run(command)

        if result.returncode == 0:
            print(f"[SUCCESS] Audio extracted:")
            print(f"         {output_path}")
            return True

        else:
            print("[ERROR] FFmpeg failed.")
            return False

    except Exception as e:
        print(f"[ERROR] {e}")
        return False


# ============================================================
# FIND VIDEOS IN FOLDER
# ============================================================

def find_videos(folder):
    """
    Find all video files directly inside a folder.
    """

    folder = Path(folder)

    if not folder.exists():
        print(f"[ERROR] Folder not found: {folder}")
        return []

    videos = []

    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(file)

    # Sort alphabetically
    videos.sort()

    return videos


# ============================================================
# EXTRACT ALL VIDEOS IN FOLDER
# ============================================================

def extract_folder(input_folder, output_folder):
    """
    Extract audio from every video in a folder.
    """

    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    videos = find_videos(input_folder)

    if not videos:
        print()
        print("[INFO] No video files found.")
        return

    print()
    print("=" * 60)
    print(f"Input folder : {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Videos found : {len(videos)}")
    print("=" * 60)

    success_count = 0
    failed_count = 0

    for video in videos:

        success = extract_audio(
            video,
            output_folder
        )

        if success:
            success_count += 1
        else:
            failed_count += 1

    print()
    print("=" * 60)
    print("FOLDER EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Successful : {success_count}")
    print(f"Failed     : {failed_count}")
    print(f"Output     : {output_folder}")
    print("=" * 60)


# ============================================================
# SINGLE VIDEO MODE
# ============================================================

def single_video_mode():
    """
    Ask the user for a video filename/path.
    Looks in the default 'Videos' folder and base directory if a relative path or filename is given.
    """

    print()
    print("=" * 60)
    print("SINGLE VIDEO MODE")
    print("=" * 60)

    video_input = input(
        "Enter video filename/path: "
    ).strip().strip('"')

    if not video_input:
        print("[ERROR] No video entered.")
        return

    video_path = Path(video_input)

    # If user only types a filename or relative path,
    # check default Videos folder first, then base directory.
    if not video_path.is_absolute():
        if (DEFAULT_VIDEO_DIR / video_path).exists():
            video_path = DEFAULT_VIDEO_DIR / video_path
        elif (BASE_DIR / video_path).exists():
            video_path = BASE_DIR / video_path
        else:
            video_path = DEFAULT_VIDEO_DIR / video_path

    # Output folder
    output_folder = DEFAULT_AUDIO_DIR

    extract_audio(
        video_path,
        output_folder
    )


# ============================================================
# FOLDER MODE
# ============================================================

def folder_mode():
    """
    Ask the user for an input folder and output folder.
    Defaults to the 'Videos' folder and 'Audio' folder.
    """

    print()
    print("=" * 60)
    print("FOLDER MODE")
    print("=" * 60)

    input_input = input(
        f"Enter video folder path (press ENTER for '{DEFAULT_VIDEO_DIR.name}'): "
    ).strip().strip('"')

    if input_input:
        input_folder = Path(input_input)
        if not input_folder.is_absolute():
            input_folder = BASE_DIR / input_folder
    else:
        input_folder = DEFAULT_VIDEO_DIR

    output_input = input(
        f"Enter output folder path (press ENTER for '{DEFAULT_AUDIO_DIR.name}'): "
    ).strip().strip('"')

    if output_input:
        output_folder = Path(output_input)
        if not output_folder.is_absolute():
            output_folder = BASE_DIR / output_folder
    else:
        output_folder = DEFAULT_AUDIO_DIR

    extract_folder(
        input_folder,
        output_folder
    )


# ============================================================
# MAIN MENU
# ============================================================

def main():

    print()
    print("=" * 60)
    print("          VIDEO -> AUDIO EXTRACTOR")
    print("=" * 60)
    print()
    print("1. Extract audio from ONE video")
    print(f"2. Extract audio from ALL videos in a folder (default: '{DEFAULT_VIDEO_DIR.name}' -> '{DEFAULT_AUDIO_DIR.name}')")
    print("3. Exit")
    print()

    choice = input("Select option [1-3]: ").strip()

    if choice == "1":
        single_video_mode()

    elif choice == "2":
        folder_mode()

    elif choice == "3":
        print("Exiting...")

    else:
        print("[ERROR] Invalid option.")


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":

    if not check_ffmpeg():
        print()
        print("=" * 60)
        print("ERROR: FFmpeg was not found.")
        print("=" * 60)
        print()
        print("Please install FFmpeg and add it to PATH.")
        print()
        sys.exit(1)

    main()