import subprocess
from pathlib import Path
from typing import Optional

TARGET_SR_HZ = 16000
TARGET_MONO = True  # always mono

# Convert input media to WAV suitable for ASR/diarization:
# - Mono (1 ch), 16 kHz, 16-bit PCM
# - High-quality soxr resampling
# - Explicitly select first audio stream
def prepare_wav(input_media: Path, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(input_media),
        "-map", "0:a:0",            # pick first audio stream
        "-vn", "-sn", "-dn",        # drop video/subtitles/data
        "-ac", "1" if TARGET_MONO else "2",
        "-ar", str(TARGET_SR_HZ),
        "-af", "aresample=resampler=soxr",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    _run(cmd)

# Get media duration in seconds (for logging/sanity checks)
def get_media_duration_seconds(input_media: Path) -> Optional[float]:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_media)
    ]
    try:
        out = _run(cmd, capture=True).strip()
        return float(out)
    except Exception:
        return None

def _run(cmd, capture: bool = False) -> str:
    if capture:
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.stdout
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return ""
