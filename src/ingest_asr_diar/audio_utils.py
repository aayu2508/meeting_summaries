# audio_utils.py

import json
import subprocess
from pathlib import Path
from typing import Optional

TARGET_SR_HZ = 16000
TARGET_MONO = True  # always mono

# Convert input media to WAV suitable for ASR/diarization:
# - Mono (1 ch), 16 kHz, 16-bit PCM
# - High-quality soxr resampling
# - Select "best" audio stream by simple heuristic (duration, then <=2 channels)
def prepare_wav(input_media: Path, out_wav: Path) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # Pick best audio stream; fallback to first if probing fails
    stream_index = _pick_best_audio_stream(input_media)
    if stream_index is None:
        stream_index = 0

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(input_media),
        "-map", f"0:a:{stream_index}",  # pick best audio stream
        "-vn", "-sn", "-dn",            # drop video/subtitles/data
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

def _pick_best_audio_stream(input_media: Path) -> Optional[int]:
    """
    Heuristic:
      1) Prefer longer duration (likely the main program/mic track).
      2) Tie-breaker: prefer <=2 channels (speech-like) over multichannel.
    Returns the stream index, or None if probing fails.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index,channels,sample_rate,duration",
        "-of", "json",
        str(input_media)
    ]
    try:
        out = _run(cmd, capture=True)
        data = json.loads(out)
        streams = data.get("streams", [])
        if not streams:
            return None

        def score(s):
            dur = float(s.get("duration") or 0.0)
            ch = int(s.get("channels") or 99)
            # Higher duration is better; <=2 channels preferred
            speechy = 1 if ch <= 2 else 0
            return (dur, speechy)

        best = max(streams, key=score)
        return int(best["index"])
    except Exception:
        return None

def _run(cmd, capture: bool = False) -> str:
    if capture:
        res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.stdout
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return ""
