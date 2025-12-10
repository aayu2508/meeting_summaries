# audio_preprocessing.py
import argparse
from pathlib import Path
from .audio_utils import prepare_wav, get_media_duration_seconds

def main():
    ap = argparse.ArgumentParser("Audio Preprocessing")
    ap.add_argument("--input", required=True, help="Path to audio/video file")
    ap.add_argument("--meeting-id", required=True, help="Meeting ID")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    assert in_path.exists(), f"Input not found: {in_path}"

    out_dir = Path("data/outputs") / args.meeting_id
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_out = out_dir / "audio_16k_mono.wav"

    print(f"[prep] converting to mono 16 kHz -> {wav_out}")
    prepare_wav(in_path, wav_out)

    # Log durations before/after
    in_dur = get_media_duration_seconds(in_path)
    out_dur = get_media_duration_seconds(wav_out)
    if in_dur:
        print(f"[prep] input duration:  {in_dur:.2f}s")
    if out_dur:
        print(f"[prep] output duration: {out_dur:.2f}s")
        if in_dur:
            print(f"[prep] duration ratio (out/in): {out_dur/in_dur:.3f}")

    print(f"[done] saved: {wav_out}")

if __name__ == "__main__":
    main()
