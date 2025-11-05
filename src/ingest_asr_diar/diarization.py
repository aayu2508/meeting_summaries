# diarization.py
import os, json
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", None)

MERGE_GAP_S = 1.5        # merge same-speaker pauses shorter than this
OVERLAP_JOIN_S = 0.05    # also merge tiny overlaps (<= 50 ms)

def merge_microturns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge adjacent segments from the SAME speaker if the silence between them is short
    (<= MERGE_GAP_S) or if there is a tiny boundary overlap (>= -OVERLAP_JOIN_S).
    - Never merges segments from different speakers.
    - Does NOT split long turns (we keep them intact).
    """
    if not turns:
        return []

    turns = sorted(turns, key=lambda t: (t["start"], t["end"]))
    merged: List[Dict[str, Any]] = []
    cur = turns[0].copy()

    for nxt in turns[1:]:
        same = nxt["speaker"] == cur["speaker"]
        gap = float(nxt["start"]) - float(cur["end"])  # negative => overlap

        # Inclusive bounds to avoid float precision edge cases.
        if same and (-OVERLAP_JOIN_S <= gap <= MERGE_GAP_S):
            cur["end"] = max(cur["end"], nxt["end"])
        else:
            merged.append(cur)
            cur = nxt.copy()

    merged.append(cur)
    return merged

def diarize(wav_path: Path, num_speakers: Optional[int] = None) -> List[Dict[str, Any]]:
    if HF_TOKEN is None:
        raise RuntimeError("HF_TOKEN not set (export HF_TOKEN=<your_hf_token>)")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=HF_TOKEN,
        cache_dir="data/cache"
    )

    with ProgressHook() as hook:
        output = (
            pipeline(str(wav_path), hook=hook, num_speakers=num_speakers)
            if num_speakers else
            pipeline(str(wav_path), hook=hook)
        )

    turns: List[Dict[str, Any]] = []
    for turn, speaker in output.speaker_diarization:
        start, end = float(turn.start), float(turn.end)
        if end > start:
            turns.append({"start": start, "end": end, "speaker": f"{speaker}"})
    return turns

def main():
    ap = argparse.ArgumentParser("======================== Speaker Diarization ========================")
    ap.add_argument("--meeting-id", required=True, help="Meeting ID")
    ap.add_argument("--num-speakers", type=int, default=None, help="Optional fixed number of speakers")
    args = ap.parse_args()

    out_dir = Path("data/outputs") / args.meeting_id
    wav = out_dir / "audio_16k_mono.wav"
    assert wav.exists(), f"Audio not found: {wav}. Run preprocessing first."

    raw_json = out_dir / "diarization_raw.json"
    clean_json = out_dir / "diarization.json"

    print(f"[diar] running diarization on {wav.name} ...")
    raw_turns = diarize(wav, num_speakers=args.num_speakers)
    print(f"[diar] raw segments: {len(raw_turns)}")
    raw_json.write_text(json.dumps(raw_turns, indent=2))

    merged = merge_microturns(raw_turns)
    print(f"[diar] merged: {len(merged)} (gap<{MERGE_GAP_S:.2f}s, overlap<{OVERLAP_JOIN_S:.2f}s)")
    clean_json.write_text(json.dumps(merged, indent=2))

    print(f"[done] raw:   {raw_json}")
    print(f"[done] clean: {clean_json}")

if __name__ == "__main__":
    main()
