# diarization.py
import os, json, math
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", None)

MERGE_GAP_S = 1.5        # merge same-speaker pauses shorter than this
OVERLAP_JOIN_S = 0.05    # also merge tiny overlaps (<= 50 ms)
MAX_SEG_S = 30.0         # split very long turns into <= 30s chunks

# def merge_microturns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Merge adjacent same-speaker turns across tiny overlaps and short gaps.
#        Then split any very long turns into <= MAX_SEG_S pieces.
#        Finally, relabel speakers deterministically S0, S1, ...
#     """
#     if not turns:
#         return []

#     # sort by time
#     turns = sorted(turns, key=lambda t: (t["start"], t["end"]))

#     # merge adjacent turns for same speaker
#     merged: List[Dict[str, Any]] = []
#     cur = turns[0].copy()
#     for nxt in turns[1:]:
#         same = nxt["speaker"] == cur["speaker"]
#         gap = nxt["start"] - cur["end"]  # negative means tiny overlap
#         if same and (-OVERLAP_JOIN_S <= gap < MERGE_GAP_S):
#             # extend current
#             cur["end"] = max(cur["end"], nxt["end"])
#         else:
#             merged.append(cur)
#             cur = nxt.copy()
#     merged.append(cur)

#     # split long turns (simple uniform split)
#     final: List[Dict[str, Any]] = []
#     for t in merged:
#         dur = t["end"] - t["start"]
#         if dur <= MAX_SEG_S:
#             final.append(t)
#             continue
#         n_parts = max(1, math.ceil(dur / MAX_SEG_S))
#         part_len = dur / n_parts
#         for i in range(n_parts):
#             s = t["start"] + i * part_len
#             e = min(t["end"], s + part_len)
#             final.append({"start": s, "end": e, "speaker": t["speaker"]})

#     # deterministic relabeling by first appearance order
#     seen = {}
#     next_id = 0
#     for t in final:
#         lab = t["speaker"]
#         if lab not in seen:
#             seen[lab] = f"S{next_id}"
#             next_id += 1
#         t["speaker"] = seen[lab]
#     return final

def merge_microturns(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not turns:
        return []

    turns = sorted(turns, key=lambda t: (t["start"], t["end"]))

    merged: List[Dict[str, Any]] = []
    cur = turns[0].copy()

    for nxt in turns[1:]:
        same = nxt["speaker"] == cur["speaker"]
        gap = float(nxt["start"]) - float(cur["end"])  # negative => overlap

        # NOTE: inclusive upper bound helps with float equality cases
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
    import argparse

    ap = argparse.ArgumentParser("Step 2: Speaker Diarization (speech-only)")
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
    print(f"[diar] merged/split: {len(merged)} (gap<{MERGE_GAP_S:.2f}s, overlap<{OVERLAP_JOIN_S:.2f}")
    clean_json.write_text(json.dumps(merged, indent=2))

    print(f"[done] raw:   {raw_json}")
    print(f"[done] clean: {clean_json}")

if __name__ == "__main__":
    main()
