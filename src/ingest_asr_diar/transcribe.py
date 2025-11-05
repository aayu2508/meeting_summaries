# transcribe.py

import os, json
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from .audio_utils import get_media_duration_seconds
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()
ASR_MODEL = os.getenv("ASR_MODEL", "medium")  # e.g., "base", "small", "medium", "large-v3"
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")   # "cpu" or "cuda"

# To check intersection between two half open time intervals
def intersects(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return not (a_end <= b_start or b_end <= a_start)

def compute_overlap_regions(
    raw_turns: List[Dict[str, Any]],
    min_overlap_ms: int = 40
) -> List[Dict[str, float]]:

    if not raw_turns:
        return []

    events = []
    for t in raw_turns:
        events.append((float(t["start"]), +1))
        events.append((float(t["end"]), -1))
    events.sort()

    regions = []
    active = 0
    cur_start = None

    for ts, delta in events:
        prev = active
        active += delta

        # rising edge into overlap
        if prev < 2 and active >= 2:
            cur_start = ts

        # falling edge out of overlap
        elif prev >= 2 and active < 2 and cur_start is not None:
            if (ts - cur_start) * 1000.0 >= min_overlap_ms:
                regions.append({"start": cur_start, "end": ts})
            cur_start = None

    return regions

def transcribe_full(wav_path: Path, model_name: str, device: str):
    compute_type = "int8" if device == "cpu" else "float16"
    print(f"[asr] loading model {model_name} ({device}, {compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    # Keep short interjections: vad_filter=False + gentle decoding thresholds.
    # initial_prompt nudges the model not to drop fillers like "yeah/okay".
    full_iter, _ = model.transcribe(
        str(wav_path),
        task="transcribe",
        vad_filter=False,                  # do not filter out short utterances
        word_timestamps=True,              # needed for word-level alignment
        temperature=0.0,
        beam_size=5,
        language="en",
        condition_on_previous_text=False, 
        no_speech_threshold=0.2, 
        log_prob_threshold=-1.5, 
        compression_ratio_threshold=2.4, 
        initial_prompt="Include all short responses like yeah, okay, mm-hmm, uh-huh, right, sure, haha."
    )
    return list(full_iter)

# Collect ASR words whose midpoints fall inside the diar window (with slack), or that overlap at least a fraction of their duration.
def fuse_asr_with_diar(
    diar_segments: List[Dict[str, Any]],
    asr_segments: List[Any],
    overlap_regions: List[Dict[str, float]],
    *,
    segment_prefix: str
) -> List[Dict[str, Any]]:

    outputs = []

    # Alignment tweaks to better capture short interjections near boundaries:
    eps_mid = 0.18   # slack around diar window for midpoint test (seconds)
    min_word_overlap_frac = 0.20  # fallback: keep word if >=20% overlaps

    for idx, seg in enumerate(diar_segments):
        start, end = float(seg["start"]), float(seg["end"])
        words_in_seg: List[str] = []
        lps: List[float] = []

        # pass 1: segment-level confidence and midpoint word assignment
        for asr_seg in asr_segments:
            a_s = asr_seg.start or 0.0
            a_e = asr_seg.end or a_s

            # collect logprobs for a soft confidence (optional)
            if hasattr(asr_seg, "avg_logprob") and asr_seg.avg_logprob is not None:
                if intersects(start, end, a_s, a_e):
                    lps.append(asr_seg.avg_logprob)

            # collect words if their midpoints fall in diar window (with slack)
            if getattr(asr_seg, "words", None):
                for w in asr_seg.words:
                    ws = w.start or 0.0
                    we = w.end or ws
                    mid = 0.5 * (ws + we)
                    if (start - eps_mid) <= mid <= (end + eps_mid):
                        wt = (w.word or "").strip()
                        if wt:
                            words_in_seg.append(wt)

        # pass 2 (fallback): if still empty, use >= min_word_overlap_frac duration overlap
        if not words_in_seg:
            for asr_seg in asr_segments:
                if getattr(asr_seg, "words", None):
                    for w in asr_seg.words:
                        ws = w.start or 0.0
                        we = w.end or ws
                        ov = min(end, we) - max(start, ws)
                        dur = max(1e-6, we - ws)
                        if ov > 0 and (ov / dur) >= min_word_overlap_frac:
                            wt = (w.word or "").strip()
                            if wt:
                                words_in_seg.append(wt)

        # finalize text + a soft confidence score
        text = " ".join(words_in_seg).strip()

        # overlapping speech indicator (useful for interactivity/visuals)
        ov = any(intersects(start, end, r["start"], r["end"]) for r in overlap_regions)

        outputs.append({
            "type": "speech",                           # speech row (silences added later)
            "segment_id": f"{segment_prefix}{idx:06d}", # stable id for later chunking
            "start": round(start, 4),
            "end": round(end, 4),
            "speaker": seg["speaker"],
            "overlap": ov,
            "text": text,
        })

    return outputs

# Build a dense timeline covering every moment of the meeting by inserting explicit "silence" rows between speech segments
def build_timeline(
    speech: List[Dict[str, Any]],
    total_dur: Optional[float],
    min_silence: float = 0.2
) -> List[Dict[str, Any]]:

    if not speech:
        return [{"type": "silence", "start": 0.0, "end": float(total_dur)}] if total_dur else []

    # Sort chronologically
    s = sorted(speech, key=lambda x: x["start"])
    timeline: List[Dict[str, Any]] = []

    # Leading silence
    if s[0]["start"] > 0 and s[0]["start"] >= min_silence:
        timeline.append({
            "type": "silence",
            "start": 0.0,
            "end": round(s[0]["start"], 2),
        })

    # Speech + inter-speech silences
    for i, seg in enumerate(s):
        # ensure type is set (safety; should already be "speech")
        seg = dict(seg)
        seg["type"] = "speech"
        timeline.append(seg)

        if i < len(s) - 1:
            gap = s[i + 1]["start"] - seg["end"]
            if gap >= min_silence:
                timeline.append({
                    "type": "silence",
                    "start": round(seg["end"], 2),
                    "end": round(s[i + 1]["start"], 2),
                })

    # Trailing silence
    if total_dur and total_dur - s[-1]["end"] >= min_silence:
        timeline.append({
            "type": "silence",
            "start": round(s[-1]["end"], 2),
            "end": round(float(total_dur), 2),
        })

    return timeline

def main():
    ap = argparse.ArgumentParser("======================== Transcription ========================")
    ap.add_argument("--meeting-id", required=True, help="Meeting ID")
    args = ap.parse_args()

    out_dir = Path("data/outputs") / args.meeting_id
    wav = out_dir / "audio_16k_mono.wav"
    diar_raw_p = out_dir / "diarization_raw.json"
    diar_merged_p = out_dir / "diarization.json"

    assert wav.exists(), f"Audio not found: {wav}"
    assert diar_raw_p.exists(), f"Raw diarization not found: {diar_raw_p}"
    assert diar_merged_p.exists(), f"Merged diarization not found: {diar_merged_p}"
    diar_raw = json.loads(diar_raw_p.read_text())
    diar_merged = json.loads(diar_merged_p.read_text())

    # Transcribe once (to speed up the process)
    print(f"[asr] transcribing full audio …")
    asr_segments = transcribe_full(wav, ASR_MODEL, ASR_DEVICE)

    # Overlap regions from RAW (most faithful to interaction)
    print(f"[asr] computing overlap regions (from RAW) …")
    overlap_regions = compute_overlap_regions(diar_raw, min_overlap_ms=40)
    total_dur = get_media_duration_seconds(wav)

    print(f"[asr] fusing ASR with RAW diarization …")
    speech_raw = fuse_asr_with_diar(
        diar_segments=diar_raw,
        asr_segments=asr_segments,
        overlap_regions=overlap_regions,
        segment_prefix="r",  # r000001, r000002, ...
    )
    timeline_raw = build_timeline(speech_raw, total_dur, min_silence=0.2)
    out_raw_json = out_dir / "transcript_raw.json"
    out_raw_json.write_text(json.dumps(timeline_raw, ensure_ascii=False, indent=2))
    print(f"[done] saved: {out_raw_json}")

    print(f"[asr] fusing ASR with MERGED diarization …")
    speech_merged = fuse_asr_with_diar(
        diar_segments=diar_merged,
        asr_segments=asr_segments,
        overlap_regions=overlap_regions,  # still computed from RAW (more precise)
        segment_prefix="m",  # m000001, m000002, ...
    )
    timeline_merged = build_timeline(speech_merged, total_dur, min_silence=0.2)
    out_merged_json = out_dir / "transcript_merged.json"
    out_merged_json.write_text(json.dumps(timeline_merged, ensure_ascii=False, indent=2))
    print(f"[done] saved: {out_merged_json}")

    # Canonical transcript: choose RAW by default (swap to merged if preferred)
    out_json = out_dir / "transcript.json"
    out_json.write_text(json.dumps(timeline_raw, ensure_ascii=False, indent=2))
    print(f"[done] canonical: {out_json} (RAW-aligned)")

if __name__ == "__main__":
    main()
