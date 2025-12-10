# transcribe.py

import os, json
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from .audio_utils import get_media_duration_seconds
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import torch  

load_dotenv()
ASR_MODEL = os.getenv("ASR_MODEL", "medium")  # e.g., "base", "small", "medium", "large-v3"
ASR_DEVICE_ENV = os.getenv("ASR_DEVICE", None)  # optional, can still use this


def pick_asr_device(cli_choice: str) -> str:
    """
    cli_choice: 'auto', 'cpu', 'cuda', or 'env'
    returns 'cpu' or 'cuda'
    """
    if cli_choice == "cpu":
        return "cpu"
    if cli_choice == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if cli_choice == "env":
        if ASR_DEVICE_ENV in ("cpu", "cuda"):
            if ASR_DEVICE_ENV == "cuda" and not torch.cuda.is_available():
                print("[asr] ASR_DEVICE=cuda but no GPU available, falling back to cpu")
                return "cpu"
            return ASR_DEVICE_ENV or "cpu"
        return "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    print(f"[asr] loading model {model_name} (device={device}, compute_type={compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    full_iter, _ = model.transcribe(
        str(wav_path),
        task="transcribe",
        vad_filter=False,
        word_timestamps=True,
        temperature=0.0,
        beam_size=5,
        language="en",
        condition_on_previous_text=False,
        no_speech_threshold=0.2,
        log_prob_threshold=-1.5,
        compression_ratio_threshold=2.4,
        initial_prompt="Include all short responses like yeah, okay, mm-hmm, uh-huh, right, sure, haha.",
    )
    return list(full_iter)

def fuse_asr_with_diar(
    diar_segments: List[Dict[str, Any]],
    asr_segments: List[Any],
    overlap_regions: List[Dict[str, float]],
    *,
    segment_prefix: str
) -> List[Dict[str, Any]]:

    outputs = []

    eps_mid = 0.18
    min_word_overlap_frac = 0.20

    for idx, seg in enumerate(diar_segments):
        start, end = float(seg["start"]), float(seg["end"])
        words_in_seg: List[str] = []
        lps: List[float] = []

        # pass 1: segment-level confidence and midpoint word assignment
        for asr_seg in asr_segments:
            a_s = asr_seg.start or 0.0
            a_e = asr_seg.end or a_s

            if hasattr(asr_seg, "avg_logprob") and asr_seg.avg_logprob is not None:
                if intersects(start, end, a_s, a_e):
                    lps.append(asr_seg.avg_logprob)

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

        text = " ".join(words_in_seg).strip()
        ov = any(intersects(start, end, r["start"], r["end"]) for r in overlap_regions)

        outputs.append({
            "type": "speech",
            "segment_id": f"{segment_prefix}{idx:06d}",
            "start": round(start, 4),
            "end": round(end, 4),
            "speaker": seg["speaker"],
            "overlap": ov,
            "text": text,
        })

    return outputs


def build_timeline(speech, total_dur, min_silence=0.2):
    if not speech:
        return [{"type": "silence", "start": 0.0, "end": float(total_dur)}] if total_dur else []

    s = sorted(speech, key=lambda x: x["start"])

    merged = []
    for seg in s:
        st, en = float(seg["start"]), float(seg["end"])
        if not merged or st > merged[-1]["end"]:
            merged.append({"start": st, "end": en})
        else:
            merged[-1]["end"] = max(merged[-1]["end"], en)

    timeline = []

    if merged[0]["start"] >= min_silence:
        timeline.append({
            "type": "silence",
            "start": 0.0,
            "end": round(merged[0]["start"], 2),
        })

    for i, win in enumerate(merged):
        if i < len(merged) - 1:
            gap = merged[i + 1]["start"] - win["end"]
            if gap >= min_silence:
                timeline.append({
                    "type": "silence",
                    "start": round(win["end"], 2),
                    "end": round(merged[i + 1]["start"], 2),
                })

    if total_dur and total_dur - merged[-1]["end"] >= min_silence:
        timeline.append({
            "type": "silence",
            "start": round(merged[-1]["end"], 2),
            "end": round(float(total_dur), 2),
        })

    for seg in s:
        seg = dict(seg)
        seg["type"] = "speech"
        timeline.append(seg)

    timeline.sort(key=lambda x: x["start"])
    return timeline


def main():
    ap = argparse.ArgumentParser("Transcription")
    ap.add_argument("--meeting-id", required=True, help="Meeting ID")
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "env"],
        help="ASR device (default: auto, env uses ASR_DEVICE)",
    )
    args = ap.parse_args()

    effective_device = pick_asr_device(args.device)
    print(f"[asr] effective device: {effective_device}  (ASR_MODEL={ASR_MODEL})")

    out_dir = Path("data/outputs") / args.meeting_id
    wav = out_dir / "audio_16k_mono.wav"
    diar_raw_p = out_dir / "diarization_raw.json"
    diar_merged_p = out_dir / "diarization.json"

    assert wav.exists(), f"Audio not found: {wav}"
    assert diar_raw_p.exists(), f"Raw diarization not found: {diar_raw_p}"
    assert diar_merged_p.exists(), f"Merged diarization not found: {diar_merged_p}"
    diar_raw = json.loads(diar_raw_p.read_text())
    diar_merged = json.loads(diar_merged_p.read_text())

    print(f"[asr] transcribing full audio …")
    asr_segments = transcribe_full(wav, ASR_MODEL, effective_device)

    print(f"[asr] computing overlap regions (from RAW) …")
    overlap_regions = compute_overlap_regions(diar_raw, min_overlap_ms=40)
    total_dur = get_media_duration_seconds(wav)

    print(f"[asr] fusing ASR with RAW diarization …")
    speech_raw = fuse_asr_with_diar(
        diar_segments=diar_raw,
        asr_segments=asr_segments,
        overlap_regions=overlap_regions,
        segment_prefix="r",
    )
    timeline_raw = build_timeline(speech_raw, total_dur, min_silence=0.2)
    out_raw_json = out_dir / "transcript_raw.json"
    out_raw_json.write_text(json.dumps(timeline_raw, ensure_ascii=False, indent=2))
    print(f"[done] saved: {out_raw_json}")

    print(f"[asr] fusing ASR with MERGED diarization …")
    speech_merged = fuse_asr_with_diar(
        diar_segments=diar_merged,
        asr_segments=asr_segments,
        overlap_regions=overlap_regions,
        segment_prefix="m",
    )
    timeline_merged = build_timeline(speech_merged, total_dur, min_silence=0.2)
    out_merged_json = out_dir / "transcript_merged.json"
    out_merged_json.write_text(json.dumps(timeline_merged, ensure_ascii=False, indent=2))
    print(f"[done] saved: {out_merged_json}")

    out_json = out_dir / "transcript.json"
    out_json.write_text(json.dumps(timeline_merged, ensure_ascii=False, indent=2))
    print(f"[done] canonical: {out_json} (merged diarization)")


if __name__ == "__main__":
    main()
