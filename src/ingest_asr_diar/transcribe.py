# transcribe.py
import os, json, math
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()
ASR_MODEL = os.getenv("ASR_MODEL", "medium")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")

# ---------- utils ----------
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def intersects(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return not (a_end <= b_start or b_end <= a_start)

def compute_overlap_regions(raw_turns: List[Dict[str, Any]], min_overlap_ms: int = 40) -> List[Dict[str, float]]:
    """Return intervals where >=2 speakers talk simultaneously (from RAW diarization)."""
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
        if prev < 2 and active >= 2:
            cur_start = ts
        elif prev >= 2 and active < 2 and cur_start is not None:
            if (ts - cur_start) * 1000.0 >= min_overlap_ms:
                regions.append({"start": cur_start, "end": ts})
            cur_start = None
    return regions

def transcribe_full(wav_path: Path, model_name: str, device: str):
    compute_type = "int8" if device == "cpu" else "float16"
    print(f"[asr] loading model {model_name} ({device}, {compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    full_iter, _ = model.transcribe(
        str(wav_path),
        task="transcribe",
        vad_filter=False,              # keep short interjections
        word_timestamps=True,
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

# ---------- ASR↔Diar fusion with fewer blanks ----------
def fuse_asr_with_diar(
    diar_segments: List[Dict[str, Any]],
    asr_segments: List[Any],
    overlap_regions: List[Dict[str, float]]
) -> List[Dict[str, Any]]:
    outputs = []
    eps_mid = 0.12  # wider slack to catch boundary words

    for idx, seg in enumerate(diar_segments):
        start, end = float(seg["start"]), float(seg["end"])
        words_in_seg, lps = [], []

        # pass 1: collect confidences + midpoint assignment
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

        # pass 2 (fallback): use >=50% duration overlap if still empty
        if not words_in_seg:
            for asr_seg in asr_segments:
                if getattr(asr_seg, "words", None):
                    for w in asr_seg.words:
                        ws = w.start or 0.0
                        we = w.end or ws
                        ov = min(end, we) - max(start, ws)
                        dur = max(1e-6, we - ws)
                        if ov > 0 and (ov / dur) >= 0.5:
                            wt = (w.word or "").strip()
                            if wt:
                                words_in_seg.append(wt)

        text = " ".join(words_in_seg).strip()
        conf = None
        if lps:
            avg_lp = sum(lps) / len(lps)
            conf = max(0.0, min(1.0, sigmoid(avg_lp + 1.0)))

        ov = any(intersects(start, end, r["start"], r["end"]) for r in overlap_regions)

        outputs.append({
            "type": "speech",
            "segment_id": f"s{idx:06d}",
            "segment_idx": idx,
            "start": round(start, 2),
            "end": round(end, 2),
            "speaker": seg["speaker"],
            "overlap": ov,
            "text": text,
            "asr_confidence": round(conf, 3) if conf is not None else None,
        })
    return outputs

# ---------- build timeline (speech + silence) ----------
def build_timeline(speech: List[Dict[str, Any]], total_dur: Optional[float], min_silence: float = 0.2) -> List[Dict[str, Any]]:
    """Combine speech and silence chronologically."""
    if not speech:
        return [{"type": "silence", "start": 0.0, "end": float(total_dur)}] if total_dur else []

    # Optional tiny-empty shard cleanup (reduces blank micro-turns)
    cleaned = []
    for i, seg in enumerate(sorted(speech, key=lambda x: x["start"])):
        dur = seg["end"] - seg["start"]
        if seg.get("text") or dur >= 0.25:
            cleaned.append(seg)
        else:
            if cleaned and cleaned[-1]["speaker"] == seg["speaker"] and (seg["start"] - cleaned[-1]["end"]) <= 0.08:
                cleaned[-1]["end"] = seg["end"]  # merge into previous
            else:
                # drop; silence will cover this gap
                pass

    timeline: List[Dict[str, Any]] = []
    s = cleaned if cleaned else []

    if not s:
        return [{"type": "silence", "start": 0.0, "end": float(total_dur)}] if total_dur else []

    # leading silence
    if s[0]["start"] > 0 and s[0]["start"] >= min_silence:
        timeline.append({"type": "silence", "start": 0.0, "end": s[0]["start"]})

    # main body
    for i, seg in enumerate(s):
        timeline.append(seg)
        if i < len(s) - 1:
            gap = s[i + 1]["start"] - seg["end"]
            if gap >= min_silence:
                timeline.append({"type": "silence", "start": seg["end"], "end": s[i + 1]["start"]})

    # trailing silence
    if total_dur and total_dur - s[-1]["end"] >= min_silence:
        timeline.append({"type": "silence", "start": s[-1]["end"], "end": float(total_dur)})

    return timeline

def main():
    import argparse
    from .audio_utils import get_media_duration_seconds

    ap = argparse.ArgumentParser("Step 3: Transcription (timeline JSON)")
    ap.add_argument("--meeting-id", required=True, help="Meeting ID")
    args = ap.parse_args()

    out_dir = Path("data/outputs") / args.meeting_id
    wav = out_dir / "audio_16k_mono.wav"
    diar_merged_p = out_dir / "diarization.json"
    diar_raw_p = out_dir / "diarization_raw.json"

    assert wav.exists(), f"Audio not found: {wav}"
    assert diar_merged_p.exists(), f"Diarization not found: {diar_merged_p}"
    assert diar_raw_p.exists(), f"Raw diarization not found: {diar_raw_p}"

    diar_merged = json.loads(diar_merged_p.read_text())
    diar_raw = json.loads(diar_raw_p.read_text())

    print(f"[asr] transcribing full audio …")
    asr_segments = transcribe_full(wav, ASR_MODEL, ASR_DEVICE)

    print(f"[asr] computing overlap regions …")
    overlap_regions = compute_overlap_regions(diar_raw, min_overlap_ms=40)

    print(f"[asr] fusing ASR with diarization …")
    speech_segments = fuse_asr_with_diar(diar_merged, asr_segments, overlap_regions)

    total_dur = get_media_duration_seconds(wav)
    timeline = build_timeline(speech_segments, total_dur, min_silence=0.2)

    # Save only the timeline (clean list with segment_id inside each speech entry)
    out_json = out_dir / "transcript.json"
    out_json.write_text(json.dumps(timeline, ensure_ascii=False, indent=2))
    print(f"[done] saved: {out_json}")

if __name__ == "__main__":
    main()
