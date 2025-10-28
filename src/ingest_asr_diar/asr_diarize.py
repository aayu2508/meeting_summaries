import os, json, math, tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv
from audio_utils import run_ffmpeg_normalize, get_media_duration_seconds
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from faster_whisper import WhisperModel

load_dotenv()  # loads .env if present

# Configs
# faster-whisper model size: tiny|base|small|medium|large-v3
ASR_MODEL = os.getenv("ASR_MODEL", "large-v3")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")
HF_TOKEN   = os.getenv("HF_TOKEN", None)
MERGE_GAP_S = 0.1
MAX_SEG_S   = 30
SAMPLE_RATE = 16000

# Cleans up diarization output
def merge_microturns(turns: List[Dict[str, Any]], merge_gap: float, max_seg: float) -> List[Dict[str, Any]]:

    if not turns: return []

    # sort (as time can never go backwards)
    turns = sorted(turns, key=lambda t: (t["start"], t["end"]))

    # merge micro gaps between neighboring segments of the same speaker to reduce the number of turns
    # if next turn has the same speaker and the gap between cur.end and next.start is non-negative and < merge_gap (e.g., 0.7 s), treat it as one continuous turn.
    merged = []
    cur = turns[0].copy()
    for nxt in turns[1:]:
        same = nxt["speaker"] == cur["speaker"]
        gap  = nxt["start"] - cur["end"]
        if same and 0 <= gap < merge_gap:
            cur["end"] = max(cur["end"], nxt["end"])
        else:
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)

    # split long segments (max_seg)
    # if a single turn exceeds, say, 30s, split it into N nearly equal slices (e.g., three ~25s pieces for a 75s monologue).
    final = []
    for t in merged:
        dur = t["end"] - t["start"]
        if dur <= max_seg:
            final.append(t)
            continue
        n_parts = math.ceil(dur / max_seg)
        part_len = dur / n_parts
        for i in range(n_parts):
            s = t["start"] + i * part_len
            e = min(t["end"], s + part_len)
            final.append({"start": s, "end": e, "speaker": t["speaker"]})

    # reassign speakers as S0, S1, ... in first-seen order 
    # need to attach speaker names to segments (maybe do it through UI)
    seen = {}
    counter = 0
    for t in final:
        lab = t["speaker"]
        if lab not in seen:
            seen[lab] = f"S{counter}"
            counter += 1
        t["speaker"] = seen[lab]
    return final

# Run pyannote diarization (who spoke what and when) and return raw speech turns.
def diarize(wav_path: Path) -> list[dict]:
    if HF_TOKEN is None:
        raise RuntimeError("HF_TOKEN not set. Create a Hugging Face read token and export HF_TOKEN.")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=HF_TOKEN)
    with ProgressHook() as hook:
        output = pipeline(str(wav_path), hook=hook)

    # convert to your turn format
    turns = []
    for turn, speaker in output.speaker_diarization:
        start, end = float(turn.start), float(turn.end)
        if end <= start:
            continue
        turns.append({"start": start, "end": end, "speaker": f"{speaker}"})
    return turns

# ASR with faster-whisper over specified segments
def transcribe_segments(wav_path, segments: List[Dict[str, Any]], model_size: str, device: str):
    model_path = Path(model_size)
    if model_path.exists() and model_path.is_dir():
        print(f"[asr] loading local model from {model_path}")
        model = WhisperModel(
            model_size_or_path=str(model_path),
            device=device,
            compute_type="int8",
            cpu_threads=4,
            num_workers=1
        )
    else:
        print(f"[asr] loading model {model_size}")
        model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8"
        )
    
    # single full transcription with word timestamps
    full_iter, _ = model.transcribe(
        str(wav_path),
        task="transcribe",
        vad_filter=False,
        word_timestamps=True,  
        temperature=0.0,
        beam_size=5,
        language="en",
        condition_on_previous_text=True,     
        no_speech_threshold=0.2,             
        log_prob_threshold=-1.5,             
        compression_ratio_threshold=2.4,     
        initial_prompt="Include all speech including yeah, okay, mm-hmm, uh-huh, right, sure, and other short responses.",
    )
    asr_segments = list(full_iter)
    
    outputs = []
    for idx, seg in enumerate(segments):
        start, end = float(seg["start"]), float(seg["end"])
        
        # collect words that fall within this segment
        words_in_seg = []
        lps = []
        
        for asr_seg in asr_segments:
            if not hasattr(asr_seg, 'words') or not asr_seg.words:
                asr_start = asr_seg.start if asr_seg.start is not None else 0
                asr_end = asr_seg.end if asr_seg.end is not None else asr_start
                
                overlap_start = max(start, asr_start)
                overlap_end = min(end, asr_end)
                overlap_dur = max(0, overlap_end - overlap_start)
                asr_dur = asr_end - asr_start
                
                if asr_dur > 0 and (overlap_dur / asr_dur) > 0.5:
                    words_in_seg.append(asr_seg.text.strip())
                    if hasattr(asr_seg, "avg_logprob") and asr_seg.avg_logprob is not None:
                        lps.append(asr_seg.avg_logprob)
                continue
            
            # use word-level timestamps for precise alignment
            for word in asr_seg.words:
                ws = word.start if word.start is not None else 0.0
                we = word.end   if word.end   is not None else ws
                seg_s, seg_e = start, end

                inter = max(0.0, min(we, seg_e) - max(ws, seg_s))
                wdur  = max(1e-6, we - ws)

                # small tolerance helps catch boundary/backchannel tokens
                tol = 0.20 if (seg_e - seg_s) < 1.0 else 0.10
                touching = (seg_s - tol <= we) and (ws <= seg_e + tol)

                # keep if a meaningful fraction of word overlaps, or it "touches" within tol
                if (inter / wdur) >= 0.35 or touching:
                    words_in_seg.append((ws, we, word.word.strip()))
            
            # add confidence from any overlapping ASR segment
            if hasattr(asr_seg, "avg_logprob") and asr_seg.avg_logprob is not None:
                asr_start = asr_seg.start if asr_seg.start is not None else 0
                asr_end = asr_seg.end if asr_seg.end is not None else asr_start
                if not (asr_end < start or asr_start > end):
                    lps.append(asr_seg.avg_logprob)
        
        # extract just the word text from tuples
        text = " ".join(w[2] for w in words_in_seg).strip()
        
        conf = None
        if lps:
            avg_lp = sum(lps) / len(lps)
            conf = 1.0 / (1.0 + math.exp(-(avg_lp + 1.0)))
        
        outputs.append({
            "segment_idx": idx,
            "start": round(start, 2),
            "end": round(end, 2),
            "speaker": seg["speaker"],
            "text": text,
            "asr_confidence": round(conf, 3) if conf is not None else None,
        })
    
    return outputs

# Merge only consecutive segments that are truly the same speaker, same text, and contiguous (no overlap).
def deduplicate_segments(segments: List[Dict]) -> List[Dict]:
    if not segments:
        return []

    result = [segments[0]]
    for seg in segments[1:]:
        prev = result[-1]

        same_speaker = seg.get("speaker") == prev.get("speaker")
        same_text = (seg.get("text") or "") == (prev.get("text") or "") and seg.get("text")
        # treat as contiguous if there is a tiny boundary gap but no overlap
        contiguous = seg["start"] >= prev["end"] and (seg["start"] - prev["end"]) <= 0.05

        if same_speaker and same_text and contiguous:
            prev["end"] = max(prev["end"], seg["end"])
        else:
            result.append(seg)

    # reindex safely
    for i, s in enumerate(result):
        s["segment_idx"] = i
        s["segment_id"] = f"s{i:06d}"
    return result

def write_jsonl(rows: List[Dict[str, Any]], out_file: Path, meeting_id: str):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            row = {
                "meeting_id": meeting_id,
                "segment_id": f"s{str(i).zfill(6)}",
                **r
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    import argparse
    ap = argparse.ArgumentParser("Step 1: Ingest -> Diarize -> ASR")
    ap.add_argument("--input", required=True, help="Path to audio/video (e.g., data/raw/meeting.mp4)")
    ap.add_argument("--meeting-id", required=True, help="Meeting ID (e.g., m1)")
    ap.add_argument("--out-dir", default="data/outputs", help="Base output dir")
    ap.add_argument("--sr", type=int, default=SAMPLE_RATE, help="Target sample rate (Hz)")
    ap.add_argument("--merge-gap", type=float, default=MERGE_GAP_S, help="Merge same-speaker micro-gap (s)")
    ap.add_argument("--max-seg", type=float, default=MAX_SEG_S, help="Max segment length (s)")
    ap.add_argument("--asr-model", default=ASR_MODEL, help="faster-whisper size (tiny|base|small|medium|large-v3)")
    ap.add_argument("--device", default=ASR_DEVICE, help="cpu|cuda")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    assert in_path.exists(), f"Input not found: {in_path}"

    meeting_id = args.meeting_id
    out_dir = Path(args.out_dir) / meeting_id
    out_jsonl = out_dir / "segments.jsonl"

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wav = tmp / "audio_16k_mono.wav"
        print(f"[prep] normalizing audio -> {wav}")
        run_ffmpeg_normalize(in_path, wav, sr_hz=args.sr, mono=True)

        dur = get_media_duration_seconds(in_path)
        if dur:
            print(f"[prep] source duration ~ {dur:.1f}s")

        print("[diar] running pyannote diarization (this can take a bit)...")
        raw_turns = diarize(wav)
        print(f"[diar] raw turns: {len(raw_turns)}")

        merged = merge_microturns(raw_turns, merge_gap=args.merge_gap, max_seg=args.max_seg)
        print(f"[diar] cleaned segments: {len(merged)}  (merge_gap={args.merge_gap}s, max_seg={args.max_seg}s)")
        
        print(f"[asr] transcribing {len(merged)} segments with faster-whisper ({args.asr_model}, {args.device})...")
        rows = transcribe_segments(wav, merged, model_size=args.asr_model, device=args.device)
        
        # Deduplicate repeated text
        rows = deduplicate_segments(rows)
        print(f"[asr] done! {len(rows)} unique segments after deduplication")
        
        write_jsonl(rows, out_jsonl, meeting_id)
        print(f"[done] wrote {out_jsonl}")

if __name__ == "__main__":
    main()
