import os, json, gc, time
from pathlib import Path
from dotenv import load_dotenv
import whisperx

# -------- config --------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", None)

device = "cpu"
audio_file = "data/raw/S16.wav"
batch_size = 4         # reduce if low on memory
compute_type = "int8"  # use "float16" or "int8" depending on device/mem

# outputs live next to this script
SCRIPT_DIR = Path(__file__).resolve().parent
F_RAW_JSON   = SCRIPT_DIR / "asr_segments_raw.json"
F_RAW_JSONL  = SCRIPT_DIR / "asr_segments_raw.jsonl"
F_ALIGN_JSON = SCRIPT_DIR / "asr_segments_aligned.json"
F_ALIGN_JSONL= SCRIPT_DIR / "asr_segments_aligned.jsonl"
F_DIAR_JSON  = SCRIPT_DIR / "diarization_segments.json"
F_DIAR_JSONL = SCRIPT_DIR / "diarization_segments.jsonl"
F_SPK_JSON   = SCRIPT_DIR / "asr_segments_speaker.json"
F_SPK_JSONL  = SCRIPT_DIR / "asr_segments_speaker.jsonl"

def write_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

start = time.time()

# 0) load audio (always needed for alignment/diar/assignment)
audio = whisperx.load_audio(audio_file)

# ------------------ SMART REUSE FOR STEPS 1 & 2 ------------------
result_aligned = None

if F_ALIGN_JSON.exists():
    # ✅ Reuse aligned ASR
    print("[reuse] Found aligned ASR → skipping transcription & alignment")
    segs_aligned = json.loads(F_ALIGN_JSON.read_text(encoding="utf-8"))
    result_aligned = {"segments": segs_aligned}

elif F_RAW_JSON.exists():
    # ✅ Reuse raw ASR and run ONLY alignment
    print("[reuse] Found raw ASR → running alignment only")
    segs_raw = json.loads(F_RAW_JSON.read_text(encoding="utf-8"))
    # If you know language, set it here; otherwise default to 'en'
    lang_code = "en"
    model_a, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
    result_aligned = whisperx.align(segs_raw, model_a, metadata, audio, device, return_char_alignments=False)
    segs_aligned = result_aligned.get("segments", []) or []
    write_json(F_ALIGN_JSON, segs_aligned)
    write_jsonl(F_ALIGN_JSONL, segs_aligned)
    print(f"  ↳ saved aligned ASR: {F_ALIGN_JSON.name}, {F_ALIGN_JSONL.name}")
    try:
        import torch
        del model_a; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

else:
    # Run full 1) Transcribe and 2) Align
    print("[1/4] Transcribing (WhisperX large-v2)…")
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=str(SCRIPT_DIR))
    result = model.transcribe(audio, batch_size=batch_size)
    segs_raw = result.get("segments", []) or []
    write_json(F_RAW_JSON, segs_raw)
    write_jsonl(F_RAW_JSONL, segs_raw)
    print(f"  ↳ saved raw ASR: {F_RAW_JSON.name}, {F_RAW_JSONL.name}")
    # free model if needed
    try:
        import torch
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    print("[2/4] Aligning words…")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_aligned = whisperx.align(
        result["segments"], model_a, metadata, audio, device, return_char_alignments=False
    )
    segs_aligned = result_aligned.get("segments", []) or []
    write_json(F_ALIGN_JSON, segs_aligned)
    write_jsonl(F_ALIGN_JSONL, segs_aligned)
    print(f"  ↳ saved aligned ASR: {F_ALIGN_JSON.name}, {F_ALIGN_JSONL.name}")
    try:
        del model_a; gc.collect()
        if 'torch' in globals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

# ------------------ 3) DIARIZATION ------------------
print("[3/4] Running diarization…")
diar_segments = None
try:
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
    # set min/max if known
    diar_segments = diarize_model(audio, min_speakers=2, max_speakers=2)

    # normalize to serializable list of dicts
    diar_list = []
    for turn, _, speaker in diar_segments.itertracks(yield_label=True):
        diar_list.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker)
        })
    write_json(F_DIAR_JSON, diar_list)
    write_jsonl(F_DIAR_JSONL, diar_list)
    print(f"  ↳ saved diarization: {F_DIAR_JSON.name}, {F_DIAR_JSONL.name} ({len(diar_list)} segments)")
except Exception as e:
    print(f"  ! diarization failed: {e}")

# ------------------ 4) SPEAKER ASSIGNMENT ------------------
print("[4/4] Assigning speakers to words…")
try:
    if diar_segments is None:
        raise RuntimeError("No diarization output available for speaker assignment.")
    result_spk = whisperx.assign_word_speakers(diar_segments, result_aligned)
    segs_spk = result_spk.get("segments", []) or []
    write_json(F_SPK_JSON, segs_spk)
    write_jsonl(F_SPK_JSONL, segs_spk)
    print(f"  ↳ saved speaker-tagged ASR: {F_SPK_JSON.name}, {F_SPK_JSONL.name}")
except Exception as e:
    print(f"  ! speaker assignment failed: {e}")
    # still dump aligned (or raw) segments so downstream can proceed
    try:
        fallback = result_aligned.get("segments", []) or []
        write_json(SCRIPT_DIR / "asr_segments_fallback.json", fallback)
        write_jsonl(SCRIPT_DIR / "asr_segments_fallback.jsonl", fallback)
        print("  ↳ saved fallback ASR: asr_segments_fallback.json/jsonl")
    except Exception as ee:
        print(f"  ! failed to save fallback ASR: {ee}")

end = time.time()
print(f"[done] Completed in {end - start:.2f}s")
