#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

EMO_TAG_MAP = {
    "<|ANGRY|>": "angry",
    "<|HAPPY|>": "happy",
    "<|SAD|>": "sad",
    "<|NEUTRAL|>": "neutral",
    "<|SURPRISE|>": "surprise",
    "<|FEAR|>": "fear",
    "<|DISGUST|>": "disgust",
    "<|OTHER|>": "other",
}

def norm_emotion(tag: Optional[str]) -> str:
    if not tag:
        return "unknown"
    return EMO_TAG_MAP.get(tag, tag.strip("<|>").lower())

def maybe_conf(d: Dict[str, Any]) -> Optional[float]:
    for k in ("emo_confidence", "emo_score", "emo_prob"):
        if k in d:
            try:
                return float(d[k])
            except Exception:
                pass
    return None

def slice_audio(wav_path: str, start: float, end: float) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    i0 = max(0, int(start * sr))
    i1 = min(len(y), int(end * sr))
    return y[i0:i1], sr

def write_tmp_wav(samples: np.ndarray, sr: int, tmp_dir: Path, stem: str) -> str:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{stem}.wav"
    sf.write(str(path), samples, sr)
    return str(path)

def audio_duration_sec(wav_path: str) -> float:
    y, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    return len(y) / float(sr)

def clamp(a: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, a))

def rms(x: np.ndarray) -> float:
    if x.size == 0: return 0.0
    return float(np.sqrt(np.mean(np.square(x))))

def gain_normalize(x: np.ndarray, target_rms: float = 0.05, floor: float = 1e-6) -> np.ndarray:
    cur = rms(x)
    if cur < floor:
        return x
    g = target_rms / cur
    return np.clip(x * g, -1.0, 1.0)

def find_key_recursive(obj, key):
    if isinstance(obj, dict):
        if key in obj: return obj[key]
        for v in obj.values():
            r = find_key_recursive(v, key)
            if r is not None: return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key_recursive(v, key)
            if r is not None: return r
    return None

def load_diarized_speech(path: str, min_dur_s: float = 0.8) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    out = []
    for seg in data:
        if seg.get("type") and seg["type"] != "speech":
            continue
        start = float(seg["start"]); end = float(seg["end"])
        if end - start < min_dur_s:
            continue
        out.append({
            "type": "speech",
            "segment_id": seg.get("segment_id"),
            "start": start,
            "end": end,
            "speaker": seg.get("speaker"),
            "text": seg.get("text"),
        })
    return out

class SenseVoiceSER:
    def __init__(
        self,
        model_dir: str = "iic/SenseVoiceSmall",
        device: str = "cuda:0",
        batch_size_s: int = 60,
        language: str = "auto",
        keep_tmp: bool = True,
    ):
        # No VAD here at all (segments-only)
        self.model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            vad_model=None,
            device=device,
        )
        self.batch_size_s = batch_size_s
        self.language = language
        self.keep_tmp = keep_tmp

    def _run_generate(self, inp: str) -> Dict[str, Any]:
        res = self.model.generate(
            input=inp,
            cache={},
            language=self.language,
            use_itn=True,            # harmless; we keep user's text; this is for debugging if needed
            batch_size_s=self.batch_size_s,
            merge_vad=False,         # ensure no internal merging
        )
        return res[0] if isinstance(res, list) and res else res

    def run_with_segments(
        self,
        audio_path: str,
        segments: List[Dict[str, Any]],
        tmp_dir: str = ".sv_tmp",
        pad_s: float = 0.35,
        min_rms: float = 0.01,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        tmp = Path(tmp_dir)
        rows: List[Dict[str, Any]] = []
        created_paths: List[str] = []
        total = audio_duration_sec(audio_path)

        for i, seg in enumerate(segments):
            start, end = float(seg["start"]), float(seg["end"])
            if end <= start:
                continue

            # pad for prosody, clamp to file bounds
            s = clamp(start - pad_s, 0.0, total)
            e = clamp(end + pad_s, 0.0, total)

            samples, sr = slice_audio(audio_path, s, e)
            if len(samples) == 0:
                continue

            # boost very quiet clips
            if rms(samples) < min_rms:
                samples = gain_normalize(samples, target_rms=0.05)

            clip_path = write_tmp_wav(samples, sr, tmp, f"seg_{i:05d}")
            created_paths.append(clip_path)

            r = self._run_generate(clip_path)

            # robust emotion extraction (handle nested structures)
            emo_tag = r.get("emo_target")
            if emo_tag is None:
                emo_tag = find_key_recursive(r, "emo_target")

            row = {
                "type": "speech",
                "segment_id": seg.get("segment_id"),
                "start": start,         # keep original bounds
                "end": end,
                "speaker": seg.get("speaker"),
                "text": seg.get("text"),
                "emotion": norm_emotion(emo_tag),
                "emotion_tag": emo_tag,
                "emotion_confidence": maybe_conf(r),
            }

            if debug:
                row["_debug_keys"] = list(r.keys()) if isinstance(r, dict) else str(type(r))
            rows.append(row)

        if not self.keep_tmp:
            for p in created_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
            try:
                tmp.rmdir()
            except Exception:
                pass

        return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="Path to meeting audio (wav/mp3/flac)")
    ap.add_argument("--segments", required=True, help="Path to diarized/transcribed JSON with windows")
    ap.add_argument("--out", required=True, help="Where to write augmented JSON")
    ap.add_argument("--device", default="cuda:0", help='"cuda:0" or "cpu"')
    ap.add_argument("--model", default="iic/SenseVoiceSmall")
    ap.add_argument("--batch_size_s", type=int, default=60)
    ap.add_argument("--language", default="auto")
    ap.add_argument("--min_dur_s", type=float, default=0.8, help="Drop segments shorter than this (seconds)")
    ap.add_argument("--pad_s", type=float, default=0.35, help="Pad each window on both sides (seconds)")
    ap.add_argument("--min_rms", type=float, default=0.01, help="Gain-normalize if RMS below this")
    ap.add_argument("--cleanup_tmp", action="store_true", help="Delete temp clips")
    ap.add_argument("--debug", action="store_true", help="Include model return keys per segment")
    args = ap.parse_args()

    segs = load_diarized_speech(args.segments, min_dur_s=args.min_dur_s)

    sv = SenseVoiceSER(
        model_dir=args.model,
        device=args.device,
        batch_size_s=args.batch_size_s,
        language=args.language,
        keep_tmp=(not args.cleanup_tmp),
    )

    rows = sv.run_with_segments(
        audio_path=args.audio,
        segments=segs,
        tmp_dir=".sv_tmp",
        pad_s=args.pad_s,
        min_rms=args.min_rms,
        debug=args.debug,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"[SER] wrote {args.out} with {len(rows)} speech segments.")

if __name__ == "__main__":
    main()
