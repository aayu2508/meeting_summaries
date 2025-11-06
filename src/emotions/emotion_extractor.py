# emotion_extractor.py
# Attach per-utterance emotion/prosody to your ASR segments JSON.
# Usage:
#   pip install librosa soundfile torch transformers tqdm
#   python emotion_extractor.py --audio meeting.wav --asr_json asr_segments.json --out_json asr_with_emotion.json \
#          --ser_model "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim" --vthr 0.15 --athr 0.15 --min_seg_s 0.6
#
# Notes:
# - Dimensional SER (recommended): use --ser_model (outputs continuous valence/arousal[/dominance]).
# - Categorical SER (fallback): use --ser_model_categorical (labels mapped to V/A/D internally).
# - If neither is given, only prosody/speech_rate are computed; V/A/D stay None.

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import librosa

# Optional: transformers/torch are loaded lazily
TRANS_AVAILABLE = True
try:
    import torch
    from transformers import (
        AutoProcessor,
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
        AutoModelForAudioFrameClassification,
    )
except Exception:
    TRANS_AVAILABLE = False

from tqdm import tqdm

# ------------------------ CONFIG (defaults; override via CLI) ------------------------
MODEL_ID = ""                # Dimensional V/A[/D] model, leave "" to disable
CATEGORICAL_MODEL_ID = ""    # Categorical emotion model, leave "" if unused
SER_PAD_S = 3.0              # seconds of context padding for SER
VTHR = 0.15                  # valence threshold ([-1,1]) for category
ATHR = 0.15                  # arousal threshold (after mapping to [-1,1]) for category
ALLOW_FEAR = False           # if True, split anger vs fear using dominance
MIN_SEG_S = 0.6              # skip SER on utterances shorter than this (keep prosody)
# ------------------------------------------------------------------------------------

FMIN_HZ = librosa.note_to_hz("C2")
FMAX_HZ = librosa.note_to_hz("C7")

# Mapping from categorical labels -> approximate (valence, arousal, dominance) in [-1,1]
CAT2VAD = {
    "anger":   (-0.7,  0.8,  0.6),
    "angry":   (-0.7,  0.8,  0.6),
    "joy":     ( 0.7,  0.7,  0.6),
    "happy":   ( 0.7,  0.7,  0.6),
    "surprise":( 0.2,  0.8,  0.4),
    "fear":    (-0.6,  0.9, -0.6),
    "sad":     (-0.7, -0.6, -0.6),
    "sadness": (-0.7, -0.6, -0.6),
    "disgust": (-0.6,  0.5,  0.2),
    "neutral": ( 0.0,  0.0,  0.0),
    "calm":    ( 0.4, -0.4,  0.4),
}

# ------------------------ helpers ------------------------
def load_audio(audio_path: str, sr: int = 16000):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    return y, sr

def crop(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    n = len(y)
    def s2i(t):
        t_clamped = float(np.clip(t, 0.0, n / sr))
        return int(t_clamped * sr)
    return y[s2i(start_s):s2i(end_s)]

def compute_rms_mean(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
    return float(np.mean(rms))

def compute_pitch_stats(y: np.ndarray, sr: int, fmin=FMIN_HZ, fmax=FMAX_HZ) -> Tuple[float, float]:
    """Return (pitch_mean_Hz, pitch_var_Hz2) or (nan, nan) if unvoiced/too-short."""
    if y.size == 0:
        return float("nan"), float("nan")
    dur = len(y) / max(sr, 1)
    if dur < 0.25:
        return float("nan"), float("nan")
    try:
        f0, _, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, frame_length=2048, hop_length=512)
        if f0 is None:
            return float("nan"), float("nan")
        f0v = f0[~np.isnan(f0)]
        if f0v.size == 0:
            return float("nan"), float("nan")
        # (Optional) median filter to reduce octave errors if scipy is available
        try:
            from scipy.ndimage import median_filter
            if f0v.size >= 5:
                f0v = median_filter(f0v, size=5)
        except Exception:
            pass
        return float(np.mean(f0v)), float(np.var(f0v))
    except Exception:
        return float("nan"), float("nan")

def count_words(text: str) -> int:
    if not text:
        return 0
    return len([t for t in text.strip().split() if t])

def speech_rate_wps(text: str, dur_s: float) -> float:
    dur_s = max(dur_s, 1e-6)
    return float(count_words(text) / dur_s)

def normalize_01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float((x - lo) / (hi - lo))

def va_to_category(valence: Optional[float], arousal01: Optional[float],
                   dominance01: Optional[float] = None,
                   v_thr: float = VTHR, a_thr: float = ATHR, allow_fear: bool = ALLOW_FEAR) -> Optional[str]:
    """
    Map continuous V ([-1,1]) + A01 ([0,1]) to discrete categories.
    Quadrants with dead-zones for 'neutral'. Optionally split anger/fear via dominance.
    """
    if valence is None or arousal01 is None:
        return None
    a = (arousal01 - 0.5) * 2.0  # map arousal to [-1,1]

    # neutral box around origin to reduce noise
    if abs(valence) <= v_thr and abs(a) <= a_thr:
        return "neutral"

    # main quadrants
    if valence >= v_thr and a >=  a_thr: return "joy"
    if valence <= -v_thr and a >=  a_thr:
        if allow_fear and dominance01 is not None and dominance01 < 0.55:
            return "fear"
        return "anger"
    if valence <= -v_thr and a <= -a_thr: return "sadness"
    if valence >=  v_thr and a <= -a_thr: return "calm"

    # fallback by stronger axis (rare)
    return "positive" if valence > 0 else "negative"

def arousal_intensity(arousal01: Optional[float]) -> Optional[str]:
    if arousal01 is None:
        return None
    d = abs(arousal01 - 0.5)  # distance from neutral arousal
    if d < 0.10: return "low"
    if d < 0.25: return "medium"
    return "high"

# ------------------------ SER wrapper ------------------------
class SERWrapper:
    def __init__(self):
        self.model = None
        self.processor = None
        self.frame_model = False
        self.categorical = False
        self.id2label = None

    def load(self, model_id: str = "", categorical_model_id: str = ""):
        if not TRANS_AVAILABLE:
            return self
        if model_id:
            # Try frame-wise regression; fallback to clip-wise
            try:
                self.model = AutoModelForAudioFrameClassification.from_pretrained(model_id)
                self.frame_model = True
            except Exception:
                self.model = AutoModelForAudioClassification.from_pretrained(model_id)
                self.frame_model = False
            try:
                self.processor = AutoProcessor.from_pretrained(model_id)
            except Exception:
                self.processor = AutoFeatureExtractor.from_pretrained(model_id)
            self.id2label = getattr(self.model.config, "id2label", None)
            self.categorical = False
            self.model.eval()
        elif categorical_model_id:
            self.model = AutoModelForAudioClassification.from_pretrained(categorical_model_id)
            try:
                self.processor = AutoProcessor.from_pretrained(categorical_model_id)
            except Exception:
                self.processor = AutoFeatureExtractor.from_pretrained(categorical_model_id)
            self.id2label = getattr(self.model.config, "id2label", None)
            self.categorical = True
            self.frame_model = False
            self.model.eval()
        return self

    def infer(self, y: np.ndarray, sr: int):
        """
        Returns: (valence[-1..1] or None, arousal[0..1] or None, dominance[0..1] or None, conf[0..1])
        """
        if self.model is None or self.processor is None or not TRANS_AVAILABLE:
            return None, None, None, 0.0

        with torch.no_grad():
            inputs = self.processor(y, sampling_rate=sr, return_tensors="pt")
            outputs = self.model(**inputs)

            # frame-wise -> mean-pool
            if self.frame_model and hasattr(outputs, "logits"):
                logits = outputs.logits.mean(dim=1)  # [B, D]
            else:
                logits = outputs.logits             # [B, D]
            logits = logits.squeeze(0)

            if not self.categorical:
                # Dimensional regression: try labeled axes; else fall back to first three
                id2label = getattr(self.model.config, "id2label", None)
                v = a = d = None
                if id2label:
                    labels = [id2label[i].lower() for i in range(len(id2label))]
                    vi = labels.index("valence") if "valence" in labels else None
                    ai = labels.index("arousal") if "arousal" in labels else None
                    di = labels.index("dominance") if "dominance" in labels else None
                    if vi is not None: v = float(torch.tanh(logits[vi]).cpu().item())
                    if ai is not None: a = float(torch.tanh(logits[ai]).cpu().item())
                    if di is not None: d = float(torch.tanh(logits[di]).cpu().item())
                if v is None and logits.shape[-1] >= 3:
                    v = float(torch.tanh(logits[0]).cpu().item())
                    a = float(torch.tanh(logits[1]).cpu().item())
                    d = float(torch.tanh(logits[2]).cpu().item())
                a01 = None if a is None else 0.5 * (a + 1.0)
                d01 = None if d is None else 0.5 * (d + 1.0)
                conf = 1.0
                return v, a01, d01, conf
            else:
                # Categorical: softmax -> label -> VAD lookup
                prob = torch.softmax(logits, dim=-1)
                idx = int(torch.argmax(prob).cpu().item())
                label = self.id2label[idx].lower() if self.id2label else "neutral"
                v, a, d = CAT2VAD.get(label, CAT2VAD["neutral"])
                a01 = 0.5 * (a + 1.0)
                d01 = 0.5 * (d + 1.0)
                conf = float(prob[idx].cpu().item())
                return v, a01, d01, conf

# ------------------------ session helpers ------------------------
def build_session_energy_stats(y: np.ndarray, sr: int, speech_segments: List[Dict[str, Any]]):
    vals = []
    for s in speech_segments:
        y_u = crop(y, sr, float(s["start"]), float(s["end"]))
        vals.append(compute_rms_mean(y_u))
    if not vals:
        return 0.0, 1.0
    return float(np.min(vals)), float(np.max(vals))

# ------------------------ main processing ------------------------
def process_segments(audio_path: str, asr_json_path: str, out_json_path: str):
    segments = json.loads(Path(asr_json_path).read_text())
    y, sr = load_audio(audio_path, sr=16000)

    speech_segments = [s for s in segments if s.get("type") == "speech"]
    session_min, session_max = build_session_energy_stats(y, sr, speech_segments)

    ser = SERWrapper().load(MODEL_ID, CATEGORICAL_MODEL_ID)

    print(f"\nProcessing {len(speech_segments)} speech segments...\n")

    for s in tqdm(segments, desc="Emotion extraction", unit="seg"):
        if s.get("type") != "speech":
            continue
        start, end = float(s["start"]), float(s["end"])
        text = s.get("text", "") or ""
        dur = max(1e-6, end - start)

        # Prosody (always)
        y_u = crop(y, sr, start, end)
        energy = compute_rms_mean(y_u)
        energy01 = normalize_01(energy, session_min, session_max)
        pmean, pvar = compute_pitch_stats(y_u, sr)
        wps = speech_rate_wps(text, dur)

        # SER (skip micro-segments)
        if dur < MIN_SEG_S:
            v, a01, d01, conf = None, None, None, 0.0
        else:
            y_pad = crop(y, sr, start - SER_PAD_S, end + SER_PAD_S)
            v, a01, d01, conf = ser.infer(y_pad, sr)

        # Category + intensity
        cat = va_to_category(v, a01, d01, v_thr=VTHR, a_thr=ATHR, allow_fear=ALLOW_FEAR)
        inten = arousal_intensity(a01)

        s["emotion"] = {
            "valence": v,
            "arousal": a01,
            "dominance": d01,
            "category": cat,
            "intensity": inten,  # optional but useful for UI/aggregation
            "prosody": {
                "pitch_mean": float(pmean) if np.isfinite(pmean) else None,
                "pitch_var": float(pvar) if np.isfinite(pvar) else None,
                "energy_mean": float(energy01),
                "speech_rate": float(wps),
            },
            "confidence": float(conf),
        }

    Path(out_json_path).write_text(json.dumps(segments, indent=2))
    print(f"\n✅ Wrote enriched file to {out_json_path}")
    return out_json_path

def main():
    import argparse
    p = argparse.ArgumentParser(description="Attach emotion/prosody to ASR segments JSON.")
    p.add_argument("--audio", required=True, help="Path to mono WAV/FLAC audio")
    p.add_argument("--asr_json", required=True, help="Path to ASR segments JSON (list of dicts)")
    p.add_argument("--out_json", required=True, help="Output path for enriched JSON")
    p.add_argument("--ser_model", default="", help="Dimensional SER model id (val/arou/dom regression)")
    p.add_argument("--ser_model_categorical", default="", help="Categorical SER model id (labels mapped to VAD)")
    p.add_argument("--pad", type=float, default=3.0, help="SER padding (seconds)")
    p.add_argument("--vthr", type=float, default=0.15, help="Valence threshold for category ([-1,1])")
    p.add_argument("--athr", type=float, default=0.15, help="Arousal threshold after mapping to [-1,1]")
    p.add_argument("--allow_fear", action="store_true", help="Split anger vs fear using dominance")
    p.add_argument("--min_seg_s", type=float, default=0.3, help="Min duration (s) to run SER (shorter -> no SER)")
    args = p.parse_args()

    global MODEL_ID, CATEGORICAL_MODEL_ID, SER_PAD_S, VTHR, ATHR, ALLOW_FEAR, MIN_SEG_S
    if args.ser_model:
        MODEL_ID = args.ser_model
    if args.ser_model_categorical:
        CATEGORICAL_MODEL_ID = args.ser_model_categorical
    SER_PAD_S = float(args.pad)
    VTHR = float(args.vthr)
    ATHR = float(args.athr)
    ALLOW_FEAR = bool(args.allow_fear)
    MIN_SEG_S = float(args.min_seg_s)

    outp = process_segments(args.audio, args.asr_json, args.out_json)
    print(f"Wrote: {outp}")

if __name__ == "__main__":
    main()
