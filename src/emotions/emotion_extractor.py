# emotion_extractor.py

import json
from pathlib import Path
from typing import Any, Tuple, Optional
import numpy as np
import librosa
from tqdm import tqdm

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

# Configs
MODEL_ID = ""                # Dimensional V/A[/D] model
CATEGORICAL_MODEL_ID = ""    # Optional: categorical emotion model (for comparison)
SER_HEAD = "dim01"           # {"dim01","dimtanh"} mapping for dimensional model heads
SER_PAD_S = 3.0              # context padding (s) for SER
VTHR = 0.15                  # valence threshold ([-1,1]) for absolute category
ATHR = 0.15                  # arousal threshold (mapped to [-1,1]) for absolute category
ALLOW_FEAR = True           # split anger vs fear via dominance<0.55
MIN_SEG_S = 0.6              # skip SER on utterances shorter than this (still compute prosody)
CKPT_EVERY = 0               # write out JSON every N processed speech segments (0 = off)
Z_MIN_UTTS = 5               # min #utterances per speaker before emitting category_z
Z_WARMUP_S = 30.0            # min cumulative speech seconds per speaker before category_z
ZVTHR = 0.8                  # z-threshold for valence (stricter -> fewer false positives)
ZATHR = 0.8                  # z-threshold for arousal
Z_REQUIRE_BOTH_AXES = True   # require BOTH |zV| & |zA| >= thresholds for basic emotions

FMIN_HZ = librosa.note_to_hz("C2")
FMAX_HZ = librosa.note_to_hz("C7")

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

# Need to fine tune these
def vad_to_category(valence: Optional[float], arousal01: Optional[float],
                   dominance01: Optional[float] = None,
                   v_thr: float = VTHR, a_thr: float = ATHR, allow_fear: bool = ALLOW_FEAR) -> Optional[str]:
    if valence is None or arousal01 is None:
        return None
    a = (arousal01 - 0.5) * 2.0  # to [-1,1]
    if abs(valence) <= v_thr and abs(a) <= a_thr:
        return "neutral"
    if valence >= v_thr and a >=  a_thr: return "joy"
    if valence <= -v_thr and a >=  a_thr:
        if allow_fear and dominance01 is not None and dominance01 < 0.55:
            return "fear"
        return "anger"
    if valence <= -v_thr and a <= -a_thr: return "sadness"
    if valence >=  v_thr and a <= -a_thr: return "calm"
    return "positive" if valence > 0 else "negative"

def arousal_intensity(arousal01: Optional[float]) -> Optional[str]:
    if arousal01 is None:
        return None
    d = abs(arousal01 - 0.5)
    if d < 0.10: return "low"
    if d < 0.25: return "medium"
    return "high"

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
            # dimensional model
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
            # categorical model
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

    def infer_dimensional(self, y: np.ndarray, sr: int, ser_head: str = "dim01"):
        if self.model is None or self.processor is None or not TRANS_AVAILABLE or self.categorical:
            return None, None, None, 0.0
        with torch.no_grad():
            inputs = self.processor(y, sampling_rate=sr, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits
            if self.frame_model and hasattr(outputs, "logits"):
                logits = logits.mean(dim=1)
            logits = logits.squeeze(0)

            id2label = getattr(self.model.config, "id2label", None)
            axes = [id2label[i].lower() for i in range(len(id2label))] if id2label else None

            def idx_of(name):
                return None if axes is None or name not in axes else axes.index(name)

            ai = idx_of("arousal"); di = idx_of("dominance"); vi = idx_of("valence")

            if ser_head == "dim01":
                sig = torch.sigmoid
                a01 = float(sig(logits[ai]).cpu().item()) if ai is not None else None
                d01 = float(sig(logits[di]).cpu().item()) if di is not None else None
                v01 = float(sig(logits[vi]).cpu().item()) if vi is not None else None
                v = None if v01 is None else (2.0 * v01 - 1.0)
                return v, a01, d01, 1.0

            if ser_head == "dimtanh":
                tnh = torch.tanh
                a  = float(tnh(logits[ai]).cpu().item()) if ai is not None else None
                d  = float(tnh(logits[di]).cpu().item()) if di is not None else None
                v  = float(tnh(logits[vi]).cpu().item()) if vi is not None else None
                a01 = None if a is None else 0.5 * (a + 1.0)
                d01 = None if d is None else 0.5 * (d + 1.0)
                return v, a01, d01, 1.0

            sig = torch.sigmoid
            a01 = float(sig(logits[0]).cpu().item()) if logits.shape[-1] > 0 else None
            d01 = float(sig(logits[1]).cpu().item()) if logits.shape[-1] > 1 else None
            v01 = float(sig(logits[2]).cpu().item()) if logits.shape[-1] > 2 else None
            v = None if v01 is None else (2.0 * v01 - 1.0)
            return v, a01, d01, 1.0

    def infer_label(self, y: np.ndarray, sr: int):
        if self.model is None or self.processor is None or not TRANS_AVAILABLE or not self.categorical:
            return None, 0.0
        with torch.no_grad():
            inputs = self.processor(y, sampling_rate=sr, return_tensors="pt")
            logits = self.model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)
            idx = int(torch.argmax(prob).cpu().item())
            label = self.id2label[idx].lower() if self.id2label else "neutral"
            conf = float(prob[0, idx].cpu().item())
            return label, conf

def collect_speaker_stats(segments):
    stats = {}
    for s in segments:
        if s.get("type") != "speech":
            continue
        spk = s.get("speaker")
        emo = s.get("emotion", {})
        v, a = emo.get("valence"), emo.get("arousal")
        conf = float(emo.get("confidence", 0.0))
        dur = float(s["end"]) - float(s["start"])

        d = stats.setdefault(spk, {"v": [], "a": [], "dur": 0.0, "n_utts": 0})
        d["dur"] += max(dur, 0.0)
        d["n_utts"] += 1

        if v is None or a is None or conf < 0.2:
            continue
        d["v"].append(float(v))
        d["a"].append(float(a))

    for spk, d in stats.items():
        for k in ["v", "a"]:
            arr = np.array(d[k], dtype=float)
            if arr.size:
                m = float(arr.mean())
                sd = float(arr.std(ddof=1)) if arr.size > 1 else 1.0
            else:
                m = 0.0 if k == "v" else 0.5
                sd = 1.0
            d[k + "_mu"] = m
            d[k + "_sd"] = sd if sd > 1e-6 else 1.0
    return stats

def add_z_categories(segments, spkstats, allow_fear=False):
    for s in segments:
        if s.get("type") != "speech":
            continue
        emo = s.get("emotion", {})
        v, a = emo.get("valence"), emo.get("arousal")
        if v is None or a is None:
            emo["category_z"] = None
            continue

        spk = s.get("speaker")
        st = spkstats.get(spk, {})
        if st.get("n_utts", 0) < Z_MIN_UTTS or st.get("dur", 0.0) < Z_WARMUP_S:
            emo["category_z"] = None
            continue

        mu_v, sd_v = float(st.get("v_mu", 0.0)), max(float(st.get("v_sd", 1.0)), 1e-6)
        mu_a, sd_a = float(st.get("a_mu", 0.5)), max(float(st.get("a_sd", 1.0)), 1e-6)
        zv = (float(v) - mu_v) / sd_v
        za = (float(a) - mu_a) / sd_a
        if abs(zv) >= ZVTHR and abs(za) >= ZATHR:
            if zv >= 0 and za >= 0:
                cat = "joy"
            elif zv <= 0 and za >= 0:
                cat = "anger" if (not allow_fear or (emo.get("dominance", 0.5) >= 0.55)) else "fear"
            elif zv <= 0 and za <= 0:
                cat = "sadness"
            else:
                cat = "calm"
        else:
            cat = "neutral"

        emo["category_z"] = cat
        emo["z_valence"] = float(zv)
        emo["z_arousal"] = float(za)

def process_segments(audio_path: str, asr_json_path: str, out_json_path: str):
    segments = json.loads(Path(asr_json_path).read_text())

    y, sr = load_audio(audio_path, sr=16000)
    speech_segments = [s for s in segments if s.get("type") == "speech"]
    ser_dim = SERWrapper().load(MODEL_ID, "")
    ser_cat = SERWrapper().load("", CATEGORICAL_MODEL_ID) if CATEGORICAL_MODEL_ID else None

    print(f"\nProcessing {len(speech_segments)} speech segments...\n")
    processed_count = 0

    for s in tqdm(segments, desc="Emotion extraction", unit="seg"):
        if s.get("type") != "speech":
            continue
        start, end = float(s["start"]), float(s["end"])
        text = s.get("text", "") or ""
        dur = max(1e-6, end - start)

        y_u = crop(y, sr, start, end)
        energy = compute_rms_mean(y_u)
        pmean, pvar = compute_pitch_stats(y_u, sr)
        wps = speech_rate_wps(text, dur)

        if dur < MIN_SEG_S:
            v, a01, d01, conf_dim = None, None, None, 0.0
        else:
            y_pad = crop(y, sr, start - SER_PAD_S, end + SER_PAD_S)
            v, a01, d01, conf_dim = ser_dim.infer_dimensional(y_pad, sr, ser_head=SER_HEAD)

        cat_abs = vad_to_category(v, a01, d01, v_thr=VTHR, a_thr=ATHR, allow_fear=ALLOW_FEAR)
        inten = arousal_intensity(a01)

        # Optional categorical model (label + confidence)
        cat_label = None
        cat_prob = 0.0
        if ser_cat is not None and dur >= MIN_SEG_S:
            y_pad = crop(y, sr, start - SER_PAD_S, end + SER_PAD_S)
            cat_label, cat_prob = ser_cat.infer_label(y_pad, sr)

        s["emotion"] = {
            "valence": v,
            "arousal": a01,
            "dominance": d01,
            "category": cat_abs,                   # from dimensional thresholds
            "category_categorical": cat_label,     # from categorical model (if provided)
            "category_confidence": float(cat_prob),
            "intensity": inten,
            "prosody": {
                "pitch_mean": float(pmean) if np.isfinite(pmean) else None,
                "pitch_var": float(pvar) if np.isfinite(pvar) else None,
                "energy_mean": float(energy),
                "speech_rate": float(wps),
            },
            "confidence": float(conf_dim),         # dimensional model "confidence" (1.0 for regressors)
        }

        processed_count += 1
        if CKPT_EVERY and (processed_count % CKPT_EVERY == 0):
            Path(out_json_path).write_text(json.dumps(segments, indent=2))

    # Speaker-relative z categories (hard-coded warmup & strict thresholds)
    spkstats = collect_speaker_stats(segments)
    add_z_categories(segments, spkstats, allow_fear=ALLOW_FEAR)

    Path(out_json_path).write_text(json.dumps(segments, indent=2))
    print(f"\nWrote file to {out_json_path}")
    return out_json_path

def main():
    import argparse
    p = argparse.ArgumentParser(description="Attach emotion/prosody to ASR segments JSON.")
    p.add_argument("--audio", required=True, help="Path to mono WAV/FLAC audio")
    p.add_argument("--asr_json", required=True, help="Path to ASR segments JSON (list of dicts)")
    p.add_argument("--out_json", required=True, help="Output path for enriched JSON")

    # models + heads
    p.add_argument("--ser_model", default="", help="Dimensional SER model id (val/arou/dom regression)")
    p.add_argument("--ser_model_categorical", default="", help="Categorical SER model id (for comparison label)")
    p.add_argument("--ser_head", choices=["dim01","dimtanh"], default="dim01",
                   help="Head mapping for dimensional SER: dim01 (0..1 targets), dimtanh (−1..1 targets).")

    # extraction knobs
    p.add_argument("--pad", type=float, default=3.0, help="SER padding (seconds)")
    p.add_argument("--vthr", type=float, default=0.15, help="Valence threshold ([-1,1]) for category")
    p.add_argument("--athr", type=float, default=0.15, help="Arousal threshold mapped to [-1,1] for category")
    p.add_argument("--allow_fear", action="store_true", help="Split anger vs fear using dominance")
    p.add_argument("--min_seg_s", type=float, default=0.6, help="Min duration (s) to run SER (shorter -> no SER)")

    # checkpoints
    p.add_argument("--ckpt_every", type=int, default=0, help="Write interim JSON every N speech segments (0=off)")

    args = p.parse_args()

    global MODEL_ID, CATEGORICAL_MODEL_ID, SER_HEAD, SER_PAD_S
    global VTHR, ATHR, ALLOW_FEAR, MIN_SEG_S, CKPT_EVERY

    MODEL_ID = args.ser_model or ""
    CATEGORICAL_MODEL_ID = args.ser_model_categorical or ""
    SER_HEAD = args.ser_head
    SER_PAD_S = float(args.pad)
    VTHR = float(args.vthr)
    ATHR = float(args.athr)
    ALLOW_FEAR = bool(args.allow_fear)
    MIN_SEG_S = float(args.min_seg_s)
    CKPT_EVERY = int(args.ckpt_every)

    outp = process_segments(args.audio, args.asr_json, args.out_json)
    print(f"Wrote: {outp}")

if __name__ == "__main__":
    main()
