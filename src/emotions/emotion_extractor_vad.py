#!/usr/bin/env python3
import json
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import librosa
from tqdm import tqdm

try:
    import torch
    from transformers import (
        AutoProcessor,
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
        AutoModelForAudioFrameClassification,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class AudioProcessor:
    def __init__(self, cfg):
        self.cfg = cfg

    def load_audio(self, path: str):
        y, sr = librosa.load(path, sr=self.cfg["sample_rate"], mono=True)
        return y.astype(np.float32), sr

    def crop(self, y: np.ndarray, sr: int, start: float, end: float):
        n = len(y)
        start_idx = int(np.clip(start * sr, 0, n))
        end_idx = int(np.clip(end * sr, 0, n))
        return y[start_idx:end_idx]

    def rms(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
        return float(np.mean(rms))

class EmotionClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.processor = None
        self.is_frame_model = False

    def load(self):
        if not TRANSFORMERS_AVAILABLE or not self.cfg["ser_model"]:
            print("Transformers unavailable or no SER model provided")
            return self

        print(f"Loading emotion model: {self.cfg['ser_model']}")

        try:
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.cfg["ser_model"]
            )
        except Exception:
            self.model = AutoModelForAudioFrameClassification.from_pretrained(
                self.cfg["ser_model"]
            )
            self.is_frame_model = True

        try:
            self.processor = AutoProcessor.from_pretrained(self.cfg["ser_model"])
        except Exception:
            self.processor = AutoFeatureExtractor.from_pretrained(self.cfg["ser_model"])

        self.model.eval()
        print("Model loaded.")
        return self

    def infer(self, y: np.ndarray, sr: int):
        """
        Return dimensional emotion as:
        - v   : valence in [-1, 1]
        - a01 : arousal in [0, 1]
        - d01 : dominance in [0, 1]
        """
        if self.model is None:
            return {"v": None, "a01": None, "d01": None}

        with torch.no_grad():
            inputs = self.processor(
                y, sampling_rate=sr, return_tensors="pt", padding=True
            )
            outputs = self.model(**inputs)
            logits = outputs.logits

            if self.is_frame_model:
                logits = logits.mean(dim=1)

            logits = logits.squeeze(0)

            id2label = getattr(self.model.config, "id2label", None)
            if id2label:
                axes = {id2label[i].lower(): i for i in range(len(id2label))}
                ai = axes.get("arousal")
                di = axes.get("dominance")
                vi = axes.get("valence")
            else:
                ai, di, vi = 0, 1, 2

            sig = torch.sigmoid

            a01 = float(sig(logits[ai])) if ai is not None else None
            d01 = float(sig(logits[di])) if di is not None else None
            if vi is not None:
                v01 = float(sig(logits[vi]))
                v = 2 * v01 - 1.0
            else:
                v = None

            return {"v": v, "a01": a01, "d01": d01}

class SpeakerStatistics:
    def __init__(self, cfg):
        self.cfg = cfg
        self.stats: Dict[str, Dict[str, float]] = {}

    def compute_statistics(self, segments: List[Dict]):
        per_spk: Dict[str, Dict[str, List[float]]] = {}

        for seg in segments:
            if seg.get("type") != "speech":
                continue

            emo = seg.get("emotion", {})
            v = emo.get("v")
            a01 = emo.get("a01")
            d01 = emo.get("d01")
            if v is None or a01 is None or d01 is None:
                continue

            if seg.get("overlap", False):
                continue

            dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
            if dur < self.cfg["min_seg_s"]:
                continue

            speaker = seg.get("speaker")
            if speaker is None:
                continue

            if speaker not in per_spk:
                per_spk[speaker] = {"v": [], "a": [], "d": []}

            per_spk[speaker]["v"].append(v)
            per_spk[speaker]["a"].append(2 * a01 - 1.0)
            per_spk[speaker]["d"].append(2 * d01 - 1.0)

        for spk, vals in per_spk.items():
            stats = {}
            for axis in ["v", "a", "d"]:
                arr = np.array(vals[axis], dtype=float)
                if arr.size == 0:
                    median = 0.0
                    sd = 1.0
                else:
                    median = float(np.median(arr))
                    mad = float(np.median(np.abs(arr - median)))
                    sd = 1.4826 * mad  # MAD -> approx std
                    if not np.isfinite(sd) or sd < 1e-6:
                        # Fallback to plain std, then to 1.0
                        sd = float(arr.std(ddof=1)) if arr.size > 1 else 1.0
                        if not np.isfinite(sd) or sd < 1e-6:
                            sd = 1.0
                stats[f"{axis}_mu"] = median
                stats[f"{axis}_sd"] = sd
            stats["n"] = len(vals["v"])
            self.stats[spk] = stats

    def compute_z(self, spk: str, v: float, a01: float, d01: float):
        st = self.stats.get(spk)
        if not st:
            return None, None, None

        a = 2 * a01 - 1.0
        d = 2 * d01 - 1.0

        def z(x, mu, sd):
            return (x - mu) / max(sd, 1e-6)

        v_z = z(v, st["v_mu"], st["v_sd"])
        a_z = z(a, st["a_mu"], st["a_sd"])
        d_z = z(d, st["d_mu"], st["d_sd"])
        return v_z, a_z, d_z


class EmotionExtractor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.audio = AudioProcessor(cfg)
        self.cls = EmotionClassifier(cfg)
        self.stats = SpeakerStatistics(cfg)

    def process(self, audio_path, asr_json, out_json):
        print(f"\nLoading segments from: {asr_json}")
        segments = json.loads(Path(asr_json).read_text())

        print(f"\nLoading audio from: {audio_path}")
        audio, sr = self.audio.load_audio(audio_path)
        print(f"   Duration: {len(audio) / sr:.2f}s @ {sr}Hz")

        self.cls.load()

        self._extract_first_pass(segments, audio, sr)
        print("\nComputing per-speaker emotion baselines...")
        self.stats.compute_statistics(segments)
        self._add_z_scores(segments)

        self._write(out_json, segments)
        print(f"\nDone. Wrote: {out_json}")

    def _extract_first_pass(self, segments, audio, sr):
        speech_segments = [s for s in segments if s.get("type") == "speech"]
        print(f"\nProcessing {len(speech_segments)} speech segments\n")

        processed = 0
        for seg in tqdm(segments, desc="Emotions", unit="seg"):
            if seg.get("type") != "speech":
                continue

            start, end = float(seg["start"]), float(seg["end"])
            dur = max(1e-6, end - start)
            text = seg.get("text", "") or ""

            y = self.audio.crop(audio, sr, start, end)

            # Prosody
            log_rms = float(np.log1p(self.audio.rms(y)))
            wps = len(text.split()) / dur if dur > 0 else 0.0
            wps = min(wps, 6.0)

            # Emotion
            if dur < self.cfg["min_seg_s"]:
                emo = {"v": None, "a01": None, "d01": None}
            else:
                pad_start = max(0.0, start - self.cfg["pad_s"])
                pad_end = end + self.cfg["pad_s"]
                y_pad = self.audio.crop(audio, sr, pad_start, pad_end)
                emo = self.cls.infer(y_pad, sr)

            seg["emotion"] = {
                "v": emo["v"],
                "a01": emo["a01"],
                "d01": emo["d01"],
                "v_z": None,
                "a_z": None,
                "d_z": None,
            }
            seg["prosody"] = {"log_rms": log_rms, "wps": wps}

            processed += 1
            if self.cfg.get("checkpoint_every") and processed % self.cfg["checkpoint_every"] == 0:
                self._write(self.cfg["checkpoint_path"], segments)

    def _add_z_scores(self, segments):
        for seg in segments:
            if seg.get("type") != "speech":
                continue

            emo = seg.get("emotion", {})
            v, a01, d01 = emo.get("v"), emo.get("a01"), emo.get("d01")
            spk = seg.get("speaker")

            if spk is None or v is None or a01 is None or d01 is None:
                continue

            v_z, a_z, d_z = self.stats.compute_z(spk, v, a01, d01)
            emo["v_z"] = v_z
            emo["a_z"] = a_z
            emo["d_z"] = d_z

    def _write(self, path, data):
        Path(path).write_text(json.dumps(self._compact(data), indent=2))

    def _compact(self, obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                v2 = self._compact(v)
                if self.cfg["drop_nulls"] and v2 in (None, {}, []):
                    continue
                out[k] = v2
            return out
        elif isinstance(obj, list):
            return [self._compact(x) for x in obj]
        elif isinstance(obj, float):
            if not math.isfinite(obj):
                return None
            return round(obj, self.cfg["round_digits"])
        return obj

def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Extract dimensional emotions and simple prosodic features from speech"
    )
    p.add_argument("--audio", required=True, help="Path to audio file")
    p.add_argument("--asr_json", required=True, help="Path to ASR segments JSON")
    p.add_argument("--out_json", required=True, help="Output JSON path")
    p.add_argument("--ser_model", required=True, help="Dimensional SER model ID")
    p.add_argument("--pad", type=float, default=0.3, help="Padding in seconds")
    p.add_argument(
        "--min_seg_s",
        type=float,
        default=0.6,
        help="Minimum segment duration for emotion inference",
    )

    args = p.parse_args()

    cfg = {
        "ser_model": args.ser_model,
        "pad_s": args.pad,
        "min_seg_s": args.min_seg_s,
        "sample_rate": 16000,
        "drop_nulls": True,
        "round_digits": 3,
        "checkpoint_every": 0,
        "checkpoint_path": args.out_json,
    }

    EmotionExtractor(cfg).process(args.audio, args.asr_json, args.out_json)

if __name__ == "__main__":
    main()