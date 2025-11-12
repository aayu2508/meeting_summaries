#!/usr/bin/env python3
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import librosa
from tqdm import tqdm

# Optional dependencies
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

@dataclass
class EmotionConfig:
    model_id: str = ""
    ser_head: str = "dim01"  # {"dim01", "dimtanh"}
    pad_seconds: float = 0.0
    min_segment_seconds: float = 0.6
    
    # Category thresholds
    valence_threshold: float = 0.15
    arousal_threshold: float = 0.15
    dominance_fear_cutoff: float = 0.55
    allow_fear: bool = True
    
    # Z-scoring thresholds
    z_valence_threshold: float = 0.8
    z_arousal_threshold: float = 0.8
    z_require_both_axes: bool = True
    
    # Output settings
    checkpoint_every: int = 0
    round_digits: int = 3
    drop_nulls: bool = True
    
    # Audio settings
    sample_rate: int = 16000
    pitch_fmin: float = librosa.note_to_hz("C2")
    pitch_fmax: float = librosa.note_to_hz("C7")

@dataclass
class DimensionalEmotion:
    valence: Optional[float] = None  # [-1, 1]
    arousal: Optional[float] = None  # [0, 1]
    dominance: Optional[float] = None  # [0, 1]
    confidence: float = 0.0
    category_absolute: Optional[str] = None
    category_relative: Optional[str] = None
    z_valence: Optional[float] = None
    z_arousal: Optional[float] = None

@dataclass
class ProsodicFeatures:
    pitch_mean: Optional[float] = None
    pitch_variance: Optional[float] = None
    log_rms: float = 0.0
    words_per_second: float = 0.0

class AudioProcessor:    
    def __init__(self, config: EmotionConfig):
        self.config = config
    
    def load_audio(self, path: str) -> Tuple[np.ndarray, int]:
        y, sr = librosa.load(path, sr=self.config.sample_rate, mono=True)
        return y.astype(np.float32), sr
    
    def crop_segment(self, y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
        n = len(y)
        start_idx = int(np.clip(start * sr, 0, n))
        end_idx = int(np.clip(end * sr, 0, n))
        return y[start_idx:end_idx]
    
    def compute_rms(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
        return float(np.mean(rms))
    
    def compute_pitch(self, y: np.ndarray, sr: int) -> Tuple[float, float]:
        if y.size == 0 or len(y) / sr < 0.25:
            return float('nan'), float('nan')
        
        f0, _, _ = librosa.pyin(
            y, 
            fmin=self.config.pitch_fmin, 
            fmax=self.config.pitch_fmax,
            frame_length=2048, 
            hop_length=512
        )
        
        if f0 is None:
            return float('nan'), float('nan')
        
        f0_valid = f0[~np.isnan(f0)]
        if f0_valid.size == 0:
            return float('nan'), float('nan')
        
        # Apply median filtering if scipy available
        try:
            from scipy.ndimage import median_filter
            if f0_valid.size >= 5:
                f0_valid = median_filter(f0_valid, size=5)
        except ImportError:
            pass
        
        return float(np.mean(f0_valid)), float(np.var(f0_valid))

class EmotionClassifier:    
    def __init__(self, config: EmotionConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.is_frame_model = False
    
    def load(self):
        """Load model and processor"""
        if not TRANSFORMERS_AVAILABLE or not self.config.model_id:
            print("⚠️  Transformers not available or no model specified")
            return self
        
        print(f"Loading emotion model: {self.config.model_id}")
        
        # Try sequence classification first (preferred)
        try:
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.config.model_id
            )
            self.is_frame_model = False
        except Exception:
            # Fallback to frame classification
            self.model = AutoModelForAudioFrameClassification.from_pretrained(
                self.config.model_id
            )
            self.is_frame_model = True
        
        # Load processor
        try:
            self.processor = AutoProcessor.from_pretrained(self.config.model_id)
        except Exception:
            self.processor = AutoFeatureExtractor.from_pretrained(self.config.model_id)
        
        self.model.eval()
        print("Model loaded successfully")
        return self
    
    def infer(self, y: np.ndarray, sr: int) -> DimensionalEmotion:
        if self.model is None or not TRANSFORMERS_AVAILABLE:
            return DimensionalEmotion()
        
        with torch.no_grad():
            inputs = self.processor(y, sampling_rate=sr, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Average frame predictions if frame model
            if self.is_frame_model:
                logits = logits.mean(dim=1)
            
            logits = logits.squeeze(0)
            
            # Get axis indices from config
            id2label = getattr(self.model.config, 'id2label', None)
            if id2label:
                axes = {id2label[i].lower(): i for i in range(len(id2label))}
                arousal_idx = axes.get('arousal')
                dominance_idx = axes.get('dominance')
                valence_idx = axes.get('valence')
            else:
                # Fallback: assume order [arousal, dominance, valence]
                arousal_idx = 0 if logits.shape[-1] > 0 else None
                dominance_idx = 1 if logits.shape[-1] > 1 else None
                valence_idx = 2 if logits.shape[-1] > 2 else None
            
            # Extract values based on head type
            if self.config.ser_head == "dim01":
                sig = torch.sigmoid
                a01 = float(sig(logits[arousal_idx])) if arousal_idx is not None else None
                d01 = float(sig(logits[dominance_idx])) if dominance_idx is not None else None
                v01 = float(sig(logits[valence_idx])) if valence_idx is not None else None
                v = (2.0 * v01 - 1.0) if v01 is not None else None
            else:  # dimtanh
                tanh = torch.tanh
                v = float(tanh(logits[valence_idx])) if valence_idx is not None else None
                a = float(tanh(logits[arousal_idx])) if arousal_idx is not None else None
                d = float(tanh(logits[dominance_idx])) if dominance_idx is not None else None
                a01 = 0.5 * (a + 1.0) if a is not None else None
                d01 = 0.5 * (d + 1.0) if d is not None else None
            
            return DimensionalEmotion(
                valence=v,
                arousal=a01,
                dominance=d01,
                confidence=1.0
            )

class CategoryMapper:    
    def __init__(self, config: EmotionConfig):
        self.config = config
    
    def va_to_category(self, emotion: DimensionalEmotion) -> Optional[str]:
        if emotion.valence is None or emotion.arousal is None:
            return None
        
        v = emotion.valence
        a = (emotion.arousal - 0.5) * 2.0  # Convert to [-1, 1]
        
        # Neutral buffer zone
        if (abs(v) <= self.config.valence_threshold and 
            abs(a) <= self.config.arousal_threshold):
            return "neutral"
        
        # Quadrant mapping
        if v >= 0 and a >= 0:
            return "joy"
        elif v >= 0 and a < 0:
            return "calm"
        elif v < 0 and a < 0:
            return "sadness"
        else:  # v < 0 and a >= 0
            # Use dominance to distinguish anger from fear
            if (self.config.allow_fear and 
                emotion.dominance is not None and 
                emotion.dominance < self.config.dominance_fear_cutoff):
                return "fear"
            return "anger"

class SpeakerStatistics:    
    def __init__(self, config: EmotionConfig):
        self.config = config
        self.stats: Dict[str, Dict[str, float]] = {}
    
    def compute_statistics(self, segments: List[Dict]) -> Dict[str, Dict[str, float]]:
        speaker_data: Dict[str, Dict[str, List[float]]] = {}
        
        for seg in segments:
            if not self._is_valid_for_baseline(seg):
                continue
            
            speaker = seg.get('speaker')
            emotion = seg['emotion']
            
            if speaker not in speaker_data:
                speaker_data[speaker] = {'v': [], 'a': []}
            
            speaker_data[speaker]['v'].append(emotion['v'])
            speaker_data[speaker]['a'].append(2.0 * emotion['a01'] - 1.0)
        
        # Compute robust statistics
        for speaker, data in speaker_data.items():
            v_array = np.array(data['v'])
            a_array = np.array(data['a'])
            
            v_mu, v_sd = self._robust_statistics(v_array)
            a_mu, a_sd = self._robust_statistics(a_array)
            
            self.stats[speaker] = {
                'v_mu': v_mu, 'v_sd': v_sd,
                'a_mu': a_mu, 'a_sd': a_sd,
                'n_segments': len(data['v'])
            }
        
        return self.stats
    
    def _is_valid_for_baseline(self, seg: Dict) -> bool:
        if seg.get('type') != 'speech':
            return False
        
        emotion = seg.get('emotion', {})
        if emotion.get('v') is None or emotion.get('a01') is None:
            return False
        
        if seg.get('overlap', False):
            return False
        
        duration = seg.get('end', 0) - seg.get('start', 0)
        if duration < self.config.min_segment_seconds:
            return False
        
        if emotion.get('conf', 0) < 0.2:
            return False
        
        return True
    
    def _robust_statistics(self, x: np.ndarray) -> Tuple[float, float]:
        if x.size == 0:
            return 0.0, 1.0
        
        median = float(np.median(x))
        mad = float(np.median(np.abs(x - median)))
        sd = 1.4826 * mad  # MAD to standard deviation
        
        # Fallback to regular std if MAD is too small
        if not np.isfinite(sd) or sd < 1e-6:
            sd = float(x.std(ddof=1)) if x.size > 1 else 1.0
            if sd < 1e-6 or not np.isfinite(sd):
                sd = 1.0
        
        return median, sd
    
    def compute_z_score(self, speaker: str, valence: float, arousal: float) -> Tuple[float, float]:
        if speaker not in self.stats:
            return 0.0, 0.0
        
        stats = self.stats[speaker]
        a = 2.0 * arousal - 1.0  # Convert to [-1, 1]
        
        z_v = (valence - stats['v_mu']) / max(stats['v_sd'], 1e-6)
        z_a = (a - stats['a_mu']) / max(stats['a_sd'], 1e-6)
        
        return float(z_v), float(z_a)
    
    def z_to_category(self, z_v: float, z_a: float, dominance: Optional[float]) -> Optional[str]:
        # Check if meets threshold
        if self.config.z_require_both_axes:
            meets_threshold = (abs(z_v) >= self.config.z_valence_threshold and 
                             abs(z_a) >= self.config.z_arousal_threshold)
        else:
            meets_threshold = (abs(z_v) >= self.config.z_valence_threshold or 
                             abs(z_a) >= self.config.z_arousal_threshold)
        
        if not meets_threshold:
            return "neutral"
        
        # Quadrant mapping
        if z_v >= 0 and z_a >= 0:
            return "joy"
        elif z_v >= 0 and z_a < 0:
            return "calm"
        elif z_v < 0 and z_a < 0:
            return "sadness"
        else:  # z_v < 0 and z_a >= 0
            if (self.config.allow_fear and dominance is not None and 
                dominance < self.config.dominance_fear_cutoff):
                return "fear"
            return "anger"

class EmotionExtractor:    
    def __init__(self, config: EmotionConfig):
        self.config = config
        self.audio_processor = AudioProcessor(config)
        self.classifier = EmotionClassifier(config)
        self.category_mapper = CategoryMapper(config)
        self.speaker_stats = SpeakerStatistics(config)
    
    def process(self, audio_path: str, segments_path: str, output_path: str):
        # Load data
        print(f"\nLoading segments from: {segments_path}")
        segments = json.loads(Path(segments_path).read_text())
        
        print(f"\nLoading audio from: {audio_path}")
        audio, sr = self.audio_processor.load_audio(audio_path)
        print(f"   Duration: {len(audio)/sr:.2f}s @ {sr}Hz")
        
        # Load model
        print(f"\nInitializing emotion classifier...")
        self.classifier.load()
        
        # First pass: extract features
        self._extract_features(segments, audio, sr, output_path)
        
        # Second pass: compute speaker statistics and relative categories
        print(f"\nComputing speaker-relative statistics...")
        self.speaker_stats.compute_statistics(segments)
        self._add_relative_categories(segments)
        
        # Write final output
        self._write_output(output_path, segments)
        print(f"\nComplete! Wrote: {output_path}")
    
    def _extract_features(self, segments: List[Dict], audio: np.ndarray, 
                         sr: int, checkpoint_path: str):
        speech_segments = [s for s in segments if s.get('type') == 'speech']
        print(f"\nProcessing {len(speech_segments)} speech segments\n")
        
        processed = 0
        for seg in tqdm(segments, desc="Extracting features", unit="seg"):
            if seg.get('type') != 'speech':
                continue
            
            start = float(seg['start'])
            end = float(seg['end'])
            duration = max(1e-6, end - start)
            text = seg.get('text', '') or ''
            
            # Extract audio segment
            y_segment = self.audio_processor.crop_segment(audio, sr, start, end)
            
            # Compute prosodic features
            prosody = self._compute_prosody(y_segment, sr, text, duration)
            
            # Compute emotional features
            if duration < self.config.min_segment_seconds:
                emotion = DimensionalEmotion()
            else:
                # Use padded segment for emotion
                pad_start = max(0, start - self.config.pad_seconds)
                pad_end = end + self.config.pad_seconds
                y_padded = self.audio_processor.crop_segment(audio, sr, pad_start, pad_end)
                emotion = self.classifier.infer(y_padded, sr)
                emotion.category_absolute = self.category_mapper.va_to_category(emotion)
            
            # Store results
            seg['emotion'] = self._emotion_to_dict(emotion)
            seg['prosody'] = self._prosody_to_dict(prosody)
            
            processed += 1
            if self.config.checkpoint_every and processed % self.config.checkpoint_every == 0:
                self._write_output(checkpoint_path, segments)
    
    def _compute_prosody(self, y: np.ndarray, sr: int, text: str, 
                        duration: float) -> ProsodicFeatures:
        energy = self.audio_processor.compute_rms(y)
        log_rms = float(np.log1p(max(energy, 0.0)))
        pitch_mean, pitch_var = self.audio_processor.compute_pitch(y, sr)
        
        # Speech rate (words per second)
        word_count = len([w for w in text.strip().split() if w])
        wps = min(word_count / duration, 6.0)  # Clip at reasonable max
        
        return ProsodicFeatures(
            pitch_mean=pitch_mean if np.isfinite(pitch_mean) else None,
            pitch_variance=pitch_var if np.isfinite(pitch_var) else None,
            log_rms=log_rms,
            words_per_second=wps
        )
    
    def _add_relative_categories(self, segments: List[Dict]):
        for seg in segments:
            if seg.get('type') != 'speech':
                continue
            
            emotion = seg.get('emotion', {})
            if emotion.get('v') is None or emotion.get('a01') is None:
                emotion['cat_z'] = None
                emotion['zv'] = None
                emotion['za'] = None
                continue
            
            speaker = seg.get('speaker')
            z_v, z_a = self.speaker_stats.compute_z_score(
                speaker, emotion['v'], emotion['a01']
            )
            
            cat_z = self.speaker_stats.z_to_category(z_v, z_a, emotion.get('d01'))
            
            emotion['cat_z'] = cat_z
            emotion['zv'] = z_v
            emotion['za'] = z_a
    
    def _emotion_to_dict(self, emotion: DimensionalEmotion) -> Dict:
        return {
            'v': emotion.valence,
            'a01': emotion.arousal,
            'd01': emotion.dominance,
            'cat': emotion.category_absolute,
            'cat_z': emotion.category_relative,
            'zv': emotion.z_valence,
            'za': emotion.z_arousal,
            'conf': emotion.confidence,
        }
    
    def _prosody_to_dict(self, prosody: ProsodicFeatures) -> Dict:
        return {
            'pitch_mu': prosody.pitch_mean,
            'pitch_var': prosody.pitch_variance,
            'log_rms': prosody.log_rms,
            'wps': prosody.words_per_second,
        }
    
    def _write_output(self, path: str, data: List[Dict]):
        compacted = self._compact(data)
        Path(path).write_text(json.dumps(compacted, indent=2))
    
    def _compact(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                compacted_v = self._compact(v)
                if self.config.drop_nulls and compacted_v in (None, {}, []):
                    continue
                result[k] = compacted_v
            return result
        elif isinstance(obj, list):
            return [self._compact(item) for item in obj]
        elif isinstance(obj, float):
            if not math.isfinite(obj):
                return None
            return round(obj, self.config.round_digits)
        return obj


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract dimensional emotions and prosodic features from speech"
    )
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--asr_json", required=True, help="Path to ASR segments JSON")
    parser.add_argument("--out_json", required=True, help="Output JSON path")
    parser.add_argument("--ser_model", default="", help="Dimensional SER model ID")
    parser.add_argument("--ser_head", choices=["dim01", "dimtanh"], default="dim01",
                       help="Head mapping: dim01 (0..1) or dimtanh (-1..1)")
    parser.add_argument("--pad", type=float, default=0.0, help="Padding in seconds")
    parser.add_argument("--min_seg_s", type=float, default=0.6, 
                       help="Minimum segment duration")
    parser.add_argument("--allow_fear", action="store_true", 
                       help="Distinguish fear from anger using dominance")
    
    args = parser.parse_args()
    
    config = EmotionConfig(
        model_id=args.ser_model,
        ser_head=args.ser_head,
        pad_seconds=args.pad,
        min_segment_seconds=args.min_seg_s,
        allow_fear=args.allow_fear
    )
    
    extractor = EmotionExtractor(config)
    extractor.process(args.audio, args.asr_json, args.out_json)


if __name__ == "__main__":
    main()