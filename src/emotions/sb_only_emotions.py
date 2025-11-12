#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import librosa
from tqdm import tqdm
from speechbrain.inference.interfaces import foreign_class

SB_MODEL_ID = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
PAD_S = 0.0          # avoid cross-speaker leakage
MIN_SEG_S = 0.6      # skip micro-utterances for stability
SR = 16000           # SpeechBrain models typically expect 16 kHz mono

def load_audio(audio_path: str, sr: int = SR):
    """Load audio file with librosa"""
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    return y.astype(np.float32, copy=False), sr

def crop(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """Extract audio segment between start_s and end_s seconds"""
    n = len(y)
    def s2i(t):
        t_clamped = float(np.clip(t, 0.0, n / sr))
        return int(t_clamped * sr)
    return y[s2i(start_s):s2i(end_s)]

class SBCategorical:
    """SpeechBrain emotion classifier wrapper using foreign_class"""
    
    def __init__(self, model_id: str = SB_MODEL_ID):
        self.model_id = model_id
        self.classifier = None

    def load(self):
        """Load SpeechBrain model using foreign_class interface"""
        print(f"Loading model from {self.model_id}...")
        
        self.classifier = foreign_class(
            source=self.model_id,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": "cpu"}
        )
        
        print("✓ Model loaded successfully")
        return self

    def infer_label(self, y: np.ndarray, sr: int) -> Tuple[Optional[str], Optional[float]]:
        """Run emotion inference on audio segment"""
        if self.classifier is None or y.size == 0:
            return None, None
        
        # Save audio to temp file (required by classify_file)
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, y, sr)
        
        # Run classification
        out_prob, score, index, text_lab = self.classifier.classify_file(tmp_path)
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)
        
        # Extract label (batch size 1)
        label = str(text_lab[0]).lower() if text_lab else None
        
        # Extract confidence
        conf = None
        if out_prob is not None and index is not None:
            b = 0
            top_idx = int(index[b])
            conf = float(out_prob[b, top_idx])
        
        return label, conf

def process(audio_path: str, asr_json_path: str, out_json_path: str,
            sb_model_id: str = SB_MODEL_ID, pad_s: float = PAD_S, min_seg_s: float = MIN_SEG_S):
    """Process audio file and add emotions to transcript segments"""
    
    # Validate inputs
    audio_path_obj = Path(audio_path)
    asr_json_path_obj = Path(asr_json_path)
    
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not asr_json_path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {asr_json_path}")
    
    # Load JSON + audio
    print(f"\nLoading transcript from: {asr_json_path}")
    segments = json.loads(asr_json_path_obj.read_text())
    print(f"   Found {len(segments)} total segments")
    
    print(f"\nLoading audio from: {audio_path}")
    y, sr = load_audio(audio_path, sr=SR)
    print(f"   Duration: {len(y)/sr:.2f}s @ {sr}Hz")

    # Load SpeechBrain model
    print(f"\nInitializing SpeechBrain emotion model...")
    sb = SBCategorical(sb_model_id).load()

    n_speech = sum(1 for s in segments if s.get("type") == "speech")
    n_skipped = sum(1 for s in segments if s.get("type") == "speech" and 
                    (float(s.get("end", 0)) - float(s.get("start", 0))) < min_seg_s)
    
    print(f"\nProcessing {n_speech} speech segments")
    print(f"   Skipping {n_skipped} segments shorter than {min_seg_s}s")
    print(f"   Using padding: {pad_s}s\n")

    processed = 0
    for s in tqdm(segments, desc="Analyzing emotions", unit="seg"):
        if s.get("type") != "speech":
            continue

        start = float(s.get("start", 0.0))
        end   = float(s.get("end", 0.0))
        dur   = max(0.0, end - start)

        # Initialize emotion fields
        emo = {"category_categorical": None, "category_confidence": None}

        if dur >= min_seg_s:
            left  = max(0.0, start - pad_s)
            right = end + pad_s
            y_win = crop(y, sr, left, right)
            label, conf = sb.infer_label(y_win, sr)
            emo["category_categorical"] = label
            emo["category_confidence"]  = conf
            if label is not None:
                processed += 1

        s["emotion"] = emo

    Path(out_json_path).write_text(json.dumps(segments, indent=2))
    print(f"\nWrote results to: {out_json_path}")
    print(f" Successfully processed: {processed}/{n_speech} segments")
    
    # Print emotion summary
    emotions = [s.get("emotion", {}).get("category_categorical") 
                for s in segments if s.get("type") == "speech"]
    emotions = [e for e in emotions if e is not None]
    
    if emotions:
        from collections import Counter
        emotion_counts = Counter(emotions)
        print(f"\nEmotion Distribution:")
        for emotion, count in emotion_counts.most_common():
            pct = count/len(emotions)*100
            bar = "█" * int(pct / 5)
            print(f"   {emotion:12s}: {count:3d} ({pct:5.1f}%) {bar}")
    else:
        print("\nNo emotions detected (all segments may be too short)")
    
    return out_json_path

def main():
    import argparse
    p = argparse.ArgumentParser(description="Add SpeechBrain categorical emotions to segments JSON.")
    p.add_argument("--audio", required=True, help="Path to mono WAV/FLAC audio (16 kHz recommended)")
    p.add_argument("--asr_json", required=True, help="Path to ASR/diarized segments JSON (list of dicts)")
    p.add_argument("--out_json", required=True, help="Output path for updated JSON")
    p.add_argument("--sb_model", default=SB_MODEL_ID, help="SpeechBrain model id or local folder")
    p.add_argument("--pad", type=float, default=PAD_S, help="Padding around segment in seconds (default 0.0)")
    p.add_argument("--min_seg_s", type=float, default=MIN_SEG_S, help="Min duration (s) to run model (default 0.6)")
    args = p.parse_args()

    process(
        audio_path=args.audio,
        asr_json_path=args.asr_json,
        out_json_path=args.out_json,
        sb_model_id=args.sb_model,
        pad_s=float(args.pad),
        min_seg_s=float(args.min_seg_s),
    )
    print("\Processing complete!")

if __name__ == "__main__":
    main()