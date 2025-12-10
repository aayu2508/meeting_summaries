#!/usr/bin/env python3
"""
End to end meeting pipeline with timings.

Stages:
1. Audio Processing      -> audio_16k_mono.wav
2. Diarization           -> diarization_raw.json, diarization.json
3. Transcription         -> transcript.json (+ raw / merged variants)
4. Chunking              -> chunks_<extract_model>.json
5. Ideas Raw             -> ideas_raw_<extract_model>.json
6. Ideas Reflected       -> ideas_reflected_<extract_model>_<reflect_model>.json

Example:
  python scripts/run_meeting_pipeline.py \
    --meeting-id ES2003b \
    --audio amicorpus/ES2003b/audio/ES2003b.Mix-Headset.wav \
    --extract-model gptnano \
    --reflect-model gptfull \
    --device auto
"""

import argparse
import os
import sys
import time
from pathlib import Path
import subprocess


# Resolve project root (one level up from this scripts directory)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Precompute PYTHONPATH that includes src/
BASE_ENV = os.environ.copy()
BASE_ENV["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + BASE_ENV.get("PYTHONPATH", "")

# Module names (not paths!) for each stage
MOD_AUDIO   = "ingest_asr_diar.audio_processing"
MOD_DIAR    = "ingest_asr_diar.diarization"
MOD_ASR     = "ingest_asr_diar.transcribe"
MOD_CHUNK   = "chunking.chunk_transcript"
MOD_IDEAS_R = "extraction.extract_ideas_raw"
MOD_IDEAS_F = "extraction.reflect_ideas"


def run_stage(label: str, module: str, args_list: list[str]) -> float:
    """
    Run `python -m <module> args...`, time it, and return elapsed seconds.
    Ensures src/ is on PYTHONPATH so relative imports (.audio_utils, .utils, etc.) work.
    """
    cmd = [sys.executable, "-m", module] + args_list
    print(f"\n[{label}] CMD: {' '.join(cmd)}")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, env=BASE_ENV, cwd=str(PROJECT_ROOT))
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"[{label}] elapsed: {elapsed:.2f} s")
    return elapsed


def main():
    ap = argparse.ArgumentParser(description="End to end meeting pipeline with timings")
    ap.add_argument("--meeting-id", required=True, help="Meeting ID, e.g., ES2003b")
    ap.add_argument("--audio", required=True, help="Path to raw audio or video file")
    ap.add_argument(
        "--extract-model",
        default="gptnano",
        choices=["gptnano", "gptfull"],
        help="Model alias for idea extraction (also used as chunk profile name).",
    )
    ap.add_argument(
        "--reflect-model",
        default="gptfull",
        help="Model alias for reflection, e.g. gptfull",
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for diarization and ASR (default: auto)",
    )
    ap.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional fixed number of speakers for diarization",
    )
    ap.add_argument(
        "--outputs-root",
        default="data/outputs",
        help="Base outputs root (default: data/outputs)",
    )
    args = ap.parse_args()

    meeting_id = args.meeting_id
    extract_model = args.extract_model   # used for chunk profile and ideas_raw
    reflect_model = args.reflect_model

    raw_audio = Path(args.audio).expanduser().resolve()
    assert raw_audio.exists(), f"Audio not found: {raw_audio}"

    outputs_root = Path(args.outputs_root).expanduser().resolve()
    out_dir = outputs_root / meeting_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] project_root: {PROJECT_ROOT}")
    print(f"[setup] src dir     : {SRC_DIR}")
    print(f"[setup] outputs dir : {out_dir}")

    timings: dict[str, float] = {}

    # 1. Audio processing
    timings["audio_preprocessing"] = run_stage(
        "audio_preprocessing",
        MOD_AUDIO,
        [
            "--input", str(raw_audio),
            "--meeting-id", meeting_id,
        ],
    )

    # 2. Diarization
    diar_args = [
        "--meeting-id", meeting_id,
        "--device", args.device,
    ]
    if args.num_speakers is not None:
        diar_args += ["--num-speakers", str(args.num_speakers)]
    timings["diarization"] = run_stage(
        "diarization",
        MOD_DIAR,
        diar_args,
    )

    # 3. Transcription
    timings["transcription"] = run_stage(
        "transcription",
        MOD_ASR,
        [
            "--meeting-id", meeting_id,
            "--device", args.device,
        ],
    )

    # 4. Chunking (uses extract_model as profile: gptnano / gptfull)
    timings["chunking"] = run_stage(
        "chunking",
        MOD_CHUNK,
        [
            "--meeting-id", meeting_id,
            "--profile", extract_model,
        ],
    )

    # 5. Ideas Raw
    timings["ideas_raw"] = run_stage(
        "ideas_raw",
        MOD_IDEAS_R,
        [
            "--meeting-id", meeting_id,
            "--model", extract_model,
        ],
    )

    # 6. Ideas Reflected
    timings["ideas_reflected"] = run_stage(
        "ideas_reflected",
        MOD_IDEAS_F,
        [
            "--meeting-id", meeting_id,
            "--extract-model", extract_model,
            "--reflect-model", reflect_model,
        ],
    )

    # Save timings as plain text only
    timings_txt = out_dir / f"timings_{extract_model}.txt"

    lines: list[str] = []
    lines.append(f"meeting_id: {meeting_id}")
    lines.append(f"audio: {raw_audio}")
    lines.append(f"extract_model: {extract_model}")
    lines.append(f"reflect_model: {reflect_model}")
    lines.append(f"device_flag: {args.device}")
    lines.append(f"num_speakers: {args.num_speakers}")
    lines.append("")
    lines.append("Stage timings (seconds):")
    for stage, sec in timings.items():
        lines.append(f"  {stage:20s} {sec:8.2f}")

    timings_txt.write_text("\n".join(lines))
    print(f"\n[done] wrote timings text to {timings_txt}")


if __name__ == "__main__":
    main()
