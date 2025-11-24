#!/bin/bash
set -euo pipefail

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)       INPUT="${2:-}"; shift 2;;
    --meeting-id)  MEETING_ID="${2:-}"; shift 2;;
    --num-speakers)NUM_SPEAKERS="${2:-}"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

[[ -n "${INPUT:-}" ]] || { echo "ERROR: --input is required"; exit 1; }
[[ -f "${INPUT}" ]]   || { echo "ERROR: file not found: ${INPUT}"; exit 1; }

if [[ -z "${MEETING_ID:-}" ]]; then
  base="$(basename -- "$INPUT")"
  MEETING_ID="${base%.*}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
OUT_DIR="${REPO_ROOT}/data/outputs/${MEETING_ID}"
mkdir -p "${OUT_DIR}"

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

echo "Running pipeline for meeting: ${MEETING_ID}"
echo "Audio: ${INPUT}"
echo "Outputs: ${OUT_DIR}"
echo

total_start=$(date +%s)

echo "[1/3] Audio preprocessing -> 16kHz mono PCM"
t1_start=$(date +%s)
python3 -m ingest_asr_diar.audio_processing --input "${INPUT}" --meeting-id "${MEETING_ID}"
t1_end=$(date +%s)

echo "[2/3] Speaker diarization (pyannote)"
t2_start=$(date +%s)
if [[ -n "${NUM_SPEAKERS:-}" ]]; then
  python3 -m ingest_asr_diar.diarization --meeting-id "${MEETING_ID}" --num-speakers "${NUM_SPEAKERS}"
else
  python3 -m ingest_asr_diar.diarization --meeting-id "${MEETING_ID}"
fi
t2_end=$(date +%s)

echo "[3/3] Transcription + ASR-Diar fusion"
t3_start=$(date +%s)
python3 -m ingest_asr_diar.transcribe --meeting-id "${MEETING_ID}"
t3_end=$(date +%s)

total_end=$(date +%s)

echo
echo "Done. Outputs in: ${OUT_DIR}"
ls -1 "${OUT_DIR}" | sed 's/^/ - /'

echo
echo "Timing (seconds):"
echo "  [1/3] audio_processing : $((t1_end - t1_start))"
echo "  [2/3] diarization      : $((t2_end - t2_start))"
echo "  [3/3] transcription    : $((t3_end - t3_start))"
echo "  TOTAL                  : $((total_end - total_start))"