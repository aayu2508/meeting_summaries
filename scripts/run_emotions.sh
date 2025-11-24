#!/bin/bash
set -euo pipefail

MEETING_ID=""
DEVICE="cpu"   # default

while [[ $# -gt 0 ]]; do
  case "$1" in
    --meeting-id) MEETING_ID="${2:-}"; shift 2;;
    --device)     DEVICE="${2:-}"; shift 2;;   # cpu or cuda
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

[[ -n "${MEETING_ID:-}" ]] || { echo "ERROR: --meeting-id is required"; exit 1; }
[[ "${DEVICE}" == "cpu" || "${DEVICE}" == "cuda" ]] \
  || { echo "ERROR: --device must be 'cpu' or 'cuda'"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/src"
OUT_DIR="${REPO_ROOT}/data/outputs/${MEETING_ID}"
mkdir -p "${OUT_DIR}"

export PYTHONPATH="${SRC_DIR}:${PYTHONPATH:-}"

AUDIO="${OUT_DIR}/audio_16k_mono.wav"
ASR_JSON="${OUT_DIR}/transcript.json"
DIM_JSON="${OUT_DIR}/asr_emotion_vad.json"
CAT_JSON="${OUT_DIR}/asr_emotion_vad_categorical.json"
MERGED_JSON="${OUT_DIR}/transcript_emotion.json"

[[ -f "${AUDIO}" ]]    || { echo "ERROR: audio not found: ${AUDIO}"; exit 1; }
[[ -f "${ASR_JSON}" ]] || { echo "ERROR: ASR JSON not found: ${ASR_JSON}"; exit 1; }

echo "Running emotion pipeline for meeting: ${MEETING_ID}"
echo "Device: ${DEVICE}"
echo

total_start=$(date +%s)

echo "[1/3] Dimensional emotion with VAD (wav2vec)"
e1_start=$(date +%s)
python3 "${SRC_DIR}/emotions/emotion_extractor_vad.py" \
  --audio "${AUDIO}" \
  --asr_json "${ASR_JSON}" \
  --out_json "${DIM_JSON}" \
  --ser_model "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
e1_end=$(date +%s)

echo "[2/3] Categorical emotions (SpeechBrain ${DEVICE})"
e2_start=$(date +%s)
python3 "${SRC_DIR}/emotions/sb_only_emotions.py" \
  --audio "${AUDIO}" \
  --asr_json "${ASR_JSON}" \
  --out_json "${CAT_JSON}" \
  --sb_model "speechbrain/emotion-recognition-wav2vec2-IEMOCAP" \
  --device "${DEVICE}"
e2_end=$(date +%s)

echo "[3/3] Merge dimensional + categorical"
e3_start=$(date +%s)
python3 "${SRC_DIR}/emotions/emotion_merged.py" \
  --dim_json "${DIM_JSON}" \
  --cat_json "${CAT_JSON}" \
  --out_json "${MERGED_JSON}"
e3_end=$(date +%s)

total_end=$(date +%s)

echo
echo "Done. Outputs in: ${OUT_DIR}"
ls -1 "${OUT_DIR}" | sed 's/^/ - /'

echo
echo "Timing (seconds):"
echo "  [1/3] dimensional (wav2vec) : $((e1_end - e1_start))"
echo "  [2/3] categorical (SB)      : $((e2_end - e2_start))"
echo "  [3/3] merge                 : $((e3_end - e3_start))"
echo "  TOTAL                       : $((total_end - total_start))"
