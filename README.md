# meeting_summaries

End‑to‑end pipeline for turning meeting audio into rich structured summaries, idea maps, and emotion / conversation‑dynamics analyses.

The project takes a meeting recording, runs ASR + diarization, and then uses a mixture of deterministic logic and LLM calls to extract:

- Core ideas and design proposals
- Unresolved open‑ended questions
- Requirements, evaluation criteria, and decisions
- Per‑idea stance / common ground
- Speaker‑ and idea‑level participation metrics
- Acoustic and text‑based emotion dynamics
- Plots for ideas, emotions, and evaluation grids

---

## 1. Quick Start

### 1.1. Environment

```bash
git clone <repo-url>
cd meeting_summaries

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env      # then edit for your keys
```

At minimum, `.env` should define:

- `HF_TOKEN` – Hugging Face token for diarization (`pyannote.audio`) if required
- `ASR_MODEL` – Faster‑Whisper model size (`small`, `medium`, `large-v3`, …)
- `ASR_DEVICE` – `cpu` or `cuda`

### 1.2. Input Data

Place your raw meeting audio under:

```text
data/raw/<MEETING_ID>/<file>.wav
```

The core tooling expects (or creates) outputs under:

```text
data/outputs/<MEETING_ID>/
```

---

## 2. High‑Level Pipeline

The typical flow is:

1. **Audio preparation**  
   Resample / convert to 16 kHz mono WAV.

   ```bash
   python -m ingest_asr_diar.audio_processing --meeting-id <MEETING_ID>
   ```

   Uses [`ingest_asr_diar.audio_processing.main`](src/ingest_asr_diar/audio_processing.py).

2. **Speaker diarization**  

   ```bash
   python -m ingest_asr_diar.diarization --meeting-id <MEETING_ID> [--num-speakers N]
   ```

   See [`ingest_asr_diar.diarization.main`](src/ingest_asr_diar/diarization.py).

3. **ASR (faster‑whisper) + diarization fusion**  

   ```bash
   python -m ingest_asr_diar.transcribe --meeting-id <MEETING_ID>
   ```

   Produces `transcript_raw.json` and `transcript.json` via
   [`ingest_asr_diar.transcribe.main`](src/ingest_asr_diar/transcribe.py).

4. **Chunking transcript for LLM passes**  

   ```bash
   python -m chunking.chunker --meeting-id <MEETING_ID> --profile gptnano
   ```

   See [`chunking.chunker.main`](src/chunking/chunker.py).  
   Writes `chunks_<profile>.json`.

5. **Idea extraction and reflection pipeline**

   - **Per‑chunk raw idea extraction**

     ```bash
     python -m extraction.extract_ideas_raw --meeting-id <MEETING_ID> --model gptnano
     ```

     [`extraction.extract_ideas_raw.main`](src/extraction/extract_ideas_raw.py) writes  
     `context_outputs/ideas_raw_<model>.json`.

   - **Reflect / consolidate into meeting‑level canonical ideas**

     ```bash
     python -m extraction.reflect_ideas --meeting-id <MEETING_ID> \
       --extract-model gptnano --reflect-model gptfull
     ```

     [`extraction.reflect_ideas.main`](src/extraction/reflect_ideas.py) writes  
     `context_outputs/ideas_reflected_<extract>_<reflect>.json`.

   - **Expand idea mentions across full transcript**

     ```bash
     python -m extraction.expand_idea_mentions --meeting-id <MEETING_ID> \
       --extract-model gptnano --reflect-model gptfull \
       --chunks-model gptnano --expansion-model gptnano
     ```

     [`extraction.expand_idea_mentions.main`](src/extraction/expand_idea_mentions.py) writes
     augmented ideas JSON with `mentions_extra` etc.

6. **Open‑ended, unresolved questions**

   ```bash
   python -m extraction.extract_open_ended_q --meeting-id <MEETING_ID> --model gptfull
   ```

   Implemented in [`extraction.extract_open_ended_q.main`](src/extraction/extract_open_ended_q.py).  
   Uses the strong system prompt in `SYSTEM_PROMPT` to output  
   `context_outputs/open_questions_<model>.json`, containing only *unanswered* or *partially answered* open questions grounded in segment IDs.


7. **Per‑idea evaluation criteria**

   ```bash
   python -m extraction.extract_evaluation_criteria --meeting-id <MEETING_ID> \
     --ideas-json <ideas_json_relative_or_absolute> \
     --model gptfull --write-csv
   ```

   Implemented in [`extraction.extract_evaluation_criteria.main`](src/extraction/extract_evaluation_criteria.py).  
   Produces a JSON and optional `eval_criteria_<model>_matrix.csv`.

8. **Emotion and prosody extraction**

    There are three emotion‑related tools:

    - **Dimensional emotion + prosody with VAD**

      ```bash
      python -m emotions.emotion_extractor_vad \
        --audio data/outputs/<MEETING_ID>/audio_16k_mono.wav \
        --asr_json data/outputs/<MEETING_ID>/transcript.json \
        --out_json data/outputs/<MEETING_ID>/asr_emotion_vad.json \
        --ser_model <HF_audio_model_id>
      ```

      See [`emotions.emotion_extractor_vad.AudioProcessor`](src/emotions/emotion_extractor_vad.py),
      [`emotions.emotion_extractor_vad.EmotionClassifier`](src/emotions/emotion_extractor_vad.py),
      and [`emotions.emotion_extractor_vad.main`](src/emotions/emotion_extractor_vad.py).

    - **Categorical emotion from SpeechBrain (IEMOCAP)**

      ```bash
      python -m emotions.sb_only_emotions \
        --audio data/outputs/<MEETING_ID>/audio_16k_mono.wav \
        --asr_json data/outputs/<MEETING_ID>/asr_emotion_vad.json \
        --out_json data/outputs/<MEETING_ID>/asr_emotion_vad_categorical.json \
        --device cpu
      ```

      Implemented around [`emotions.sb_only_emotions.process`](src/emotions/sb_only_emotions.py).

    - **Text‑only 7‑class emotion (CardiffNLP RoBERTa)**

      ```bash
      python -m emotions.text_emotion_extractor \
        --asr_json data/outputs/<MEETING_ID>/transcript.json \
        --out_json data/outputs/<MEETING_ID>/transcript_with_text_emotion.json
      ```

      See [`emotions.text_emotion_extractor.main`](src/emotions/text_emotion_extractor.py).

    A convenience script ties the first two together and merges them:

    ```bash
    bash scripts/run_emotions.sh --meeting-id <MEETING_ID> --device cpu  # or cuda
    ```

    Defined in [scripts/run_emotions.sh](scripts/run_emotions.sh).

9. **Conversation dynamics (ideas & meeting)**

    - Idea‑centric:

      ```bash
      python -m analysis.conversation_dynamics_ideas \
        --meeting-id <MEETING_ID> \
        --extract-model gptnano --reflect-model gptfull
      ```

      [`analysis.conversation_dynamics_ideas.main`](src/analysis/conversation_dynamics_ideas.py) reads
      `ideas_reflected_*.json` and writes:
      - `conversation_dynamics_ideas.json`
      - `conversation_dynamics_ideas.txt`
      - plots in `plots/` (timeline, bipartite speaker–idea, etc.)

    - Meeting‑level turn dynamics:

      ```bash
      python -m analysis.conversation_dynamics_meeting --meeting-id <MEETING_ID>
      ```

      See [`analysis.conversation_dynamics_meeting.main`](src/analysis/conversation_dynamics_meeting.py).

10. **Emotion dynamics analysis & plots**

    ```bash
    python -m analysis.emotion_meeting_analysis --meeting-id <MEETING_ID> \
      --emotion-json-name transcript_emotion.json
    ```

    Implemented in [`analysis.emotion_meeting_analysis.main`](src/analysis/emotion_meeting_analysis.py).
    Produces:
    - `emotion_dynamics_meeting.txt`
    - `emotion_dynamics_meeting.json`
    - plots in `plots/`.

---

## 3. LLM Client & Common Utilities

All LLM‑based extractors share a small wrapper that handles JSON‑mode, retries, and model aliases:

- [`extraction.utils.llm_client.chat_json`](src/extraction/utils/llm_client.py)
- [`extraction.utils.llm_client.init_client`](src/extraction/utils/llm_client.py)

Common helpers live in:

- [`extraction.utils.common`](src/extraction/utils/common.py), including:
  - [`extraction.utils.common.get_meeting_base_dir`](src/extraction/utils/common.py)
  - [`extraction.utils.common.load_metadata`](src/extraction/utils/common.py)
  - [`extraction.utils.common.norm_key`](src/extraction/utils/common.py)
  - [`extraction.utils.common.canonical_idea_text`](src/extraction/utils/common.py)

---

## 4. Key JSON Artifacts (per meeting)

Under `data/outputs/<MEETING_ID>/` you will typically see:

- `audio_16k_mono.wav` – preprocessed audio
- `diarization_raw.json`, `diarization.json`
- `transcript_raw.json`, `transcript.json`
- `chunks_<profile>.json`
- `context_outputs/`
  - `ideas_raw_<model>.json`
  - `ideas_reflected_<extract>_<reflect>.json`
  - `ideas_windows_<model>.json`
  - `ideas_expanded_*.json` (depending on pipeline)
  - `open_questions_<model>.json`
  - `eval_criteria_<model>.json` and `eval_criteria_<model>_matrix.csv`
- Emotion:
  - `asr_emotion_vad.json`
  - `asr_emotion_vad_categorical.json`
  - `transcript_emotion.json`
- Analysis & plots:
  - `conversation_dynamics_*.json` / `.txt`
  - `emotion_dynamics_meeting.*`
  - `plots/*.png`

---

## 5. Notes and Tips

- Most scripts are intended to be run as modules from repo root with `PYTHONPATH` including `src/`.  
  The bash helper [scripts/run_emotions.sh](scripts/run_emotions.sh) shows the pattern:

  ```bash
  export PYTHONPATH="src:${PYTHONPATH:-}"
  ```

- Many extraction scripts accept a `--model` argument as an **alias** (e.g. `gptnano`, `gpt3.5`, `gptfull`); resolution happens in [`extraction.utils.llm_client._resolve_model`](src/extraction/utils/llm_client.py).

- Open‑ended question extraction is deliberately conservative.  
  See `SYSTEM_PROMPT` in
  [`extraction.extract_open_ended_q`](src/extraction/extract_open_ended_q.py) for the exact behavior.

- If a step fails, check the expected input file paths in that script’s `main()`; most contain explicit `assert` / `SystemExit` messages.

---