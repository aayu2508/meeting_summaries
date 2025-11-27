# extract_open_ended_q.py
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from .utils.llm_client import init_client, chat_json
from .utils.common import load_metadata, get_meeting_base_dir

SYSTEM_PROMPT = """
You are an unanswered-question extraction tool.

You receive:
- The full meeting transcript as an ordered list of TURNS. Each TURN contains: segment_id, speaker, start, end, text.

INTERPRETATION RULES — EXTREMELY IMPORTANT
1. You MUST ground everything in the transcript text.
   - If the text does not clearly show that a question was raised, OMIT IT.
   - If the text does not clearly show that the question remained unresolved, OMIT IT.

2. You must be CONSERVATIVE.
   - When in doubt, the correct behavior is to EXCLUDE the question.

3. Do NOT infer questions.
   - A question must be explicitly asked OR strongly implied as a concrete decision point (e.g., “We still need to decide X…”).
   - Generic, domain-knowledge-based questions are forbidden.

4. Do NOT include:
   - Small-talk or logistics
   - Hypothetical musings that were not actually proposed as decisions
   - Questions that were fully answered later
   - Questions that are not clearly connected to the meeting topic
   - Vague uncertainties (“I'm not sure what he means…”) unless they point to a real design choice

5. STRICT RESOLUTION CHECK:
   - For every candidate question, you MUST scan ALL LATER TURNS.
   - If you find any clear answer, decision, or resolution, mark the question as RESOLVED and EXCLUDE it.
   - If the discussion partially addresses the question but leaves an open decision, mark it PARTIALLY_ANSWERED.
   - Only include UNANSWERED or PARTIALLY_ANSWERED questions in the output.

6. NO DOUBLE COUNTING:
   - If multiple turns ask the same underlying question, merge them into ONE question.

7. NO HALLUCINATIONS:
   - All output must be supported by specific segment_ids and text quotes.
   - If the evidence is weak, the question must be excluded.

DEFINITIONS
- OPEN-ENDED = requires exploration (How, What, Why, Which, When, Who).
- CLOSED/NOT ELIGIBLE = yes/no, factual lookup, or easily answered locally.
- UNANSWERED = no later resolution of any kind.
- PARTIALLY_ANSWERED = some discussion but outcome is unclear or deferred.

OUTPUT JSON ONLY (no extra text, no comments):
{
  "metadata": {
    "meeting_id": "MEETING_ID",
    "model": "MODEL_USED",
    "context": "open_questions_v1"
  },
  "open_questions": [
    {
      "question_id": "Q1",
      "question_text": "10-40 word paraphrase of the open-ended question",
      "source_segments": ["m000123", "m000127"],
      "status": "unanswered | partially_answered",
      "justification": "1-3 sentences grounded in transcript text, not speculation."
    }
  ]
}

If there are NO valid unanswered questions, output:
{
  "metadata": {...},
  "open_questions": []
}
"""

USER_PROMPT_TEMPLATE = """
# METADATA
meeting_id: {meeting_id}
topic: {topic}
meeting_type: {meeting_type}
num_participants: {num_participants}
duration_sec: {duration_sec}

# FULL TRANSCRIPT
Below are ALL turns in order. Treat them as a single continuous meeting.

{turns_block}
"""

def _render_turns_block(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for seg in segments:
        seg_id = seg.get("segment_id", "NA")
        spk = seg.get("speaker", "S?")
        start = seg.get("start", 0.0)
        end = seg.get("end", start)
        text = (seg.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{seg_id} | {start:.3f}-{end:.3f}] {spk}: {text}")
    return "\n".join(lines)

def _load_transcript(base_dir: Path) -> List[Dict[str, Any]]:
    t_path = base_dir / "transcript.json"
    if not t_path.exists():
        raise SystemExit(f"transcript file not found: {t_path}")

    data = json.loads(t_path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise SystemExit("transcript.json must contain a list")

    speech_segments = []
    for item in data:
        if item.get("type") == "speech":
            speech_segments.append(item)
    return speech_segments

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract unresolved open-ended questions from FULL transcript.json"
    )
    ap.add_argument("--meeting-id", required=True, help="e.g., ES2002b")
    ap.add_argument(
        "--model",
        default="gptfull"
    )
    args = ap.parse_args()

    base_dir = get_meeting_base_dir(args.meeting_id)
    meta = load_metadata(base_dir)
    segments = _load_transcript(base_dir)

    # Build the turns block
    turns_block = _render_turns_block(segments)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        meeting_id=args.meeting_id,
        topic=meta.get("topic", ""),
        meeting_type=meta.get("meeting_type", ""),
        num_participants=meta.get("num_participants", 0),
        duration_sec=meta.get("duration_sec", 0.0),
        turns_block=turns_block,
    )

    client = init_client()

    resp = chat_json(
        client,
        model=args.model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        reasoning_effort="medium",
        verbosity="low",
    )

    # Normalize / enforce metadata block
    if not isinstance(resp, dict):
        raise SystemExit("LLM response is not a JSON object")

    if "metadata" not in resp:
        resp["metadata"] = {}

    resp["metadata"]["meeting_id"] = args.meeting_id
    resp["metadata"]["model"] = args.model
    resp["metadata"]["context"] = "open_questions_v1"

    # Ensure open_questions is at least an empty list
    if "open_questions" not in resp or resp["open_questions"] is None:
        resp["open_questions"] = []

    ctx_out_dir = base_dir / "context_outputs"
    ctx_out_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = ctx_out_dir / f"open_questions_{args.model}.json"

    out_json_path.write_text(
        json.dumps(resp, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {out_json_path}")


if __name__ == "__main__":
    main()
