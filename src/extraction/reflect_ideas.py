#reflect_ideas.py
import json
import argparse
from typing import Dict, Any
from .utils.llm_client import init_client, chat_json
from .utils.common import get_meeting_base_dir, load_metadata

REFLECTION_SYSTEM_PROMPT = """
You are an idea reflection and consolidation tool for a single meeting.

You receive:
- METADATA about the meeting
- A JSON object with candidate ideas per CHUNK, where each idea has:
  - idea_local_id (unique within meeting)
  - idea_text (short paraphrase of the idea)
  - mentions: list of segments with
      segment_id, speaker, start, end, text, and chunk_id context

Your job is to:
1) Decide which candidate ideas are genuine, on-topic ideas and which should be discarded.
2) Merge semantically equivalent or very similar ideas across chunks into consolidated ideas.
3) Provide supporting quotes for each consolidated idea from the underlying mentions.

Return ONLY valid JSON with this exact schema (no extra keys, no comments):

{
  "metadata": {
    "meeting_id": "BAC",
    "extract_model": "gptnano",
    "reflect_model": "gptfull",
    "context": "ideas_reflection_v1"
  },
  "ideas": [
    {
      "idea_id": "I1",
      "canonical_idea": "short canonical description of the idea (5-20 words)",
      "source_idea_ids": ["BAC#chunkA_I1", "BAC#chunkB_I3"],
      "source_idea_texts": [
        "Use biological waste in a separate fermentation tank to generate natural gas",
        "Large-scale digestion tanks to turn mixed bio waste into fuel"
      ],
      "mentions": [
        {
          "segment_id": "m000186",
          "chunk_id": "BAC#abcd1234abcd1234",
          "speaker": "SPEAKER_00",
          "start": 857.686,
          "end": 865.482,
          "text": "So if we're thinking about like, what we wanna do if we're designing for a system...",
          "supporting_quote": "So if we're thinking about like, what we wanna do if we're designing for a system..."
        }
      ]
    }
  ],
  "discarded_ideas": [
    {
      "source_idea_ids": ["BAC#chunkX_I2"],
      "source_idea_texts": ["some text here"],
      "reason": "why these did not count as valid ideas or were off-topic"
    }
  ]
}

FILTERING AND MERGING IDEAS
- Treat idea_local_id as a unique handle for each raw idea.
- Two ideas belong to the same consolidated idea if, in your judgment, they describe essentially the same proposal or design, even if wording differs.
- Do NOT merge ideas that describe distinct features, design directions, or implementation variants.
- If a candidate is clearly not an idea related to the meeting topic, place it under discarded_ideas with a clear reason.

CANONICAL IDEA
- Canonical_idea should be a clear, concise phrase (5-20 words).
- Describe the central proposal in a general way that would still make sense outside this specific meeting.
- Order consolidated ideas by the earliest timestamp among their mentions.

SOURCE FIELDS
- source_idea_ids: ALL idea_local_id values that you merged into this consolidated idea.
- source_idea_texts: the corresponding idea_text strings.

MENTIONS AND SUPPORTING QUOTES
- For each consolidated idea, keep at mentions.
- For each mention, you MUST set:
  - segment_id (string)
  - chunk_id (string)
  - speaker (string)
  - start (number, seconds)
  - end (number, seconds)
  - supporting_quote (string)
- supporting_quote should:
  - Come from the underlying segment text.
  - Either be the full text of the segment OR a focused excerpt that strongly evidences the idea.
  - Be a clean, readable snippet (you may lightly clean filler words if needed, but do not change meaning).

DISCARDED IDEAS
- source_idea_ids and source_idea_texts should list all the ideas that you are discarding in that entry.
- Discard only if the idea is clearly off-topic, non-actionable, purely conversational, or not a solution/design proposal.
- reason should be specific and human-readable.

GENERAL RULES
- Use ONLY information present in the provided JSON and metadata.
- Do NOT invent new segment_id, chunk_id, or timestamps.
- Do NOT fabricate mentions or quotes; all must be grounded in the provided text.
- If there are no valid ideas, return "ideas": [] and put everything in discarded_ideas.
"""

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reflect and consolidate raw ideas into meeting-level canonical ideas"
    )
    ap.add_argument("--meeting-id", required=True, help="e.g., BAC")
    ap.add_argument(
        "--extract-model",
        default="gptnano",
        help="Model alias used for extraction (to locate ideas_raw_<extract-model>.json)",
    )
    ap.add_argument(
        "--reflect-model",
        default="gptfull",
        help="LLM alias for reflection, e.g. gptfull (mapped to gpt-5)",
    )
    args = ap.parse_args()

    base_dir = get_meeting_base_dir(args.meeting_id)
    ctx_dir = base_dir / "context_outputs"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    ideas_raw_path = ctx_dir / f"ideas_raw_{args.extract_model}.json"
    if not ideas_raw_path.exists():
        raise SystemExit(f"ideas_raw file not found: {ideas_raw_path}")

    raw_text = ideas_raw_path.read_text(encoding="utf-8")
    raw_data = json.loads(raw_text)

    # Prefer metadata from ideas_raw; fall back to metadata.json if missing
    meta: Dict[str, Any] = raw_data.get("metadata") or {}
    if not meta:
        meta = load_metadata(base_dir)

    meeting_id = meta.get("meeting_id", args.meeting_id)
    topic = meta.get("topic", "")
    meeting_type = meta.get("meeting_type", "")
    num_participants = meta.get("num_participants", 0)
    duration_sec = meta.get("duration_sec", 0.0)

    user_prompt = (
        "# METADATA\n"
        f"meeting_id: {meeting_id}\n"
        f"topic: {topic}\n"
        f"meeting_type: {meeting_type}\n"
        f"num_participants: {num_participants}\n"
        f"duration_sec: {duration_sec}\n\n"
        "# RAW_CANDIDATE_IDEAS_JSON\n"
        "Below is the full JSON of candidate ideas per chunk, exactly as produced by the extraction step.\n"
        "Use this as your ONLY source of ideas and mentions.\n\n"
        "```json\n"
        f"{raw_text}\n"
        "```\n"
    )

    client = init_client()

    resp = chat_json(
        client,
        model=args.reflect_model,
        system_prompt=REFLECTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        reasoning_effort="minimal",
        verbosity="low",
    )

    if isinstance(resp, dict):
        resp.setdefault("metadata", {})
        resp["metadata"]["meeting_id"] = meeting_id
        resp["metadata"]["extract_model"] = args.extract_model
        resp["metadata"]["reflect_model"] = args.reflect_model
        resp["metadata"]["topic"] = topic
        resp["metadata"]["context"] = "ideas_reflection_v1"

    out_path = ctx_dir / f"ideas_reflected_{args.extract_model}_{args.reflect_model}.json"
    out_path.write_text(json.dumps(resp, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {out_path}")

if __name__ == "__main__":
    main()
