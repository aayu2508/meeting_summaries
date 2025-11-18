#!/usr/bin/env python3
import json
import argparse
from typing import Dict, Any, List
from .utils.llm_client import init_client, chat_json
from .utils.common import load_metadata, get_meeting_base_dir

SYSTEM_PROMPT = """
You are an idea extraction tool.
Return valid JSON with this schema (no extra keys, no comments):

{
  "chunk_id": "chunk identifier",
  "ideas": [
    {
      "idea_local_id": "local id within this chunk",
      "idea_text": "short description of the idea",
      "mentions": [
        {
          "segment_id": "segment id copied from a TURN line"
        }
      ]
    }
  ]
}

DEFINITIONS
- Extract proposals, solutions, suggestions, or actionable concepts that relate to the meeting topic.
- Skip purely procedural comments (scheduling, breaks) and complete off-topic tangents.
- Be faithful to the source.
- This chunk is only part of the meeting. Extract ideas based ONLY on the turns in this chunk.
- Include things that might be ideas even if you're uncertain - it's better to be inclusive.

RULES FOR TURNS AND MENTIONS
- For each mention, copy the segment_id EXACTLY from the TURN lines.
- A single idea can have multiple mentions (different segment_id).
- Do NOT invent new segment_id values.
- Do NOT reference text that is not present in the provided TURNS.
- Do NOT output an idea if you cannot attach at least one valid segment_id in mentions.

RULES FOR IDEA TEXT
- idea_text should be a clear, concise paraphrase of the idea in 5 to 15 words.
- Do NOT include speaker names or timestamps in idea_text.
- Do NOT invent ideas that are not supported by the TURNS.

RULES FOR IDS
- chunk_id in the output MUST match the provided chunk_id.
- idea_local_id must be unique within this chunk. You may use simple labels like "I1", "I2", "I3", etc.

OTHER RULES
- Use ONLY the provided TURNS and METADATA.
- Stay focused on the meeting TOPIC; if no valid ideas appear in this chunk, return "ideas": [].
- Output STRICT JSON only, with the exact schema above.
"""

USER_TEMPLATE = """# METADATA
meeting_id: {meeting_id}
topic: {topic}
meeting_type: {meeting_type}
num_participants: {num_participants}
duration_sec: {duration_sec}

# CHUNK
chunk_id: {chunk_id}
window: {start:.3f}s-{end:.3f}s

# TURNS
{turns_block}
"""

def _render_turns_block(spans: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for t in spans or []:
        seg_id = t.get("segment_id") or "NA"
        s = float(t.get("start", 0.0))
        e = float(t.get("end", s))
        spk = t.get("speaker", "S?")
        txt = (t.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[segment_id={seg_id} | {s:.3f}-{e:.3f}] {spk}: {txt}")
    return "\n".join(lines)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract raw candidate ideas per chunk (mentions with segment_ids + metadata)"
    )
    ap.add_argument("--meeting-id", required=True, help="e.g., BAC")
    ap.add_argument(
        "--model",
        default="gptnano",
        help="LLM alias for extraction, e.g. gptnano (mapped to gpt-5-nano)",
    )
    args = ap.parse_args()

    base_dir = get_meeting_base_dir(args.meeting_id)
    meta = load_metadata(base_dir)

    chunks_path = base_dir / f"chunks_{args.model}.json"
    if not chunks_path.exists():
        raise SystemExit(f"chunks file not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if isinstance(chunks, dict) and "chunks" in chunks:
        chunks = chunks["chunks"]

    ctx_out_dir = base_dir / "context_outputs"
    ctx_out_dir.mkdir(parents=True, exist_ok=True)
    out_json_path = ctx_out_dir / f"ideas_raw_{args.model}.json"

    client = init_client()

    state: Dict[str, Any] = {
        "metadata": {
            **meta,
            "meeting_id": args.meeting_id,
            "model": args.model,
            "context": "ideas_extraction_raw",
        },
        "chunks": [],
    }

    for c in chunks:
        chunk_id = c.get("chunk_id")
        spans_this_chunk: List[Dict[str, Any]] = c.get("spans", []) or []
        if not chunk_id or not spans_this_chunk:
            continue

        # Build lookup from segment_id -> span dict
        seg_lookup: Dict[str, Dict[str, Any]] = {
            t["segment_id"]: t
            for t in spans_this_chunk
            if t.get("segment_id")
        }

        turns_block = _render_turns_block(spans_this_chunk)

        user_prompt = USER_TEMPLATE.format(
            meeting_id=args.meeting_id,
            topic=meta.get("topic", ""),
            meeting_type=meta.get("meeting_type", ""),
            num_participants=meta.get("num_participants", 0),
            duration_sec=meta.get("duration_sec", 0.0),
            chunk_id=chunk_id,
            start=float(c.get("start", 0.0)),
            end=float(c.get("end", 0.0)),
            turns_block=turns_block,
        )

        resp = chat_json(
            client,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            reasoning_effort="minimal",
            verbosity="low",
        )

        ideas_in = []
        if isinstance(resp, dict):
            ideas_in = resp.get("ideas") or []

        valid_seg_ids = set(seg_lookup.keys())
        clean_ideas: List[Dict[str, Any]] = []

        for idx, it in enumerate(ideas_in):
            idea_text = (it.get("idea_text") or it.get("idea") or "").strip()
            if not idea_text:
                continue

            raw_mentions = it.get("mentions") or []
            clean_mentions: List[Dict[str, Any]] = []
            for m in raw_mentions:
                seg_id = (m.get("segment_id") or "").strip()
                if not seg_id or seg_id not in valid_seg_ids:
                    continue
                seg_info = seg_lookup.get(seg_id, {})
                clean_mentions.append(
                    {
                        "segment_id": seg_id,
                        "speaker": seg_info.get("speaker"),
                        "start": seg_info.get("start"),
                        "end": seg_info.get("end"),
                        "text": (seg_info.get("text") or "").strip(),
                    }
                )

            if not clean_mentions:
                continue

            idea_local_id = f"{chunk_id}_I{idx + 1}"

            clean_ideas.append(
                {
                    "idea_local_id": idea_local_id,
                    "idea_text": idea_text,
                    "mentions": clean_mentions,
                }
            )

        state["chunks"].append(
            {
                "chunk_id": chunk_id,
                "start": float(c.get("start", 0.0)),
                "end": float(c.get("end", 0.0)),
                "ideas": clean_ideas,
            }
        )

    out_json_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {out_json_path}")

if __name__ == "__main__":
    main()