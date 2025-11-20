# expand_idea_mentions.py
from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Set

from .utils.llm_client import init_client, chat_json
from .utils.common import get_meeting_base_dir

SYSTEM_PROMPT = """
You identify all transcript segments that refer to this idea, either explicitly or implicitly.”

You receive:
- A SINGLE IDEA description for a meeting.
- A list of segment_ids that are ALREADY known to explicitly mention this idea.
- The FULL MEETING transcript as a list of TURNS, where each TURN has a segment_id, timestamps, speaker, and text.

Your tasks:
1) EXTRA EXPLICIT MENTIONS
- Find additional TURNS that clearly refer to the SAME IDEA.
- These may:
  - restate the idea,
  - paraphrase it,
  - refer back to it.
- Tag these as role = "explicit".
- DO NOT include any segment_id that is already in the known core list.
- DO NOT include generic talk that could be about many or other ideas.

2) CONTEXT SEGMENTS
- Find TURNS that do NOT directly describe the idea but provide IMPORTANT CONTEXT for evaluating it.
- Context includes constraints, properties, or tradeoffs that obviously matter for this idea.
Examples:
  - Earlier discussion about things involved in the idea.
  - Safety, regulations, cost structure, or user constraints that are clearly relevant.
- Tag these as role = "context".

IMPORTANT RULES
- If you are unsure if it is about the idea, DO NOT tag the segment.
- Use ONLY segment_id values that appear in the TURNS section.
- Never invent new segment_id values.
- If you find nothing, return empty lists.

OUTPUT JSON (no extra keys, no comments):
{
  "idea_id": "I1",
  "extra_mentions": [
    { "segment_id": "m000190", "role": "explicit" },
    { "segment_id": "m000050", "role": "context" }
  ]
}
"""

USER_TEMPLATE = """# IDEA
idea_id: {idea_id}
canonical_idea: {canonical_idea}

source_idea_texts:
{source_idea_texts_block}

# KNOWN_CORE_MENTIONS
{known_core_block}

# TURNS
{turns_block}
"""

def _render_turns_block(turns: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for t in turns:
        seg_id = t.get("segment_id") or "NA"
        s = float(t.get("start", 0.0))
        e = float(t.get("end", s))
        spk = t.get("speaker", "S?")
        txt = (t.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[segment_id={seg_id} | {s:.3f}-{e:.3f}] {spk}: {txt}")
    return "\n".join(lines)


def _build_turns_from_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for c in chunks:
        chunk_id = c.get("chunk_id")
        spans = c.get("spans") or []
        for s in spans:
            seg_id = s.get("segment_id")
            if not seg_id:
                continue
            start = float(s.get("start", 0.0))
            end = float(s.get("end", start))
            turns.append(
                {
                    "segment_id": seg_id,
                    "chunk_id": chunk_id,
                    "speaker": s.get("speaker", "S?"),
                    "start": start,
                    "end": end,
                    "text": (s.get("text") or "").strip(),
                }
            )
    turns.sort(key=lambda t: float(t.get("start", 0.0)))
    return turns

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Expand idea mentions by scanning ALL chunks"
        )
    )
    ap.add_argument("--meeting-id", required=True, help="e.g., BAC")
    ap.add_argument(
        "--extract-model",
        default="gptnano",
        help="Model alias used for idea extraction (for reflected filename).",
    )
    ap.add_argument(
        "--reflect-model",
        default="gptfull",
        help="Model alias used for reflection (for reflected filename).",
    )
    ap.add_argument(
        "--chunks-model",
        default="gptnano",
        help="Model alias used for chunking (to locate chunks_<model>.json). "
    )
    ap.add_argument(
        "--expansion-model",
        default="gptnano",
        help="LLM alias used for mention expansion, e.g., gptnano.",
    )
    args = ap.parse_args()

    base_dir = get_meeting_base_dir(args.meeting_id)
    ctx_dir = base_dir / "context_outputs"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    # Load reflected ideas
    ideas_path = ctx_dir / f"ideas_reflected_{args.extract_model}_{args.reflect_model}.json"
    if not ideas_path.exists():
        raise SystemExit(f"reflected ideas file not found: {ideas_path}")

    ideas_doc = json.loads(ideas_path.read_text(encoding="utf-8"))
    ideas = ideas_doc.get("ideas") or []
    metadata = ideas_doc.get("metadata") or {}

    if not ideas:
        raise SystemExit(f"No ideas found in {ideas_path}")

    chunks_path = base_dir / f"chunks_{args.chunks_model}.json"
    if not chunks_path.exists():
        raise SystemExit(f"chunks file not found: {chunks_path}")

    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks: List[Dict[str, Any]] = raw_chunks or []
    if not chunks:
        raise SystemExit(f"No chunks found in {chunks_path}")

    turns = _build_turns_from_chunks(chunks)
    turns_block = _render_turns_block(turns)
    seg_lookup: Dict[str, Dict[str, Any]] = {t["segment_id"]: t for t in turns}

    client = init_client()

    # Iterate over ideas and expand mentions for each
    for idea in ideas:
        idea_id = idea.get("idea_id") or "NO_ID"
        canonical_idea = (idea.get("canonical_idea") or "").strip()
        source_texts = idea.get("source_idea_texts") or []

        # Known core mentions from reflection
        core_mentions_in = idea.get("mentions") or []
        core_ids: List[str] = []
        known_core_lines: List[str] = []

        for m in core_mentions_in:
            sid = (m.get("segment_id") or "").strip()
            if not sid:
                continue
            core_ids.append(sid)

            # Prefer text in the idea object, else fall back to seg_lookup
            txt = (m.get("text") or "").strip()
            if not txt:
                seg = seg_lookup.get(sid)
                if seg:
                    txt = (seg.get("text") or "").strip()
            if not txt:
                txt = "(no_text_available)"

            known_core_lines.append(f"{sid}: {txt}")

        known_core_block = "\n".join(known_core_lines) if known_core_lines else "(none)"

        source_idea_texts_block = "\n".join(
            f"- {t}" for t in source_texts if t
        ) or "- (none)"

        user_prompt = USER_TEMPLATE.format(
            idea_id=idea_id,
            canonical_idea=canonical_idea or "(none)",
            source_idea_texts_block=source_idea_texts_block,
            known_core_block=known_core_block,
            turns_block=turns_block,
        )

        resp = chat_json(
            client,
            model=args.expansion_model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            reasoning_effort="minimal",
            verbosity="low",
        )

        extra_mentions: List[Dict[str, Any]] = []

        if isinstance(resp, dict):
            if resp.get("idea_id") and str(resp.get("idea_id")) != str(idea_id):
                pass

            for em in (resp.get("extra_mentions") or []):
                sid = (em.get("segment_id") or "").strip()
                if not sid:
                    continue

                # Skip if already in core
                if sid in core_ids:
                    continue

                role_raw = (em.get("role") or "explicit").strip().lower()
                role = "context" if role_raw == "context" else "explicit"

                seg = seg_lookup.get(sid)
                if not seg:
                    continue

                extra_mentions.append(
                    {
                        "segment_id": sid,
                        "chunk_id": seg.get("chunk_id"),
                        "speaker": seg.get("speaker", "S?"),
                        "start": float(seg.get("start", 0.0)),
                        "end": float(seg.get("end", 0.0)),
                        "text": (seg.get("text") or "").strip(),
                        "role": role,
                    }
                )

        # Attach to idea
        idea["mentions_extra"] = extra_mentions

    # Update metadata
    metadata["context"] = "ideas_reflection_with_expanded_mentions"
    metadata["expansion_model"] = args.expansion_model

    out_doc = {
        "metadata": metadata,
        "ideas": ideas,
    }

    out_name = f"ideas_reflected_expanded_{args.extract_model}_{args.reflect_model}_{args.expansion_model}.json"
    out_path = ctx_dir / out_name
    out_path.write_text(
        json.dumps(out_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[done] wrote expanded ideas to {out_path}")

if __name__ == "__main__":
    main()
