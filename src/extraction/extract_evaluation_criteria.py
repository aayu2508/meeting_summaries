#extract_evaluation_criteria.py
from __future__ import annotations

import json, argparse, csv
from pathlib import Path
from typing import Dict, Any, List, Set

from .utils.llm_client import init_client, chat_json
from .utils.common import get_meeting_base_dir, load_metadata


SYSTEM_PROMPT = """
You are an evaluation-dimension extractor for a SINGLE IDEA in a meeting at a time.

WHAT YOU RECEIVE
- A single IDEA (canonical_idea and optional source_idea_texts).
- A list of PREVIOUS_DIMENSIONS but you don't need to reuse it if you don't find it being talked about in an idea.
- A block of TURNS. Each TURN has segment_id, timestamps, speaker, and text.
- These TURNS are ALL the segments that are about this idea or closely related context.

YOUR JOB
For THIS idea only, infer a SET of evaluation DIMENSIONS that participants actually use
(or clearly imply) when they talk about the idea.

A dimension is an abstract lens used to judge, reason or evaluate about the idea, such as:
  - expressing a benefit or advantage,
  - noting a drawback, risk, or difficulty,
  - questioning practicality or scale,
  - comparing against another approach,
  - talking about whether people would adopt/use it.
- If you cannot find clear evaluative content for a candidate lens, do NOT include that dimension.
- It is acceptable to return 0 dimensions if the TURNS do not contain any real evaluation.
- Prefer generalizable names that make sense (e.g., feasibility, cost, risk, user_adoption, regulatory, maintainability, sustainability, partner_dependency, etc.).

QUOTES AND EVIDENCE
- For each dimension:
  - Provide a short rationale (<= 30 words) that explains why this dimension matters for this idea,
    grounded in what people actually say.
  - Provide 1-3 verbatim quotes as evidence.
- Quotes MUST follow all of these rules:
  - They must be exact substrings of the TURNS text (you may trim small fillers, but do not rewrite).
  - They must come ONLY from the TURNS block, not from metadata or source_idea_texts.
  - Each quote must include: quote, start, end, speaker, segment_id.
- Do NOT invent segment_id values.
- Do NOT treat source_idea_texts as if they were spoken quotes; they are background only.

OUTPUT JSON SCHEMA (no extra keys, no comments):
{
  "idea": "short description of the idea",
  "dimensions": [
    {
      "name": "concise_snake_case_name",
      "rationale": "why this dimension is relevant for evaluating this idea",
      "quotes": [
        {
          "quote": "...",
          "start": 0.0,
          "end": 0.0,
          "speaker": "SPEAKER_00",
          "segment_id": "..."
        }
      ]
    }
  ]
}
"""

USER_TEMPLATE = """# IDEA
idea_id: {idea_id}
canonical_idea: {canonical_idea}

source_idea_texts:
{source_idea_texts_block}

# PREVIOUS_DIMENSIONS
(reuse these names when appropriate; add new ones only if needed)
{prev_dims_block}

# TURNS
(use ONLY these; verbatim quotes must be exact substrings of this text)
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


def _build_spans_from_mentions(idea: Dict[str, Any]) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for key in ("mentions", "mentions_extra"):
        for m in (idea.get(key) or []):
            sid = (m.get("segment_id") or "").strip()
            if not sid or sid in seen:
                continue
            seen.add(sid)
            start = float(m.get("start", 0.0))
            end = float(m.get("end", start))
            spans.append(
                {
                    "segment_id": sid,
                    "start": start,
                    "end": end,
                    "speaker": m.get("speaker", "S?"),
                    "text": (m.get("text") or "").strip(),
                }
            )

    spans.sort(key=lambda t: float(t.get("start", 0.0)))
    return spans


def _normalize_result(result: Dict[str, Any], canonical_idea: str) -> Dict[str, Any]:
    dims: List[Dict[str, Any]] = []
    for d in (result.get("dimensions") or []):
        name_raw = (d.get("name") or "dimension").strip()
        name = name_raw.lower().replace(" ", "_")

        rationale = (d.get("rationale") or "").strip()

        qlist: List[Dict[str, Any]] = []
        for e in (d.get("quotes") or []):
            q = (e.get("quote") or "").strip()
            if not q:
                continue
            qlist.append(
                {
                    "quote": q,
                    "start": float(e.get("start", 0.0)),
                    "end": float(e.get("end", 0.0)),
                    "speaker": (e.get("speaker") or "").strip(),
                    "segment_id": (e.get("segment_id") or "").strip(),
                }
            )

        dims.append(
            {
                "name": name,
                "rationale": rationale,
                "quotes": qlist[:3],
            }
        )

    return {
        "idea": canonical_idea,
        "dimensions": dims,
    }


def _aggregate_csv_rows(ideas: List[Dict[str, Any]]) -> List[List[Any]]:
    names: List[str] = []
    seen: Set[str] = set()
    for it in ideas:
        for d in (it.get("dimensions") or []):
            n = d.get("name")
            if n and n not in seen:
                seen.add(n)
                names.append(n)
    names.sort()

    rows: List[List[Any]] = [["idea_id", "idea"] + names]
    for it in ideas:
        idea_id = it.get("idea_id", "")
        idea_str = it.get("idea", "")
        present = {d.get("name") for d in (it.get("dimensions") or [])}
        rows.append(
            [idea_id, idea_str] + [1 if n in present else "" for n in names]
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract evaluation dimensions per idea from mentions-only JSON."
    )
    ap.add_argument("--meeting-id", required=True, help="e.g., BAC")
    ap.add_argument(
        "--ideas-json",
        required=True,
        help="Path (relative to context_outputs or absolute) to ideas JSON with mentions + mentions_extra.",
    )
    ap.add_argument(
        "--model",
        default="gptfull",
        help="LLM alias for evaluation, e.g., gptfull (mapped to gpt-5).",
    )
    ap.add_argument(
        "--write-csv",
        action="store_true",
        help="Emit a wide CSV of dimension presence across ideas.",
    )
    args = ap.parse_args()

    base_dir = get_meeting_base_dir(args.meeting_id)
    ctx_out = base_dir / "context_outputs"
    ctx_out.mkdir(parents=True, exist_ok=True)

    # Resolve ideas JSON path
    ideas_path = Path(args.ideas_json)
    if not ideas_path.is_absolute():
        ideas_path = ctx_out / ideas_path

    if not ideas_path.exists():
        raise SystemExit(f"ideas file not found: {ideas_path}")

    ideas_doc = json.loads(ideas_path.read_text(encoding="utf-8"))
    ideas_list = ideas_doc.get("ideas") or []

    if not ideas_list:
        raise SystemExit(f"No ideas found in {ideas_path}")

    # Metadata: prefer ideas_doc metadata, fall back to metadata.json
    meta: Dict[str, Any] = ideas_doc.get("metadata") or {}
    if not meta:
        meta = load_metadata(base_dir)

    meeting_id = meta.get("meeting_id", args.meeting_id)
    topic = meta.get("topic", "")
    meeting_type = meta.get("meeting_type", "")
    num_participants = meta.get("num_participants", 0)
    duration_sec = meta.get("duration_sec", 0.0)

    client = init_client()

    running_dims: List[str] = []
    running_dims_set: Set[str] = set()

    combined_items: List[Dict[str, Any]] = []

    for idea in ideas_list:
        idea_id = idea.get("idea_id") or "NO_ID"
        canonical_idea = (idea.get("canonical_idea") or "").strip() or idea_id

        # Build spans from mentions & mentions_extra
        spans = _build_spans_from_mentions(idea)
        if not spans:
            # No text to show; skip to avoid hallucinated dimensions
            continue

        turns_block = _render_turns_block(spans)

        # Previous dimensions: one per line, or (none)
        prev_dims_block = (
            "\n".join(running_dims) if running_dims else "(none)"
        )

        source_texts = idea.get("source_idea_texts") or []
        source_idea_texts_block = (
            "\n".join(f"- {t}" for t in source_texts if t) or "- (none)"
        )

        user_prompt = USER_TEMPLATE.format(
            idea_id=idea_id,
            canonical_idea=canonical_idea,
            source_idea_texts_block=source_idea_texts_block,
            prev_dims_block=prev_dims_block,
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

        if not isinstance(resp, Dict):
            continue

        norm = _normalize_result(resp, canonical_idea)
        norm["idea_id"] = idea_id
        combined_items.append(norm)

        # Update running dimension list (order-preserving)
        for d in norm.get("dimensions") or []:
            n = (d.get("name") or "").strip()
            if n and n not in running_dims_set:
                running_dims.append(n)
                running_dims_set.add(n)

    # --- Combined meeting-level JSON ---
    out_json = ctx_out / f"eval_criteria_{args.model}.json"
    out_payload = {
        "metadata": {
            "meeting_id": meeting_id,
            "model": args.model,
            "topic": topic,
            "meeting_type": meeting_type,
            "num_participants": num_participants,
            "duration_sec": duration_sec,
            "context": "idea_eval_dimensions_v1",
        },
        "ideas": combined_items,
    }
    out_json.write_text(
        json.dumps(out_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if getattr(args, "write_csv", False):
        rows = _aggregate_csv_rows(combined_items)
        csv_path = out_json.with_name(out_json.stem + "_matrix.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"[csv] wrote dimension presence matrix: {csv_path}")

    print("[dims]", json.dumps(running_dims, ensure_ascii=False))
    print(f"[done] {out_json}")


if __name__ == "__main__":
    main()
