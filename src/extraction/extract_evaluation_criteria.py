#!/usr/bin/env python3
from __future__ import annotations

import json, argparse, csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from .llm_helper import init_client, chat_json

SYSTEM_PROMPT = """You are an evaluation analyst. Return ONLY valid JSON matching the schema and rules.

GOAL
- For a SINGLE idea, induce sensible evaluation DIMENSIONS from the discussion and assess each with a brief rationale.
- Provide both paraphrased key phrases and verbatim quotes (with timestamps/speakers) as evidence.
- Only consider content clearly about the IDEA; ignore other ideas even if they occur in the same time window.
- When possible, REUSE any previously found criteria listed in the prompt; also ADD new criteria if the evidence supports them. Output must still match the schema.

DIMENSIONS
- Do NOT assume a predefined list. Name dimensions concisely in snake_case.
- Prefer generalizable names (e.g., feasibility, cost, risk, user_adoption, regulatory, maintainability, sustainability, partner_dependency, etc.).
- 3-8 dimensions is typical; include more only if clearly justified by the text.

EVIDENCE RULES
- Use ONLY the provided TURNS. Do NOT invent content.
- Include 1-3 VERBATIM quotes per dimension (exact substrings) with start, end, speaker.
- Include 1-3 PARAPHRASED phrases (<= 12 words each) that capture the gist.
- Keep rationales concise (<= 30 words) and grounded in the evidence.

OUTPUT JSON SCHEMA (no extra keys, no comments):
{
  "idea": "...",
  "dimensions": [
    {
      "name": "concise_snake_case_name",
      "rationale": "...",
      "score": 3,
      "quotes": [
        {"quote": "...", "start": 0.0, "end": 0.0, "speaker": "SPEAKER_00", "segment_id": "..."}
      ],
      "paraphrases": ["short phrase", "..."]
    }
  ]
}
"""

USER_TEMPLATE = """# IDEA
{idea}

# PREVIOUS_DIMENSIONS (reuse if relevant; also add new)
{prev_dims_block}

# TURNS (use ONLY these; verbatim quotes must be exact substrings)
{turns_block}
"""

def _render_turns_block(spans: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for t in spans or []:
        s = float(t.get("start", 0.0))
        e = float(t.get("end", s))
        spk = t.get("speaker", "S?")
        txt = (t.get("text") or "").strip().replace("\n", " ")
        cid = t.get("segment_id") or "NA"
        lines.append(f"[{s:.3f}-{e:.3f}] {spk} (chunk={cid}): {txt}")
    return "\n".join(lines)

def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 <= b0 or a0 >= b1)

def _build_other_ideas_masks(
    ideas_doc: Dict[str, Any],
    transcript_by_id: Dict[str, Dict[str, Any]],
    current_idea_index: int
) -> List[Tuple[float, float]]:
    masks: List[Tuple[float, float]] = []
    ideas = ideas_doc.get("ideas", []) or []
    for j, other in enumerate(ideas):
        if j == current_idea_index:
            continue
        for m in (other.get("mentions") or []):
            segid = m.get("segment_id")
            if not segid:
                continue
            seg = transcript_by_id.get(segid)
            if not seg:
                continue
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", s))
            if e > s:
                masks.append((s, e))
    return masks

def _build_turns_from_mentions_transcript_exclusive(
    idea: Dict[str, Any],
    ideas_doc: Dict[str, Any],
    transcript: List[Dict[str, Any]],
    idea_index: int
) -> str:
    
    # Index transcript by segment_id for O(1) lookups
    transcript_by_id: Dict[str, Dict[str, Any]] = {
        t.get("segment_id"): t for t in (transcript or []) if t.get("type") == "speech"
    }

    # Build masks from other ideas' mentions
    other_masks = _build_other_ideas_masks(ideas_doc, transcript_by_id, idea_index)

    spans: List[Dict[str, Any]] = []
    for m in (idea.get("mentions") or []):
        segid = m.get("segment_id")
        if not segid:
            continue
        seg = transcript_by_id.get(segid)
        if not seg:
            continue
        s = float(seg.get("start", 0.0))
        e = float(seg.get("end", s))
        # Exclusive: skip this segment if it overlaps any other-idea mask
        overlapped = False
        for (o0, o1) in other_masks:
            if _intervals_overlap(s, e, o0, o1):
                overlapped = True
                break
        if overlapped:
            continue
        spans.append({
            "start": s,
            "end": e,
            "speaker": seg.get("speaker", "S?"),
            "text": (seg.get("text") or "").strip(),
            "segment_id": seg.get("segment_id") or "T"
        })

    return _render_turns_block(spans)

def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    dims: List[Dict[str, Any]] = []
    for d in (result.get("dimensions") or []):
        # quotes
        qlist: List[Dict[str, Any]] = []
        for e in (d.get("quotes") or []):
            q = (e.get("quote") or "").strip()
            if not q:
                continue
            qlist.append({
                "quote": q,
                "start": float(e.get("start", 0.0)),
                "end": float(e.get("end", 0.0)),
                "speaker": e.get("speaker", ""),
                "segment_id": e.get("segment_id", "")
            })
        # score
        sc = d.get("score")
        try:
            sc = int(sc) if sc is not None else None
        except Exception:
            sc = None
        if sc is not None:
            sc = max(1, min(5, sc))
        dims.append({
            "name": (d.get("name") or "dimension").strip().lower().replace(" ", "_"),
            "score": sc,
            "rationale": (d.get("rationale") or "").strip(),
            "quotes": qlist[:3],
            "paraphrases": [ (p or "").strip() for p in (d.get("paraphrases") or []) ][:3]
        })
    return {"idea": result.get("idea", ""), "dimensions": dims}

def _aggregate_csv_rows(ideas: List[Dict[str, Any]]) -> List[List[Any]]:
    names: List[str] = []
    seen: Set[str] = set()
    for it in ideas:
        for d in it.get("dimensions", []) or []:
            n = d.get("name")
            if n and n not in seen:
                seen.add(n); names.append(n)
    names.sort()
    rows: List[List[Any]] = [["idea"] + names]
    for it in ideas:
        scores = {d.get("name"): d.get("score") for d in it.get("dimensions", []) or []}
        rows.append([it.get("idea", "")] + [scores.get(n) for n in names])
    return rows

def main():
    ap = argparse.ArgumentParser(description="Extract evaluation dimensions per idea (mentions-only, exclusive, chained criteria)")
    ap.add_argument("--meeting-id", required=True, help="e.g., S16")
    ap.add_argument("--model", default="gptfull", help="LLM: gptnano | gpt3.5 | gptfull (aliases supported)")
    ap.add_argument("--write-csv", action="store_true", help="emit a wide CSV of scores across discovered dimensions")
    args = ap.parse_args()

    base_dir = Path("data/outputs") / args.meeting_id
    ctx_out = base_dir / "context_outputs"
    ctx_out.mkdir(parents=True, exist_ok=True)

    ideas_path = ctx_out / f"ideas_windows_{args.model}.json"
    if not ideas_path.exists():
        raise SystemExit(f"ideas file not found: {ideas_path}")

    transcript_path = base_dir / "transcript_final.json"
    if not transcript_path.exists():
        raise SystemExit(f"transcript file not found: {transcript_path}")

    # Output paths
    out_json = ctx_out / f"eval_criteria_{args.model}.json"
    per_idea_dir = ctx_out / f"eval_by_idea_{args.model}"
    per_idea_dir.mkdir(parents=True, exist_ok=True) 

    # Load inputs
    ideas_doc = json.loads(ideas_path.read_text(encoding="utf-8"))
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    if isinstance(transcript, dict) and "segments" in transcript:
        transcript = transcript["segments"]

    client = init_client()

    running_criteria: List[str] = []
    running_criteria_set: Set[str] = set()

    combined_items: List[Dict[str, Any]] = []

    # Iterate ideas in the given order
    for idx, idea in enumerate(ideas_doc.get("ideas", []) or []):
        # Build mentions-only, exclusive turns from transcript
        turns_block = _build_turns_from_mentions_transcript_exclusive(
            idea=idea,
            ideas_doc=ideas_doc,
            transcript=transcript,
            idea_index=idx
        )

        # If nothing to show (e.g., fully overlapped), skip to avoid random outputs
        if not turns_block.strip():
            continue

        prev_dims_block = "\n".join(running_criteria) if running_criteria else "(none)"

        user_prompt = USER_TEMPLATE.format(
            idea=idea.get("idea", ""),
            prev_dims_block=prev_dims_block,
            turns_block=turns_block,
        )

        resp = chat_json(
            client,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            reasoning_effort="minimal",
            verbosity="low"
        )

        if not isinstance(resp, dict):
            continue

        norm = _normalize_result(resp)
        combined_items.append(norm)

        fname = (idea.get("idea", "idea").strip() or "idea").lower()
        fname = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in fname)[:80]
        (per_idea_dir / f"{fname}.json").write_text(
            json.dumps({
                "metadata": {"meeting_id": args.meeting_id, "model": args.model, "context": "transcript_mentions_exclusive"},
                "idea": norm.get("idea", ""),
                "dimensions": norm.get("dimensions", []),
            }, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        # Update running criteria (order-preserving)
        for d in norm.get("dimensions", []) or []:
            n = (d.get("name") or "").strip()
            if n and n not in running_criteria_set:
                running_criteria.append(n)
                running_criteria_set.add(n)

    # --- Combined meeting-level JSON ---
    out_json.write_text(
        json.dumps({
            "metadata": {"meeting_id": args.meeting_id, "model": args.model, "context": "transcript_mentions_exclusive"},
            "ideas": combined_items
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if getattr(args, "write_csv", False):
        rows = _aggregate_csv_rows(combined_items)
        csv_path = out_json.with_name(out_json.stem + "_scores.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"Wrote matrix CSV: {csv_path}")

    print("[criteria_json]", json.dumps(running_criteria, ensure_ascii=False))
    print(f"[done] {out_json}")


if __name__ == "__main__":
    main()
