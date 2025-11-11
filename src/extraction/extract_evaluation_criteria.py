#!/usr/bin/env python3
from __future__ import annotations

import json, re, argparse, csv
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from .llm_helper import init_client, chat_json

SYSTEM_PROMPT = """You are an evaluation analyst. Return ONLY valid JSON matching the schema and rules.

GOAL
- For a SINGLE idea, induce sensible evaluation DIMENSIONS from the discussion and assess each with a brief rationale. 
- Provide both paraphrased key phrases and verbatim quotes (with timestamps/speakers) as evidence.

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
        {"quote": "...", "start": 0.0, "end": 0.0, "speaker": "SPEAKER_00", "chunk_id": "..."}
      ],
      "paraphrases": ["short phrase", "..."]
    }
  ]
}
"""

USER_TEMPLATE = """# IDEA
{idea}

# TURNS (use ONLY these; verbatim quotes must be exact substrings)
{turns_block}
"""

def _render_turns_block(spans: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for t in spans or []:
        if t.get("type") and t["type"] != "speech":
            continue
        s = float(t.get("start", 0.0))
        e = float(t.get("end", s))
        spk = t.get("speaker", "S?")
        txt = (t.get("text") or "").strip().replace("\n", " ")
        cid = t.get("chunk_id") or t.get("cid") or t.get("segment_id") or "NA"
        lines.append(f"[{s:.3f}-{e:.3f}] {spk} (chunk={cid}): {txt}")
    return "\n".join(lines)


def _clip_spans_to_window(spans: List[Dict[str, Any]], w0: float, w1: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in spans or []:
        st = float(s.get("start", 0.0)); en = float(s.get("end", st))
        if en < w0 or st > w1:
            continue
        out.append({**s, "start": max(st, w0), "end": min(en, w1)})
    return out


def _build_turns_from_chunks(idea: Dict[str, Any], chunk_by_id: Dict[str, Any]) -> str:
    lines: List[str] = []
    for win in idea.get("windows", []) or []:
        if len(win) < 3:
            continue
        w0, w1, cid = float(win[0]), float(win[1]), str(win[2])
        ch = chunk_by_id.get(cid)
        if not ch:
            continue
        spans = [s for s in ch.get("spans", []) if s.get("type") == "speech"]
        spans = _clip_spans_to_window(spans, w0, w1)
        block = _render_turns_block(spans)
        if block:
            lines.append(block)
    return "\n".join(lines)


def _build_turns_from_transcript(idea: Dict[str, Any], transcript: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for win in idea.get("windows", []) or []:
        if len(win) < 2:
            continue
        w0, w1 = float(win[0]), float(win[1])
        spans: List[Dict[str, Any]] = []
        for t in transcript or []:
            if t.get("type") and t["type"] != "speech":
                continue
            s = float(t.get("start", 0.0)); e = float(t.get("end", s))
            if e < w0 or s > w1:
                continue
            spans.append({
                "start": max(s, w0), "end": min(e, w1),
                "speaker": t.get("speaker", "S?"),
                "text": (t.get("text") or "").strip(),
                "chunk_id": t.get("segment_id") or "T"
            })
        block = _render_turns_block(spans)
        if block:
            lines.append(block)
    return "\n".join(lines)


def _normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    dims: List[Dict[str, Any]] = []
    for d in (result.get("dimensions") or []):
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
                "chunk_id": e.get("chunk_id", "")
            })
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
    ap = argparse.ArgumentParser(description="Extract evaluation dimensions per idea (quotes + paraphrases)")
    ap.add_argument("--meeting-id", required=True, help="e.g., S16")
    ap.add_argument("--model", default="gptnano", help="LLM: gptnano | gpt3.5 | gptfull (aliases supported)")
    ap.add_argument("--context", choices=["chunks", "transcript"], required=True)
    ap.add_argument("--per-idea-dir", action="store_true", help="also write per-idea JSON files")
    ap.add_argument("--write-csv", action="store_true", help="emit a wide CSV of scores across discovered dimensions")
    args = ap.parse_args()

    base_dir = Path("data/outputs") / args.meeting_id
    ctx_out = base_dir / "context_outputs"
    ctx_out.mkdir(parents=True, exist_ok=True)

    ideas_path = ctx_out / f"ideas_windows_{args.model}.json"
    if not ideas_path.exists():
        raise SystemExit(f"ideas file not found: {ideas_path}")

    if args.context == "chunks":
        context_path = base_dir / f"chunks_{args.model}.json"
        if not context_path.exists():
            raise SystemExit(f"chunks file not found: {context_path}")
    else:
        context_path = base_dir / f"transcript_final.json"
        if not context_path.exists():
            raise SystemExit(f"transcript file not found: {context_path}")

    out_json = ctx_out / f"eval_criteria_{args.model}.json"
    per_idea_dir = ctx_out / f"eval_by_idea_{args.model}"

    ideas_doc = json.loads(ideas_path.read_text(encoding="utf-8"))

    if args.context == "chunks":
        ctx = json.loads(context_path.read_text(encoding="utf-8"))
        if isinstance(ctx, dict) and "chunks" in ctx:
            ctx = ctx["chunks"]
        chunk_by_id: Dict[str, Any] = {}
        for ch in ctx or []:
            cid = ch.get("chunk_id") or ch.get("id")
            if cid:
                chunk_by_id[str(cid)] = ch
    else:
        transcript = json.loads(context_path.read_text(encoding="utf-8"))
        if isinstance(transcript, dict) and "segments" in transcript:
            transcript = transcript["segments"]

    client = init_client()

    combined_items: List[Dict[str, Any]] = []
    if args.per_idea_dir:
        per_idea_dir.mkdir(parents=True, exist_ok=True)

    for idea in ideas_doc.get("ideas", []) or []:
        if args.context == "chunks":
            turns_block = _build_turns_from_chunks(idea, chunk_by_id)
        else:
            turns_block = _build_turns_from_transcript(idea, transcript)

        user_prompt = USER_TEMPLATE.format(
            idea=idea.get("idea", ""),
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

        if args.per_idea_dir:
            fname = (idea.get("idea", "idea").strip() or "idea").lower()
            fname = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in fname)[:80]
            (per_idea_dir / f"{fname}.json").write_text(
                json.dumps({
                    "metadata": {"meeting_id": args.meeting_id, "model": args.model, "context": args.context},
                    "idea": norm.get("idea", ""),
                    "dimensions": norm.get("dimensions", []),
                }, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

    out_json.write_text(
        json.dumps({
            "metadata": {"meeting_id": args.meeting_id, "model": args.model, "context": args.context},
            "ideas": combined_items
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if args.write_csv:
        rows = _aggregate_csv_rows(combined_items)
        csv_path = out_json.with_name(out_json.stem + "_scores.csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"Wrote matrix CSV: {csv_path}")

    print(f"[done] {out_json}")


if __name__ == "__main__":
    main()
