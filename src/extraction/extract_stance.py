#!/usr/bin/env python3
from __future__ import annotations
import json, argparse, re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from utils.llm_client import init_client, chat_json


STANCE_SYSTEM = """Label each SPEAKER's stance toward the IDEA.

Rules:
- Use ONLY the provided SEGMENTS (mentions for this idea + a few segments right after).
- Return ONE entry per speaker label that appears in SEGMENTS (use the exact keys).
- Polarity can be any sensible string (e.g., positive, negative, mixed, neutral, unknown).
- Provide quotes (<=20-word verbatim) for the speaker (may be empty if none) and the segment_id it came from.

Return ONLY JSON like:
{"stance": { "<speaker>": {"polarity":"...", "quote":"...", "segment_id":"..."}, ... }}
"""

COMMON_GROUND_SYSTEM = """Decide if speakers reached COMMON GROUND on the IDEA.

Lenient rule:
- reached=true if at least two different speakers show positive/mixed stance;
  OR if one is clearly positive and another gives explicit acknowledgment/commitment in adjacent segments.
- Otherwise false.

Return ONLY:
{"common_ground": {"reached": true, "summary": "<1–2 sentences>"}}
"""


def _intervals_overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return not (a1 <= b0 or a0 >= b1)

def _index_transcript_by_id(transcript: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        t.get("segment_id"): t
        for t in (transcript or [])
        if t and t.get("type", "speech") == "speech" and t.get("segment_id")
    }

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

def _mentions_exclusive_segments(
    idea: Dict[str, Any],
    ideas_doc: Dict[str, Any],
    transcript: List[Dict[str, Any]],
    idea_index: int
) -> List[Dict[str, Any]]:

    transcript_by_id = _index_transcript_by_id(transcript)
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
        overlapped = any(_intervals_overlap(s, e, o0, o1) for (o0, o1) in other_masks)
        if overlapped:
            continue
        spans.append({
            "start": s,
            "end": e,
            "speaker": seg.get("speaker", "S?"),
            "text": (seg.get("text") or "").strip(),
            "segment_id": seg.get("segment_id")
        })
    spans.sort(key=lambda t: (t["start"], t["end"]))
    return spans

def _collect_adjacent_segments_after(
    segments_all: List[Dict[str, Any]],
    last_end: float,
    limit_n: int,
    max_window: float,
    exclude_ids: Set[str]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in segments_all:
        if s.get("type","speech") != "speech":
            continue
        if float(s.get("start", 0.0)) + 1e-9 < last_end:
            continue
        if s.get("segment_id") in exclude_ids:
            continue
        if float(s.get("start", 0.0)) - last_end > max_window:
            break
        out.append({
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "speaker": s.get("speaker", "S?"),
            "text": (s.get("text") or "").strip(),
            "segment_id": s.get("segment_id")
        })
        if len(out) >= limit_n:
            break
    return out

def build_segments_for_idea_with_adjacent(
    idea: Dict[str, Any],
    ideas_doc: Dict[str, Any],
    transcript: List[Dict[str, Any]],
    idea_index: int,
    adjacent_segments: int,
    adjacent_window: float
) -> List[Dict[str, Any]]:
    core = _mentions_exclusive_segments(idea, ideas_doc, transcript, idea_index)
    if not core:
        return []
    last_end = max(float(s["end"]) for s in core)
    exclude_ids = {s["segment_id"] for s in core if s.get("segment_id")}
    # transcript assumed chronological; if not, sort by start
    transcript_sorted = sorted(
        (t for t in transcript if t and t.get("type","speech")=="speech"),
        key=lambda t: (float(t.get("start",0.0)), float(t.get("end",0.0)))
    )
    tail = _collect_adjacent_segments_after(
        transcript_sorted, last_end, adjacent_segments, adjacent_window, exclude_ids
    )
    merged = core + tail
    # de-dup just in case
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for s in merged:
        sid = s.get("segment_id")
        if sid and sid in seen:
            continue
        seen.add(sid)
        out.append(s)
    out.sort(key=lambda t: (t["start"], t["end"]))
    return out

def _speakers_in(segments: List[Dict[str, Any]]) -> List[str]:
    return sorted({s.get("speaker","") for s in segments if s.get("speaker")})

def _ensure_all_present_speakers(stance_obj: Dict[str, Any], speakers: List[str]) -> Dict[str, Any]:
    stance_obj = stance_obj if isinstance(stance_obj, dict) else {}
    stance = stance_obj.get("stance")
    if not isinstance(stance, dict):
        stance = {}
    for spk in speakers:
        if spk not in stance:
            stance[spk] = {"polarity":"unknown","quote":"","segment_id":""}
    return {"stance": stance}

ACK_PATTERNS = [r"\byes\b", r"\byep\b", r"\byeah\b", r"\bright\b", r"\bexactly\b",
                r"\bagreed?\b", r"\bmakes sense\b", r"\blet'?s\b", r"\bwe('ll| will)\b", r"\bok(ay)?\b"]

def _is_ack_text(t: str) -> bool:
    t = (t or "").lower()
    return any(re.search(p, t) for p in ACK_PATTERNS)

def _heuristic_common_ground(stance_obj: Dict[str, Any], segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Broad + lenient: >=2 speakers with positive/mixed OR positive + explicit ack/commitment in last two segments.
    stance = (stance_obj or {}).get("stance", {})
    pos_like: Set[str] = set()
    for spk, v in stance.items():
        pol = str((v or {}).get("polarity","")).lower()
        if "positive" in pol or "mixed" in pol or pol in {"pos","mix"}:
            pos_like.add(spk)
    if len(pos_like) >= 2:
        return {"common_ground": {"reached": True, "summary": "At least two speakers show positive/mixed stance."}}
    last_two = segments[-2:] if len(segments) >= 2 else segments[-1:]
    ack = any(_is_ack_text(s.get("text","")) for s in last_two)
    if len(pos_like) >= 1 and ack:
        return {"common_ground": {"reached": True, "summary": "Positive stance with explicit acknowledgment/commitment."}}
    # fallback negatives
    if not stance:
        return {"common_ground": {"reached": False, "summary": "No stance information."}}
    if len(stance) == 1:
        return {"common_ground": {"reached": False, "summary": "Only one speaker evaluated; no alignment evidence."}}
    return {"common_ground": {"reached": False, "summary": "Insufficient alignment across speakers."}}


def _slug(s: str, maxlen: int = 80) -> str:
    s = (s or "").strip().lower()
    s = "".join(c if c.isalnum() or c in (" ","-","_") else "-" for c in s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s[:maxlen] or "idea"

def process_single_idea(
    client,
    model: str,
    idea: Dict[str, Any],
    ideas_doc: Dict[str, Any],
    transcript: List[Dict[str, Any]],
    idea_index: int,
    adjacent_segments: int,
    adjacent_window: float,
    use_heuristic_cg: bool
) -> Dict[str, Any]:
    idea_text = idea.get("idea","").strip()
    segments = build_segments_for_idea_with_adjacent(
        idea=idea,
        ideas_doc=ideas_doc,
        transcript=transcript,
        idea_index=idea_index,
        adjacent_segments=adjacent_segments,
        adjacent_window=adjacent_window
    )
    if not segments:
        return {
            "idea": idea_text,
            "stance": {},
            "common_ground": {"reached": False, "summary": "No segments available for this idea (mentions-exclusive)."}
        }

    spk_list = _speakers_in(segments)

    # --- Pass A: Stance (lenient, multi-speaker) ---
    stance_user = json.dumps({
        "IDEA": idea_text,
        "INSTRUCTION": "Use the exact speaker labels present in SEGMENTS. Do not invent speakers.",
        "SEGMENTS": segments,
        "OUTPUT_EXAMPLE": {"stance": {spk: {"polarity":"positive","quote":"...","segment_id":"..."} for spk in spk_list}}
    }, ensure_ascii=False)

    stance_raw = chat_json(
        client,
        model=model,
        system_prompt=STANCE_SYSTEM,
        user_prompt=stance_user,
        reasoning_effort="minimal",
        verbosity="low"
    )
    stance_obj = _ensure_all_present_speakers(stance_raw, spk_list)

    # --- Pass B: Common Ground (lenient) ---
    if use_heuristic_cg:
        cg_obj = _heuristic_common_ground(stance_obj, segments)
    else:
        cg_user = json.dumps({
            "IDEA": idea_text,
            "STANCE": stance_obj,     # whatever model returned (now with all present speakers)
            "SEGMENTS": segments,
            "OUTPUT_EXAMPLE": {"common_ground": {"reached": True, "summary": "Both endorse feasibility."}}
        }, ensure_ascii=False)

        cg_raw = chat_json(
            client,
            model=model,
            system_prompt=COMMON_GROUND_SYSTEM,
            user_prompt=cg_user,
            reasoning_effort="minimal",
            verbosity="low"
        )
        # very light merge
        cg = cg_raw.get("common_ground") if isinstance(cg_raw, dict) else {}
        if not isinstance(cg, dict):
            cg = {}
        reached = bool(cg.get("reached", False))
        summary = str(cg.get("summary","")).strip() or "No summary."
        cg_obj = {"common_ground": {"reached": reached, "summary": summary}}

    return {"idea": idea_text, "stance": stance_obj.get("stance", {}), "common_ground": cg_obj.get("common_ground", {})}


def main():
    ap = argparse.ArgumentParser(description="Per-idea stance + common-ground (lenient, multi-speaker, mentions-only)")
    ap.add_argument("--meeting-id", required=True, help="e.g., BAC")
    ap.add_argument("--model", default="gptfull", help="LLM alias (e.g., gptnano | gpt3.5 | gptfull)")
    ap.add_argument("--adjacent-segments", type=int, default=1, help="# of segments after last mention to include")
    ap.add_argument("--adjacent-window", type=float, default=15.0, help="Max seconds after last mention")
    ap.add_argument("--use-heuristic-cg", action="store_true", help="Compute common ground with heuristic instead of LLM")
    ap.add_argument("--out-dir", default=None, help="Defaults to data/outputs/<MEETING_ID>/context_outputs")
    args = ap.parse_args()

    base_dir = Path("data/outputs") / args.meeting_id
    ctx_out = Path(args.out_dir) if args.out_dir else (base_dir / "context_outputs")
    ctx_out.mkdir(parents=True, exist_ok=True)

    ideas_path = ctx_out / f"ideas_windows_{args.model}.json"
    if not ideas_path.exists():
        raise SystemExit(f"ideas file not found: {ideas_path}")

    transcript_path = base_dir / "transcript_final.json"
    if not transcript_path.exists():
        raise SystemExit(f"transcript file not found: {transcript_path}")

    ideas_doc = json.loads(ideas_path.read_text(encoding="utf-8"))
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    if isinstance(transcript, dict) and "segments" in transcript:
        transcript = transcript["segments"]

    client = init_client()

    per_idea_dir = ctx_out / f"stance_by_idea_{args.model}"
    per_idea_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for idx, idea in enumerate(ideas_doc.get("ideas", []) or []):
        merged = process_single_idea(
            client=client,
            model=args.model,
            idea=idea,
            ideas_doc=ideas_doc,
            transcript=transcript,
            idea_index=idx,
            adjacent_segments=args.adjacent_segments,
            adjacent_window=args.adjacent_window,
            use_heuristic_cg=args.use_heuristic_cg
        )
        slug = _slug(idea.get("idea","idea"))
        (per_idea_dir / f"{slug}.json").write_text(
            json.dumps({
                "metadata": {
                    "meeting_id": args.meeting_id,
                    "model": args.model,
                    "context": "transcript_mentions_exclusive+adjacent"
                },
                "idea": merged.get("idea",""),
                "stance": merged.get("stance", {}),
                "common_ground": merged.get("common_ground", {})
            }, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        results.append(merged)

    combined_out = ctx_out / f"stance_common_ground_{args.model}.json"
    combined_out.write_text(
        json.dumps({
            "metadata": {
                "meeting_id": args.meeting_id,
                "model": args.model,
                "context": "transcript_mentions_exclusive+adjacent"
            },
            "ideas": results
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"[done] Wrote {len(results)} ideas")
    print(f"Combined: {combined_out}")
    print(f"Per-idea: {per_idea_dir}")

if __name__ == "__main__":
    main()
