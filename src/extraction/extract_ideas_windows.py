# extract_ideas_windows.py
import json, re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from .llm_helper import init_client, chat_json, norm_key, canonical_idea_text

SYSTEM_PROMPT = """You extract MAJOR IDEAS from meeting chunks and cite verbatim evidence spans.

Output STRICTLY as JSON:
{
  "ideas": [
    {
      "idea": "concise major idea (<= 8 words; canonical phrasing)",
      "mentions": [
        { "start": 430.012, "end": 432.145, "speaker": "SPEAKER_00", "quote": "verbatim <=200 chars" }
      ]
    }
  ]
}

Definitions & rules:
- The idea label should be a well-phrased canonical summary, not necessarily a direct quote.
- Include only ideas that directly contribute to the provided topic (strict topic anchoring).
- Be faithful to the text; DO NOT invent content not present in the chunk.
- Normalize and dedupe near-duplicates (see normalization).
- Exclude meta/administrative chatter and vague sentiments without a concrete claim.
- Exclude criteria-only statements unless the underlying idea is explicit.

Normalization:
- Convert to a short noun phrase or imperative (<= 8 words).
- Lowercase except proper nouns/acronyms (e.g., "AWS", "HIPAA").
- Remove filler/hedges (“maybe”, “please”, “I think”).
- Merge equivalent phrasings conservatively: { "kanban board", "task board" } -> "task board".
- Prefer patterns like "do X with Y" / "use X for Y" / "turn X into Y" when present.

Evidence requirements:
- For each idea, provide 1-3 mentions where the idea is stated, advanced, or explicitly endorsed.
- Each mention MUST include exact start/end/speaker and a verbatim quote (<=200 chars).
- If the idea spans adjacent text by the same speaker, pick a window that covers the unified utterance.
- Prefer quotes containing decision/evaluation language (“we can…”, “let’s…”, “turn X into Y”).
"""

USER_TEMPLATE = """# METADATA
topic: {topic}
meeting_type: {meeting_type}
num_participants: {num_participants}
duration_sec: {duration_sec}

# CHUNK
chunk_id: {cid} | window: {start:.3f}s–{end:.3f}s

# TURNS (use these exact spans when citing mentions)
{turns_block}

#TOPIC ANCHORING
Only extract ideas that contribute to: "{topic}".
If uncertain whether an idea is on-topic, exclude it.
"""

# ------------------------ Helpers ------------------------

# Loads simple key=value metadata file
def load_kv_file(path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            v = float(v) if "." in v else int(v)
        except ValueError:
            pass
        meta[k] = v
    return meta

def _render_turns_block(spans: List[Dict[str, Any]]) -> str:
    lines = []
    for t in spans:
        s = float(t.get("start", 0.0))
        e = float(t.get("end", s))
        spk = t.get("speaker", "S?")
        txt = (t.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{s:.3f}–{e:.3f}] {spk}: {txt}")
    return "\n".join(lines)

def _merge_windows(windows: List[List[float]], eps: float = 0.25) -> List[List[float]]:
    if not windows:
        return []
    ws = sorted([[float(a), float(b), int(c)] for a, b, c in windows], key=lambda w: (w[0], w[1], w[2]))
    merged = []
    for w in ws:
        if not merged:
            merged.append(w); continue
        prev = merged[-1]
        if w[0] <= prev[1] + eps:  # overlap/adjacent -> extend
            prev[1] = max(prev[1], w[1])
        else:
            merged.append(w)
    return merged

def _overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    start = max(a0, b0)
    end = min(a1, b1)
    return max(0.0, end - start)

def _best_span_key_for_mention(spans: List[Dict[str, Any]],
                               mention_s: float,
                               mention_e: float,
                               mention_spk: str,
                               chunk_id: int) -> Optional[str]:
    best_key = None
    best_overlap = 0.0
    best_span_len = 0.0
    mention_sid = mention_spk.strip() if mention_spk else ""

    for idx, sp in enumerate(spans or []):
        if sp.get("type") != "speech":
            continue
        sp_s = float(sp.get("start", 0.0))
        sp_e = float(sp.get("end", sp_s))
        if sp_e < sp_s:
            sp_s, sp_e = sp_e, sp_s

        ov = _overlap_len(mention_s, mention_e, sp_s, sp_e)
        if ov <= 0.0:
            continue

        sp_spk = (sp.get("speaker") or "").strip()
        same_speaker = (not mention_sid) or (sp_spk == mention_sid)
        span_len = sp_e - sp_s

        take = False
        if best_key is None:
            take = True
        else:
            # Prefer same-speaker if possible, otherwise maximize overlap then length
            if same_speaker and best_overlap == 0.0:
                take = True
            elif ov > best_overlap or (ov == best_overlap and span_len > best_span_len):
                take = True

        if take:
            best_key = f"{chunk_id}#{idx}"
            best_overlap = ov
            best_span_len = span_len

    return best_key

def merge_ideas_windows(state: Dict[str, Any],
                        ideas_payload: List[Dict[str, Any]],
                        cid: int,
                        eps: float = 0.25) -> Dict[str, Any]:

    out = dict(state)
    # Build index of existing ideas by normalized key
    by_key = {}
    for i in out.get("ideas", []):
        key = norm_key(canonical_idea_text(i.get("idea","")))
        if not key:
            continue
        by_key[key] = i

    for row_in in (ideas_payload or []):
        idea_txt = canonical_idea_text((row_in.get("idea") or "").strip())
        if not idea_txt:
            continue

        k = norm_key(idea_txt)
        row = by_key.get(k)
        if not row:
            row = {
                "idea": idea_txt,
                "first_seen": float("inf"),
                "last_seen": float("-inf"),
                "windows": [],
                "mentions": []
            }
            by_key[k] = row

        # collect windows AND mentions from this chunk
        for m in (row_in.get("mentions") or []):
            s = float(m.get("start", 0.0))
            e = float(m.get("end", s))
            if e < s:
                s, e = e, s
            row["windows"].append([s, e, int(cid)])
            row["first_seen"] = min(row["first_seen"], s)
            row["last_seen"]  = max(row["last_seen"],  e)

            # persist mention with cid, segment_id, and optional quote
            mention = {
                "start": s,
                "end": e,
                "speaker": (m.get("speaker") or "").strip(),
                "cid": int(cid)  # keep for traceability/back-compat
            }
            if m.get("segment_id"):
                mention["segment_id"] = m["segment_id"]

            q = (m.get("quote") or "").strip()
            if q:
                mention["quote"] = q[:200]
            row["mentions"].append(mention)

    # coalesce windows and normalize first/last per idea
    ideas_out: List[Dict[str, Any]] = []
    for row in by_key.values():
        merged_windows = _merge_windows(row["windows"], eps=eps)
        row["windows"] = merged_windows
        if merged_windows:
            row["first_seen"] = merged_windows[0][0]
            row["last_seen"]  = merged_windows[-1][1]
        else:
            row["first_seen"] = float("inf")
            row["last_seen"]  = float("-inf")

        # sort mentions by time for readability
        row["mentions"] = sorted(row["mentions"], key=lambda m: (m["start"], m["end"]))
        ideas_out.append(row)

    out["ideas"] = sorted(ideas_out, key=lambda r: (r.get("first_seen", 1e18), r.get("idea","")))
    return out

def main():
    ap = argparse.ArgumentParser(description="Extract ideas from meeting chunks with window tracking + segment ids")
    ap.add_argument("--meeting-id", required=True, help="e.g., ES2002a")
    ap.add_argument("--model", default="gpt-3.5-turbo")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    meta: Dict[str, Any] = {}
    base_dir = Path("data/outputs") / args.meeting_id

    for ext in (".txt", ".json"):
        candidate = base_dir / f"metadata{ext}"
        if candidate.exists():
            meta_path = candidate
            if meta_path.suffix.lower() == ".json":
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                meta = load_kv_file(meta_path)
            print(f"[info] Loaded metadata from {meta_path}")
            break

    base_dir = Path("data/outputs") / args.meeting_id
    chunks_path = base_dir / "chunks.json"
    out_dir = base_dir / "context_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise SystemExit(f"chunks.json not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    client = init_client()

    state: Dict[str, Any] = {
        "metadata": {**meta, "meeting_id": args.meeting_id, "model": args.model},
        "ideas": []
    }

    for c in chunks:
        turns_block = _render_turns_block(c.get("spans", []))

        user_prompt = USER_TEMPLATE.format(
            topic=meta.get("topic", ""),
            meeting_type=meta.get("meeting_type", ""),
            num_participants=meta.get("num_participants", 0),
            duration_sec=meta.get("duration_sec", 0.0),
            cid=c["chunk_id"],
            start=c["start"],
            end=c["end"],
            turns_block=turns_block,
        )

        step = chat_json(
            client,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=args.temperature
        )

        step_ideas = step.get("ideas") if isinstance(step, dict) else []
        if isinstance(step_ideas, list) and step_ideas and isinstance(step_ideas[0], str):
            step_ideas = [
                {"idea": s, "mentions": [{"start": c["start"], "end": c["end"], "speaker": ""}]}
                for s in step_ideas
            ]

        if not isinstance(step_ideas, list):
            step_ideas = []

        clean_payload = []
        spans_this_chunk = c.get("spans", []) or []
        for it in step_ideas:
            idea = (it.get("idea") or "").strip()
            mentions = []
            for m in (it.get("mentions") or []):
                try:
                    s = float(m.get("start", 0.0))
                    e = float(m.get("end", s))
                    spk = (m.get("speaker") or "").strip()
                    if e < s:
                        s, e = e, s

                    rec = {"start": s, "end": e, "speaker": spk}

                    # attach best-matching segment_id from this chunk
                    seg_id = _best_span_key_for_mention(
                        spans=spans_this_chunk,
                        mention_s=s, mention_e=e,
                        mention_spk=spk,
                        chunk_id=c["chunk_id"]
                    )
                    if seg_id:
                        rec["segment_id"] = seg_id

                    # keep cid for traceability/back-compat
                    rec["cid"] = int(c["chunk_id"])

                    q = (m.get("quote") or "").strip()
                    if q:
                        rec["quote"] = q[:200]
                    mentions.append(rec)
                except Exception:
                    continue
            if idea and mentions:
                clean_payload.append({"idea": idea, "mentions": mentions})

        state = merge_ideas_windows(state, clean_payload, c["chunk_id"])

    out_json = out_dir / "ideas_windows.json"
    out_json.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {out_json}")

if __name__ == "__main__":
    main()
