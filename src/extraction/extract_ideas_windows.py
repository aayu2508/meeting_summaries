# extract_ideas_windows.py
import json, argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from .utils.llm_client import init_client, chat_json
from .utils.common import load_metadata, get_meeting_base_dir
from .utils.common import norm_key, canonical_idea_text

SYSTEM_PROMPT = """You extract MAJOR IDEAS from meeting chunks and provide mentions with timestamps and speaker labels.
Return ONLY this JSON:
{
  "ideas": [
    {
      "idea": "concise major idea (<= 10 words)",
      "mentions": [
        { "start": 430.012, "end": 432.145, "speaker": " " }
      ]
    }
  ]
}

Extraction rules:
- Extract all actionable ideas that advance the topic; do not omit relevant ideas.
- Exclude problems, risks, complaints, logistics.
- Paraphrase each major idea into a concise, clear phrase with proper grammar.
- Be faithful to the source; do not invent any idea.
- Provide all mentions per idea with accurate timestamps and speaker labels (no quotes, no extra fields).
- SPEAKER COVERAGE: If multiple speakers paraphrase or build on the SAME idea within this chunk, include a separate mention for EACH speaker with accurate time spans.
- TIMESTAMPS: Mentions must align with supplied TURNS; do not cite text outside the chunk window.
- OUTPUT: JSON only. No extra keys. No commentary.

# NOTE
If the speakers briefly affirms an idea within a few seconds (e.g., “yeah/okay/right/exactly”), include that as a separate mention for that speaker.
"""

USER_TEMPLATE = """# METADATA
topic: {topic}
meeting_type: {meeting_type}
num_participants: {num_participants}
duration_sec: {duration_sec}

# TOPIC HINT (exclude clearly off-topic content): "{topic}"

# CHUNK
chunk_id: {cid} | window: {start:.3f}s-{end:.3f}s

# TURNS (use these exact spans when citing mentions)
{turns_block}
"""

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
    for t in spans or []:
        s = float(t.get("start", 0.0))
        e = float(t.get("end", s))
        spk = t.get("speaker", "S?")
        txt = (t.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{s:.3f}-{e:.3f}] {spk}: {txt}")
    return "\n".join(lines)

def _best_segment_id(spans: List[Dict[str, Any]],
                     mention_s: float,
                     mention_e: float,
                     mention_spk: str) -> Optional[str]:
    best_sid = None
    best_overlap = 0.0
    best_span_len = 0.0
    want_spk = (mention_spk or "").strip()

    for sp in spans or []:
        if sp.get("type") and sp["type"] != "speech":
            continue
        s0 = float(sp.get("start", 0.0)); s1 = float(sp.get("end", s0))
        if s1 < s0: s0, s1 = s1, s0
        
        ov = max(0.0, min(mention_e, s1) - max(mention_s, s0))
        if ov <= 0.0:
            continue
        same_spk = (not want_spk) or ((sp.get("speaker") or "").strip() == want_spk)
        span_len = s1 - s0

        take = (best_sid is None) or \
               (same_spk and best_overlap == 0.0) or \
               (ov > best_overlap) or \
               (ov == best_overlap and span_len > best_span_len)
        if take:
            sid = sp.get("segment_id")
            if sid:  # we assume segment_id is always present in your pipeline
                best_sid = sid
                best_overlap = ov
                best_span_len = span_len

    return best_sid

def _merge_windows(windows: List[List[float]], eps: float = 0.25) -> List[List[float]]:
    if not windows:
        return []
    ws = sorted([[float(a), float(b), c] for a, b, c in windows], key=lambda w: (w[0], w[1], w[2]))
    merged: List[List[float]] = []
    for w in ws:
        if not merged:
            merged.append(w); continue
        prev = merged[-1]
        if w[0] <= prev[1] + eps:
            prev[1] = max(prev[1], w[1])
        else:
            merged.append(w)
    return merged

def merge_ideas_windows(state: Dict[str, Any],
                        ideas_payload: List[Dict[str, Any]],
                        cid: str,
                        eps: float = 0.25) -> Dict[str, Any]:
    out = dict(state)
    by_key: Dict[str, Dict[str, Any]] = {}
    for i in out.get("ideas", []):
        key = norm_key(canonical_idea_text(i.get("idea","")))
        if key:
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

        for m in (row_in.get("mentions") or []):
            s = float(m.get("start", 0.0)); e = float(m.get("end", s))
            if e < s: s, e = e, s
            row["windows"].append([s, e, cid])
            row["first_seen"] = min(row["first_seen"], s)
            row["last_seen"]  = max(row["last_seen"],  e)

            mention = {
                "start": s,
                "end": e,
                "speaker": (m.get("speaker") or "").strip(),
                "cid": cid
            }
            # always present per your pipeline guarantee
            if m.get("segment_id"):
                mention["segment_id"] = m["segment_id"]
            row["mentions"].append(mention)

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
        row["mentions"] = sorted(row["mentions"], key=lambda m: (m["start"], m["end"]))
        ideas_out.append(row)

    out["ideas"] = sorted(ideas_out, key=lambda r: (r.get("first_seen", 1e18), r.get("idea","")))
    return out

def main():
    ap = argparse.ArgumentParser(description="Extract ideas from meeting chunks with time windows + segment ids")
    ap.add_argument("--meeting-id", required=True, help="e.g., BAC")
    ap.add_argument("--model", default="gptnano", help="LLM: gptnano | gpt3.5 | gptfull (aliases supported)")
    args = ap.parse_args()

    base_dir = Path("data/outputs") / args.meeting_id
    meta: Dict[str, Any] = {}
    for ext in (".json", ".txt"):
        p = base_dir / f"metadata{ext}"
        if p.exists():
            meta = json.loads(p.read_text(encoding="utf-8")) if p.suffix == ".json" else load_kv_file(p)
            print(f"[info] Loaded metadata from {p}")
            break

    chunks_path = base_dir / f"chunks_{args.model}.json"
    out_dir = base_dir / "context_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"ideas_windows_{args.model}.json"

    if not chunks_path.exists():
        raise SystemExit(f"chunks file not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if isinstance(chunks, dict) and "chunks" in chunks:
        chunks = chunks["chunks"]

    client = init_client()

    state: Dict[str, Any] = {
        "metadata": {**meta, "meeting_id": args.meeting_id, "model": args.model},
        "ideas": []
    }

    for c in chunks:
        spans_this_chunk = c.get("spans", []) or []
        turns_block = _render_turns_block(spans_this_chunk)

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

        resp = chat_json(
            client,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            reasoning_effort="minimal",
            verbosity="low"
        )

        step_ideas = resp.get("ideas") if isinstance(resp, dict) else []
        if not isinstance(step_ideas, list):
            step_ideas = []

        clean_payload: List[Dict[str, Any]] = []
        for it in step_ideas:
            idea = (it.get("idea") or "").strip()
            if not idea:
                continue

            mentions: List[Dict[str, Any]] = []
            for m in (it.get("mentions") or []):
                try:
                    s = float(m.get("start", 0.0)); e = float(m.get("end", s))
                    spk = (m.get("speaker") or "").strip()
                    if e < s: s, e = e, s

                    seg_id = _best_segment_id(
                        spans=spans_this_chunk,
                        mention_s=s, mention_e=e,
                        mention_spk=spk
                    )
                    if seg_id is None:
                        continue

                    mentions.append({
                        "start": s, "end": e, "speaker": spk,
                        "segment_id": seg_id, "cid": c["chunk_id"]
                    })
                except Exception:
                    continue

            if mentions:
                clean_payload.append({"idea": idea, "mentions": mentions})

        state = merge_ideas_windows(state, clean_payload, str(c["chunk_id"]))

    out_json.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {out_json}")

if __name__ == "__main__":
    main()
