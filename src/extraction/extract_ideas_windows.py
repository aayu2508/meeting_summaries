# extract_ideas_windows.py
import json, re
import argparse
from pathlib import Path
from typing import Dict, Any, List
from .llm_helper import init_client, chat_json, norm_key, canonical_idea_text

SYSTEM_PROMPT = """You extract MAJOR IDEAS from meeting chunks.
Output STRICTLY as JSON:
{
  "ideas": [
    {
      "idea": "concise idea",
      "mentions": [
        { "start": 430.012, "end": 432.145, "speaker": "SPEAKER_00" }
      ]
    }
  ]
}

Definitions & rules:
- "Major idea" = a concise, self-contained proposal/claim/feature/decision (≤ 8 words).
- Only include ideas DIRECTLY about the provided topic.
- Be faithful to the text; DO NOT invent.
- Normalize wording and dedupe near-duplicates (see normalization).
- Exclude meta/administrative chatter and vague sentiments without a concrete claim.
- Exclude criteria-only statements unless the idea itself is explicit.

Normalization:
- Convert to a short noun phrase or imperative (≤ 8 words).
- Lowercase except proper nouns/acronyms (keep “AWS”, “HIPAA”).
- Remove filler words/hedges (“maybe”, “please”, “I think”).
- Merge equivalent phrasings conservatively: { "kanban board", "task board" } → "task board".

Important:
- Identify the *exact* spans (start/end/speaker) where each idea is actually stated or advanced.
- Use ONLY the provided turns when creating mentions.
- Return at most 8 ideas per chunk; prefer precision over recall.
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

def merge_ideas_windows(state: Dict[str, Any],
                        ideas_payload: List[Dict[str, Any]],
                        cid: int,
                        eps: float = 0.25) -> Dict[str, Any]:
    """
    state = { "metadata": {...}, "ideas": [ {idea, first_seen, last_seen, windows:[[s,e,cid], ...]} ] }
    ideas_payload = [
      { "idea": "…", "mentions": [ {"start":..,"end":..,"speaker":"…"}, ... ] },
      ...
    ]
    """
    out = dict(state)
    by_key = {norm_key(canonical_idea_text(i["idea"])): i
              for i in out.get("ideas", []) if i.get("idea")}

    for row_in in (ideas_payload or []):
        idea_txt = canonical_idea_text((row_in.get("idea") or "").strip())
        if not idea_txt:
            continue
        k = norm_key(idea_txt)
        row = by_key.get(k)
        if not row:
            row = {"idea": idea_txt, "first_seen": float("inf"), "last_seen": float("-inf"), "windows": []}
            by_key[k] = row

        # collect windows from mentions
        for m in (row_in.get("mentions") or []):
            s = float(m.get("start", 0.0))
            e = float(m.get("end", s))
            if e < s:
                s, e = e, s
            row["windows"].append([s, e, int(cid)])
            row["first_seen"] = min(row["first_seen"], s)
            row["last_seen"]  = max(row["last_seen"],  e)

    # coalesce windows per idea
    for row in by_key.values():
        ws = sorted(row["windows"], key=lambda w: (w[0], w[1], w[2]))
        merged = []
        for w in ws:
            if not merged:
                merged.append(w); continue
            prev = merged[-1]
            if w[0] <= prev[1] + eps:         # overlap/adjacent -> extend
                prev[1] = max(prev[1], w[1])
            else:
                merged.append(w)
        row["windows"] = merged
        if merged:
            row["first_seen"] = merged[0][0]
            row["last_seen"]  = merged[-1][1]
        else:
            row["first_seen"] = float("inf")
            row["last_seen"]  = float("-inf")

    out["ideas"] = sorted(by_key.values(), key=lambda r: (r.get("first_seen", 1e18), r.get("idea", "")))
    return out

def main():
    ap = argparse.ArgumentParser(description="Extract ideas from meeting chunks with window tracking")
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

        # Expected: { "ideas": [ { "idea": "...", "mentions": [ {start,end,speaker}, ... ] } ] }
        step_ideas = step.get("ideas") if isinstance(step, dict) else []

        # Back-compat fallback: if it's a list of strings, treat the whole chunk as one window
        if isinstance(step_ideas, list) and step_ideas and isinstance(step_ideas[0], str):
            step_ideas = [
                {"idea": s, "mentions": [{"start": c["start"], "end": c["end"], "speaker": ""}]}
                for s in step_ideas
            ]

        if not isinstance(step_ideas, list):
            step_ideas = []

        # Keep only well-formed mentions
        clean_payload = []
        for it in step_ideas:
            idea = (it.get("idea") or "").strip()
            mentions = []
            for m in (it.get("mentions") or []):
                try:
                    s = float(m.get("start", 0.0))
                    e = float(m.get("end", s))
                    spk = (m.get("speaker") or "").strip()
                    if e < s: s, e = e, s
                    mentions.append({"start": s, "end": e, "speaker": spk})
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
