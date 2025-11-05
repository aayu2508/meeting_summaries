# extract_ideas_windows.py
import json, re
import argparse
from pathlib import Path
from typing import Dict, Any, List
from .llm_helper import init_client, chat_json, norm_key

SYSTEM_PROMPT = """You extract MAJOR IDEAS from meeting chunks.

Output STRICTLY as JSON:
{ "ideas": ["idea a", "idea b"] }

Definitions & rules:
- "Major idea" = a concise, self-contained proposal/claim/feature/decision (≤ 8 words).
- Only include ideas that are DIRECTLY about the meeting topic provided in the metadata block.
- If the chunk contains no topic-relevant ideas, return { "ideas": [] }.
- Be faithful to the chunk; DO NOT invent or generalize beyond the text.
- Normalize wording and dedupe near-duplicates (see normalization).
- Exclude meta/administrative chatter (greetings, turn-taking, “let’s circle back”, etc.).
- Exclude vague sentiments without a concrete claim (“this is interesting”, “maybe later”).
- Exclude criteria-only statements unless the idea itself is explicit.

Normalization:
- Convert to a short noun phrase or imperative (≤ 8 words).
- Lowercase except proper nouns/acronyms (keep “AWS”, “HIPAA”).
- Remove filler words, hedges, and politeness (“maybe”, “please”, “I think”).
- Merge equivalent phrasings conservatively: { "kanban board", "task board" } → "task board".

Formatting:
- JSON only, no comments, no trailing commas, no extra keys.
- Return at most 8 ideas per chunk; prefer precision over recall.
"""

USER_TEMPLATE = """# METADATA
topic: {topic}
meeting_type: {meeting_type}
num_participants: {num_participants}
duration_sec: {duration_sec}

# CHUNK
chunk_id: {cid}
window: {start:.1f}s–{end:.1f}s
text:
{transcript}
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

def _canonical_idea_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[-_]", " ", s) 
    return s

def merge_ideas_windows(state: Dict[str, Any], step_ideas: List[str],
                        start: float, end: float, cid: int,
                        eps: float = 0.5) -> Dict[str, Any]:
    out = dict(state)
    by_key = {norm_key(_canonical_idea_text(i["idea"])): i
              for i in out.get("ideas", []) if i.get("idea")}

    for idea_txt in (step_ideas or []):
        idea = _canonical_idea_text((idea_txt or "").strip())
        if not idea:
            continue
        k = norm_key(idea)
        row = by_key.get(k)
        if not row:
            row = {
                "idea": idea,
                "first_seen": float(start),
                "last_seen": float(end),
                "windows": []
            }
            by_key[k] = row
        else:
            row["first_seen"] = min(row["first_seen"], float(start))
            row["last_seen"]  = max(row["last_seen"],  float(end))

        row["windows"].append([float(start), float(end), int(cid)])

    # If an idea appears in back-to-back chunks whose windows nearly touch
    for row in by_key.values():
        ws = sorted(row["windows"], key=lambda w: (w[0], w[1], w[2]))
        merged = []
        for w in ws:
            if not merged:
                merged.append(w); continue
            prev = merged[-1]
            if w[0] <= prev[1] + eps:               # overlap/adjacent
                prev[1] = max(prev[1], w[1])        # extend end
            else:
                merged.append(w)
        row["windows"] = merged
        if merged:
            row["first_seen"] = merged[0][0]
            row["last_seen"]  = merged[-1][1]

    out["ideas"] = sorted(by_key.values(), key=lambda r: (r.get("first_seen", 1e18), r.get("idea","")))
    return out

def main():
    ap = argparse.ArgumentParser(description="Extract ideas + time windows (chunk-based)")
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
        user_prompt = USER_TEMPLATE.format(
            topic=meta.get("topic", ""),
            meeting_type=meta.get("meeting_type", ""),
            num_participants=meta.get("num_participants", 0),
            duration_sec=meta.get("duration_sec", 0.0),
            cid=c["chunk_id"],
            start=c["start"],
            end=c["end"],
            transcript=c["text"],
        )

        step = chat_json(
            client,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=args.temperature
        )

        step_ideas = step.get("ideas") if isinstance(step, dict) else []
        if not isinstance(step_ideas, list):
            step_ideas = []

        state = merge_ideas_windows(state, step_ideas, c["start"], c["end"], c["chunk_id"])

    out_json = out_dir / "ideas_windows.json"
    out_json.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {out_json}")

if __name__ == "__main__":
    main()
