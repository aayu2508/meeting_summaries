#!/usr/bin/env python3
import json, argparse, hashlib, re, math
from pathlib import Path
from typing import List, Dict, Any

PROFILES = {
    "gptnano": {"target_tokens": 900,  "hard_max": 1200, "overlap": 0.20},
    "gpt35":   {"target_tokens": 900,  "hard_max": 1200, "overlap": 0.25},
    "gptfull": {"target_tokens": 3000, "hard_max": 3600, "overlap": 0.12},
}

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[ \t]+(\n|$)", r"\1", s)
    return s

def turn_line(t: Dict[str, Any]) -> str:
    return f"{t.get('speaker','S?')}: {(t.get('text') or '').strip()}\n"

def content_hash(lines: List[str]) -> str:
    text_norm = normalize_text("".join(lines))
    return hashlib.md5(text_norm.encode("utf-8")).hexdigest()

def _approx_tokens(s: str) -> int:
    return max(1, math.ceil(len(re.findall(r"\S+", s)) * 1.3))

def chunk_by_turns(turns: List[Dict[str, Any]], *, target_tokens: int, hard_max: int,
                   overlap_ratio: float, meeting_id: str) -> List[Dict[str, Any]]:
    turns = [t for t in turns if t.get("type") == "speech" and (t.get("text") or "").strip()]
    if not turns:
        return []

    turns.sort(key=lambda t: (float(t.get("start", 0.0)), float(t.get("end", 0.0))))

    chunks, i, n = [], 0, len(turns)
    while i < n:
        cur, tokens, j = [], 0, i
        while j < n:
            line = turn_line(turns[j])
            tcount = _approx_tokens(line)  # good enough for packing
            if not cur and tcount > hard_max:
                cur.append(turns[j]); j += 1; break
            if cur and (tokens + tcount) > target_tokens:
                if (tokens + tcount) > hard_max:
                    break
                cur.append(turns[j]); j += 1; break
            cur.append(turns[j]); tokens += tcount; j += 1

        lines = [turn_line(t) for t in cur]
        htxt = content_hash(lines)
        chunk_id = f"{meeting_id}#{htxt[:16]}"

        start = float(cur[0].get("start", 0.0))
        end   = float(cur[-1].get("end", start))

        spans = []
        for t in cur:
            spans.append({
                "segment_id": t.get("segment_id"),
                "start": round(float(t.get("start", 0.0)), 3),
                "end":   round(float(t.get("end", 0.0)), 3),
                "speaker": t.get("speaker", "S?"),
                "text": (t.get("text") or "").strip()
            })

        chunks.append({
            "chunk_id": chunk_id,
            "start": round(start, 3),
            "end":   round(end, 3),
            "spans": spans
        })

        if j >= n:
            break
        overlap_turns = max(1, int(len(cur) * overlap_ratio))
        i = max(i + len(cur) - overlap_turns, i + 1)

    return chunks

def main():
    ap = argparse.ArgumentParser(description="Minimal chunker for idea-first pipeline")
    ap.add_argument("--meeting-id", type=str, required=True, help="Meeting ID (e.g., BAC)")
    ap.add_argument("--profile", type=str, default="gptnano", help="gptnano | gpt35 | gptfull")
    args = ap.parse_args()

    profile = args.profile.lower()
    cfg = PROFILES[profile]

    root = Path("data/outputs") / args.meeting_id
    in_path  = root / "transcript_final.json"
    out_path = root / f"chunks_{profile}.json"

    turns = json.loads(in_path.read_text(encoding="utf-8"))
    chunks = chunk_by_turns(
        turns,
        target_tokens=cfg["target_tokens"],
        hard_max=cfg["hard_max"],
        overlap_ratio=cfg["overlap"],
        meeting_id=args.meeting_id
    )

    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[summary] meeting={args.meeting_id} | profile={profile} | chunks={len(chunks)}")
    print(f"[wrote] {out_path}")

if __name__ == "__main__":
    main()
