#!/usr/bin/env python3
import json, math, argparse
from pathlib import Path

TARGET_TOKENS = 1200    # soft cap per chunk
HARD_MAX_TOKENS = 1500  # never exceed this (except single-turn overflow; see note)
OVERLAP_RATIO = 0.20    # 20% of previous turns are overlapped into next chunk

def approx_tokens(s: str) -> int:
    # simple, stable heuristic
    return max(1, math.ceil(len(s) / 4))

def turn_line(t: dict) -> str:
    spk = t.get("speaker", "S?")
    txt = (t.get("text") or "").strip()
    return f"{spk}: {txt}\n"

def chunk_by_turns(
    turns: list,
    target_tokens: int = TARGET_TOKENS,
    hard_max: int = HARD_MAX_TOKENS,
    overlap_ratio: float = OVERLAP_RATIO
) -> list:
    # keep speech with non-empty text
    turns = [t for t in turns if t.get("type") == "speech" and (t.get("text") or "").strip()]
    if not turns:
        return []

    # ensure chronological order (defensive)
    turns = sorted(
        turns,
        key=lambda t: (float(t.get("start", 0.0)), float(t.get("end", 0.0)))
    )

    chunks, i, n = [], 0, len(turns)
    while i < n:
        cur, tokens = [], 0
        j = i
        while j < n:
            line = turn_line(turns[j])
            tcount = approx_tokens(line)

            # If a single turn alone exceeds hard max and cur is empty,
            # accept it as its own chunk (documented single-turn overflow).
            # This avoids infinite loops and preserves turn coherence.
            if not cur and tcount > hard_max:
                cur.append(turns[j]); tokens += tcount; j += 1
                break

            # Stop at soft cap; allow one overflow turn unless it breaks hard cap
            if cur and (tokens + tcount) > target_tokens:
                if (tokens + tcount) > hard_max:
                    break
                # allow one overflow turn to keep coherence
                cur.append(turns[j]); tokens += tcount; j += 1
                break

            # normal add
            cur.append(turns[j]); tokens += tcount; j += 1

        # finalize this chunk
        start = float(cur[0].get("start", 0.0))
        end   = float(cur[-1].get("end", start))
        text  = "".join(turn_line(t) for t in cur).strip()
        speakers = sorted({t.get("speaker") for t in cur})

        # include SER & other useful fields per span (if present)
        spans = []
        for t in cur:
            span = {
                "start": round(float(t.get("start", 0.0)), 3),
                "end":   round(float(t.get("end", 0.0)), 3),
                "speaker": t.get("speaker", "S?"),
                "text": (t.get("text") or "").strip()
            }
            # carry-through fields if available in the input:
            # (safe to include; absent keys just won't appear)
            for k in ("segment_id", "overlap", "type"):
                if k in t:
                    span[k] = t[k]
            if "emotion" in t and isinstance(t["emotion"], dict):
                # keep the full SER object (valence, arousal, dominance, prosody, z-scores, etc.)
                span["emotion"] = t["emotion"]
            spans.append(span)

        chunk = {
            "chunk_id": len(chunks),
            "start": round(start, 3),
            "end":   round(end, 3),
            "speakers": speakers,
            "num_turns": len(cur),
            "approx_tokens": tokens,
            "spans": spans,          # now includes emotion if present on the input turns
            "text": text,            # keep for quick inspection/back-compat
        }
        chunks.append(chunk)

        # compute next start index with overlap on turn count
        if j >= n:
            break
        overlap_turns = max(1, int(len(cur) * overlap_ratio))
        i = max(i + len(cur) - overlap_turns, i + 1)

    return chunks

def main():
    ap = argparse.ArgumentParser(description="Minimal turn-based chunker (preserves SER in spans if present)")
    ap.add_argument("--in",  dest="in_path",  type=Path, required=True, help="Input JSON (turns with text, optionally with SER)")
    ap.add_argument("--out", dest="out_path", type=Path, required=True, help="Output JSON (chunks)")
    ap.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    ap.add_argument("--hard-max",      type=int, default=HARD_MAX_TOKENS)
    ap.add_argument("--overlap",       type=float, default=OVERLAP_RATIO)
    args = ap.parse_args()

    turns = json.loads(args.in_path.read_text(encoding="utf-8"))
    chunks = chunk_by_turns(
        turns,
        target_tokens=args.target_tokens,
        hard_max=args.hard_max,
        overlap_ratio=args.overlap
    )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[summary] created {len(chunks)} chunks\n")
    for c in chunks:
        spk = ",".join(c["speakers"])
        print(
            f"chunk {c['chunk_id']:>2} | {c['num_turns']:>2} turns | "
            f"{c['approx_tokens']:>4} tokens | {c['start']:>6.1f}s→{c['end']:>6.1f}s | {spk}"
        )
    print(f"\n[wrote] {args.out_path}\n")

if __name__ == "__main__":
    main()
