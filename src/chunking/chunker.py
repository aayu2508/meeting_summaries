import argparse, json
from pathlib import Path
from typing import List, Dict, Any, Tuple

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)

# quick-and-safe token estimate for budgeting
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    chars = len(text)
    words = max(1, len(text.split()))
    return max(words, int(round(chars / 4.0)))

def seconds_to_clock(x: float) -> str:
    x = max(0.0, float(x))
    h = int(x // 3600); m = int((x % 3600) // 60); s = x % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}" if h > 0 else f"{m:02d}:{s:05.2f}"

def build_chunk_text(segs: List[Dict[str, Any]]) -> str:
    lines = []
    for s in segs:
        t0 = seconds_to_clock(float(s["start"]))
        t1 = seconds_to_clock(float(s["end"]))
        spk = s.get("speaker", "S?")
        txt = (s.get("text") or "").strip()
        lines.append(f"[{spk} {t0}–{t1}] {txt}")
    return "\n".join(lines).strip()

# Fixed-length windows with overlap. Ensures coverage even if the file is shorter than chunk_len.
def make_chunks(segments: List[Dict[str, Any]], chunk_len: float, stride: float) -> List[Tuple[float,float]]:
    if not segments:
        return []
    min_t = min(float(s["start"]) for s in segments)
    max_t = max(float(s["end"]) for s in segments)

    windows = []
    t = min_t

    # normal sweep
    while not windows or t + chunk_len <= max_t:
        windows.append((t, t + chunk_len))
        t += stride

    # ensure the tail is covered
    if windows[-1][1] < max_t:
        windows.append((max_t - chunk_len, max_t))

    # handle very short audio (max_t - min_t < chunk_len)
    if not windows:
        windows = [(min_t, min_t + chunk_len)]
    if windows and windows[0][0] < min_t:
        windows[0] = (min_t, min_t + chunk_len)

    # dedupe with rounding
    seen, out = set(), []
    for w in sorted(windows):
        key = (round(w[0], 3), round(w[1], 3))
        if key not in seen:
            seen.add(key); out.append(w)
    return out

def segments_for_windows_sweep(
    segments: List[Dict[str, Any]],
    windows: List[Tuple[float,float]],
    min_overlap_sec: float = 0.0
) -> List[List[Dict[str, Any]]]:
    segs = sorted(segments, key=lambda s: (float(s["start"]), float(s["end"])))
    wins = sorted(windows, key=lambda w: (w[0], w[1]))

    result: List[List[Dict[str, Any]]] = []
    i = 0  # pointer into segments

    for (w_start, w_end) in wins:
        cur: List[Dict[str, Any]] = []
        # advance i until segs[i].end >= w_start
        while i < len(segs) and float(segs[i]["end"]) < w_start:
            i += 1
        j = i
        while j < len(segs) and float(segs[j]["start"]) <= w_end:
            s = segs[j]
            # compute actual overlap with this window
            ov = max(0.0, min(float(s["end"]), w_end) - max(float(s["start"]), w_start))
            if ov >= min_overlap_sec:
                cur.append(s)
            j += 1
        result.append(cur)
    return result

# acceptance checks (just for validation)
def assert_every_segment_included(segments: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> None:
    covered = set()
    for c in chunks:
        covered.update(c["segment_ids"])
    missing = [s["segment_id"] for s in segments if s["segment_id"] not in covered]
    if missing:
        raise AssertionError(f"{len(missing)} segments not covered by any chunk, e.g. {missing[:5]}")

def avg_tokens_within_budget(chunks: List[Dict[str, Any]], budget: int) -> bool:
    if not chunks:
        return True
    avg = sum(c.get("est_tokens", 0) for c in chunks) / len(chunks)
    return avg <= budget

def main():
    ap = argparse.ArgumentParser("Step 3: Chunking (time windows + overlap)")
    ap.add_argument("--segments", required=True, help="Path to segments.jsonl (speaker turns with text)")
    ap.add_argument("--meeting-id", required=True, help="Meeting ID (e.g., m1)")
    ap.add_argument("--chunk-len", type=float, default=150.0, help="Window length (seconds)")
    ap.add_argument("--stride", type=float, default=30.0, help="Stride between windows (seconds)")
    ap.add_argument("--min-overlap-sec", type=float, default=0.0, help="Min segment-window overlap to include")
    ap.add_argument("--out-dir", default="data/outputs", help="Base output dir")
    ap.add_argument("--write-texts", action="store_true", help="Also write per-chunk .txt files")
    ap.add_argument("--max-chunks", type=int, default=0, help="Cap number of chunks (0=no cap)")
    ap.add_argument("--token-budget", type=int, default=6000, help="Soft avg token budget per chunk")
    args = ap.parse_args()

    seg_path = Path(args.segments).expanduser().resolve()
    out_dir = Path(args.out_dir) / args.meeting_id
    out_chunks = out_dir / "chunks.jsonl"
    chunks_txt_dir = out_dir / "chunks"

    segments = read_jsonl(seg_path)
    if not segments:
        raise SystemExit(f"No segments in {seg_path}")

    # keep only this meeting_id (if present)
    if "meeting_id" in segments[0]:
        segments = [s for s in segments if s.get("meeting_id") == args.meeting_id]
        if not segments:
            raise SystemExit(f"No segments for meeting_id={args.meeting_id}")

    # coerce & sort, and ensure segment_id exists
    for idx, s in enumerate(segments):
        s["start"] = float(s["start"]); s["end"] = float(s["end"])
        s["text"] = (s.get("text") or "").strip()
        if "segment_id" not in s:
            s["segment_id"] = f"s{idx:06d}"
    segments.sort(key=lambda x: (x["start"], x["end"]))

    # Precompute per-segment token estimates
    for s in segments:
        s["_est_tokens"] = estimate_tokens(s["text"])

    # windows & sweep selection
    windows = make_chunks(segments, args.chunk_len, args.stride)
    win_segments = segments_for_windows_sweep(segments, windows, min_overlap_sec=args.min_overlap_sec)

    # build rows
    rows: List[Dict[str, Any]] = []
    for i, ((w_start, w_end), segs) in enumerate(zip(windows, win_segments)):
        if not segs:
            continue
        chunk_id = f"c{str(i).zfill(4)}"
        est_toks = sum(ss["_est_tokens"] for ss in segs)

        row = {
            "chunk_id": chunk_id,
            "meeting_id": args.meeting_id,
            "start": round(w_start, 2),
            "end": round(w_end, 2),
            "segment_ids": [s["segment_id"] for s in segs],
            "segment_count": len(segs),
            "est_tokens": est_toks
        }
        rows.append(row)

        if args.write_texts:
            write_text(build_chunk_text(segs), chunks_txt_dir / f"{chunk_id}.txt")

        if args.max_chunks and len(rows) >= args.max_chunks:
            break

    # acceptance: every segment appears ≥1 time
    assert_every_segment_included(segments, rows)

    ok_budget = avg_tokens_within_budget(rows, args.token_budget)
    avg_toks = int(sum(c["est_tokens"] for c in rows) / max(1, len(rows)))
    print(f"[chunks] {len(rows)} chunks; avg est tokens/chunk ≈ {avg_toks} (budget={args.token_budget})",
          "OK" if ok_budget else "⚠︎ high")

    write_jsonl(rows, out_chunks)
    print(f"[done] wrote {out_chunks}")

if __name__ == "__main__":
    main()
