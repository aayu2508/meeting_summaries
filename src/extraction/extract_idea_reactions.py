# run_reactions_from_mentions.py
import json, argparse, re
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
from statistics import mean
from .llm_helper import init_client, chat_json

LLM_SYSTEM = """You are given meeting segments with ASR text and SER (speech emotion recognition) features.

Each speech segment includes:
- start, end (seconds)
- speaker: "SPEAKER_XX"
- text: ASR transcript
- emotion: valence [-1,1], arousal [-1,1], dominance [0,1],
           category, category_categorical, category_confidence, intensity,
           prosody {pitch_mean, pitch_var, energy_mean, speech_rate},
           z_valence, z_arousal, category_z.

Interpretation:
1) Combine content and delivery. Text meaning takes priority; use SER when text is minimal/ambiguous.
2) Heuristics: valence>0.3 → positive; < -0.3 → negative. arousal>0.3 → high intensity. dominance>0.6 → assertive.
   Use z-scores to compare a speaker to their own baseline.
3) Stance categories:
   - pro: supports/builds on the idea
   - con: criticizes/blocks/raises risks
   - neutral: clarifies or off-topic
   Usually exactly one of {pro, con, neutral} is true (allow mixed only with clear conflicting evidence).
4) Labels:
   - "felt_positive", "felt_negative", "high_intensity", "enthusiastic", "skeptical", "assertive"
   Do NOT include "neutral" in labels (neutral is only a stance option).
5) Evidence:
   - Use at most 2–3 short, verbatim quotes per speaker; prefer evaluative/decision language.
   - Never invent content; only use provided quotes.
   - IMPORTANT: A speaker may ONLY use quotes listed under THAT SAME speaker. If a speaker has no quotes, leave reasoning empty for that speaker.

Output ONLY JSON in this exact schema (no extra keys):
{
  "idea_id": "I-##",
  "title": "string",
  "window": [startSec, endSec],
  "idea_reactions": [
    {
      "speaker_id": "S#",
      "stance_distribution": {"pro": true|false, "con": true|false, "neutral": true|false},
      "labels": ["felt_positive" | "felt_negative" | "high_intensity" | "enthusiastic" | "skeptical" | "assertive", ...],
      "reasoning": [{"quote": "<<=200 chars exact quote>"} , ...]
    }
  ]
}
"""

LLM_USER_TPL = """Infer stances and concise evidence for this idea window.

# IDEA (fixed — do not change wording)
- idea_id: {idea_id}
- title: {title}
- window: {w0}–{w1} (seconds)

# SPEAKERS (SER aggregates computed from spans inside the window)
{speakers_block}

# QUOTES_BY_SPEAKER (JSON; use only quotes under that speaker; do not copy across speakers)
{quotes_block}

Return ONLY the JSON described above.
"""

ALLOWED_LABELS = {"felt_positive","felt_negative","high_intensity","enthusiastic","skeptical","assertive"}

BAD_QUOTE_PAT = re.compile(r'^\.*$')  # matches "", ".", "...", etc.

def _sanitize_quote(q: Optional[str]) -> Optional[str]:
    if not q: return None
    q = q.strip()
    if not q or q.lower() in {"n/a", "na"}: return None
    if BAD_QUOTE_PAT.match(q): return None
    if len(q) < 4: return None
    if len(q) > 200: q = q[:197] + "..."
    return q

def _speaker_short_id(s: str) -> str:
    m = re.search(r"SPEAKER_(\d+)", s or "")
    return f"S{int(m.group(1))}" if m else (s or "S?")

def _slice_spans(spans: List[Dict[str,Any]], w0: float, w1: float) -> List[Dict[str,Any]]:
    out=[]
    for t in spans:
        if t.get("type") != "speech": continue
        if t.get("text") is None: continue
        s, e = float(t.get("start", 0.0)), float(t.get("end", 0.0))
        if not (e < w0 or s > w1):
            out.append(t)
    return out

def _agg_mean(vals):
    vals=[v for v in vals if v is not None]
    return mean(vals) if vals else None

def _aggregate_ser(segs: List[Dict[str,Any]]) -> Dict[str, float]:
    v, a, zv, za, d = [], [], [], [], []
    en, pv, sr = [], [], []
    for s in segs:
        emo = s.get("emotion") or {}
        v.append(emo.get("valence"))
        a.append(emo.get("arousal"))
        zv.append(emo.get("z_valence"))
        za.append(emo.get("z_arousal"))
        d.append(emo.get("dominance"))
        pros = emo.get("prosody") or {}
        en.append(pros.get("energy_mean"))
        pv.append(pros.get("pitch_var"))
        sr.append(pros.get("speech_rate"))
    return {
        "valence_mean": _agg_mean(v),
        "arousal_mean": _agg_mean(a),
        "z_valence_mean": _agg_mean(zv),
        "z_arousal_mean": _agg_mean(za),
        "dominance_mean": _agg_mean(d),
        "energy_mean": _agg_mean(en),
        "pitch_var_mean": _agg_mean(pv),
        "speech_rate_mean": _agg_mean(sr),
    }

def _render_speakers_block(packets: List[Dict[str,Any]]) -> str:
    lines=[]
    for p in packets:
        lines.append(f"- speaker_id: {p['speaker_id']}")
        lines.append(f"  ser_aggregate: {p['ser_aggregate']}")
    return "\n".join(lines)

def _render_quotes_json(quotes_by_spk: Dict[str, List[str]]) -> str:
    safe = {sid: [q for q in (_sanitize_quote(q) for q in qs) if q]
            for sid, qs in quotes_by_spk.items()}
    safe = {sid: qs for sid, qs in safe.items() if qs}
    return json.dumps(safe, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser(description="Add reactions to existing ideas using SER; leaves ideas/mentions unchanged.")
    ap.add_argument("--meeting-id", required=True)
    ap.add_argument("--model", default="gpt-3.5-turbo")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--pad", type=float, default=0.5, help="seconds to pad idea window for SER aggregation (quotes still come from mentions)")
    ap.add_argument("--out", default=None, help="Output JSON path (default: context_outputs/idea_reactions.json)")
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id
    ideas_path = base / "context_outputs" / "ideas_windows.json"
    chunks_path = base / "chunks_ser.json"
    out_path = Path(args.out) if args.out else (base / "context_outputs" / "idea_reactions.json")

    assert ideas_path.exists(), f"Missing: {ideas_path}"
    assert chunks_path.exists(), f"Missing: {chunks_path}"

    ideas_doc = json.loads(ideas_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    client = init_client()

    results=[]
    for idea_idx, idea in enumerate(ideas_doc.get("ideas", []), start=1):
        title = (idea.get("idea") or "").strip()
        idea_id = f"I-{idea_idx:02d}"

        # Build speaker->quotes from the existing mentions (unchanged, but sanitized)
        quotes_from_mentions: Dict[str, List[str]] = defaultdict(list)
        for m in idea.get("mentions", []) or []:
            spk = (m.get("speaker") or "").strip() or "SPEAKER_??"
            sid = _speaker_short_id(spk)
            q = _sanitize_quote(m.get("quote") or "")
            if q:
                quotes_from_mentions[sid].append(q)

        for w in (idea.get("windows") or []):
            w0, w1, cid = float(w[0]), float(w[1]), int(w[2])

            chunk = chunks_by_id.get(cid)
            if not chunk:
                results.append({"idea_id": idea_id, "title": title, "window": [w0, w1], "idea_reactions": []})
                continue

            # SER aggregation window (pad & clamp); quotes remain from mentions
            cw0 = max(chunk["start"], w0 - args.pad)
            cw1 = min(chunk["end"],   w1 + args.pad)
            spans = _slice_spans(chunk.get("spans", []), cw0, cw1)

            # Group spans by speaker for SER aggregation
            by_spk = defaultdict(list)
            for s in spans:
                spk = (s.get("speaker") or "").strip()
                by_spk[spk].append(s)

            # Build speaker packets using SER only; quotes strictly from mentions
            packets=[]
            seen_sids=set()
            for spk, segs in by_spk.items():
                sid = _speaker_short_id(spk)
                agg = _aggregate_ser(segs)
                packets.append({"speaker_id": sid, "ser_aggregate": agg})
                seen_sids.add(sid)

            # If a speaker has mention quotes but no spans (edge case), still pass them so LLM can mark neutral
            for sid in quotes_from_mentions.keys():
                if sid not in seen_sids:
                    packets.append({"speaker_id": sid, "ser_aggregate": {}})

            # Nothing to say
            if not packets and not quotes_from_mentions:
                results.append({"idea_id": idea_id, "title": title, "window": [w0, w1], "idea_reactions": []})
                continue

            speakers_block = _render_speakers_block(packets)
            quotes_block   = _render_quotes_json(quotes_from_mentions)

            user_prompt = LLM_USER_TPL.format(
                idea_id=idea_id, title=title, w0=w0, w1=w1,
                speakers_block=speakers_block, quotes_block=quotes_block
            )

            resp = chat_json(
                client,
                model=args.model,
                system_prompt=LLM_SYSTEM,
                user_prompt=user_prompt,
                temperature=args.temperature
            )

            # ----- Post-validation & label whitelist -----
            record = {"idea_id": idea_id, "title": title, "window": [w0, w1], "idea_reactions": []}
            if isinstance(resp, dict) and isinstance(resp.get("idea_reactions"), list):
                # Whitelist map of allowed quotes per speaker
                try:
                    allowed_quotes_map = json.loads(quotes_block) if quotes_block else {}
                except Exception:
                    allowed_quotes_map = {}

                clean_reactions = []
                for r in resp["idea_reactions"]:
                    sid = r.get("speaker_id")
                    # labels whitelist
                    r["labels"] = [l for l in (r.get("labels") or []) if l in ALLOWED_LABELS]

                    # stance_distribution sanity (ensure keys exist & booleans)
                    sd = r.get("stance_distribution") or {}
                    r["stance_distribution"] = {
                        "pro": bool(sd.get("pro", False)),
                        "con": bool(sd.get("con", False)),
                        "neutral": bool(sd.get("neutral", False)),
                    }

                    # keep only quotes that belong to this speaker and pass sanitization
                    wl = set(allowed_quotes_map.get(sid, []))
                    kept_reasoning = []
                    for it in (r.get("reasoning") or []):
                        q = _sanitize_quote((it or {}).get("quote"))
                        if q and (sid in allowed_quotes_map) and (q in wl):
                            kept_reasoning.append({"quote": q})
                    r["reasoning"] = kept_reasoning

                    clean_reactions.append(r)

                resp["idea_reactions"] = clean_reactions
                record = resp

            results.append(record)

    final = {
        "metadata": {
            "meeting_id": ideas_doc.get("metadata", {}).get("meeting_id"),
            "model": args.model,
            "source": str(ideas_path)
        },
        "items": results
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote: {out_path}")

if __name__ == "__main__":
    main()
