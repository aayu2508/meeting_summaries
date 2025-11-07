# run_reactions_from_mentions.py
import json, argparse, re
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
from statistics import mean
from .llm_helper import init_client, chat_json

LLM_SYSTEM = """You infer stances and evidence for the idea window, using QUOTES for reasoning and SER aggregates as context.
Do NOT invent content. Keep outputs concise and schema-accurate.

### Stance categories
- pro: supports or advances the idea
- con: raises obstacles/risks or argues against
- neutral: neither supports nor rejects; clarifying or off-topic
Usually exactly one is true (allow mixed only with clear conflicting evidence).

### Labels (do not include "neutral" here)
Use any of: felt_positive, felt_negative, high_intensity, enthusiastic, skeptical, assertive.

### Emotion from text only
Infer *emotion_text* strictly from the provided quotes (ignore acoustics):
- polarity: positive | negative | neutral
- intensity: low | medium | high
If quotes are purely factual/unclear → polarity=neutral, intensity=low.

### Evidence selection
Use at most 2-3 short, verbatim quotes per speaker (<=200 chars); prefer evaluative/decision language.

### Output (NO extra keys):
{
  "idea_id": "I-##",
  "title": "string",
  "window": [startSec, endSec],
  "idea_reactions": [
    {
      "speaker_id": "S#",
      "stance_distribution": {"pro": true|false, "con": true|false, "neutral": true|false},
      "labels": ["felt_positive" | "felt_negative" | "high_intensity" | "enthusiastic" | "skeptical" | "assertive", ...],
      "reasoning": [{"quote": "<<=200 chars exact quote>"} , ...],
      "emotion_text": { "polarity": "positive|negative|neutral", "intensity": "low|medium|high" }
    }
  ]
}
"""

LLM_USER_TPL = """Infer stance/labels and text-only emotion for this idea window.

# IDEA (fixed — do not change wording)
- idea_id: {idea_id}
- title: {title}
- window: {w0}–{w1} (seconds)

# SPEAKERS (SER aggregates are context only; use quotes for reasoning)
{speakers_block}

# QUOTES (use only these verbatim; at most 2–3 per speaker)
{quotes_block}

Return ONLY the JSON described above.
"""

# --------- LLM (SER-only inference) ----------
LLM_SYSTEM_SER = """You will infer emotion FROM SER ONLY (no text semantics).
You are given per-speaker acoustic aggregates for a time window:
- valence_mean ∈ [-1,1] (pleasantness)
- arousal_mean ∈ [-1,1] (activation)
- dominance_mean ∈ [0,1] (control/assertiveness)
- z_valence_mean, z_arousal_mean: speaker-relative z-scores (above baseline if >0)

Task: For EACH speaker, output:
- polarity: positive | neutral | negative   (from valence + z_valence)
- intensity: low | medium | high            (from arousal + z_arousal + energy)
- assertiveness: low | medium | high        (from dominance)
- rationale: ONE short sentence that ties these together (no raw numbers)

Heuristics (guidelines, not rules):
- polarity:  valence>0.3 or z_valence>0.6 → positive; valence<-0.3 or z_valence<-0.6 → negative; else neutral
- intensity: arousal>0.3 or |z_arousal|>0.8 or energy_mean>0.6 → high; arousal<-0.3 → low; else medium
- assertiveness: dominance>0.6 → high; dominance<0.4 → low; else medium

Do NOT use any text content. Do NOT mention numbers. Keep to one sentence per speaker.

Return ONLY JSON:
{
  "idea_id": "I-##",
  "ser_only": [
    { "speaker_id": "S#", "polarity": "positive|neutral|negative",
      "intensity": "low|medium|high", "assertiveness": "low|medium|high",
      "rationale": "one sentence" }
  ]
}
"""

LLM_USER_TPL_SER = """Infer SER-only emotion for this idea window. Ignore text; use only the aggregates.

- idea_id: {idea_id}
- window: {w0}–{w1} (seconds)

# SER AGGREGATES (per speaker)
{ser_block}

Return ONLY the JSON described above.
"""

ALLOWED_LABELS = {"felt_positive","felt_negative","high_intensity","enthusiastic","skeptical","assertive"}

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
        pros = (emo.get("prosody") or {})
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

def _render_quotes_block(quotes_by_spk: Dict[str, List[str]]) -> str:
    lines=[]
    for sid, qs in quotes_by_spk.items():
        for q in qs[:3]:
            q = q.strip()
            if len(q) > 200: q = q[:197] + "..."
            lines.append(f"- {sid}: {q}")
    return "\n".join(lines) if lines else "(no quotes)"

def _render_ser_block(packets: List[Dict[str,Any]]) -> str:
    # compact YAML-like listing with only numeric fields present
    lines=[]
    for p in packets:
        agg = p.get("ser_aggregate") or {}
        if not agg: 
            # still include speaker_id with empty aggregate so LLM can say medium/neutral cautiously
            lines.append(f"- speaker_id: {p['speaker_id']}")
            lines.append(f"  valence_mean: null")
            lines.append(f"  arousal_mean: null")
            lines.append(f"  dominance_mean: null")
            lines.append(f"  z_valence_mean: null")
            lines.append(f"  z_arousal_mean: null")
            lines.append(f"  energy_mean: null")
            continue
        lines.append(f"- speaker_id: {p['speaker_id']}")
        for k in ["valence_mean","arousal_mean","dominance_mean","z_valence_mean","z_arousal_mean","energy_mean"]:
            v = agg.get(k)
            lines.append(f"  {k}: {('null' if v is None else round(v,4))}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Add reactions with text-only emotion and SER-only inference (ideas/mentions unchanged).")
    ap.add_argument("--meeting-id", required=True)
    ap.add_argument("--model", default="gpt-3.5-turbo")
    ap.add_argument("--model-ser", default=None, help="Optional different model for SER-only inference")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--pad", type=float, default=0.5, help="seconds to pad idea window for SER aggregation")
    ap.add_argument("--out", default=None, help="Output JSON path (default: context_outputs/idea_reactions.json)")
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id
    ideas_path = base / "context_outputs" / "ideas_windows.json"
    chunks_path = base / "chunks.json"
    out_path = Path(args.out) if args.out else (base / "context_outputs" / "idea_reactions.json")

    assert ideas_path.exists(), f"Missing: {ideas_path}"
    assert chunks_path.exists(), f"Missing: {chunks_path}"

    ideas_doc = json.loads(ideas_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks_by_id = {c["chunk_id"]: c for c in chunks}

    client = init_client()

    results=[]
    idea_idx = 1
    for idea in ideas_doc.get("ideas", []):
        title = (idea.get("idea") or "").strip()

        # quotes from mentions (unchanged)
        quotes_from_mentions: Dict[str, List[str]] = defaultdict(list)
        for m in (idea.get("mentions") or []):
            spk = (m.get("speaker") or "").strip() or "SPEAKER_??"
            sid = _speaker_short_id(spk)
            q = (m.get("quote") or "").strip()
            if q:
                quotes_from_mentions[sid].append(q)

        for w in (idea.get("windows") or []):
            w0, w1, cid = float(w[0]), float(w[1]), int(w[2])
            idea_id = f"I-{idea_idx:02d}"
            idea_idx += 1

            chunk = chunks_by_id.get(cid)
            if not chunk:
                results.append({"idea_id": idea_id, "title": title, "window": [w0, w1], "idea_reactions": []})
                continue

            # SER aggregation window (pad & clamp). Quotes remain from mentions.
            cw0 = max(chunk["start"], w0 - args.pad)
            cw1 = min(chunk["end"],   w1 + args.pad)
            spans = _slice_spans(chunk.get("spans", []), cw0, cw1)

            # group spans by speaker for SER aggregation
            by_spk = defaultdict(list)
            for s in spans:
                spk = (s.get("speaker") or "").strip()
                by_spk[spk].append(s)

            # packets: for reactions (context) AND for SER-only pass
            packets=[]
            for spk, segs in by_spk.items():
                sid = _speaker_short_id(spk)
                agg = _aggregate_ser(segs)
                packets.append({"speaker_id": sid, "ser_aggregate": agg})
            # include speakers who have quotes but no spans (so they still appear)
            for sid, qs in quotes_from_mentions.items():
                if not any(p["speaker_id"] == sid for p in packets):
                    packets.append({"speaker_id": sid, "ser_aggregate": {}})

            # if no data at all
            if not packets and not quotes_from_mentions:
                results.append({"idea_id": idea_id, "title": title, "window": [w0, w1], "idea_reactions": []})
                continue

            # ------- 1) main reactions call (stance + labels + emotion_text) -------
            speakers_block = _render_speakers_block(packets)
            quotes_block   = _render_quotes_block(quotes_from_mentions)
            user_prompt = LLM_USER_TPL.format(
                idea_id=idea_id, title=title, w0=w0, w1=w1,
                speakers_block=speakers_block, quotes_block=quotes_block
            )
            resp = chat_json(
                client,
                model=args.model,
                system_prompt=LLM_SYSTEM,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=800
            )

            record = {"idea_id": idea_id, "title": title, "window": [w0, w1], "idea_reactions": []}
            if isinstance(resp, dict) and isinstance(resp.get("idea_reactions"), list):
                # labels whitelist, default emotion_text
                for r in resp["idea_reactions"]:
                    r["labels"] = [l for l in (r.get("labels") or []) if l in ALLOWED_LABELS]
                    if "emotion_text" not in r or not isinstance(r["emotion_text"], dict):
                        r["emotion_text"] = {"polarity":"neutral","intensity":"low"}
                record = resp

            # ------- 2) SER-only inference call (LLM interprets SER) -------
            if packets:
                ser_block = _render_ser_block(packets)
                user_ser = LLM_USER_TPL_SER.format(idea_id=idea_id, w0=w0, w1=w1, ser_block=ser_block)
                resp_ser = chat_json(
                    client,
                    model=(args.model_ser or args.model),
                    system_prompt=LLM_SYSTEM_SER,
                    user_prompt=user_ser,
                    temperature=args.temperature,
                    max_tokens=500
                )
                # merge SER-only results into reactions
                ser_map = {}
                if isinstance(resp_ser, dict) and isinstance(resp_ser.get("ser_only"), list):
                    for row_ser in resp_ser["ser_only"]:
                        sid = row_ser.get("speaker_id")
                        if sid:
                            ser_map[sid] = {
                                "polarity": row_ser.get("polarity"),
                                "intensity": row_ser.get("intensity"),
                                "assertiveness": row_ser.get("assertiveness"),
                                "rationale": row_ser.get("rationale")
                            }
                # attach per speaker
                for r in record.get("idea_reactions", []):
                    sid = r.get("speaker_id")
                    if sid in ser_map:
                        r["ser_inferred"] = ser_map[sid]

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
