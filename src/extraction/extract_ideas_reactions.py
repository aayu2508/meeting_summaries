# run_reactions_from_mentions.py
import json, argparse, re
from pathlib import Path
from typing import Dict, Any, List
from .llm_helper import init_client, chat_json

ALLOWED_LABELS = {
    "felt_positive","felt_negative","high_intensity",
    "enthusiastic","skeptical","assertive"
}
BAD_QUOTE_PAT = re.compile(r'^\.*$')

LLM_SYSTEM = """You will classify a single speaker's stance towards a given idea using ONLY the provided quote.
Do NOT invent content.

Stance categories (exactly one true unless there is clear conflict):
- pro: supports/advances the idea
- con: argues against or raises blockers/risks
- neutral: neither supports nor rejects (clarifying/off-topic)

Labels (optional; do not include "neutral"):
- Choose zero or more from: felt_positive, felt_negative, high_intensity, enthusiastic, skeptical, assertive.

Output JSON ONLY (no extra keys):
{
  "stance_distribution": {"pro": true|false, "con": true|false, "neutral": true|false},
  "labels": ["felt_positive" | "felt_negative" | "high_intensity" | "enthusiastic" | "skeptical" | "assertive", ...]
}
"""

LLM_USER = """# IDEA
{title}

# QUOTE (single-speaker; use ONLY this)
{quote}
"""

def _sanitize_quote(q: str) -> str | None:
    if not q: return None
    q = q.strip()
    if not q or q.lower() in {"n/a","na"}: return None
    if BAD_QUOTE_PAT.match(q): return None
    return q[:200]

def main():
    ap = argparse.ArgumentParser(
        description="Augment each mention in ideas_windows.json with SER emotion (by segment_id) and text-only stance/labels. Writes to a new file."
    )
    ap.add_argument("--meeting-id", required=True, help="e.g., S14")
    ap.add_argument("--model", default="gpt-3.5-turbo")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id
    ideas_path = base / "context_outputs" / "ideas_windows.json"
    chunks_path = base / "chunks_ser.json"
    out_path = base / "context_outputs" / "ideas_with_reactions.json"

    assert ideas_path.exists(), f"Missing: {ideas_path}"
    assert chunks_path.exists(), f"Missing: {chunks_path}"

    ideas_doc = json.loads(ideas_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    # Build span/emotion lookup by "chunkId#spanIndex" and native segment_id
    emotion_by_key: Dict[str, Dict[str,Any]] = {}
    for c in chunks:
        cid = c["chunk_id"]
        for idx, sp in enumerate(c.get("spans", []) or []):
            emo = sp.get("emotion")
            if isinstance(emo, dict):
                emotion_by_key[f"{cid}#{idx}"] = emo
                seg_native = sp.get("segment_id")
                if isinstance(seg_native, str):
                    emotion_by_key[seg_native] = emo

    client = init_client()

    new_doc = json.loads(json.dumps(ideas_doc))  # deep copy
    for idea in new_doc.get("ideas", []):
        title = (idea.get("idea") or "").strip()
        for m in (idea.get("mentions") or []):
            # 1) Attach SER emotion from segment_id
            segk = m.get("segment_id")
            if isinstance(segk, str) and segk in emotion_by_key:
                m["emotion"] = emotion_by_key[segk]

            # 2) Infer stance/labels using LLM
            quote = _sanitize_quote(m.get("quote") or "")
            if not quote:
                m["stance_distribution"] = {"pro": False, "con": False, "neutral": True}
                m["labels"] = []
                continue

            user_prompt = LLM_USER.format(title=title, quote=quote)
            resp = chat_json(
                client,
                model=args.model,
                system_prompt=LLM_SYSTEM,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=250
            )

            stance = {"pro": False, "con": False, "neutral": True}
            labels: List[str] = []

            if isinstance(resp, dict):
                sd = resp.get("stance_distribution") or {}
                stance = {
                    "pro": bool(sd.get("pro", False)),
                    "con": bool(sd.get("con", False)),
                    "neutral": bool(sd.get("neutral", False)),
                }
                labels = [l for l in (resp.get("labels") or []) if l in ALLOWED_LABELS]

            m["stance_distribution"] = stance
            m["labels"] = labels

    out_path.write_text(json.dumps(new_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote augmented file: {out_path}")

if __name__ == "__main__":
    main()
