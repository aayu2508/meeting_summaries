# extract_all_info.py

import json, argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, DefaultDict
from collections import defaultdict
from .llm_helper import init_client, chat_json, norm_key


CATEGORIES: List[Tuple[str, str]] = [
    ("ideas", "Proposals/solutions that advance the topic"),
    ("problems", "Issues, pain points, constraints"),
    ("decisions", "Choices agreed in the meeting"),
    ("actions", "Concrete to-dos or assignments"),
    ("risks", "Potential downsides or uncertainties"),
    ("criteria", "Evaluation rubrics or success measures"),
    ("questions", "Open questions raised"),
    ("facts/definitions", "Informative statements or clarifications"),
    ("data/metrics", "Numbers, KPIs, targets"),
    ("tools/tech", "Tools, vendors, integrations, partners"),
    ("deadlines/timelines", "Dates, milestones, timeframes"),
    ("stakeholders/owners", "People/roles/orgs responsible/affected"),
    ("other", "Doesn't fit elsewhere"),
]

CATEGORY_SET = {c for c, _ in CATEGORIES}
CATEGORY_ALIASES = {
    "fact": "facts/definitions",
    "facts": "facts/definitions",
    "definitions": "facts/definitions",
    "definition": "facts/definitions",
    "metric": "data/metrics",
    "metrics": "data/metrics",
    "data": "data/metrics",
    "tool": "tools/tech",
    "tools": "tools/tech",
    "tech": "tools/tech",
    "technology": "tools/tech",
    "timeline": "deadlines/timelines",
    "timelines": "deadlines/timelines",
    "deadline": "deadlines/timelines",
    "deadlines": "deadlines/timelines",
    "stakeholders": "stakeholders/owners",
    "owners": "stakeholders/owners",
}

SYSTEM_PROMPT = """You will read a meeting CHUNK (several turns) and list broad topics discussed.
Return ONLY JSON with this shape:

{
  "items": [
    { "type": "<one of: ideas | problems | decisions | actions | risks | criteria | questions | facts/definitions | data/metrics | tools/tech | deadlines/timelines | stakeholders/owners | other>",
      "summary": "<paraphrase why the topic appeared; no quotes/speakers.>"}

  ]
}

Rules:
- NO evidence, NO quotes, NO speakers, NO timestamps.
- Use short, normalized labels (lowercase except proper nouns; remove hedges like "maybe").
- Be broad, not micro-keyphrases. Prefer concepts users would recognize on a summary map.
- Deduplicate within this chunk; emit at most 6 items per category (skip the rest).
- If in doubt about category, choose "other" (sparingly).
"""

USER_TEMPLATE = """# CHUNK CONTEXT
chunk_id: {cid}
window: {start:.3f}s-{end:.3f}s

# TEXT (speaker-prefixed turns)
{turns_block}
"""

def _render_turns_block(spans: List[Dict[str, Any]]) -> str:
    lines = []
    for t in spans or []:
        s = float(t.get("start", 0.0))
        e = float(t.get("end", s))
        spk = t.get("speaker", "S?")
        txt = (t.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{s:.3f}–{e:.3f}] {spk}: {txt}")
    return "\n".join(lines)

def _canon_category(s: str) -> str:
    s0 = (s or "").strip().lower()
    s1 = CATEGORY_ALIASES.get(s0, s0)
    return s1 if s1 in CATEGORY_SET else "other"

def main():
    ap = argparse.ArgumentParser(description="Chunk-level taxonomy of 'all things discussed' (summary-only, no mentions)")
    ap.add_argument("--meeting-id", required=True, help="e.g., BAC")
    ap.add_argument("--model", default="gptnano", help="LLM: gptnano | gpt3.5 | gptfull (aliases supported)")
    ap.add_argument("--max_tokens", type=int, default=900)
    args = ap.parse_args()

    base_dir = Path("data/outputs") / args.meeting_id
    chunks_path = base_dir / f"chunks_{args.model}.json"
    out_dir = base_dir / "context_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_info.json"

    if not chunks_path.exists():
        raise SystemExit(f"chunks file not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if isinstance(chunks, dict) and "chunks" in chunks:
        chunks = chunks["chunks"]

    client = init_client()

    per_chunk_items: List[Dict[str, str]] = []
    for c in chunks:
        turns_block = _render_turns_block(c.get("spans", []))
        user_prompt = USER_TEMPLATE.format(
            cid=c["chunk_id"], start=c["start"], end=c["end"], turns_block=turns_block
        )
        resp = chat_json(
            client,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=args.max_tokens,
            reasoning_effort="minimal",
            verbosity="low"
        )
        if isinstance(resp, dict) and "_error" in resp:
            print("[llm_error]", resp["_error"])
            continue

        items = resp.get("items") if isinstance(resp, dict) else []
        if not isinstance(items, list):
            print("[warn] non-list items; raw:", resp)
            items = []

        seen = set()
        clean: List[Dict[str, str]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            cat = _canon_category(it.get("type", "other"))
            summary = (it.get("summary") or "").strip()
            if not summary:
                continue
            key = (cat, norm_key(summary))
            if key in seen:
                continue
            seen.add(key)
            clean.append({"type": cat, "summary": summary})
        per_chunk_items.extend(clean)

    # Global consolidation across chunks
    support: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    for it in per_chunk_items:
        key = (it["type"], norm_key(it["summary"]))
        support[key] += 1

    final_items: List[Dict[str, Any]] = []
    by_category_count: DefaultDict[str, int] = defaultdict(int)
    for (cat, nk), count in sorted(support.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        summary = next(it["summary"] for it in per_chunk_items if it["type"] == cat and norm_key(it["summary"]) == nk)
        final_items.append({"type": cat, "summary": summary, "support_chunks": int(count)})
        by_category_count[cat] += 1

    out = {
        "categories": [{"type": c, "description": d} for c, d in CATEGORIES],
        "items": final_items,
        "stats": {
            "total_items": len(final_items),
            "by_category": dict(sorted(by_category_count.items()))
        }
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {out_path} | items={len(final_items)}")

if __name__ == "__main__":
    main()