# extract_sub_ideas.py
import json, argparse, re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set
from .llm_helper import chat_json, init_client, canonical_idea_text

SYSTEM = """You will group a flat list of ideas into major ideas and their sub-ideas.
Return ONLY JSON in this shape:
{
  "groups": [
    { "major": "concise major idea (<= 8 words)",
      "subs": ["sub idea 1", "sub idea 2", "..."] }
  ]
}
Rules:
- Have atmost 7 major ideas.
- Sub-ideas must be chosen from the provided list ONLY (no new text).
- A “major idea” is a high-level proposal/decision that directly advances the topic.
- Sub-ideas are implementations, variants, or examples under a major idea.
- Prefer precision over recall. It’s OK to leave some items unassigned if they’re noise.
- JSON only; no comments, no trailing commas, no extra keys.
"""

USER_TPL = """# TOPIC
{topic}

# IDEAS (one per line; choose majors and map subs from this set only)
{bullets}
"""

def _merge_windows(windows: List[List[float]], eps: float = 0.5) -> List[List[float]]:
    if not windows:
        return []
    ws = sorted([[float(a), float(b)] for a, b in windows])
    merged = [ws[0]]
    for w in ws[1:]:
        prev = merged[-1]
        if w[0] <= prev[1] + eps:
            prev[1] = max(prev[1], w[1])
        else:
            merged.append(w)
    return merged

def main():
    ap = argparse.ArgumentParser(description="Group flat ideas into major → sub-ideas and build timelines")
    ap.add_argument("--meeting-id", required=True)
    ap.add_argument("--model", default="gpt-3.5-turbo")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id
    in_path = base / "context_outputs" / "ideas_windows.json"
    if not in_path.exists():
        raise SystemExit(f"Not found: {in_path}")

    doc = json.loads(in_path.read_text(encoding="utf-8"))
    topic = (doc.get("metadata") or {}).get("topic", "")
    ideas_list: List[Dict[str, Any]] = doc.get("ideas", [])

    # Build lookup: canonical text -> original idea records (there can be near-dups)
    by_text: Dict[str, Dict[str, Any]] = {}
    for it in ideas_list:
        txt = canonical_idea_text(it.get("idea", ""))
        if not txt:
            continue

        # if same text appears multiple times, merge their windows immediately
        if txt not in by_text:
            by_text[txt] = {"idea": it["idea"], "windows": list(it.get("windows", []))}
        else:
            by_text[txt]["windows"].extend(it.get("windows", []))

    # Prepare the flat list for LLM (unique canonical texts, but display original)
    unique_ideas = sorted({v["idea"] for v in by_text.values()})
    bullets = "\n".join(f"- {s}" for s in unique_ideas)

    client = init_client()
    user = USER_TPL.format(topic=topic or "(no topic provided)", bullets=bullets)
    resp = chat_json(
        client,
        model=args.model,
        system_prompt=SYSTEM,
        user_prompt=user,
        temperature=args.temperature,
    )

    groups = resp.get("groups") if isinstance(resp, dict) else []
    if not isinstance(groups, list):
        groups = []

    # Build output: for each group, compute major windows as the union of all member windows (major + subs)
    out: Dict[str, Any] = {
        "metadata": {
            "meeting_id": args.meeting_id,
            "source": str(in_path),
            "topic": topic,
            "model": args.model
        },
        "groups": []
    }

    for g in groups:
        major = canonical_idea_text(g.get("major", ""))
        subs_in = g.get("subs", []) if isinstance(g.get("subs"), list) else []

        # Collect members as canonical keys
        member_keys: Set[str] = set()
        if major:
            member_keys.add(major)
        for s in subs_in:
            member_keys.add(canonical_idea_text(s))

        # Gather original windows for each member
        members: List[Dict[str, Any]] = []
        all_spans: List[List[float]] = []   # CID-agnostic (start,end)
        all_windows_with_cid: List[List[float]] = []  # keep raw [start,end,cid] for traceability

        for mk in member_keys:
            rec = by_text.get(mk)
            if not rec:
                continue
            member_windows = rec.get("windows", [])
            # append to union accumulators
            for w in member_windows:
                # w = [start, end, cid]
                all_windows_with_cid.append([float(w[0]), float(w[1]), int(w[2])])
                all_spans.append([float(w[0]), float(w[1])])

            members.append({
                "idea": rec["idea"],
                "windows": member_windows
            })

        # Coalesce into a clean timeline (list of [start,end])
        timeline = _merge_windows(all_spans, eps=0.5)

        # Human-facing major label: prefer the original text if present; else use the normalized
        major_label = by_text.get(major, {"idea": g.get("major", "")})["idea"]

        out["groups"].append({
            "idea": major_label,
            "timeline": timeline,                # merged spans for visualization
            "windows": all_windows_with_cid,     # raw spans with cids for traceability
            "members": members                   # sub-ideas and their own windows
        })

    # Optional: include unassigned ideas as their own singleton groups
    assigned: Set[str] = set()
    for g in out["groups"]:
        assigned.add(canonical_idea_text(g["idea"]))
        for m in g["members"]:
            assigned.add(canonical_idea_text(m["idea"]))

    for txt, rec in by_text.items():
        if txt not in assigned:
            # treat as standalone minor; give it its own timeline
            spans = [[float(w[0]), float(w[1])] for w in rec.get("windows", [])]
            out["groups"].append({
                "idea": rec["idea"],
                "timeline": _merge_windows(spans),
                "windows": rec.get("windows", []),
                "members": []
            })

    out_path = base / "context_outputs" / "ideas_grouped.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {out_path}")

if __name__ == "__main__":
    main()
