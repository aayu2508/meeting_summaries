#!/usr/bin/env python3
"""
Run GPT-3.5 over meeting chunks with rolling memory AND speaker-linked ideas.

Outputs:
- data/outputs/<meeting_id>/context_outputs/attrib_chunk_XXX.json   (per-chunk JSON)
- data/outputs/<meeting_id>/context_outputs/final_state.json         (running state after last chunk)
- data/outputs/<meeting_id>/context_outputs/speaker_ideas.json       (per-speaker aggregation)
"""
import os, json, argparse, hashlib
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are an analyst of brainstorming meetings.
You receive the previous state and a new transcript chunk. Update the state and return ONLY JSON.
Schema (keys required):
{
  "summary_so_far": "short running summary in <= 8 bullets or a compact paragraph",
  "ideas": [
    {
      "idea": "short idea title or claim",
      "speakers": ["S0","S1", ...],
      "evidence": [
        {"speaker":"S0","quote":"short quote","timecodes":[[start,end]]}
      ]
    }
  ],
  "open_questions": ["..."],
  "topics": ["optional tags/areas, short strings"]
}
Rules:
- Preserve previous ideas; do NOT drop earlier ideas unless contradicted.
- When a new chunk adds support or mentions an existing idea, append speakers and evidence.
- Keep quotes short; include timecodes if present in the chunk context.
"""

USER_TEMPLATE = """Previous state JSON (may be empty on first chunk):
{prev_state}

New meeting chunk (chunk_id={cid}, {start:.1f}s–{end:.1f}s):
{transcript}

Update the JSON state according to the schema and rules. Respond with JSON only.
"""

def json_from_model(model: str, system_prompt: str, user_prompt: str, temperature: float = 0.3):
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    txt = r.choices[0].message.content.strip()
    # tolerate code fences
    if txt.startswith("```"):
        txt = txt.strip("`")
        if "\n" in txt:
            txt = txt.split("\n", 1)[1]
        if txt.endswith("```"):
            txt = txt[:-3]
    try:
        return json.loads(txt)
    except Exception:
        # if parsing fails, wrap raw for debugging but keep pipeline moving
        return {"_raw": txt}

def normalize_idea_key(s: str) -> str:
    base = " ".join((s or "").lower().strip().split())
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def merge_states(prev_state: dict, new_state: dict) -> dict:
    out = {
        "summary_so_far": new_state.get("summary_so_far") or prev_state.get("summary_so_far") or "",
        "ideas": [],
        "open_questions": [],
        "topics": [],
    }

    # collect all ideas from both
    pool = []
    for src in (prev_state.get("ideas") or []):
        pool.append(src)
    for src in (new_state.get("ideas") or []):
        pool.append(src)

    # merge ideas by key
    by_key = {}
    for it in pool:
        idea_txt = it.get("idea", "").strip()
        if not idea_txt:
            continue
        k = normalize_idea_key(idea_txt)
        dst = by_key.setdefault(k, {"idea": idea_txt, "speakers": set(), "evidence": []})
        for sp in it.get("speakers") or []:
            if sp: dst["speakers"].add(sp)
        for ev in it.get("evidence") or []:
            # keep small evidence dicts
            dst["evidence"].append({
                "speaker": ev.get("speaker"),
                "quote": (ev.get("quote") or "")[:240],
                "timecodes": ev.get("timecodes") or [],
            })

    out["ideas"] = [
        {"idea": v["idea"], "speakers": sorted(list(v["speakers"])), "evidence": v["evidence"]}
        for v in by_key.values()
    ]

    # open questions & topics (dedupe)
    oq = (prev_state.get("open_questions") or []) + (new_state.get("open_questions") or [])
    tp = (prev_state.get("topics") or []) + (new_state.get("topics") or [])
    out["open_questions"] = sorted(set([q for q in oq if q]))
    out["topics"] = sorted(set([t for t in tp if t]))
    return out

def build_per_speaker(state: dict) -> dict:
    per = {}
    for it in state.get("ideas") or []:
        for sp in it.get("speakers") or []:
            per.setdefault(sp, []).append({
                "idea": it.get("idea"),
                "evidence": it.get("evidence") or [],
            })
    return per

def main():
    ap = argparse.ArgumentParser(description="Context runner with speaker-linked ideas")
    ap.add_argument("--meeting-id", required=True, help="e.g., m1, m2")
    ap.add_argument("--model", default="gpt-3.5-turbo")
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    base_dir = Path("data/outputs") / args.meeting_id
    chunks_path = base_dir / "chunks.json"
    out_dir = base_dir / "context_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise SystemExit(f"chunks.json not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"[load] {len(chunks)} chunks from {chunks_path}")

    # rolling JSON state
    state = {"summary_so_far": "", "ideas": [], "open_questions": [], "topics": []}

    for c in chunks:
        user_prompt = USER_TEMPLATE.format(
            prev_state=json.dumps(state, ensure_ascii=False, indent=2),
            cid=c["chunk_id"], start=c["start"], end=c["end"],
            transcript=c["text"],
        )

        print(f"[chunk {c['chunk_id']}] → {args.model}")
        new_state = json_from_model(args.model, SYSTEM_PROMPT, user_prompt, temperature=args.temperature)
        # write the model's direct output for inspection
        (out_dir / f"attrib_chunk_{c['chunk_id']:03d}.json").write_text(
            json.dumps(new_state, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # merge defensively with our previous state
        state = merge_states(state, new_state)

    # save final state
    (out_dir / "final_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    # build per-speaker mapping
    per_speaker = build_per_speaker(state)
    (out_dir / "speaker_ideas.json").write_text(json.dumps(per_speaker, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] final_state.json and speaker_ideas.json written to {out_dir}")

if __name__ == "__main__":
    main()
