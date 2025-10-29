#!/usr/bin/env python3
"""
Rolling context over chunks to extract:
Idea -> Evaluation Criteria -> Statements (quote + stance).

No speaker attribution or timecodes.
"""

import os, json, argparse, hashlib
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Prompt ----------
SYSTEM_PROMPT = """You analyze brainstorming meetings and identify ideas, their evaluation criteria,
and statements made about those criteria.

Given a previous STATE and a new TRANSCRIPT CHUNK, update STATE and return ONLY JSON.

Schema (required):
{
  "ideas": [
    {
      "idea": "short idea name or claim",
      "criteria": [
        {
          "name": "evaluation criterion (e.g., cost, feasibility, risk, timeline, performance, ROI, usability)",
          "statements": [
            {
              "quote": "short, faithful excerpt or paraphrase (<=200 chars)",
              "stance": "pro|con|neutral"
            }
          ]
        }
      ]
    }
  ],
  "open_questions": ["optional questions asked verbatim or near-verbatim"],
  "summary_so_far": "<=8 short bullets or compact paragraph summarizing progress"
}

Rules:
- DO NOT INVENT anything not found in the chunk.
- Preserve previous ideas/criteria/statements.
- Append new statements if new evidence appears.
- Keep quotes short and factual; classify stance as pro, con, or neutral.
"""

USER_TEMPLATE = """Previous STATE JSON (may be empty on first chunk):
{prev_state}

New TRANSCRIPT CHUNK (chunk_id={cid}, {start:.1f}s–{end:.1f}s):
{transcript}

Update the STATE according to the schema and rules. Respond with JSON only.
"""

# ---------- Helpers ----------
def md5_key(s: str) -> str:
    base = " ".join((s or "").lower().strip().split())
    return hashlib.md5(base.encode("utf-8")).hexdigest()

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
    if txt.startswith("```"):
        txt = txt.strip("`")
        if "\n" in txt:
            txt = txt.split("\n", 1)[1]
        if txt.endswith("```"):
            txt = txt[:-3]
    try:
        return json.loads(txt)
    except Exception:
        return {"_raw": txt}

def _merge_lists_dedup(a, b):
    seen = set()
    out = []
    for s in (a or []) + (b or []):
        if not s: 
            continue
        key = " ".join(s.lower().strip().split())
        if key in seen:
            continue
        seen.add(key)
        out.append(s.strip())
    return out

def merge_states(prev_state: dict, new_state: dict) -> dict:
    """Merge ideas + criteria + statements defensively."""
    out = {
        "summary_so_far": new_state.get("summary_so_far") or prev_state.get("summary_so_far") or "",
        "open_questions": _merge_lists_dedup(prev_state.get("open_questions"), new_state.get("open_questions")),
        "ideas": []
    }

    pool = (prev_state.get("ideas") or []) + (new_state.get("ideas") or [])
    idea_map = {}

    for it in pool:
        idea_txt = (it.get("idea") or "").strip()
        if not idea_txt:
            continue
        ik = md5_key(idea_txt)
        dst = idea_map.setdefault(ik, {"idea": idea_txt, "criteria": {}})

        for c in (it.get("criteria") or []):
            cname = (c.get("name") or "").strip()
            if not cname:
                continue
            ck = md5_key(cname)
            crit = dst["criteria"].setdefault(ck, {"name": cname, "statements": []})

            # merge statements
            existing_quotes = {s["quote"].lower().strip() for s in crit["statements"] if "quote" in s}
            for st in (c.get("statements") or []):
                q = (st.get("quote") or "").strip()
                stance = (st.get("stance") or "neutral").lower()
                if not q or q.lower().strip() in existing_quotes:
                    continue
                crit["statements"].append({"quote": q[:200], "stance": stance})

    # flatten criteria
    for v in idea_map.values():
        v["criteria"] = list(v["criteria"].values())
        out["ideas"].append(v)

    return out

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Criteria extractor (quotes only, no speakers/timecodes)")
    ap.add_argument("--meeting-id", required=True, help="e.g., m1, m2")
    ap.add_argument("--model", default="gpt-3.5-turbo")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    base_dir = Path("data/outputs") / args.meeting_id
    chunks_path = base_dir / "chunks.json"
    out_dir = base_dir / "context_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise SystemExit(f"❌ chunks.json not found: {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"[load] {len(chunks)} chunks from {chunks_path}")

    state = {"ideas": [], "open_questions": [], "summary_so_far": ""}

    for c in chunks:
        user_prompt = USER_TEMPLATE.format(
            prev_state=json.dumps(state, ensure_ascii=False, indent=2),
            cid=c["chunk_id"], start=c["start"], end=c["end"],
            transcript=c["text"],
        )
        print(f"[chunk {c['chunk_id']}] → {args.model}")
        step_json = json_from_model(args.model, SYSTEM_PROMPT, user_prompt, temperature=args.temperature)

        # save raw model output for inspection
        (out_dir / f"crit_chunk_{c['chunk_id']:03d}.json").write_text(
            json.dumps(step_json, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # merge with prior state
        state = merge_states(state, step_json)

    # save final merged structure
    (out_dir / "final_criteria.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] final_criteria.json written to {out_dir}")

if __name__ == "__main__":
    main()
