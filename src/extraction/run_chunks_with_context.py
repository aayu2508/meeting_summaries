#!/usr/bin/env python3
"""
Sequentially send each chunk to GPT-3.5-Turbo with running context memory.
Each new chunk includes the previous summary so the model "remembers" ideas.
"""

import os, json, argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def main():
    ap = argparse.ArgumentParser(description="Run GPT-3.5 over meeting chunks with context memory")
    ap.add_argument("--meeting-id", required=True, help="Meeting ID (e.g., m1, m2, etc.)")
    ap.add_argument("--model", default="gpt-3.5-turbo", help="Model name")
    ap.add_argument("--temperature", type=float, default=0.4, help="Response randomness")
    args = ap.parse_args()

    # --- paths ---
    base_dir = Path("data/outputs") / args.meeting_id
    chunks_path = base_dir / "chunks.json"
    out_dir = base_dir / "context_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load chunks ---
    if not chunks_path.exists():
        raise SystemExit(f"chunks.json not found for meeting {args.meeting_id} → {chunks_path}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"[gpt] Loaded {len(chunks)} chunks for meeting {args.meeting_id}")

    # --- prompts ---
    SYSTEM_PROMPT = """You are an intelligent meeting summarizer and idea tracker.
        As you read each chunk of a brainstorming meeting, update the running list of ideas, topics, and open questions.
        Preserve all previously discussed ideas unless explicitly contradicted or replaced.
        Output in JSON with keys: summary_so_far, ideas, open_questions.
        """

    running_context = ""  # memory starts empty
    all_results = []

    # --- main loop ---
    for i, c in enumerate(chunks):
        user_prompt = f"""Previous context:
{running_context or "None yet"}

New meeting chunk ({c['chunk_id']}, {c['start']:.1f}s–{c['end']:.1f}s):

{c['text']}

Please update the running summary and idea list."""
        print(f"[chunk {c['chunk_id']}] sending to GPT...")

        response = client.chat.completions.create(
            model=args.model,
            temperature=args.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )

        output = response.choices[0].message.content.strip()
        out_file = out_dir / f"chunk_{c['chunk_id']:03d}.txt"
        out_file.write_text(output, encoding="utf-8")
        print(f"  → saved {out_file.name}")

        # update memory
        running_context = output
        all_results.append({"chunk_id": c["chunk_id"], "result": output})

    # --- save final summary ---
    final_summary_path = out_dir / "final_summary.txt"
    final_summary_path.write_text(running_context, encoding="utf-8")
    print(f"\nAll chunks processed.")
    print(f"[final summary] saved to {final_summary_path}")

if __name__ == "__main__":
    main()
