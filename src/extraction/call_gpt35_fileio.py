#!/usr/bin/env python3
"""
Call GPT-3.5-Turbo using text files for input/output for testing.

Usage:
  python call_gpt35_fileio.py \
      --system system_prompt.txt \
      --prompt user_prompt.txt \
      --out output.txt
"""

import os, argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Call GPT-3.5-Turbo using text files.")
    parser.add_argument("--system", type=Path, required=True, help="Path to system prompt .txt file")
    parser.add_argument("--prompt", type=Path, required=True, help="Path to user prompt .txt file")
    parser.add_argument("--out", type=Path, required=True, help="Where to save the model response")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.3, help="Response randomness")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENAI_API_KEY in your environment.")

    client = OpenAI(api_key=api_key)

    # read inputs 
    system_prompt = args.system.read_text(encoding="utf-8").strip()
    user_prompt = args.prompt.read_text(encoding="utf-8").strip()

    print(f"[gpt] Sending to {args.model} ...")
    response = client.chat.completions.create(
        model=args.model,
        temperature=args.temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    output_text = response.choices[0].message.content.strip()
    args.out.write_text(output_text, encoding="utf-8")

    print(f"[done] Response written to: {args.out}")

if __name__ == "__main__":
    main()
