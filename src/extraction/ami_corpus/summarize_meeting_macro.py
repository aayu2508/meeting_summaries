#!/usr/bin/env python3
import json
import argparse
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "gpt-4.1-mini"

MACRO_SUMMARY_PROMPT = """
You are a meeting summarization assistant.

Your job is to read the entire meeting as one continuous event and produce a clean, high-level summary in plain English.

You MUST:
- Break the summary into logically meaningful sections.
- Create your own section headers based on the natural phases of the meeting.
- Identify major shifts in the meeting.
- Use short section headers in plain English: not technical, not forced, and not template-driven.
- Have bullet points under each section that capture the key content if needed.
- Write the summary as if you were explaining the meeting to someone who missed it.
- Make it as detailed as possible while still being high-level.

You MUST NOT:
- Rely on a predefined template.
- Use JSON.
- Provide timestamps.
- Use extremely fine-grained segmentation.
- Over-explain or list every micro event.

The tone should be concise, structured, and story-like. The goal is to give a strong high-level sense of what happened and how the meeting unfolded, including which parts were substantive vs social or procedural.
"""

def load_transcript(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_speaker_transcript(segments):
    lines = []
    for seg in segments:
        if seg.get("type") != "speech":
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker", "UNKNOWN")
        # Simple format: SPEAKER: text
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def call_model(client, transcript_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": MACRO_SUMMARY_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "Here is the full meeting transcript. "
                "Please produce a single high-level summary text file as described.\n\n"
                + transcript_text
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()


def main():
    ap = argparse.ArgumentParser(
        description="Generate a high-level meeting overview from transcript.json"
    )
    ap.add_argument(
        "--transcript",
        required=True,
        help="Path to transcript.json (AMI style diarized ASR)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Output .txt path (default: meeting_overview.txt next to transcript)",
    )
    args = ap.parse_args()

    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        raise SystemExit(f"Transcript file not found: {transcript_path}")

    out_path = (
        Path(args.output)
        if args.output
        else transcript_path.with_name("meeting_overview.txt")
    )

    segments = load_transcript(transcript_path)
    speaker_text = build_speaker_transcript(segments)
    if not speaker_text.strip():
        raise SystemExit("No speech text found in transcript.json")

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)  # expects OPENAI_API_KEY in env
    summary = call_model(client, speaker_text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(summary + "\n")

    print(f"Wrote meeting overview to: {out_path}")


if __name__ == "__main__":
    main()
