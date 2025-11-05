#!/usr/bin/env python3

import os, json, hashlib
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)

def _unwrap_code_fence(s: str) -> str:
    s = s.strip()
    if not s.startswith("```"):
        return s
    s = s.strip("`")
    if "\n" in s:
        s = s.split("\n", 1)[1]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()

def chat_json(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """Send prompt to OpenAI and safely parse JSON reply."""
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    txt = _unwrap_code_fence(r.choices[0].message.content or "")
    try:
        return json.loads(txt)
    except Exception:
        return {"_raw": txt}

def norm_key(s: str) -> str:
    """Normalize a string to a stable key for deduplication."""
    base = " ".join((s or "").lower().strip().split())
    return hashlib.md5(base.encode("utf-8")).hexdigest()
