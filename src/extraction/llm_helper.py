# llm_helper.py
import os, json, re, time, hashlib
from typing import Any, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initializes the open ai client which is authenticated
def init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)

# Extracts JSON content from the model's response
def _extract_json(txt: str) -> str:
    s = (txt or "").strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    m = re.findall(r"```(?:json)?\n([\s\S]*?)```", s)
    if m:
        return m[0].strip()
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i:j+1].strip()
    return s  

def _json_loads_loose(s: str) -> Dict[str, Any]:
    s2 = re.sub(r",\s*([}\]])", r"\1", s) 
    s2 = s2.replace("NaN", "null").replace("Infinity", "null").replace("-Infinity", "null")
    return json.loads(s2)

# Sends a chat completion request to the OpenAI API and attempts to parse the response as JSON.
def chat_json(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_retries: int = 3,
    backoff: float = 0.6
) -> Dict[str, Any]:
    last_err = None
    for _ in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = r.choices[0].message.content or ""
            try:
                return _json_loads_loose(_extract_json(raw))
            except Exception:
                # Attempt to repair malformed JSON using the model itself
                repair = client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": "You fix JSON. Return ONLY valid JSON."},
                        {"role": "user", "content": f"Fix to valid JSON only:\n\n{raw}"},
                    ],
                )
                return _json_loads_loose(_extract_json(repair.choices[0].message.content or ""))
        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff *= 2
    return {"_error": str(last_err) if last_err else "unknown_error"}

# Eliminates redundant data by storing only one unique copy of a data block or file
def norm_key(s: str) -> str:
    base = " ".join((s or "").lower().strip().split())
    return hashlib.md5(base.encode("utf-8")).hexdigest()
