# llm_helper.py
import os, json, re, time, hashlib
from typing import Any, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Incase odd model naming is used, map to canonical model ids
_MODEL_ALIASES = {
    "gpt3.5": "gpt-3.5-turbo",
    "gpt-3.5": "gpt-3.5-turbo",
    "gpt35": "gpt-3.5-turbo",
    "gptnano": "gpt-5-nano",
    "gpt-5-nano": "gpt-5-nano",
    "gptfull": "gpt-5",
    "gpt-5": "gpt-5",
}

def _resolve_model(name: str) -> str:
    return _MODEL_ALIASES.get(name, name)

# GPT-5 family supports response_format={"type":"json_object"} in Chat Completions, which strongly biases valid JSON.
# For non-GPT-5 models (e.g., 3.5), it transparently falls back to normal responses.
def _supports_json_mode(model: str) -> bool:
    return model.startswith("gpt-5")

# GPT-5 family expects 'max_completion_tokens' instead of 'max_tokens'
def _uses_max_completion_tokens(model_id: str) -> bool:
    return model_id.startswith("gpt-5")

def init_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)

# Json extraction helpers
def _extract_json(txt: str) -> str:
    s = (txt or "").strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    m = re.findall(r"```(?:json)?\n([\s\S]*?)```", s, flags=re.IGNORECASE)
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

# Build a chat completion with JSON response parsing and retries
def chat_json(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2048,
    max_retries: int = 3,
    backoff: float = 0.6,
    reasoning_effort: Optional[str] = None,   # e.g., "minimal"
    verbosity: Optional[str] = None           # e.g., "low" | "medium" | "high"
) -> Dict[str, Any]:
    
    last_err = None
    model_id = _resolve_model(model)

    # Base payload used across attempts
    base_payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    if _uses_max_completion_tokens(model_id):
        base_payload["max_completion_tokens"] = max_tokens
    else:
        base_payload["max_tokens"] = max_tokens

    json_mode_payload = dict(base_payload)
    if _supports_json_mode(model_id):
        json_mode_payload["response_format"] = {"type": "json_object"}
        if reasoning_effort is not None:
            json_mode_payload["reasoning_effort"] = reasoning_effort
        if verbosity is not None:
            json_mode_payload["verbosity"] = verbosity

    for _ in range(max_retries):
        try:
            try:
                r = client.chat.completions.create(**json_mode_payload)
            except Exception as e:
                r = client.chat.completions.create(**base_payload)

            raw = r.choices[0].message.content or ""
            try:
                return _json_loads_loose(_extract_json(raw))
            except Exception:
                repair_payload = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": "You fix JSON. Return ONLY valid JSON."},
                        {"role": "user", "content": f"Fix to valid JSON only:\n\n{raw}"},
                    ],
                }

                if _uses_max_completion_tokens(model_id):
                    repair_payload["max_completion_tokens"] = max_tokens
                else:
                    repair_payload["max_tokens"] = max_tokens

                if _supports_json_mode(model_id):
                    repair_payload["response_format"] = {"type": "json_object"}
                    if reasoning_effort is not None:
                        repair_payload["reasoning_effort"] = reasoning_effort
                    if verbosity is not None:
                        repair_payload["verbosity"] = verbosity
                repair = client.chat.completions.create(**repair_payload)
                return _json_loads_loose(_extract_json(repair.choices[0].message.content or ""))

        except Exception as e:
            last_err = e
            time.sleep(backoff)
            backoff *= 2

    return {"_error": str(last_err) if last_err else "unknown_error"}

def norm_key(s: str) -> str:
    base = " ".join((s or "").lower().strip().split())
    return hashlib.md5(base.encode("utf-8")).hexdigest()

def canonical_idea_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[-_]", " ", s)
    return s