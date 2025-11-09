# extract_meeting_insights.py
import json, re, argparse, difflib, hashlib
from pathlib import Path
from typing import Dict, Any, List
from .llm_helper import init_client, chat_json

SYSTEM_PROMPT = """You are an information extractor. Return ONLY valid JSON that matches the schema below.
NO extra keys, NO comments, NO markdown, NO prose outside JSON.

SCOPE
- Extract ONLY project/decision content: purpose/plan, explicit roles, requested/desired changes, evaluation criteria/targets.
- Ignore icebreakers, small talk, jokes, equipment/recording chatter, and room logistics.

EVIDENCE
- Any item with a "quote" must be a verbatim substring from TURNS.
- Include exactly one "times": [[start, end]] copied from the same line; if timestamps absent, use [].
- quotes ≤ 200 chars.

SPEAKERS/ROLES
- Use speaker names exactly as in TURNS.
- Assign a role ONLY when it is explicitly stated (e.g., “I'm the project manager”). Otherwise role=null and role_confidence="low".

PARAPHRASE
- For every quote include a 1-2 sentence paraphrase that clarifies without adding new facts.

DESIRED CHANGES
- Observations made by the particpants.

EVIDENCE CRITERIA
- Identify as many distinct evaluation criteria as possible.
- If a participant thinks something is important to judge but gives no concrete targets/evidence, still include it with empty targets and evidence.

SCHEMA (match exactly)
{
  "chunk_id": "string",
  "time_range_sec": [0.0, 0.0],
  "meeting_about": "string|null",
  "participants": [
    { "name": "string", "role": "string|null", "role_confidence": "high|medium|low" }
  ],
  "requirements": [
    {
      "speaker": "string",
      "quote": "string",
      "paraphrase": "string",
      "times": [[start_float, end_float]] | [],
      "rationale": "string|null",
      "acceptance_criterion": "string|null"
    }
  ],
  "intention": [
      {
        "speaker": "string",
        "quote": "string",
        "paraphrase": "string",
        "times": [[start_float, end_float]],
        "rationale": "string"
      },
  ],
  "evaluation_criteria": [
    {
      "name": "string",
      "definition": "string",
      "targets": ["string"] | [],
      "evidence": [
        { "speaker": "string", "quote": "string", "paraphrase": "string", "times": [[start_float, end_float]] | [] }
      ]
    }
  ]
}

OUTPUT
- Return a SINGLE JSON object that matches the schema exactly.
"""

USER_TEMPLATE = """# METADATA
meeting_type: {meeting_type}
topic: {topic}
num_participants: {num_participants}
duration_sec: {duration_sec}
facilitator: {facilitator}
objectives: {objectives}
expected_outputs: {expected_outputs}
org: {org}
team: {team}

# CHUNK
chunk_id: {cid}
time_range_sec: [{start:.3f}, {end:.3f}]

# TURNS
(Quotes must be verbatim substrings; include matching times [[start,end]] from the same line.)
{turns_block}

# TASKS
1) meeting_about — If purpose/plan is stated here, summarize in ≤2 sentences; else null.
2) participants — Only speakers in these TURNS; roles only when explicitly stated; else role=null, role_confidence="low".
3) requirements — Requested features/behaviors/constraints. Include speaker, quote, paraphrase, times; add rationale/acceptance_criterion only if explicitly stated or clearly implied in the same turn.
4) evaluation_criteria — How options will be judged (e.g., cost/budget, timeline/schedule, performance, reliability, scalability, security/privacy, usability/accessibility, compliance, interoperability, internationalization/localization, maintainability/operability, success metrics/impact, risk/feasibility). Provide name, concise definition, any explicit numeric/threshold/date targets as strings, and ≥1 evidence quote with times. Infer as mnay criterias as you can.

# OUTPUT
Return ONLY the JSON object conforming to the schema.
"""

def load_kv_file(path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if (v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}")):
            try:
                meta[k] = json.loads(v); continue
            except json.JSONDecodeError:
                pass
        try:
            meta[k] = float(v) if "." in v else int(v)
        except ValueError:
            meta[k] = v
    return meta

def render_turns_block(spans: List[Dict[str, Any]]) -> str:
    lines = []
    for t in spans or []:
        if t.get("type") and t["type"] != "speech":
            continue
        s = float(t.get("start", 0.0)); e = float(t.get("end", s))
        spk = (t.get("speaker") or "S?").strip()
        txt = (t.get("text") or "").strip().replace("\n", " ")
        lines.append(f"[{s:.3f}-{e:.3f}] {spk}: {txt}")
    return "\n".join(lines)

def speakers_in_turns(spans):
    return { (t.get("speaker") or "").strip() for t in spans if (t.get("type") in (None, "speech")) }

def turn_index(spans):
    idx = {}
    for t in spans or []:
        if t.get("type") and t["type"] != "speech":
            continue
        spk = (t.get("speaker") or "").strip()
        s = float(t.get("start", 0.0)); e = float(t.get("end", s))
        txt = (t.get("text") or "").strip()
        idx.setdefault(spk, []).append((s, e, txt))
    return idx

def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").lower()).strip()

def diffsim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=_norm(a), b=_norm(b)).ratio()

def quote_verbatim_in_turns(speaker, quote, times, idx) -> bool:
    if not speaker or not quote: return False
    turns = idx.get(speaker, [])
    if not turns: return False
    qnorm = " ".join(quote.split())
    if times and isinstance(times, list) and times and len(times[0]) == 2:
        qs, qe = float(times[0][0]), float(times[0][1])
        for (s,e,txt) in turns:
            if s - 1e-6 <= qs and qe <= e + 1e-6 and qnorm in " ".join(txt.split()):
                return True
        return False
    else:
        return any(qnorm in " ".join(txt.split()) for (_,_,txt) in turns)

import re

# Compile once
_RE_GREETING = re.compile(r"\b(hi|hello|hey|good (morning|afternoon|evening)|thanks|thank you|welcome)\b", re.I)
_RE_CLOSING  = re.compile(r"\b(thanks.*(everyone|all)|see you|bye|goodbye|let's wrap|wrapping up)\b", re.I)
_RE_AV       = re.compile(r"\b(mic|microphone|mute|unmute|camera|record(ing|er)?|screen share|projector|speaker|volume|battery|adapter|hdmi)\b", re.I)
_RE_TIMEKEEP = re.compile(r"\b(\d+\s*(minutes?|mins?)\s*(left|to go)|time (left|check)|we are (out of|over) time|five minutes to (the )?end)\b", re.I)
_RE_BREAK    = re.compile(r"\b(coffee|bio|lunch) break|stretch break|take five\b", re.I)
_RE_META     = re.compile(r"\b(off the record|on the record|for the recording|start the recording|is this being recorded)\b", re.I)
_RE_SMALL    = re.compile(r"\b(weather|traffic|weekend|sports game|holidays?)\b", re.I)
_RE_LAUGH    = re.compile(r"\b(lol|haha|laughs|laughter)\b", re.I)

_WHITELIST_TERMS = {
    "requirement","requirements","acceptance criterion","acceptance criteria","deliverable","scope","out of scope",
    "must","should","needs to","needs-to","need to","shall","constraint","tradeoff","trade-off",
    "budget","cost","price","timeline","deadline","milestone","estimate","eta",
    "perf","performance","latency","throughput","reliability","availability","uptime","sla",
    "security","privacy","compliance","gdpr","soc 2","hipaa","pci",
    "usability","accessibility","a11y","ux","ui",
    "scalability","maintainability","operability","observability","monitoring","logging","alerting",
    "integration","api","sdk","protocol","interoperability",
    "localization","internationalization","i18n","l10n",
    "risk","feasibility","assumption","dependency","blocker",
}

def _contains_any(text: str, regexes) -> bool:
    return any(r.search(text) for r in regexes)

def _has_whitelist_signal(text: str, whitelist_terms=None) -> bool:
    wl = set(_WHITELIST_TERMS)
    if whitelist_terms:
        wl |= {w.lower() for w in whitelist_terms}
    t = text.lower()
    return any(w in t for w in wl)

def is_hard_negative(
    text: str,
    *,
    custom_stoplist: list[str] | None = None,
    whitelist_terms: list[str] | None = None
) -> bool:
    if not text:
        return False

    t = text.strip()
    if _has_whitelist_signal(t, whitelist_terms):
        return False

    if _contains_any(t, (_RE_GREETING, _RE_CLOSING, _RE_AV, _RE_TIMEKEEP, _RE_BREAK, _RE_META, _RE_SMALL, _RE_LAUGH)):
        return True

    if custom_stoplist:
        tl = t.lower()
        for s in custom_stoplist:
            if s and s.lower() in tl:
                return True

    return False

def midpoint(a: float, b: float) -> float:
    return 0.5 * (a + b)

def overlap_len(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def merge_participants(dst: List[Dict[str, Any]], src: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by = {p["name"]: p for p in dst}
    rank = {"low": 0, "medium": 1, "high": 2}
    for p in src or []:
        nm = p.get("name")
        if not nm: continue
        if nm not in by:
            by[nm] = dict(p)
        else:
            cur = by[nm]
            if (p.get("role") and not cur.get("role")) or (rank.get(p.get("role_confidence","low"),0) > rank.get(cur.get("role_confidence","low"),0)):
                by[nm] = dict(p)
    return sorted(by.values(), key=lambda x: x["name"])

def merge_evidence(dst: List[Dict[str, Any]], src: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def h(ev):
        t = ev.get("times") or [[None,None]]
        s,e = t[0]
        m = None
        if isinstance(s,(int,float)) and isinstance(e,(int,float)):
            m = round(midpoint(float(s), float(e)), 1)
        key = f"{ev.get('speaker','')}|{_norm(ev.get('quote',''))}|{m}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()
    out = list(dst or [])
    seen = {h(x) for x in out}
    for e in src or []:
        he = h(e)
        if he in seen: continue
        out.append(e); seen.add(he)
    return out

def merge_lists(dst: List[Dict[str, Any]], src: List[Dict[str, Any]], is_eval: bool=False) -> List[Dict[str, Any]]:
    if is_eval:
        def slug(s): return re.sub(r"[^a-z0-9]+", "-", (s or "").strip().lower()).strip("-")
        by = {slug(i["name"]): dict(i) for i in (dst or []) if i.get("name")}
        for it in src or []:
            nm = it.get("name")
            if not nm: continue
            k = slug(nm)
            if k not in by:
                by[k] = dict(it)
            else:
                if len((it.get("definition") or "")) > len((by[k].get("definition") or "")):
                    by[k]["definition"] = it.get("definition")
                tgt = set(by[k].get("targets") or [])
                tgt |= {str(t).strip() for t in (it.get("targets") or []) if str(t).strip()}
                by[k]["targets"] = sorted(tgt)
                by[k]["evidence"] = merge_evidence(by[k].get("evidence") or [], it.get("evidence") or [])
        return sorted(by.values(), key=lambda x: x["name"])
    else:
        out = list(dst or [])
        for it in src or []:
            spk = it.get("speaker",""); q = (it.get("quote") or "").strip()
            if not spk or not q or is_hard_negative(q): continue
            times = it.get("times") or []
            s,e = (times[0] if (times and len(times[0])==2) else (None,None))
            dup = None
            for idx, ex in enumerate(out):
                if ex.get("speaker") != spk: continue
                q2 = (ex.get("quote") or "").strip()
                simv = diffsim(q, q2)
                t2 = ex.get("times") or []
                s2,e2 = (t2[0] if (t2 and len(t2[0])==2) else (None,None))
                close = False
                if all(isinstance(x,(int,float)) for x in [s,e,s2,e2]):
                    ov = overlap_len(s,e,s2,e2)
                    close = (ov >= 0.6 * (e - s + 1e-9)) or (abs(midpoint(s,e) - midpoint(s2,e2)) <= 1.5)
                if simv >= 0.85 and (close or (s is None and s2 is None)):
                    dup = idx; break
            if dup is None:
                out.append(dict(it))
            else:
                ex = out[dup]
                if len(q) > len(ex.get("quote","")):
                    ex["quote"] = q
                    if it.get("paraphrase"): ex["paraphrase"] = it["paraphrase"]
                for k in ("rationale","acceptance_criterion"):
                    if it.get(k) and not ex.get(k): ex[k] = it[k]
                if (not ex.get("times")) and it.get("times"): ex["times"] = it["times"]
        return out

def validate_chunk_output(step: Dict[str,Any], spans: List[Dict[str,Any]], require_times: bool=True) -> Dict[str,Any]:
    if not isinstance(step, dict): return {}
    speakers = speakers_in_turns(spans); idx = turn_index(spans)

    def fix_list(items, fields):
        out = []
        for it in items or []:
            spk = (it.get("speaker") or "").strip()
            q = (it.get("quote") or "").strip()
            times = it.get("times") or []
            if not spk or spk not in speakers: continue
            if not q or is_hard_negative(q): continue
            if require_times and not times: continue
            if not quote_verbatim_in_turns(spk, q, times, idx): continue
            clean = {k: it.get(k) for k in fields}
            if "paraphrase" in fields and not (clean.get("paraphrase") or "").strip(): continue
            out.append(clean)
        return out

    part = []
    seen = set()
    for p in step.get("participants") or []:
        nm = (p.get("name") or "").strip()
        if not nm or nm not in speakers: continue
        role = p.get("role") if isinstance(p.get("role"), str) and p.get("role").strip() else None
        rc = p.get("role_confidence","low") if role else "low"
        key = (nm, role or "")
        if key in seen: continue
        seen.add(key)
        part.append({"name": nm, "role": role, "role_confidence": rc})
    step["participants"] = part

    step["desired_changes"] = fix_list(
        step.get("desired_changes"),
        ["speaker","quote","paraphrase","times","rationale","acceptance_criterion"]
    )

    ev_clean = []
    for ev in step.get("evaluation_criteria") or []:
        name = (ev.get("name") or "").strip()
        definition = (ev.get("definition") or "").strip()
        targets = [str(t).strip() for t in (ev.get("targets") or []) if str(t).strip()]
        ev_evd = fix_list(ev.get("evidence"), ["speaker","quote","paraphrase","times"])
        if name and ev_evd:
            ev_clean.append({"name": name, "definition": definition, "targets": targets, "evidence": ev_evd})
    step["evaluation_criteria"] = ev_clean

    about = (step.get("meeting_about") or "").strip()
    if about and is_hard_negative(about): step["meeting_about"] = None
    return step

def main():
    ap = argparse.ArgumentParser(description="Meeting extractor (domain-agnostic)")
    ap.add_argument("--meeting-id", required=True)
    ap.add_argument("--model", default="gpt-3.5-turbo")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--require-times", action="store_true")
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id
    chunks_path = base / "chunks.json"
    out_dir = base / "context_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not chunks_path.exists():
        raise SystemExit(f"chunks.json not found: {chunks_path}")

    # Optional metadata (not required)
    meta = {}
    for ext in (".json", ".txt"):
        p = base / f"metadata{ext}"
        if p.exists():
            meta = json.loads(p.read_text(encoding="utf-8")) if ext == ".json" else load_kv_file(p)
            break

    client = init_client()
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    state: Dict[str, Any] = {
        "metadata": {**meta, "meeting_id": args.meeting_id, "model": args.model},
        "meeting_about": None,
        "participants": [],
        "desired_changes": [],
        "evaluation_criteria": []
    }

    for c in chunks:
        spans = c.get("spans", []) or []
        turns_block = render_turns_block(spans)
        user_prompt = USER_TEMPLATE.format(
            meeting_type=meta.get("meeting_type",""),
            topic=meta.get("topic",""),
            num_participants=meta.get("num_participants", 0),
            duration_sec=meta.get("duration_sec", 0.0),
            facilitator=meta.get("facilitator",""),
            objectives=json.dumps(meta.get("objectives", []), ensure_ascii=False),
            expected_outputs=json.dumps(meta.get("expected_outputs", []), ensure_ascii=False),
            org=meta.get("org",""),
            team=meta.get("team",""),
            cid=c["chunk_id"],
            start=c["start"],
            end=c["end"],
            turns_block=turns_block,
        )

        step = chat_json(
            client,
            model=args.model,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=args.temperature
        )

        step = validate_chunk_output(step, spans=spans, require_times=args.require_times)
        if not isinstance(step, dict):
            step = {}

        about = step.get("meeting_about")
        if isinstance(about, str) and about.strip():
            if len(about) > len(state.get("meeting_about") or ""):
                state["meeting_about"] = about.strip()

        if isinstance(step.get("participants"), list):
            state["participants"] = merge_participants(state["participants"], step["participants"])

        if isinstance(step.get("desired_changes"), list):
            state["desired_changes"] = merge_lists(state["desired_changes"], step["desired_changes"], is_eval=False)

        if isinstance(step.get("evaluation_criteria"), list):
            state["evaluation_criteria"] = merge_lists(state["evaluation_criteria"], step["evaluation_criteria"], is_eval=True)

    # Final ordering
    state["participants"] = sorted(state["participants"], key=lambda p: p["name"])

    def first_time(item):
        t = item.get("times") or []
        return t[0][0] if (t and t[0] and len(t[0]) == 2) else 1e18

    state["desired_changes"]     = sorted(state["desired_changes"], key=first_time)
    state["evaluation_criteria"] = sorted(state["evaluation_criteria"], key=lambda x: x.get("name",""))

    out_path = out_dir / "meeting_insights.json"
    out_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {out_path}")

if __name__ == "__main__":
    main()
