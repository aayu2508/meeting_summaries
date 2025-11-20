#!/usr/bin/env python3
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# -------------------------
# Helpers and utilities
# -------------------------

def _safe_model_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "model"


def _load_reflected_ideas(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Reflected ideas file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read JSON: {path}\n{e}")


def speaker_short_id(s: str) -> str:
    m = re.search(r"SPEAKER_(\d+)", s or "")
    return f"S{int(m.group(1))}" if m else (s or "S?")


# -------------------------
# Idea-level participation stats
# -------------------------

def _compute_idea_stats(ideas_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    From ideas_reflected_<extract>_<reflect>.json, compute:
      - which speakers contributed to each idea
      - per-speaker mention counts and speaking time
      - who brought up the idea (first mention)
      - who is primary contributor (most mentions, then speaking time)
      - first_seen / last_seen for idea span
    Assumes schema per idea:
      {
        "idea_id": "I1",
        "canonical_idea": "...",
        "mentions": [
          {
            "segment_id": "m0001",
            "speaker": "SPEAKER_00",
            "start": 857.686,
            "end": 865.482,
            "text": "...",
            ...
          },
          ...
        ]
      }
    """
    out: List[Dict[str, Any]] = []
    ideas_list = ideas_doc.get("ideas") or []

    for idea in ideas_list:
        idea_id = idea.get("idea_id") or ""
        title = (idea.get("canonical_idea") or idea.get("idea") or idea_id or "").strip()
        mentions = idea.get("mentions") or []

        # If no mentions, record minimal info and continue
        if not mentions:
            out.append({
                "idea_id": idea_id,
                "idea": title,
                "num_mentions": 0,
                "num_speakers": 0,
                "first_seen_sec": None,
                "last_seen_sec": None,
                "span_duration_sec": 0.0,
                "initiator_speaker_full": None,
                "initiator_speaker_short": None,
                "primary_contributor_speaker_full": None,
                "primary_contributor_speaker_short": None,
                "speakers": [],
            })
            continue

        # Aggregate per-speaker stats
        per_spk: Dict[str, Dict[str, Any]] = {}
        first_seen = float("inf")
        last_seen = 0.0

        for m in mentions:
            spk_full = str(m.get("speaker", "S?"))
            start = float(m.get("start", 0.0))
            end = float(m.get("end", start))
            dur = max(0.0, end - start)

            first_seen = min(first_seen, start)
            last_seen = max(last_seen, end)

            if spk_full not in per_spk:
                per_spk[spk_full] = {
                    "speaker_full": spk_full,
                    "speaker_short": speaker_short_id(spk_full),
                    "num_mentions": 0,
                    "total_speaking_time_sec": 0.0,
                    "first_mention_time_sec": start,
                }

            st = per_spk[spk_full]
            st["num_mentions"] += 1
            st["total_speaking_time_sec"] += dur
            st["first_mention_time_sec"] = min(st["first_mention_time_sec"], start)

        # Determine initiator (brought up) = earliest first_mention_time
        initiator_full = None
        initiator_short = None
        if per_spk:
            initiator_full = min(
                per_spk.values(), key=lambda d: float(d["first_mention_time_sec"])
            )["speaker_full"]
            initiator_short = speaker_short_id(initiator_full)

        # Determine primary contributor = highest num_mentions (tie-break by speaking time)
        primary_full = None
        primary_short = None
        if per_spk:
            primary_full = max(
                per_spk.values(),
                key=lambda d: (int(d["num_mentions"]), float(d["total_speaking_time_sec"]))
            )["speaker_full"]
            primary_short = speaker_short_id(primary_full)

        # Serialize per-speaker list
        speakers_list = sorted(
            per_spk.values(),
            key=lambda d: (-d["num_mentions"], d["first_mention_time_sec"])
        )

        span_duration = max(0.0, last_seen - first_seen) if first_seen < float("inf") else 0.0

        out.append({
            "idea_id": idea_id,
            "idea": title,
            "num_mentions": int(sum(d["num_mentions"] for d in per_spk.values())),
            "num_speakers": len(per_spk),
            "first_seen_sec": None if first_seen == float("inf") else first_seen,
            "last_seen_sec": last_seen,
            "span_duration_sec": span_duration,
            "initiator_speaker_full": initiator_full,
            "initiator_speaker_short": initiator_short,
            "primary_contributor_speaker_full": primary_full,
            "primary_contributor_speaker_short": primary_short,
            "speakers": speakers_list,
        })

    return out


# -------------------------
# JSON + text summaries
# -------------------------

def _write_ideas_json_summary(
    path: Path,
    meeting_id: str,
    ideas_summary: List[Dict[str, Any]],
    extract_model: str,
    reflect_model: str,
) -> None:
    payload = {
        "metadata": {
            "meeting_id": meeting_id,
            "extract_model": extract_model,
            "reflect_model": reflect_model,
            "context": "conversation_dynamics_ideas_v1",
            "num_ideas": len(ideas_summary),
        },
        "ideas": ideas_summary,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] wrote idea-level JSON summary → {path}")


def _write_ideas_text_summary(
    path: Path,
    meeting_id: str,
    ideas_summary: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append(f"Meeting ID: {meeting_id}")
    lines.append(f"Number of ideas: {len(ideas_summary)}")
    lines.append("")

    for it in ideas_summary:
        lines.append(f"Idea {it.get('idea_id', '')}: {it.get('idea', '')}")
        lines.append(f"  Speakers: {it.get('num_speakers', 0)}")
        lines.append(f"  Mentions: {it.get('num_mentions', 0)}")
        fs = it.get("first_seen_sec")
        le = it.get("last_seen_sec")
        if fs is not None and le is not None:
            lines.append(f"  Span: {fs:.2f}s → {le:.2f}s (duration {it.get('span_duration_sec', 0.0):.2f}s)")
        initiator = it.get("initiator_speaker_short") or it.get("initiator_speaker_full")
        primary = it.get("primary_contributor_speaker_short") or it.get("primary_contributor_speaker_full")
        if initiator:
            lines.append(f"  Brought up by: {initiator}")
        if primary and primary != initiator:
            lines.append(f"  Primary contributor: {primary}")
        elif primary:
            lines.append(f"  Primary contributor: {primary} (same as initiator)")

        lines.append("  Speaker contributions:")
        for sp in it.get("speakers", []):
            lines.append(
                f"    - {sp['speaker_short']}: {sp['num_mentions']} mentions, "
                f"{sp['total_speaking_time_sec']:.2f}s (first at {sp['first_mention_time_sec']:.2f}s)"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote idea-level text summary → {path}")


# -------------------------
# Idea timeline plot
# -------------------------

def _plot_idea_timeline(
    ideas_summary: List[Dict[str, Any]],
    title: str,
    save_path: Path,
) -> None:
    """
    Plot each idea as a horizontal bar from first_seen_sec to last_seen_sec.
    """
    ideas = [it for it in ideas_summary if it.get("first_seen_sec") is not None]

    if not ideas:
        print("[warn] no ideas with time spans to plot.")
        return

    ideas.sort(key=lambda it: float(it.get("first_seen_sec", 1e18)))

    fig_h = max(3.5, 0.6 * len(ideas))
    fig, ax = plt.subplots(figsize=(12, fig_h))

    cmap = plt.get_cmap("tab10")

    for idx, it in enumerate(ideas):
        color = cmap(idx % 10)
        start = float(it.get("first_seen_sec", 0.0))
        end = float(it.get("last_seen_sec", start))
        if end < start:
            start, end = end, start
        ax.hlines(y=idx, xmin=start, xmax=end, color=color, linewidth=6)

    ax.set_yticks(range(len(ideas)))
    ax.set_yticklabels([it.get("idea", f"Idea {i}") for i, it in enumerate(ideas)])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ideas")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    xmax = max((float(it.get("last_seen_sec", 0.0)) for it in ideas), default=1.0)
    ax.set_xlim(0.0, max(xmax, 1.0))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] saved idea timeline → {save_path}")


# -------------------------
# Speaker–Idea bipartite plot
# -------------------------

MIN_EDGE = 1.0
CHARS_PER_LINE = 26
X_GAP = 5.0
FONTSIZE_IDEA = 13
FONTSIZE_SPEAKER = 13

def _wrap(text: str, width: int) -> str:
    import textwrap
    text = (text or "").strip()
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) if text else ""


def load_mentions_from_reflected(ideas_doc: Dict[str, Any]) -> pd.DataFrame:
    """
    From ideas_reflected_<extract>_<reflect>.json build (speaker, idea) rows.
    """
    rows: List[Dict[str, Any]] = []
    for idea in (ideas_doc.get("ideas") or []):
        title = (idea.get("canonical_idea") or idea.get("idea") or "").strip()
        for m in (idea.get("mentions") or []):
            spk = speaker_short_id(m.get("speaker", ""))
            rows.append({"idea": title, "speaker": spk})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No (speaker, idea) mentions found in ideas_reflected JSON")
    return df


def build_support_count(df: pd.DataFrame) -> pd.DataFrame:
    support = df.groupby(["speaker", "idea"]).size().unstack(fill_value=0).astype(float)
    support = support.sort_index(axis=0).sort_index(axis=1)
    return support


def _draw_curved_edge(ax, x0, y0, x1, y1, color, lw, alpha, rad):
    patch = FancyArrowPatch(
        (x0, y0), (x1, y1),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle='-',
        linewidth=lw,
        color=color,
        alpha=alpha,
        zorder=1
    )
    ax.add_patch(patch)


def plot_bipartite_centered(support: pd.DataFrame, out_path: Path):
    speakers = list(support.index)
    ideas = list(support.columns)

    R = len(ideas)
    left_count = int(math.ceil(R / 2.0))
    ideas_left = ideas[:left_count]
    ideas_right = ideas[left_count:]

    totals = support.sum(axis=0).sort_values(ascending=False)
    ideas_left = sorted(ideas_left, key=lambda k: -float(totals.get(k, 0)))
    ideas_right = sorted(ideas_right, key=lambda k: -float(totals.get(k, 0)))

    L = len(speakers)
    y_margin = 0.06
    y_speakers = np.linspace(y_margin, 1 - y_margin, L) if L else np.array([])
    y_left = np.linspace(y_margin, 1 - y_margin, len(ideas_left)) if ideas_left else np.array([])
    y_right = np.linspace(y_margin, 1 - y_margin, len(ideas_right)) if ideas_right else np.array([])

    x_center = 0.0
    x_left, x_right = -X_GAP, X_GAP

    wrapped_left = [_wrap(i, CHARS_PER_LINE) for i in ideas_left]
    wrapped_right = [_wrap(i, CHARS_PER_LINE) for i in ideas_right]
    max_lines_left = max((w.count("\n") + 1) for w in wrapped_left) if wrapped_left else 1
    max_lines_right = max((w.count("\n") + 1) for w in wrapped_right) if wrapped_right else 1
    max_lines = max(max_lines_left, max_lines_right)

    fig_w = max(12, 10 + 0.30 * (len(ideas)) + 0.55 * (max_lines - 1))
    fig_h = max(7.5, 0.5 * max(L, len(ideas)) + 0.35 * (max_lines - 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    left_pad = 1.2 + 0.12 * (max_lines_left - 1)
    right_pad = 1.2 + 0.12 * (max_lines_right - 1)
    ax.set_xlim(x_left - left_pad, x_right + right_pad)
    ax.set_ylim(0, 1)

    palette = ["#2f6db3", "#be5a2f", "#5aa469", "#9b59b6", "#f39c12",
               "#16a085", "#34495e", "#d35400", "#7f8c8d", "#8e44ad"]
    colors_by_speaker = {spk: palette[i % len(palette)] for i, spk in enumerate(speakers)}

    ax.scatter([x_center] * L, y_speakers, s=950,
               c=[colors_by_speaker[s] for s in speakers], alpha=0.98,
               edgecolors="white", zorder=3)
    ax.scatter([x_left] * len(ideas_left), y_left, s=950, c="#315a7d",
               alpha=0.98, edgecolors="white", zorder=3)
    ax.scatter([x_right] * len(ideas_right), y_right, s=950, c="#315a7d",
               alpha=0.98, edgecolors="white", zorder=3)

    for i, spk in enumerate(speakers):
        ax.text(x_center, y_speakers[i], spk, va="center", ha="center",
                fontsize=FONTSIZE_SPEAKER, weight="bold", color="#111")
    for j, lab in enumerate(wrapped_left):
        ax.text(x_left - 0.28, y_left[j], lab, va="center", ha="right",
                fontsize=FONTSIZE_IDEA, color="#222", linespacing=1.15)
    for j, lab in enumerate(wrapped_right):
        ax.text(x_right + 0.28, y_right[j], lab, va="center", ha="left",
                fontsize=FONTSIZE_IDEA, color="#222", linespacing=1.15)

    sub = support.reindex(index=speakers, columns=ideas, fill_value=0.0)
    v = sub.values
    finite = np.isfinite(v)
    vmin = float(np.nanmin(v[finite])) if finite.any() else 0.0
    vmax = float(np.nanmax(v[finite])) if finite.any() else 1.0
    if vmin == vmax:
        vmin, vmax = 0.0, max(1.0, vmax)

    left_index = {idea: j for j, idea in enumerate(ideas_left)}
    right_index = {idea: j for j, idea in enumerate(ideas_right)}

    for i, spk in enumerate(speakers):
        for idea in ideas:
            val = float(sub.loc[spk, idea])
            if val < MIN_EDGE:
                continue
            width = np.interp(val, (vmin, vmax), (0.8, 4.0))
            x0, y0 = x_center, y_speakers[i]
            if idea in left_index:
                j = left_index[idea]
                x1, y1 = x_left, y_left[j]
                rad = -0.12 if (i % 2 == 0) else -0.20
            else:
                j = right_index[idea]
                x1, y1 = x_right, y_right[j]
                rad = 0.12 if (i % 2 == 0) else 0.20
            _draw_curved_edge(ax, x0, y0, x1, y1,
                              color=colors_by_speaker[spk], lw=width, alpha=0.55, rad=rad)

    ax.set_title("Speaker ↔ Idea Mentions (centered speakers)", fontsize=20, weight="bold", pad=22)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close()
    print(f"[done] {out_path}")


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute idea-level conversation dynamics and plots from ideas_reflected_<extract>_<reflect>.json"
    )
    ap.add_argument("--meeting-id", required=True, help="e.g., BAC")
    ap.add_argument(
        "--extract-model",
        default="gptnano",
        help="Model alias used for idea extraction (for locating ideas_reflected file).",
    )
    ap.add_argument(
        "--reflect-model",
        default="gptfull",
        help="Model alias used for idea reflection (for locating ideas_reflected file).",
    )
    ap.add_argument(
        "--title-timeline",
        default="Ideas Timeline",
        help="Title for the ideas timeline plot.",
    )
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id
    ctx_dir = base / "context_outputs"
    plots_dir = base / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    extract_tag = _safe_model_tag(args.extract_model)
    reflect_tag = _safe_model_tag(args.reflect_model)

    ideas_path = ctx_dir / f"ideas_reflected_{extract_tag}_{reflect_tag}.json"
    ideas_doc = _load_reflected_ideas(ideas_path)

    # 1) Compute idea-level participation stats
    ideas_summary = _compute_idea_stats(ideas_doc)

    # 2) Write JSON + text summaries
    json_out = base / "conversation_dynamics_ideas.json"
    txt_out = base / "conversation_dynamics_ideas.txt"
    _write_ideas_json_summary(json_out, args.meeting_id, ideas_summary, extract_tag, reflect_tag)
    _write_ideas_text_summary(txt_out, args.meeting_id, ideas_summary)

    # 3) Ideas timeline plot
    tl_out = plots_dir / f"{args.meeting_id}_ideas_timeline_{extract_tag}_{reflect_tag}.png"
    _plot_idea_timeline(ideas_summary, args.title_timeline, tl_out)

    # 4) Speaker–Idea bipartite plot
    df = load_mentions_from_reflected(ideas_doc)
    support = build_support_count(df)
    bp_out = plots_dir / f"{args.meeting_id}_speaker_idea_bipartite_{extract_tag}_{reflect_tag}.png"
    plot_bipartite_centered(support, bp_out)


if __name__ == "__main__":
    main()
