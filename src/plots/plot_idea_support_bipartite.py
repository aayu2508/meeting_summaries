# speaker_idea_bipartite_centered.py
import json, argparse, textwrap, re, math
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# -------------------------
# Hard-coded display params
# -------------------------
MIN_EDGE = 1.0
CHARS_PER_LINE = 26
X_GAP = 5.0
FONTSIZE_IDEA = 13
FONTSIZE_SPEAKER = 13

# -------------------------

def speaker_short_id(s: str) -> str:
    m = re.search(r"SPEAKER_(\d+)", s or "")
    return f"S{int(m.group(1))}" if m else (s or "S?")

def load_mentions(doc: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idea_idx, idea in enumerate(doc.get("ideas", []), start=1):
        title = idea.get("idea", "")
        for m in (idea.get("mentions") or []):
            spk = speaker_short_id(m.get("speaker", ""))
            sd = m.get("stance_distribution") or {}
            rows.append({
                "idea_idx": idea_idx,
                "idea": title,
                "speaker": spk,
                "pro": bool(sd.get("pro", False)),
                "con": bool(sd.get("con", False)),
                "neutral": bool(sd.get("neutral", False)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No mentions found in ideas_with_reactions.json")
    return df

def build_support_count(df: pd.DataFrame) -> pd.DataFrame:
    # Count of pro-mentions per (speaker, idea)
    return df[df["pro"]].groupby(["speaker", "idea"]).size().unstack(fill_value=0)

def _wrap(text: str, width: int) -> str:
    text = (text or "").strip()
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) if text else ""

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

    # Split ideas evenly left/right
    R = len(ideas)
    left_count = int(math.ceil(R / 2.0))
    ideas_left = ideas[:left_count]
    ideas_right = ideas[left_count:]

    # Rank ideas by total support (count) just for label order nicety (optional)
    # (We keep sides as defined above; if you prefer rank-based split, reorder before split.)
    totals = support.sum(axis=0).sort_values(ascending=False)
    # Reorder inside each side by totals (optional but helpful)
    ideas_left = sorted(ideas_left, key=lambda k: -float(totals.get(k, 0)))
    ideas_right = sorted(ideas_right, key=lambda k: -float(totals.get(k, 0)))

    # Positions
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

    # Figure size (heuristic)
    fig_w = max(12, 10 + 0.30 * (len(ideas)) + 0.55 * (max_lines - 1))
    fig_h = max(7.5, 0.5 * max(L, len(ideas)) + 0.35 * (max_lines - 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    left_pad = 1.2 + 0.12 * (max_lines_left - 1)
    right_pad = 1.2 + 0.12 * (max_lines_right - 1)
    ax.set_xlim(x_left - left_pad, x_right + right_pad)
    ax.set_ylim(0, 1)

    # Colors per speaker
    colors_by_speaker = {}
    palette = ["#2f6db3", "#be5a2f", "#5aa469", "#9b59b6", "#f39c12", "#16a085"]
    for idx, spk in enumerate(speakers):
        colors_by_speaker[spk] = palette[idx % len(palette)]

    # Nodes
    ax.scatter([x_center] * L, y_speakers, s=950, c="#5a88d1", alpha=0.98, edgecolors="white", zorder=3)
    ax.scatter([x_left] * len(ideas_left), y_left, s=950, c="#315a7d", alpha=0.98, edgecolors="white", zorder=3)
    ax.scatter([x_right] * len(ideas_right), y_right, s=950, c="#315a7d", alpha=0.98, edgecolors="white", zorder=3)

    # Labels
    for i, spk in enumerate(speakers):
        ax.text(x_center, y_speakers[i], spk, va="center", ha="center",
                fontsize=FONTSIZE_SPEAKER, weight="bold", color="#111")

    for j, lab in enumerate(wrapped_left):
        ax.text(x_left - 0.28, y_left[j], lab, va="center", ha="right",
                fontsize=FONTSIZE_IDEA, color="#222", linespacing=1.15)
    for j, lab in enumerate(wrapped_right):
        ax.text(x_right + 0.28, y_right[j], lab, va="center", ha="left",
                fontsize=FONTSIZE_IDEA, color="#222", linespacing=1.15)

    # Edge scaling (using counts)
    sub = support  # all ideas on one page
    v = sub.values
    finite = np.isfinite(v)
    vmin = float(np.nanmin(v[finite])) if finite.any() else 0.0
    vmax = float(np.nanmax(v[finite])) if finite.any() else 1.0
    if vmin == vmax:
        vmin, vmax = 0.0, max(1.0, vmax)

    left_index = {idea: j for j, idea in enumerate(ideas_left)}
    right_index = {idea: j for j, idea in enumerate(ideas_right)}

    # Draw edges (only if count >= MIN_EDGE)
    for i, spk in enumerate(speakers):
        for idea in ideas:
            val = sub.loc[spk, idea] if idea in sub.columns else np.nan
            if pd.isna(val) or val < MIN_EDGE:
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

    ax.set_title("Speaker ↔ Idea Support Network (centered speakers)", fontsize=20, weight="bold", pad=22)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240)
    plt.close()
    print(f"[done] {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Speaker–Idea support visualization (centered, single page)")
    ap.add_argument("--meeting-id", required=True)
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id / "context_outputs"
    in_path = base / "ideas_with_reactions.json"
    assert in_path.exists(), f"Missing file: {in_path}"

    doc = json.loads(in_path.read_text(encoding="utf-8"))
    df = load_mentions(doc)
    support = build_support_count(df)  # metric = count (fixed)

    out_path = base / f"{args.meeting_id}_bipartite_support_centered.png"
    plot_bipartite_centered(support, out_path)

if __name__ == "__main__":
    main()
