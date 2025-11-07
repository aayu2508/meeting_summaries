import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def speaker_short_id(s: str) -> str:
    m = re.search(r"SPEAKER_(\\d+)", s or "")
    return f"S{int(m.group(1))}" if m else (s or "S?")


def load_mentions(doc: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idea_idx, idea in enumerate(doc.get("ideas", []), start=1):
        title = idea.get("idea", "")
        for m in (idea.get("mentions") or []):
            s = float(m.get("start", 0.0))
            e = float(m.get("end", s))
            if e < s:
                s, e = e, s
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


def build_support(df: pd.DataFrame, metric: str = "count") -> pd.DataFrame:
    if metric == "count":
        return df[df["pro"]].groupby(["speaker", "idea"]).size().unstack(fill_value=0)
    else:
        stance_code = np.where(df["pro"], 1.0, np.where(df["con"], -1.0, 0.0))
        df["stance_code"] = stance_code
        return df.groupby(["speaker", "idea"])["stance_code"].mean().unstack(fill_value=np.nan)


def plot_bipartite(support: pd.DataFrame, out_path: Path, metric: str = "count", min_edge: float = 1.0):
    speakers = list(support.index)
    ideas = list(support.columns)

    # Layout and spacing
    x_left, x_right = 0.0, 2.5   # <- increased distance between speaker and idea sides
    y_left = np.linspace(0, 1, len(speakers))
    y_right = np.linspace(0, 1, len(ideas))

    # Larger figure with adaptive size
    fig_w = max(12, 0.22 * len(ideas) + 6)
    fig_h = max(8, 0.4 * max(len(speakers), len(ideas)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    # Draw nodes
    ax.scatter([x_left] * len(speakers), y_left, s=800, c="#4B8BBE", alpha=0.9, edgecolors="white", zorder=3)
    ax.scatter([x_right] * len(ideas), y_right, s=800, c="#306998", alpha=0.9, edgecolors="white", zorder=3)

    # Speaker labels (left)
    for i, spk in enumerate(speakers):
        ax.text(x_left - 0.15, y_left[i], spk, va="center", ha="right", fontsize=13, weight="bold", color="#111111")

    # Idea labels (right) — full text, multiline if long
    for j, idea in enumerate(ideas):
        ax.text(
            x_right + 0.15,
            y_right[j],
            idea,
            va="center",
            ha="left",
            fontsize=12,
            color="#222222",
            wrap=True,
        )

    # Draw edges — thinner, spaced, with transparency
    for i, spk in enumerate(speakers):
        for j, idea in enumerate(ideas):
            val = support.iloc[i, j]
            if pd.isna(val):
                continue

            draw = False
            width = 0.0
            if metric == "count":
                if val >= min_edge:
                    draw = True
                    width = np.interp(val, (support.values.min(), support.values.max()), (0.4, 2.0))
            else:
                if val > 0:
                    draw = True
                    width = max(0.4, 2.5 * float(val))

            if draw:
                ax.plot(
                    [x_left, x_right],
                    [y_left[i], y_right[j]],
                    linewidth=width,
                    color="#7F7F7F",
                    alpha=0.5,
                    zorder=1,
                )

    ax.set_title("Speaker ↔ Idea Support Network", fontsize=18, weight="bold", pad=20)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[done] saved bipartite plot: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Speaker–Idea support visualization (spacious version)")
    ap.add_argument("--meeting-id", required=True, help="e.g., S14")
    ap.add_argument("--metric", choices=["count", "mean"], default="count")
    ap.add_argument("--min-edge", type=float, default=1.0)
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id / "context_outputs"
    in_path = base / "ideas_with_reactions.json"
    assert in_path.exists(), f"Missing file: {in_path}"

    doc = json.loads(in_path.read_text(encoding="utf-8"))
    df = load_mentions(doc)
    support = build_support(df, metric=args.metric)

    out_path = base / f"{args.meeting_id}_bipartite_support_readable.png"
    plot_bipartite(support, out_path, metric=args.metric, min_edge=args.min_edge)


if __name__ == "__main__":
    main()
