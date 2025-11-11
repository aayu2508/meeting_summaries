#!/usr/bin/env python3
import json
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def _safe_model_tag(s: str) -> str:
    # keep letters, numbers, underscore, dash; collapse others to "_"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_") or "model"

def _windows_extent(ideas: List[Dict[str, Any]]) -> float:
    mx = 0.0
    for it in ideas:
        for w in it.get("windows", []):
            if isinstance(w, (list, tuple)) and len(w) >= 2:
                try:
                    mx = max(mx, float(w[1]))
                except Exception:
                    pass
    return mx

def plot_ideas_windows_timeline(ideas_json_path: Path, title: str, save_path: Path) -> None:
    data = json.loads(ideas_json_path.read_text(encoding="utf-8"))

    # expect: {"ideas": [ {"idea": str, "first_seen": float, "last_seen": float, "windows": [[s,e,cid], ...]} ]}
    ideas = data.get("ideas", [])
    if not ideas:
        raise SystemExit(f"No ideas found in {ideas_json_path}")

    # Sort ideas by first appearance for a readable top→bottom order
    ideas.sort(key=lambda it: float(it.get("first_seen", 1e18)))

    print(f"[plot] {len(ideas)} ideas; plotting windows from {ideas_json_path}")

    # Figure height scales with number of ideas
    fig_h = max(3.5, 0.6 * len(ideas))
    fig, ax = plt.subplots(figsize=(12, fig_h))

    cmap = plt.get_cmap("tab10")

    for idx, it in enumerate(ideas):
        color = cmap(idx % 10)
        windows = it.get("windows", [])

        # Draw each [start, end, cid] as a horizontal segment at y=idx
        for w in windows:
            if not isinstance(w, (list, tuple)) or len(w) < 2:
                continue
            try:
                start, end = float(w[0]), float(w[1])
            except Exception:
                continue
            if end < start:
                start, end = end, start
            ax.hlines(y=idx, xmin=start, xmax=end, color=color, linewidth=6)

    # Y axis: idea labels
    ax.set_yticks(range(len(ideas)))
    ax.set_yticklabels([it.get("idea", f"Idea {i}") for i, it in enumerate(ideas)])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ideas")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Tight x-limits based on windows
    xmax = max(_windows_extent(ideas), 1.0)
    ax.set_xlim(0.0, xmax)

    # Nice padding around content
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] saved → {save_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot idea timelines from ideas_windows_<MODEL>.json (per-window spans)"
    )
    ap.add_argument("--meeting-id", required=True, help="e.g., ES2002a")
    ap.add_argument("--model", required=True,
                    help="Model tag to use in filename, e.g., gpt-5, gpt-3.5, roberta7")
    ap.add_argument("--title", default="Ideas Timeline (windows)")
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id
    model_tag = _safe_model_tag(args.model)

    ideas_file = base / "context_outputs" / f"ideas_windows_{model_tag}.json"
    assert ideas_file.exists(), f"File not found: {ideas_file}"

    out_png = base / "plots" / f"{args.meeting_id}_ideas_windows_{model_tag}.png"
    plot_ideas_windows_timeline(ideas_file, title=args.title, save_path=out_png)
