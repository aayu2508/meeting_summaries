#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def plot_ideas_timeline(ideas_json_path: Path, title: str = "Ideas Timeline"):
    # Load JSON
    data = json.loads(ideas_json_path.read_text(encoding="utf-8"))
    groups = data.get("groups", [])
    if not groups:
        raise SystemExit(f"No idea groups found in {ideas_json_path}")

    print(f"[plot] {len(groups)} ideas found")

    # Sort ideas by first appearance (optional)
    groups.sort(key=lambda g: g["timeline"][0][0] if g.get("timeline") else 1e9)

    # Create figure
    fig_h = max(3.5, 0.6 * len(groups))
    fig, ax = plt.subplots(figsize=(12, fig_h))

    # Assign colors to each idea
    cmap = plt.get_cmap("tab10")
    for idx, g in enumerate(groups):
        color = cmap(idx % 10)
        idea = g.get("idea", f"Idea {idx}")
        for (start, end) in g.get("timeline", []):
            ax.hlines(
                y=idx,
                xmin=start,
                xmax=end,
                color=color,
                linewidth=6,
                label=idea if idx == 0 else None,
            )

    # Set labels
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g["idea"] for g in groups])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ideas")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot idea timelines from grouped JSON")
    ap.add_argument("--meeting-id", required=True, help="e.g., ES2002a")
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id
    ideas_file = base / "context_outputs" / "ideas_grouped.json"
    assert ideas_file.exists(), f"File not found: {ideas_file}"
    plot_ideas_timeline(ideas_file)
