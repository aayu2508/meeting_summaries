#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def plot_ideas_windows_timeline(ideas_json_path: Path, title: str = "Ideas Timeline (windows)"):
    # Load JSON
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
        label = it.get("idea", f"Idea {idx}")
        windows = it.get("windows", [])

        # Draw each [start, end, cid] as a horizontal segment at y=idx
        for w in windows:
            # tolerate malformed items
            if not isinstance(w, (list, tuple)) or len(w) < 2:
                continue
            start, end = float(w[0]), float(w[1])
            ax.hlines(
                y=idx,
                xmin=start,
                xmax=end,
                color=color,
                linewidth=6,
            )

        # (Optional) annotate idea text on the left edge for long labels
        # ax.text(x=ax.get_xlim()[0], y=idx+0.1, s=label, fontsize=9)

    # Y axis: idea labels
    ax.set_yticks(range(len(ideas)))
    ax.set_yticklabels([it.get("idea", f"Idea {i}") for i, it in enumerate(ideas)])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Ideas")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Nice padding around content
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot idea timelines from ideas_windows.json (per-window spans)")
    ap.add_argument("--meeting-id", required=True, help="e.g., ES2002a")
    ap.add_argument("--title", default="Ideas Timeline (windows)")
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id
    ideas_file = base / "context_outputs" / "ideas_windows.json"
    assert ideas_file.exists(), f"File not found: {ideas_file}"

    plot_ideas_windows_timeline(ideas_file, title=args.title)
