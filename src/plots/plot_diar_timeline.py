#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read JSON: {path}\n{e}")

def _speaker_order(turns):
    seen, order = set(), []
    for t in sorted(turns, key=lambda x: (float(x["start"]), float(x["end"]))):
        spk = str(t.get("speaker", "S?"))
        if spk not in seen:
            seen.add(spk)
            order.append(spk)
    return order

def plot_diar_timeline(diar_turns, title="Conversation Timeline", min_visible=0.08):
    """
    diar_turns: list of {start,end,speaker}
    min_visible: minimum width (sec) to render so micro-turns are visible
    """
    if not diar_turns:
        print("[warn] no diarization turns to plot.")
        return

    order = _speaker_order(diar_turns)
    y_index = {spk: i for i, spk in enumerate(order)}

    # choose consistent colors per speaker
    cmap = plt.get_cmap("tab20")
    colors = {spk: cmap(i % 20) for i, spk in enumerate(order)}

    # canvas
    fig_h = max(3.0, 0.7 * len(order))
    fig, ax = plt.subplots(figsize=(12, fig_h))

    total_max = 0.0
    seg_count = 0

    for seg in diar_turns:
        spk = str(seg.get("speaker", "S?"))
        y = y_index.get(spk, 0)
        x0 = float(seg.get("start", 0.0))
        x1 = float(seg.get("end", x0))
        w = max(min_visible, x1 - x0)  # ensure visibility for micro-turns
        total_max = max(total_max, x1)
        seg_count += 1

        rect = patches.Rectangle(
            (x0, y - 0.35), w, 0.7,
            linewidth=0,
            facecolor=colors[spk],
            alpha=0.75
        )
        ax.add_patch(rect)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(left=0, right=max(total_max, 1.0))
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    # baseline lines per row to make empty rows obvious
    for i in range(len(order)):
        ax.hlines(i, 0, total_max, linewidth=0.5, alpha=0.2)

    plt.tight_layout()
    plt.show()

    print(f"[summary] speakers: {len(order)} → {order}")
    print(f"[summary] segments: {seg_count}, duration ~ {total_max:.1f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Plot meeting timelines")
    ap.add_argument("--meeting-id", required=True, help="e.g., ES2002a")
    ap.add_argument("--diar-source", choices=["raw", "merged"], default="raw",
                    help="Which diarization file to plot")
    ap.add_argument("--min-visible", type=float, default=0.08,
                    help="Minimum visible width (seconds) for micro-turns")
    args = ap.parse_args()

    base = Path("data/outputs") / args.meeting_id
    diar_file = base / ("diarization_raw.json" if args.diar_source == "raw" else "diarization.json")
    assert diar_file.exists(), f"Not found: {diar_file}"
    diar_turns = _load_json(diar_file)
    plot_diar_timeline(diar_turns, title=f"Speaker Timeline • {args.meeting_id} • {args.diar_source}",
                       min_visible=args.min_visible)
