#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from typing import List, Dict, Any, Union, Optional

def _load_json(path: Union[str, Path]) -> Any:
    """Load and parse JSON from a file path."""
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to read JSON: {p}\n{e}")

def _speaker_order(turns: List[Dict[str, Any]]) -> List[str]:
    seen, order = set(), []
    for t in sorted(turns, key=lambda x: (float(x.get("start", 0.0)),
                                          float(x.get("end", 0.0)))):
        spk = str(t.get("speaker", "S?"))
        if spk not in seen:
            seen.add(spk)
            order.append(spk)
    return order

def plot_diar_timeline(
    diar_turns: List[Dict[str, Any]],
    title: str = "Conversation Timeline",
    min_visible: float = 0.08,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    diar_turns: list of {start,end,speaker}
    min_visible: minimum width (sec) to render so micro-turns are visible
    save_path: if provided, save the figure there; otherwise show interactively
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

    # for x-axis extent & summaries, use true end times (not min_visible)
    max_time = 0.0
    seg_count = 0

    for seg in diar_turns:
        spk = str(seg.get("speaker", "S?"))
        y = y_index.get(spk, 0)
        x0 = float(seg.get("start", 0.0))
        x1 = float(seg.get("end", x0))
        w = max(min_visible, x1 - x0)  # ensure visibility for micro-turns

        max_time = max(max_time, x1)
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
    ax.set_xlim(left=0, right=max(max_time, 1.0))
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)

    # baseline lines per row to make empty rows obvious
    for i in range(len(order)):
        ax.hlines(i, 0, max_time, linewidth=0.5, alpha=0.2)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] saved plot → {save_path}")
    else:
        plt.show()

    print(f"[summary] speakers: {len(order)} → {order}")
    print(f"[summary] segments: {seg_count}, duration ~ {max_time:.1f}s")

def main():
    parser = argparse.ArgumentParser(description="Plot a diarization timeline per speaker.")
    parser.add_argument("--meeting-id", required=True, help="Meeting ID under data/outputs/<MEETING_ID>/")
    parser.add_argument("--title", default="Conversation Timeline",help="Plot title.")
    args = parser.parse_args()

    base = Path("data/outputs") / args.meeting_id
    in_path = base / "diarization_raw.json"
    assert in_path.exists(), f"Missing file: {in_path}"

    diar_turns = _load_json(in_path)

    out_path = base / "plots" / f"{args.meeting_id}_diarization.png"
    plot_diar_timeline(diar_turns, title=args.title, save_path=out_path)

if __name__ == "__main__":
    main()
