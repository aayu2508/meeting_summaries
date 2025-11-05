# plot_timeline.py
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

def plot_diar_timeline(diar_path: Path, title: str = "Conversation Timeline"):
    data = json.loads(diar_path.read_text())
    # unique speakers in deterministic order of first appearance
    seen, order = set(), []
    for d in data:
        spk = d["speaker"]
        if spk not in seen:
            seen.add(spk)
            order.append(spk)
    y_index = {spk: i for i, spk in enumerate(order)}

    fig, ax = plt.subplots(figsize=(12, max(3, 0.6*len(order))))
    for seg in data:
        y = y_index[seg["speaker"]]
        x0, x1 = float(seg["start"]), float(seg["end"])
        ax.add_patch(patches.Rectangle((x0, y - 0.35), x1 - x0, 0.7))

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.set_ylim(-0.8, len(order)-0.2)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Plot diarization timeline (RAW)")
    ap.add_argument("--meeting-id", required=True)
    args = ap.parse_args()
    path = Path("data/outputs") / args.meeting_id / "diarization_raw.json"
    assert path.exists(), f"Not found: {path}"
    plot_diar_timeline(path, title=f"Timeline • {args.meeting_id}")
