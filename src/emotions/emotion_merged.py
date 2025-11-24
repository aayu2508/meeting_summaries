#!/usr/bin/env python3
import json
from pathlib import Path
import argparse

def merge_emotion_json(dim_json_path, cat_json_path, out_json_path):
    dim_segments = json.loads(Path(dim_json_path).read_text())
    cat_segments = json.loads(Path(cat_json_path).read_text())

    if len(dim_segments) != len(cat_segments):
        raise ValueError("Mismatch: the two JSON files have different number of segments")

    merged = []
    for dim_seg, cat_seg in zip(dim_segments, cat_segments):

        # Copy the dimensional segment fully
        out = dim_seg

        # Ensure emotion field exists
        emo_dim = out.get("emotion", {})
        emo_cat = cat_seg.get("emotion", {})

        # Add categorical fields (do NOT overwrite dim fields)
        for key, value in emo_cat.items():
            if key not in emo_dim or emo_dim[key] is None:
                emo_dim[key] = value

        out["emotion"] = emo_dim
        merged.append(out)

    Path(out_json_path).write_text(json.dumps(merged, indent=2))
    print(f"Merged JSON saved to: {out_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge VA + categorical emotion JSONs.")
    parser.add_argument("--dim_json", required=True, help="Dimensional emotion JSON")
    parser.add_argument("--cat_json", required=True, help="Categorical emotion JSON")
    parser.add_argument("--out_json", required=True, help="Output merged JSON path")
    args = parser.parse_args()

    merge_emotion_json(args.dim_json, args.cat_json, args.out_json)

if __name__ == "__main__":
    main()
