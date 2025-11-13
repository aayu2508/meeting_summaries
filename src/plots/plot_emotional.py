#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import json
from pathlib import Path
import math
import numpy as np
import csv

ap = argparse.ArgumentParser(description="Plot emotion/prosody with idea timeline markers")
ap.add_argument("--meeting-id", required=True, help="e.g., BAC")
ap.add_argument("--ideas-json", required=True, help="Path to ideas_windows.json")
args = ap.parse_args()

# -----------------------------
# Paths
# -----------------------------
base = os.path.join("data", "outputs", args.meeting_id)
INPUT_JSON = os.path.join(base, "asr_emotion_vad.json")

# Create output directory for plots
PLOTS_DIR = os.path.join(base, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def fmt_hms(seconds: float) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "0:00"
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def context_from_df(df: pd.DataFrame):
    n_speakers = df["speaker"].nunique()
    n_segments = len(df)
    duration = (df["time_end"].max() if "time_end" in df.columns else df["time_mid"].max())
    return n_speakers, n_segments, duration

def z_per_speaker(df: pd.DataFrame, col: str, out_col: str):
    def z(group):
        x = group[col].astype(float)
        mu = x.mean()
        sd = x.std(ddof=0)
        if sd and not math.isclose(sd, 0.0):
            return (x - mu) / sd
        return pd.Series(np.zeros(len(x)), index=group.index)
    df[out_col] = df.groupby("speaker", group_keys=False).apply(z)
    return df

def load_segments(path: str) -> pd.DataFrame:
    data = json.loads(Path(path).read_text())
    rows = []
    for seg in data:
        if seg.get("type") != "speech":
            continue
        emo = seg.get("emotion", {}) or {}
        pros = seg.get("prosody", {}) or {}
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        rows.append({
            "time_mid": (start + end) / 2.0,
            "time_start": start,
            "time_end": end,
            "speaker": seg.get("speaker", "UNKNOWN"),
            "valence": float(emo.get("v")) if emo.get("v") is not None else None,
            "arousal": float(emo.get("a01")) if emo.get("a01") is not None else None,
            "dominance": float(emo.get("d01")) if emo.get("d01") is not None else None,
            "cat": emo.get("cat"),
            "conf": float(emo.get("conf")) if emo.get("conf") is not None else None,
            "pitch_mu": float(pros.get("pitch_mu")) if pros.get("pitch_mu") is not None else None,
            "pitch_var": float(pros.get("pitch_var")) if pros.get("pitch_var") is not None else None,
            "log_rms": float(pros.get("log_rms")) if pros.get("log_rms") is not None else None,
            "wps": float(pros.get("wps")) if pros.get("wps") is not None else None,
        })
    df = pd.DataFrame(rows)
    return df.dropna(subset=["speaker", "time_mid"])

def load_ideas(path: str):
    data = json.loads(Path(path).read_text())
    ideas = data.get("ideas", [])
    out = []
    for i, it in enumerate(ideas, start=1):
        first = float(it.get("first_seen", None)) if it.get("first_seen") is not None else None
        windows = it.get("windows", []) or []
        out.append({
            "idx": i,
            "idea": it.get("idea", f"idea_{i}"),
            "first_seen": first,
            "windows": [(float(w[0]), float(w[1])) for w in windows if len(w) >= 2]
        })
    return out

def save_idea_index_mapping(ideas, path_csv):
    # Save mapping of index -> first_seen -> idea text
    with open(path_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "first_seen_sec", "first_seen_hms", "idea"])
        for it in ideas:
            fs = it["first_seen"]
            writer.writerow([it["idx"], fs, fmt_hms(fs if fs is not None else 0), it["idea"]])

def unicode_circled_number(n: int) -> str:
    # 1..20 circled numbers; beyond that, show plain number
    circled = {
        1:"①",2:"②",3:"③",4:"④",5:"⑤",6:"⑥",7:"⑦",8:"⑧",9:"⑨",10:"⑩",
        11:"⑪",12:"⑫",13:"⑬",14:"⑭",15:"⑮",16:"⑯",17:"⑰",18:"⑱",19:"⑲",20:"⑳"
    }
    return circled.get(n, str(n))

def add_idea_markers(ax, ideas, y_top_pad=0.04, annotate=True, show_windows=False):
    """
    Draw vertical dashed lines at first_seen for each idea.
    Optionally shade [start,end] windows.

    We avoid custom colors; using defaults/linestyles only.
    """
    # Determine y-limits to place annotations just above the plot area
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    y_annot = ymax - y_top_pad * yrange

    # Vertical lines at first_seen
    for it in ideas:
        fs = it["first_seen"]
        if fs is None:
            continue
        ax.axvline(fs, linestyle="--", linewidth=1, alpha=0.9)
        if annotate:
            ax.text(fs, y_annot, unicode_circled_number(it["idx"]),
                    rotation=90, va="top", ha="center")

    # Optional shaded windows
    if show_windows:
        for it in ideas:
            for (s, e) in it["windows"]:
                ax.axvspan(s, e, alpha=0.1)  # light shading without specifying color

# -----------------------------
# Plots
# -----------------------------
def plot_valence_over_time(df: pd.DataFrame, ideas, outpath, meeting_id: str):
    n_spk, n_seg, dur = context_from_df(df)
    subtitle = f"{meeting_id} • {n_spk} speakers • {fmt_hms(dur)} • {n_seg} segments"

    plt.figure(figsize=(9, 4.8))
    ax = plt.gca()
    for spk, g in df.dropna(subset=["valence"]).sort_values("time_mid").groupby("speaker"):
        ax.plot(g["time_mid"], g["valence"], marker="o", linestyle="-", label=spk, alpha=0.8)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    add_idea_markers(ax, ideas, annotate=True, show_windows=False)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Valence (−1 … +1)")
    ax.set_title(f"Valence over time (by speaker)\n{subtitle}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_arousal_over_time(df: pd.DataFrame, ideas, outpath, meeting_id: str):
    n_spk, n_seg, dur = context_from_df(df)
    subtitle = f"{meeting_id} • {n_spk} speakers • {fmt_hms(dur)} • {n_seg} segments"

    plt.figure(figsize=(9, 4.8))
    ax = plt.gca()
    for spk, g in df.dropna(subset=["arousal"]).sort_values("time_mid").groupby("speaker"):
        ax.plot(g["time_mid"], g["arousal"], marker="o", linestyle="-", label=spk, alpha=0.8)
    add_idea_markers(ax, ideas, annotate=True, show_windows=False)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Arousal (0 … 1)")
    ax.set_title(f"Arousal over time (by speaker)\n{subtitle}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_wps_over_time(df: pd.DataFrame, ideas, outpath, meeting_id: str):
    n_spk, n_seg, dur = context_from_df(df)
    subtitle = f"{meeting_id} • {n_spk} speakers • {fmt_hms(dur)} • {n_seg} segments"

    plt.figure(figsize=(9, 4.8))
    ax = plt.gca()
    for spk, g in df.dropna(subset=["wps"]).sort_values("time_mid").groupby("speaker"):
        ax.plot(g["time_mid"], g["wps"], marker="o", linestyle="-", label=spk, alpha=0.8)
    add_idea_markers(ax, ideas, annotate=True, show_windows=False)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Words per second")
    ax.set_title(f"Speech rate (WPS) over time (by speaker)\n{subtitle}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def plot_engagement_over_time(df: pd.DataFrame, ideas, outpath, meeting_id: str):
    work = df.copy()
    work = z_per_speaker(work, "arousal", "z_arousal")
    work = z_per_speaker(work, "log_rms", "z_rms")
    work = z_per_speaker(work, "pitch_var", "z_pitchvar")
    work["engagement_index"] = work[["z_arousal", "z_rms", "z_pitchvar"]].mean(axis=1)

    n_spk, n_seg, dur = context_from_df(work)
    subtitle = f"{meeting_id} • {n_spk} speakers • {fmt_hms(dur)} • {n_seg} segments"

    plt.figure(figsize=(9, 4.8))
    ax = plt.gca()
    for spk, g in work.dropna(subset=["engagement_index"]).sort_values("time_mid").groupby("speaker"):
        ax.plot(g["time_mid"], g["engagement_index"], marker="o", linestyle="-", label=spk, alpha=0.8)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    add_idea_markers(ax, ideas, annotate=True, show_windows=False)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Engagement index (z, per speaker)")
    ax.set_title(f"Engagement over time (by speaker)\n{subtitle}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df = load_segments(INPUT_JSON)
    if df.empty:
        raise SystemExit(f"No speech segments found in {INPUT_JSON}")
    df = df.sort_values("time_mid")

    ideas = load_ideas(args.ideas_json)
    if not ideas:
        raise SystemExit(f"No ideas found in {args.ideas_json}")

    # Save mapping (index → idea text/time) for reference
    idea_map_csv = os.path.join(PLOTS_DIR, "idea_index_mapping.csv")
    save_idea_index_mapping(ideas, idea_map_csv)

    # Output paths
    out_val = os.path.join(PLOTS_DIR, "valence_over_time.png")
    out_ar = os.path.join(PLOTS_DIR, "arousal_over_time.png")
    out_wps = os.path.join(PLOTS_DIR, "wps_over_time.png")
    out_eng = os.path.join(PLOTS_DIR, "engagement_over_time.png")

    # Plots with idea markers
    plot_valence_over_time(df, ideas, out_val, args.meeting_id)
    plot_arousal_over_time(df, ideas, out_ar, args.meeting_id)
    plot_wps_over_time(df, ideas, out_wps, args.meeting_id)
    plot_engagement_over_time(df, ideas, out_eng, args.meeting_id)

    print("Saved plots to:")
    print(f" - {out_val}")
    print(f" - {out_ar}")
    print(f" - {out_wps}")
    print(f" - {out_eng}")
    print(f"Idea index mapping: {idea_map_csv}")
