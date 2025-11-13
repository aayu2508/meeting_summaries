#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import json, os, math, numpy as np, csv
from pathlib import Path
import argparse

ap = argparse.ArgumentParser(description="Plot emotional/prosodic trends with idea markers")
ap.add_argument("--segments", required=True, help="Path to segments JSON with emotion/prosody")
ap.add_argument("--ideas", required=True, help="Path to ideas_windows JSON")
args = ap.parse_args()

SEG_PATH = Path(args.segments).expanduser().resolve()
IDEA_PATH = Path(args.ideas).expanduser().resolve()
OUT_DIR = SEG_PATH.parent  # save in same directory
MEETING_ID = SEG_PATH.parent.name

def fmt_hms(sec):
    if sec is None or math.isnan(sec): return "0:00"
    sec = int(round(sec)); m, s = divmod(sec, 60); h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def z_per_speaker(df, col, out_col):
    def z(g):
        x = g[col].astype(float)
        mu, sd = x.mean(), x.std(ddof=0)
        return (x - mu) / sd if sd > 1e-9 else pd.Series(np.zeros(len(x)), index=g.index)
    df[out_col] = df.groupby("speaker", group_keys=False).apply(z)
    return df

def load_segments(path):
    data = json.loads(path.read_text())
    rows=[]
    for seg in data:
        if seg.get("type")!="speech": continue
        emo, pros = seg.get("emotion",{}), seg.get("prosody",{})
        s,e=float(seg["start"]),float(seg["end"])
        rows.append({
            "time_mid":(s+e)/2,"time_start":s,"time_end":e,
            "speaker":seg.get("speaker","UNK"),
            "valence":emo.get("v"),"arousal":emo.get("a01"),
            "log_rms":pros.get("log_rms"),"pitch_var":pros.get("pitch_var"),"wps":pros.get("wps")
        })
    return pd.DataFrame(rows).dropna(subset=["speaker","time_mid"])

def load_ideas(path):
    data=json.loads(path.read_text()); out=[]
    for i,it in enumerate(data.get("ideas",[]),1):
        out.append({"idx":i,"idea":it["idea"],"first_seen":it.get("first_seen"),
                    "windows":[(float(w[0]),float(w[1])) for w in it.get("windows",[])]})
    return out

def add_markers(ax, ideas, show_windows=True):
    """Draw vertical dashed lines + optional shaded windows for each idea."""
    ymin, ymax = ax.get_ylim()
    y_annot = ymax - (ymax - ymin) * 0.04

    circ = {
        1:"①",2:"②",3:"③",4:"④",5:"⑤",6:"⑥",7:"⑦",8:"⑧",9:"⑨",10:"⑩",
        11:"⑪",12:"⑫",13:"⑬",14:"⑭",15:"⑮",16:"⑯",17:"⑰",18:"⑱",19:"⑲",20:"⑳"
    }

    # Vertical dashed lines and circled numbers
    for it in ideas:
        fs = it.get("first_seen")
        if fs is None:
            continue
        ax.axvline(fs, linestyle="--", linewidth=1, alpha=0.8, color="black")
        ax.text(fs, y_annot, circ.get(it["idx"], str(it["idx"])),
                rotation=90, va="top", ha="center", fontsize=9)

    # Optional shaded windows
    if show_windows:
        # Use alternating colors to distinguish ideas
        colors = plt.cm.tab20(np.linspace(0, 1, len(ideas)))
        for i, it in enumerate(ideas):
            for (s, e) in it.get("windows", []):
                ax.axvspan(s, e, alpha=0.08, color=colors[i % len(colors)])

# -----------------------------
# Plot functions
# -----------------------------
def plot_metric(df,ideas,col,label,ylab,fname):
    nspk=df["speaker"].nunique()
    dur=df["time_end"].max(); nseg=len(df)
    plt.figure(figsize=(9,4.5)); ax=plt.gca()
    for spk,g in df.dropna(subset=[col]).sort_values("time_mid").groupby("speaker"):
        ax.plot(g["time_mid"],g[col],marker="o",ls="-",label=spk,alpha=0.8)
    if col=="valence": ax.axhline(0,ls="--",lw=1)
    add_markers(ax,ideas)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab)
    ax.set_title(f"{label} over time ({MEETING_ID}) • {nspk} spk • {fmt_hms(dur)} • {nseg} seg")
    ax.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR/fname,dpi=160); plt.close()

# -----------------------------
# Main
# -----------------------------
df=load_segments(SEG_PATH)
ideas=load_ideas(IDEA_PATH)
z_per_speaker(df,"arousal","z_arousal")
z_per_speaker(df,"log_rms","z_rms")
z_per_speaker(df,"pitch_var","z_pitchvar")
df["engagement_index"]=df[["z_arousal","z_rms","z_pitchvar"]].mean(axis=1)

plot_metric(df,ideas,"valence","Valence","Valence (−1…+1)","valence_over_time.png")
plot_metric(df,ideas,"arousal","Arousal","Arousal (0…1)","arousal_over_time.png")
plot_metric(df,ideas,"wps","Speech rate","Words per second","wps_over_time.png")
plot_metric(df,ideas,"engagement_index","Engagement","Engagement (z, per spk)","engagement_over_time.png")

print(f"Plots saved in: {OUT_DIR}")
