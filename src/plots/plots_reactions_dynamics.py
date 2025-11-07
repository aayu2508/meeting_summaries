# plots_reactions_dynamics.py
import json
from pathlib import Path
from typing import Dict, Any, List
import re

import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def speaker_short_id(s: str) -> str:
    m = re.search(r"SPEAKER_(\d+)", s or "")
    return f"S{int(m.group(1))}" if m else (s or "S?")

def load_mentions(ideas_with_reactions_path: Path) -> pd.DataFrame:
    doc = json.loads(ideas_with_reactions_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for idea_idx, idea in enumerate(doc.get("ideas", []), start=1):
        idea_title = idea.get("idea", "")
        for m in (idea.get("mentions") or []):
            s = float(m.get("start", 0.0))
            e = float(m.get("end", s))
            mid = (s + e) / 2.0
            spk = speaker_short_id(m.get("speaker", ""))
            emo = m.get("emotion") or {}
            pros = (emo.get("prosody") or {})
            sd = m.get("stance_distribution") or {}
            labels = m.get("labels") or []

            rows.append({
                "idea_idx": idea_idx,
                "idea": idea_title,
                "start": s,
                "end": e,
                "mid": mid,
                "speaker_id": spk,
                "segment_id": m.get("segment_id"),
                "valence": emo.get("valence"),
                "arousal": emo.get("arousal"),
                "dominance": emo.get("dominance"),
                "z_valence": emo.get("z_valence"),
                "z_arousal": emo.get("z_arousal"),
                "energy_mean": pros.get("energy_mean"),
                "speech_rate": pros.get("speech_rate"),
                "stance_pro": bool(sd.get("pro", False)),
                "stance_con": bool(sd.get("con", False)),
                "stance_neutral": bool(sd.get("neutral", False)),
                "labels": labels,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No mentions found in ideas_with_reactions.json")
    return df

def stance_code(row) -> int:
    # map stance to {pro:+1, neutral:0, con:-1}
    if row.get("stance_pro"):
        return 1
    if row.get("stance_con"):
        return -1
    return 0

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

# ---------- main plotting ----------
def make_plots(meeting_id: str):
    base = Path("data/outputs") / meeting_id / "context_outputs"
    in_path = base / "ideas_with_reactions.json"
    assert in_path.exists(), f"Missing file: {in_path}"

    df = load_mentions(in_path)
    df["stance_code"] = df.apply(stance_code, axis=1)

    # 1) Stance timeline (scatter; marker encodes stance, size ~ energy)
    out1 = base / f"{meeting_id}_stance_timeline.png"
    ensure_dir(out1)
    plt.figure()
    for spk, d in df.groupby("speaker_id"):
        # choose marker by stance
        dd_pro = d[d["stance_code"] == 1]
        dd_neu = d[d["stance_code"] == 0]
        dd_con = d[d["stance_code"] == -1]

        if not dd_pro.empty:
            plt.scatter(dd_pro["mid"], [spk]*len(dd_pro), marker="^",
                        s=(dd_pro["energy_mean"].fillna(0.02) * 1500).clip(lower=20, upper=300))
        if not dd_neu.empty:
            plt.scatter(dd_neu["mid"], [spk]*len(dd_neu), marker="o",
                        s=(dd_neu["energy_mean"].fillna(0.02) * 1500).clip(lower=20, upper=300))
        if not dd_con.empty:
            plt.scatter(dd_con["mid"], [spk]*len(dd_con), marker="v",
                        s=(dd_con["energy_mean"].fillna(0.02) * 1500).clip(lower=20, upper=300))

    plt.xlabel("time (s)")
    plt.ylabel("speaker")
    plt.title("Stance over time (marker: ^ pro, o neutral, v con; size ~ energy)")
    plt.tight_layout()
    plt.savefig(out1, dpi=160)
    plt.close()

    # 2) Valence over time, per speaker (line)
    out2 = base / f"{meeting_id}_valence_over_time.png"
    plt.figure()
    for spk, d in df.sort_values("mid").groupby("speaker_id"):
        plt.plot(d["mid"], d["valence"], label=spk)
    plt.xlabel("time (s)")
    plt.ylabel("valence")
    plt.title("Valence over time (per speaker)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out2, dpi=160)
    plt.close()

    # 4) Stance mix by speaker (stacked bars)
    stance_counts = df.groupby("speaker_id")[["stance_pro","stance_neutral","stance_con"]].sum().astype(int)
    out4 = base / f"{meeting_id}_stance_mix_by_speaker.png"
    plt.figure()
    idx = range(len(stance_counts))
    pro = stance_counts["stance_pro"].tolist()
    neu = stance_counts["stance_neutral"].tolist()
    con = stance_counts["stance_con"].tolist()

    import numpy as np
    idx = np.arange(len(stance_counts))
    plt.bar(idx, pro, label="pro")
    plt.bar(idx, neu, bottom=pro, label="neutral")
    plt.bar(idx, con, bottom=(np.array(pro)+np.array(neu)), label="con")
    plt.xticks(idx, stance_counts.index.tolist())
    plt.ylabel("# mentions")
    plt.title("Stance mix by speaker")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out4, dpi=160)
    plt.close()

    # 5) Label counts by speaker (bar)
    label_rows = []
    for _, r in df.iterrows():
        for lab in (r["labels"] or []):
            label_rows.append({"speaker_id": r["speaker_id"], "label": lab})
    if label_rows:
        lab_df = pd.DataFrame(label_rows)
        out5 = base / f"{meeting_id}_label_counts_by_speaker.png"
        plt.figure()
        counts = lab_df.groupby(["speaker_id","label"]).size().unstack(fill_value=0)
        counts.plot(kind="bar")
        plt.ylabel("# labels")
        plt.title("Behavioral labels by speaker")
        plt.tight_layout()
        plt.savefig(out5, dpi=160)
        plt.close()

    print("[done] wrote plots:")
    for p in [out1, out2, out3, out4] + ([out5] if label_rows else []):
        print(" -", p)

if __name__ == "__main__":
    # example: python plots_reactions_dynamics.py
    #          (edit below or wrap with argparse if you prefer)
    MEETING_ID = "S14"
    make_plots(MEETING_ID)
