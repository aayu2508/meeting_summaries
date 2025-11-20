#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(description="Plot idea - criteria table from meeting CSV")
ap.add_argument("--meeting-id", required=True, help="e.g., BAC")
ap.add_argument(
    "--model",
    default="gptfull",
    help="model name used in the CSV filename (default: gptfull)",
)
args = ap.parse_args()

base_dir = Path("data") / "outputs" / args.meeting_id
ctx_dir = base_dir / "context_outputs"
plots_dir = base_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

csv_path = ctx_dir / f"eval_criteria_{args.model}_matrix.csv"
if not csv_path.exists():
    raise SystemExit(f"CSV not found: {csv_path}")

def humanize_snake(s: str) -> str:
    s = s.replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else s

def wrap_by_words(text: str, words_per_line: int) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    words = text.split()
    if not words:
        return ""
    lines = []
    for i in range(0, len(words), words_per_line):
        lines.append(" ".join(words[i : i + words_per_line]))
    return "\n".join(lines)

df_raw = pd.read_csv(csv_path).fillna("")

# Drop any Unnamed index columns
cols = [c for c in df_raw.columns if not str(c).startswith("Unnamed")]
if not cols:
    raise SystemExit("No usable columns in CSV")

# Find idea TEXT column: prefer column literally named "idea"
idea_col = None
for c in cols:
    if str(c).strip().lower() == "idea":
        idea_col = c
        break
if idea_col is None:
    # Fallback to the first column if "idea" not found
    idea_col = cols[0]

# Meta columns we NEVER want in the matrix
meta_names = {"idea", "idea id", "idea_id", "id"}
meta_cols = {idea_col}
for c in cols:
    if str(c).strip().lower() in meta_names:
        meta_cols.add(c)

# Criteria columns = everything except meta columns
criteria_cols = [c for c in cols if c not in meta_cols]
if not criteria_cols:
    raise SystemExit("No criteria columns found after removing meta columns")

# Wrap idea text for row labels (from idea_col)
idea_labels = df_raw[idea_col].apply(lambda x: wrap_by_words(str(x), 3)).tolist()

# Matrix contains only criteria columns
df = df_raw[criteria_cols].copy()

for c in criteria_cols:
    def _norm(v):
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none"):
            return ""
        return "1"
    df[c] = df[c].apply(_norm)

# Human readable dimension headers
criteria_headers_human = [humanize_snake(c) for c in criteria_cols]
criteria_headers_wrapped = ["\n".join(h.split()) for h in criteria_headers_human]

n_rows, n_cols = df.shape

# Equal width for all criteria columns
col_units = [1.0] * n_cols
col_sum = float(sum(col_units))
col_widths = [u / col_sum for u in col_units]

# Estimate row height from wrapped idea labels
row_line_counts = [lbl.count("\n") + 1 for lbl in idea_labels]
avg_lines = max(1, np.mean(row_line_counts))

fig_w = max(10, 0.8 * n_cols + 4)
fig_h = max(6, 0.35 * n_rows * (1 + 0.25 * (avg_lines - 1)))

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

# cellText: ONLY criteria matrix
# rowLabels: idea text (not part of matrix; no "Idea id" anywhere)
table = ax.table(
    cellText=df.values,
    rowLabels=idea_labels,
    colLabels=criteria_headers_wrapped,
    colWidths=col_widths,
    cellLoc="center",
    rowLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)

table.auto_set_font_size(False)
table.set_fontsize(9)

# Header formatting
for (r, c), cell in table.get_celld().items():
    if r == 0:  # header row (only criteria headers)
        cell.set_text_props(weight="bold")
        cell.set_height(0.06)

# Ensure row labels show idea text, left aligned, never colored
for r in range(1, n_rows + 1):
    if (r, -1) in table.get_celld():
        label_cell = table[(r, -1)]
        label_cell.get_text().set_text(idea_labels[r - 1])
        label_cell._loc = "left"
        label_cell.get_text().set_ha("left")
        label_cell.set_facecolor("white")

# Dynamic row heights (header is row 0)
base_h = 0.045
for r in range(1, n_rows + 1):
    label_lines = idea_labels[r - 1].count("\n") + 1
    max_lines = label_lines
    for c in range(0, n_cols):
        txt = table[(r, c)].get_text().get_text()
        max_lines = max(max_lines, (txt.count("\n") + 1) if txt else 1)
    for c in [-1] + list(range(0, n_cols)):
        if (r, c) in table.get_celld():
            table[(r, c)].set_height(base_h * (1 + 0.30 * (max_lines - 1)))

# Thin grid all around
for cell in table.get_celld().values():
    cell.set_linewidth(0.5)

green = mcolors.to_rgba("#c6efce")  # discussed
red   = mcolors.to_rgba("#ffc7ce")  # not discussed

# Data rows start at 1, data columns are 0..n_cols-1 (pure criteria)
for r in range(1, n_rows + 1):
    for c in range(0, n_cols):
        txt = table[(r, c)].get_text().get_text().strip()
        if txt == "1":
            table[(r, c)].set_facecolor(green)
        else:
            table[(r, c)].set_facecolor(red)

plt.tight_layout()

out_path = plots_dir / f"evaluation_table_{args.meeting_id}_{args.model}.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[ok] saved evaluation table → {out_path}")
