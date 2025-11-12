import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser(description="Plot idea and criteria table from meeting CSV")
ap.add_argument("--meeting-id", required=True, help="e.g., BAC")
ap.add_argument("--model", default="gptfull", help="model name used in the CSV filename (default: gptfull)")
args = ap.parse_args()

base = os.path.join(
    "data", "outputs", args.meeting_id, "context_outputs",
)
CSV = os.path.join(
    base,
    f"eval_criteria_{args.model}_scores.csv"
)

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
        lines.append(" ".join(words[i:i+words_per_line]))
    return "\n".join(lines)

# ---------- Load & clean ----------
df = pd.read_csv(CSV).fillna("")
idea_col_raw = df.columns[0]
score_cols_raw = df.columns[1:]

# Cast scores to int strings or blank
for c in score_cols_raw:
    s = pd.to_numeric(df[c], errors="coerce")
    df[c] = s.apply(lambda v: "" if pd.isna(v) else str(int(v)))

# ---------- Build headers with your wrapping rule ----------
idea_header = "Idea"
criteria_headers_wrapped = [
    "\n".join(humanize_snake(c).split()) for c in score_cols_raw
]
df.columns = [idea_header] + [humanize_snake(c) for c in score_cols_raw]

# ---------- Wrap cell content ----------
df[idea_header] = df[idea_header].apply(lambda x: wrap_by_words(str(x), 3))

# ---------- Compute layout ----------
n_rows, n_cols = df.shape

# Column widths (slightly wider Idea col; enough for criteria too)
col_units = [1.7] + [0.8]*(n_cols-1)
col_sum = float(sum(col_units))
col_widths = [u/col_sum for u in col_units]

# Row height based on max wrapped lines
row_line_counts = []
for _, row in df.iterrows():
    max_lines = 1
    for val in row:
        txt = str(val)
        max_lines = max(max_lines, (txt.count("\n")+1) if txt else 1)
    row_line_counts.append(max_lines)
avg_lines = max(1, np.mean(row_line_counts))

# Figure size heuristic
fig_w = max(12, 1.2 * n_cols + 2)
fig_h = max(6, 0.38 * n_rows * (1 + 0.25 * (avg_lines - 1)))

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

wrapped_headers = [idea_header] + criteria_headers_wrapped

table = ax.table(
    cellText=df.values,
    colLabels=wrapped_headers,
    colWidths=col_widths,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1]
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(10)

# Bold header + a bit taller
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_text_props(weight="bold")
        cell.set_height(0.065)

# Left-align idea column; keep others centered
for r in range(1, n_rows + 1):
    cell = table[(r, 0)]
    cell._loc = 'left'
    cell.get_text().set_ha('left')

# Dynamic row heights
base_h = 0.05
for r in range(1, n_rows + 1):
    max_lines = 1
    for c in range(n_cols):
        txt = table[(r, c)].get_text().get_text()
        max_lines = max(max_lines, (txt.count("\n")+1) if txt else 1)
    for c in range(n_cols):
        table[(r, c)].set_height(base_h * (1 + 0.30 * (max_lines - 1)))

# Thin grid for readability
for cell in table.get_celld().values():
    cell.set_linewidth(0.6)

plt.tight_layout()
path = base + f"/evaluation_table_{args.meeting_id}_{args.model}.png"
plt.savefig(path, dpi=300, bbox_inches="tight")
