"""Plot Figure 3: higher/equal/lower accuracy relative to random order.

Replication counts come from analysis/rq1_insights.json. Original-study counts
are computed from the GraphDO Table 1 transcription in plot_final_accuracy.py.
All bars contain 100 comparisons: 5 tasks x 5 prompt styles x 4 orders.
"""

import ast
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import OUTCOME_COLORS, PALETTE, apply_style, style_axes


HERE = Path(__file__).resolve().parent
results = json.loads((HERE.parent / "analysis/rq1_insights.json").read_text())
tree = ast.parse((HERE / "plot_final_accuracy.py").read_text())
original = next(
    ast.literal_eval(node.value) for node in tree.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "ORIGINAL"
            for target in node.targets)
)
baseline = [value for row in original["Random"] for value in row]
original_outcomes = Counter()
for order, rows in original.items():
    if order == "Random":
        continue
    for value, reference in zip([v for row in rows for v in row], baseline):
        original_outcomes["higher" if value > reference else
                          "equal" if value == reference else "lower"] += 1

model_ids = ["deepseek-v4.1-flash", "glm-5.3", "gpt-6-luna"]
labels = ["GPT-3.5-Turbo-0613\n(original study)",
          "DeepSeek V4.1 Flash", "GLM-5.3", "GPT-6 Luna"]
rows = [original_outcomes] + [
    results["traditional"][model]["higher_equal_lower"] for model in model_ids
]
assert all(sum(row.values()) == 100 for row in rows)
total = Counter()
for row in rows[1:]:
    total.update(row)
assert total == results["traditional_summary"]["higher_equal_lower"]

apply_style()
fig, ax = plt.subplots(figsize=(12.2, 4.8))
categories = [
    ("higher", "Higher accuracy", OUTCOME_COLORS["higher"], PALETTE["background"]),
    ("equal", "Equal accuracy", OUTCOME_COLORS["equal"], PALETTE["text"]),
    ("lower", "Lower accuracy", OUTCOME_COLORS["lower"], PALETTE["background"]),
]
left = [0] * len(rows)
for key, label, color, text_color in categories:
    widths = [row.get(key, 0) for row in rows]
    ax.barh(range(len(rows)), widths, left=left, height=0.58,
            color=color, edgecolor=PALETTE["background"], linewidth=1.2, label=label, zorder=3)
    for i, (start, width) in enumerate(zip(left, widths)):
        if width:
            ax.text(start + width / 2, i, str(width), ha="center", va="center",
                    color=text_color, fontsize=15, fontweight="medium")
    left = [start + width for start, width in zip(left, widths)]

ax.set_yticks(range(len(rows)), labels, fontsize=12)
ax.set_ylim(len(rows) - 0.5, -0.5)
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 20))
ax.set_xlabel("Number of comparisons (100 per model)", fontsize=12, labelpad=10)
style_axes(ax, grid_axis="x")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
          frameon=False, fontsize=12, handlelength=1.5, columnspacing=2.0)
fig.subplots_adjust(left=0.235, right=0.975, top=0.84, bottom=0.18)

for extension in ["png", "svg"]:
    destination = HERE / f"graphdo-order-consistency.{extension}"
    fig.savefig(destination)
    print(destination)
plt.close(fig)
