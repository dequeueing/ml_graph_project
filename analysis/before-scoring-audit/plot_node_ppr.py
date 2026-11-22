"""Render Figure 5: PPR effects by task family, task, and citation dataset.

Traditional-task rows average five prompt styles. Dataset rows average both
subgraph samplers. All changes use correct-answer counts in the final analysis.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

from figure_style import MODEL_COLORS, PALETTE, apply_style, style_axes


HERE = Path(__file__).resolve().parent
results = json.loads((HERE.parent / "analysis/rq1_insights.json").read_text())
models = [("deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),
          ("gpt-6-luna", "GPT-6 Luna")]
tasks = [("connectivity", "Connectivity"),
         ("cycle", "Cycle detection"),
         ("shortest_path", "Weighted shortest path"),
         ("hamilton", "Hamiltonian path"),
         ("topology", "Topological sorting")]
datasets = [("cora", "Cora"), ("citeseer", "CiteSeer"), ("pubmed", "PubMed")]
row_positions = [0, 1, 2, 3, 4, 5, 7.3, 8.3, 9.3, 10.3]
row_labels = ["Family mean"] + [label for _, label in tasks]
row_labels += ["Family mean"] + [label for _, label in datasets]
summary_indices = {0, 6}


def accuracy_change(rows):
    return 100 * sum(row["correct"] - row["baseline_correct"] for row in rows) / sum(
        row["n"] for row in rows)


apply_style()
fig, ax = plt.subplots(figsize=(12.2, 8.4))
fig.subplots_adjust(left=0.245, right=0.975, top=0.85, bottom=0.085)
style_axes(ax, grid_axis="x")

# Extend summary shading through the label column to make the hierarchy clear.
row_transform = blended_transform_factory(ax.transAxes, ax.transData)
for index in summary_indices:
    y = row_positions[index]
    ax.fill_between([-0.31, 1], y - 0.43, y + 0.43,
                    color=PALETTE["grid"], alpha=0.65,
                    transform=row_transform, clip_on=False, linewidth=0, zorder=0)

for model_index, (model, label) in enumerate(models):
    traditional = [row for row in results["paired_comparisons"]
                   if row["model"] == model and row["order"] == "ppr"]
    assert len(traditional) == 25 and all(row["n"] == 140 for row in traditional)
    traditional_mean = accuracy_change(traditional)
    assert math.isclose(traditional_mean,
                        results["traditional"][model]["mean_delta_pp"]["ppr"],
                        abs_tol=1e-10)
    task_deltas = []
    for task, _ in tasks:
        rows = [row for row in traditional if row["task"] == task]
        assert len(rows) == 5 and len({row["prompt"] for row in rows}) == 5
        task_deltas.append(accuracy_change(rows))
    assert math.isclose(sum(task_deltas) / len(task_deltas), traditional_mean,
                        abs_tol=1e-10)

    node = results["node"][model]["ppr"]
    assert len(node) == 6 and all(row["n"] == 75 for row in node)
    for row in node:
        assert math.isclose(accuracy_change([row]), row["delta_pp"], abs_tol=1e-10)
    node_mean = accuracy_change(node)
    dataset_deltas = []
    for dataset, _ in datasets:
        rows = [row for row in node if row["group"].split("/")[0] == dataset]
        assert {row["group"] for row in rows} == {
            f"{dataset}/ego", f"{dataset}/forest_fire"}
        dataset_deltas.append(accuracy_change(rows))
    assert math.isclose(sum(dataset_deltas) / len(dataset_deltas), node_mean,
                        abs_tol=1e-10)

    deltas = [traditional_mean] + task_deltas + [node_mean] + dataset_deltas
    positions = [y + (model_index - 0.5) * 0.31 for y in row_positions]
    bars = ax.barh(positions, deltas, height=0.25,
                   color=MODEL_COLORS[model], label=label, zorder=3)
    for index, (bar, value) in enumerate(zip(bars, deltas)):
        ax.annotate(f"{value:+.2f}",
                    xy=(value, bar.get_y() + bar.get_height() / 2),
                    xytext=(5 if value >= 0 else -5, 0),
                    textcoords="offset points", va="center",
                    ha="left" if value >= 0 else "right",
                    fontsize=10.5,
                    fontweight="bold" if index in summary_indices else "normal",
                    color=PALETTE["text"])
    print(label, dict(zip(["Traditional mean"] + [name for _, name in tasks]
                         + ["Node classification mean"] + [name for _, name in datasets],
                         [round(value, 4) for value in deltas])))

ax.axvline(0, color=PALETTE["muted"], linewidth=1.2, zorder=2)
ax.set_yticks(row_positions, row_labels, fontsize=11)
ax.tick_params(axis="y", pad=14)
for index, tick in enumerate(ax.get_yticklabels()):
    if index in summary_indices:
        tick.set_fontweight("bold")
ax.set_ylim(10.95, -1.55)
ax.set_xlim(-24, 12.5)
ax.set_xticks([-20, -15, -10, -5, 0, 5, 10])
ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.0f}" if value else "0"))
ax.set_xlabel("Accuracy change: PPR − random (percentage points)", labelpad=12)

# Each family has a plain-language heading and an explicit averaging rule.
for y, title, explanation in [
    (-1.17, "Traditional graph tasks", "Each task averages five prompt styles"),
    (6.13, "Node classification", "Each citation dataset averages two sampling methods"),
]:
    ax.text(-0.30, y, title, transform=row_transform, clip_on=False,
            fontsize=13, fontweight="bold", va="center")
    ax.text(-0.30, y + 0.39, explanation, transform=row_transform, clip_on=False,
            fontsize=10, color=PALETTE["muted"], va="center")

fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
           bbox_to_anchor=(0.54, 0.985), ncol=2, frameon=False,
           fontsize=12, handlelength=1.5, columnspacing=2)
ax.text(-0.5, 1.055, "← Lower accuracy", transform=ax.get_xaxis_transform(),
        ha="right", va="center", fontsize=10.5, color=PALETTE["muted"])
ax.text(0.5, 1.055, "Higher accuracy →", transform=ax.get_xaxis_transform(),
        ha="left", va="center", fontsize=10.5, color=PALETTE["muted"])

for extension in ["png", "svg"]:
    destination = HERE / f"graphdo-ppr-task-dataset-breakdown.{extension}"
    fig.savefig(destination)
    print(destination)
plt.close(fig)
