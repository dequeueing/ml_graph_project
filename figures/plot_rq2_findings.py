"""Render Figures 7 and 8 from the frozen, accepted RQ2 results.

Run independent figures in parallel with --figure ranking or --figure prompting.
This script checks plotted aggregates, but never rescores model responses.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import pvariance

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from figure_style import MODEL_COLORS, PALETTE, apply_style, style_axes


HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent / "analysis/releases/scoring-v1-20261122"
MODELS = [
    ("deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),
    ("glm-5.3", "GLM-5.3"),
    ("gpt-6-luna", "GPT-6 Luna"),
]
PROMPTS = ["zero_shot", "zero_shot_cot", "few_shot", "cot", "cot_bag"]
PROMPT_LABELS = ["Zero-shot", "Zero-shot CoT", "Few-shot", "CoT", "CoT-BAG"]
TASKS = ["connectivity", "cycle", "shortest_path", "hamilton", "topology"]
SEQUENCE_TASKS = [
    ("shortest_path", "Weighted\nshortest path"),
    ("hamilton", "Hamiltonian\npath"),
    ("topology", "Topological\nsorting"),
]
# The direction colors describe sensitivity, so green means lower variance.
CHANGE_COLORS = {
    "lower": PALETTE["green"],
    "equal": PALETTE["light_gray"],
    "higher": PALETTE["orange"],
}


def load_data():
    source = RELEASE / "analysis/rq2_insights.json"
    manifest = json.loads((RELEASE / "MANIFEST.json").read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == manifest["files_sha256"]["analysis/rq2_insights.json"]
    results = json.loads(source.read_text())
    cells = {(r["model"], r["task"], r["prompt"]): r for r in results["cells"]}
    assert len(cells) == 75
    for row in cells.values():
        n = row["questions_per_order"]
        assert n == (78 if row["task"] == "hamilton" else 140)
        accuracies = [100 * count / n for count in row["correct_by_order"].values()]
        assert len(accuracies) == 5
        assert math.isclose(pvariance(accuracies), row["variance_pp2"], abs_tol=1e-10)
    print(f"Source SHA256 {digest}")
    return results, cells


def save_figure(fig, name):
    # Catch labels outside the canvas before writing either output format.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in fig.findobj(match=matplotlib.text.Text):
        if not label.get_visible() or not label.get_text().strip():
            continue
        box = label.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1, label.get_text()
        assert box.x1 <= fig.bbox.width + 1, label.get_text()
        assert box.y1 <= fig.bbox.height + 1, label.get_text()
    for extension in ["png", "svg"]:
        destination = HERE / f"{name}.{extension}"
        fig.savefig(destination)
        print(destination)
    plt.close(fig)


def plot_ranking(results, cells):
    """Trace the identity of the most sensitive task, not its numeric rank."""
    rankings = {(r["model"], r["prompt"]): r for r in results["rankings"]}
    winners = {}
    for model, _ in MODELS:
        for prompt in PROMPTS:
            task_values = {task: cells[model, task, prompt]["variance_pp2"] for task in TASKS}
            maximum = max(task_values.values())
            highest = [task for task, value in task_values.items()
                       if math.isclose(value, maximum, abs_tol=1e-10)]
            assert len(highest) == 1
            assert highest == rankings[model, prompt]["highest_tasks"]
            winners[model, prompt] = highest[0]
    counts = Counter(winners.values())
    assert counts["shortest_path"] == 4
    assert counts["shortest_path"] == results["findings"]["shortest_path_highest_settings"]
    assert winners["gpt-6-luna", "zero_shot"] == "hamilton"
    assert winners["gpt-6-luna", "zero_shot_cot"] == "shortest_path"
    print("Most-sensitive-task counts", dict(counts))

    fig, ax = plt.subplots(figsize=(12.8, 5.1))
    fig.subplots_adjust(left=0.19, right=0.975, top=0.78, bottom=0.18)
    style_axes(ax, grid_axis="x")
    task_y = {task: i for i, (task, _) in enumerate(SEQUENCE_TASKS)}
    # Task rows are nominal categories. Within-row offsets keep coincident
    # model results visible, and do not encode differences in variance.
    for y in task_y.values():
        ax.axhspan(y - 0.31, y + 0.31, color=PALETTE["grid"], alpha=0.52, zorder=0)
    for i, (model, label) in enumerate(MODELS):
        y_values = [task_y[winners[model, prompt]] + (i - 1) * 0.12
                    for prompt in PROMPTS]
        ax.plot(range(5), y_values, marker="o", markersize=9,
                markeredgecolor="white", markeredgewidth=1.1,
                color=MODEL_COLORS[model], linewidth=2.2, label=label, zorder=3)
    ax.set_xticks(range(5), PROMPT_LABELS, fontsize=11)
    ax.set_yticks(range(3), [label for _, label in SEQUENCE_TASKS], fontsize=12)
    ax.set_xlim(-0.12, 4.12)
    ax.set_ylim(2.47, -0.47)
    ax.tick_params(axis="y", pad=19)
    ax.set_xlabel("Prompt style", labelpad=12)
    ax.text(-0.185, 1.055, "Most sensitive task", transform=ax.transAxes,
            fontsize=12, fontweight="bold", ha="left")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.57, 0.98), ncol=3, frameon=False,
               fontsize=11, handlelength=1.7, columnspacing=2)
    save_figure(fig, "graphdo-reviewed-sensitivity-leaders")


def variance_direction(before, after):
    if math.isclose(before, after, abs_tol=1e-10):
        return "equal"
    return "lower" if after < before else "higher"


def plot_prompting(results, cells):
    contrasts = [
        ("zero_shot", "zero_shot_cot", "Add a reasoning instruction", "Zero-shot → Zero-shot CoT"),
        ("few_shot", "cot", "Add worked reasoning examples", "Few-shot → CoT"),
        ("cot", "cot_bag", "Add graph-construction instructions", "CoT → CoT-BAG"),
    ]
    saved = {(r["before"], r["after"]): r for r in results["prompt_contrasts"]}
    counts = []
    for before, after, _, _ in contrasts:
        directions = Counter(
            variance_direction(cells[model, task, before]["variance_pp2"],
                               cells[model, task, after]["variance_pp2"])
            for model, _ in MODELS for task in TASKS
        )
        assert sum(directions.values()) == 15
        assert dict(directions) == saved[before, after]["variance_changes"]
        counts.append(directions)
        print(before, "→", after, dict(directions))

    example = [cells["gpt-6-luna", "hamilton", prompt] for prompt in ["cot", "cot_bag"]]
    variances = [row["variance_pp2"] for row in example]
    accuracies = [row["mean_accuracy_pct"] for row in example]
    assert variances[1] / variances[0] > 2
    print("Luna Hamiltonian variance", variances, "mean accuracy", accuracies)

    fig, (summary, detail) = plt.subplots(
        1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.85, 1]}
    )
    fig.subplots_adjust(left=0.305, right=0.98, top=0.73, bottom=0.23, wspace=0.43)
    style_axes(summary, grid_axis="x")
    style_axes(detail, grid_axis="y")

    for y, count in enumerate(counts):
        left = 0
        for direction in ["lower", "equal", "higher"]:
            width = count[direction]
            summary.barh(y, width, left=left, height=0.52,
                         color=CHANGE_COLORS[direction], edgecolor="white", linewidth=1.2,
                         zorder=3)
            summary.text(left + width / 2, y, str(width), ha="center", va="center",
                         fontsize=13, fontweight="bold",
                         color=PALETTE["text"] if direction == "equal" else "white")
            left += width
        summary.text(-0.075, y - 0.12, contrasts[y][2],
                     transform=summary.get_yaxis_transform(), ha="right", va="center",
                     fontsize=10.8, fontweight="bold", clip_on=False)
        summary.text(-0.075, y + 0.16, contrasts[y][3],
                     transform=summary.get_yaxis_transform(), ha="right", va="center",
                     fontsize=10.5, color=PALETTE["muted"], clip_on=False)
    summary.set_xlim(0, 15)
    summary.set_xticks([0, 3, 6, 9, 12, 15])
    summary.set_yticks([])
    summary.set_ylim(2.55, -0.55)
    summary.set_xlabel("Number of model–task combinations", fontsize=11, labelpad=12)
    summary.set_title("Changes in order sensitivity", fontsize=12, fontweight="bold", pad=18)

    bars = detail.bar([0, 1], variances, width=0.58,
                      color=[PALETTE["gray"], PALETTE["orange"]], zorder=3)
    detail.bar_label(bars, labels=[f"{value:.2f}" for value in variances],
                     padding=5, fontsize=12, fontweight="bold")
    detail.set_xticks([0, 1], ["CoT", "CoT-BAG"], fontsize=11)
    detail.set_ylim(0, 103)
    detail.set_yticks([0, 25, 50, 75, 100])
    detail.set_xlim(-0.6, 1.6)
    detail.set_title("Luna on Hamiltonian path", color=MODEL_COLORS["gpt-6-luna"],
                     fontsize=12, fontweight="bold", pad=18)
    detail.set_ylabel("Variance (squared percentage points)", fontsize=10.5, labelpad=8)
    detail.text(0.5, -0.26, f"Mean accuracy {accuracies[0]:.2f}% → {accuracies[1]:.2f}%",
                transform=detail.transAxes, ha="center", fontsize=10.5,
                color=PALETTE["muted"])

    handles = [Patch(facecolor=CHANGE_COLORS[key], label=label)
               for key, label in [("lower", "Lower variance"),
                                  ("equal", "Unchanged"), ("higher", "Higher variance")]]
    # Keep the legend with the summary panel, as the right panel shows absolute values.
    summary.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.22),
                   ncol=3, frameon=False, fontsize=10.5, handlelength=1.2, columnspacing=1.5)
    save_figure(fig, "graphdo-reviewed-reasoning-sensitivity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=["ranking", "prompting"], required=True)
    args = parser.parse_args()
    results, cells = load_data()
    apply_style()
    if args.figure == "ranking":
        plot_ranking(results, cells)
    else:
        plot_prompting(results, cells)


if __name__ == "__main__":
    main()
