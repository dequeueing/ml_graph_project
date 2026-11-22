"""Render Figure 6 from the accepted scoring release without rescoring answers.

Small points retain all five prompt styles for each task. Diamonds summarize
their mean variance, matching Table 8. Model colors come from the report palette.
"""

import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pvariance

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from figure_style import MODEL_COLORS, PALETTE, apply_style, style_axes


HERE = Path(__file__).resolve().parent
RELEASE = HERE.parent / "analysis/releases/scoring-v1-20261122"
SOURCE = RELEASE / "analysis/rq2_insights.json"
MODELS = [
    ("deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),
    ("glm-5.3", "GLM-5.3"),
    ("gpt-6-luna", "GPT-6 Luna"),
]
TASKS = [
    ("connectivity", "Connectivity"),
    ("cycle", "Cycle detection"),
    ("shortest_path", "Weighted shortest path"),
    ("hamilton", "Hamiltonian path"),
    ("topology", "Topological sorting"),
]
PROMPTS = ["zero_shot", "zero_shot_cot", "few_shot", "cot", "cot_bag"]
ROW_POSITIONS = [0, 1, 2.6, 3.6, 4.6]
POINT_OFFSETS = [-0.18, -0.09, 0, 0.09, 0.18]


def load_variances():
    """Check the frozen source and the quantities actually plotted."""
    manifest = json.loads((RELEASE / "MANIFEST.json").read_text())
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == manifest["files_sha256"]["analysis/rq2_insights.json"]
    results = json.loads(SOURCE.read_text())
    cells = {(r["model"], r["task"], r["prompt"]): r for r in results["cells"]}
    summaries = {(r["model"], r["task"]): r for r in results["task_summary"]}
    assert len(cells) == 75 and len(summaries) == 15
    values = {}
    for model, _ in MODELS:
        for task, _ in TASKS:
            task_values = []
            for prompt in PROMPTS:
                row = cells[model, task, prompt]
                n = row["questions_per_order"]
                assert n == (78 if task == "hamilton" else 140)
                accuracies = [100 * count / n for count in row["correct_by_order"].values()]
                variance = pvariance(accuracies)
                assert math.isclose(variance, row["variance_pp2"], abs_tol=1e-10)
                task_values.append(variance)
            assert math.isclose(mean(task_values),
                                summaries[model, task]["mean_variance_pp2"], abs_tol=1e-10)
            values[model, task] = task_values

    exceptions = []
    for model, _ in MODELS:
        for i, prompt in enumerate(PROMPTS):
            binary_max = max(values[model, task][i] for task, _ in TASKS[:2])
            sequence_min = min(values[model, task][i] for task, _ in TASKS[2:])
            if binary_max >= sequence_min:
                exceptions.append((model, prompt))
    assert exceptions == [("glm-5.3", "zero_shot")]
    assert 15 - len(exceptions) == results["findings"]["binary_tasks_least_variable_settings"]
    print(f"Verified 75 plotted variances and 15 means against {SOURCE}")
    print(f"Source SHA256 {digest}")
    print("Binary tasks below all sequence tasks in 14 of 15 settings")
    print("GLM zero-shot exception", {
        task: round(values["glm-5.3", task][0], 4) for task, _ in TASKS
    })
    return values


def main():
    values = load_variances()
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 6.3), sharey=True)
    fig.subplots_adjust(left=0.235, right=0.98, top=0.79, bottom=0.16, wspace=0.16)

    for index, (ax, (model, name)) in enumerate(zip(axes, MODELS)):
        color = MODEL_COLORS[model]
        style_axes(ax, grid_axis="x")
        row_transform = blended_transform_factory(ax.transAxes, ax.transData)
        # Shade both binary-task rows as one group, including their labels.
        ax.fill_between([-0.99 if index == 0 else 0, 1], -0.48, 1.48,
                        transform=row_transform, clip_on=False,
                        color=PALETTE["grid"], alpha=0.62, linewidth=0, zorder=0)
        for (task, _), y in zip(TASKS, ROW_POSITIONS):
            row = values[model, task]
            # Fixed vertical offsets separate coincident values, not uncertainty.
            ax.scatter(row, [y + offset for offset in POINT_OFFSETS],
                       s=27, color=color, alpha=0.48, linewidths=0, zorder=3)
            ax.scatter([mean(row)], [y], marker="D", s=76, color=color,
                       edgecolors="white", linewidths=1.1, zorder=4)

        ax.set_title(name, color=color, fontsize=13, fontweight="bold", pad=34)
        ax.set_ylim(5.15, -1.03)
        if index < 2:
            ax.set_xlim(-0.45, 14)
            ax.set_xticks([0, 4, 8, 12])
        else:
            ax.set_xlim(-2.9, 90)
            ax.set_xticks([0, 30, 60, 90])
            ax.text(0.5, 1.04, "Wider horizontal scale", transform=ax.transAxes,
                    ha="center", fontsize=10, color=PALETTE["muted"])
        ax.axvline(0, color=PALETTE["light_gray"], linewidth=0.9, zorder=1)
        ax.set_yticks(ROW_POSITIONS, [label for _, label in TASKS], fontsize=11)
        ax.tick_params(axis="y", pad=14)

    label_transform = blended_transform_factory(axes[0].transAxes, axes[0].transData)
    for y, label in [(-0.78, "Binary graph questions"),
                     (1.93, "Vertex-sequence tasks")]:
        axes[0].text(-0.94, y, label, transform=label_transform, clip_on=False,
                     fontsize=12, fontweight="bold", va="center")

    legend = [
        Line2D([], [], linestyle="none", marker="o", markersize=5.5,
               markerfacecolor=PALETTE["muted"], markeredgewidth=0, alpha=0.48,
               label="One prompt style"),
        Line2D([], [], linestyle="none", marker="D", markersize=7,
               markerfacecolor=PALETTE["muted"], markeredgecolor="white",
               label="Mean across five prompt styles"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.58, 0.99),
               ncol=2, frameon=False, fontsize=11, handlelength=1, columnspacing=2)
    fig.text(0.60, 0.055, "Variance across edge orders (squared percentage points)",
             ha="center", fontsize=12)
    for extension in ["png", "svg"]:
        destination = HERE / f"graphdo-reviewed-task-sensitivity.{extension}"
        fig.savefig(destination)
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
