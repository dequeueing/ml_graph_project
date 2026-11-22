"""Render RQ3 Figures 9–11 and an optional traversal comparison.

Use analysis/rq3_insights.json. Each --figure option is independent and can run
in parallel after analyze_rq3.py. The traversal comparison is not in the report.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import blended_transform_factory

from figure_style import MODEL_COLORS, ORDER_COLORS, PALETTE, apply_style, style_axes


HERE = Path(__file__).resolve().parent
MODELS = [("deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),
          ("glm-5.3", "GLM-5.3"), ("gpt-6-luna", "GPT-6 Luna")]
TASKS = [("connectivity", "Connectivity"), ("cycle", "Cycle detection"),
         ("shortest_path", "Weighted shortest path"),
         ("hamilton", "Hamiltonian path"), ("topology", "Topological sorting")]
PROMPTS = ["zero_shot", "zero_shot_cot", "few_shot", "cot", "cot_bag"]
ORDER_LABELS = {"random": "Random", "bfs": "BFS", "dfs": "DFS", "pagerank": "PR", "ppr": "PPR"}


def load_results():
    results = json.loads((HERE.parent / "analysis/rq3_insights.json").read_text())
    release = HERE.parent / "analysis/releases" / results["provenance"]["release"]
    for name, digest in results["provenance"]["files_sha256"].items():
        assert hashlib.sha256((release / name).read_bytes()).hexdigest() == digest
    return results


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in fig.findobj(match=matplotlib.text.Text):
        if label.get_visible() and label.get_text().strip():
            box = label.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, label.get_text()
            assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1, label.get_text()
    for suffix in ["png", "svg"]:
        path = HERE / f"graphdo-reviewed-{name}.{suffix}"
        fig.savefig(path)
        print(path)
    plt.close(fig)


def traversal(results):
    lookup = {(r["model"], r["task"]): r for r in results["dfs_vs_bfs"]}
    fig, ax = plt.subplots(figsize=(12.8, 6.3))
    fig.subplots_adjust(left=0.235, right=0.79, top=0.78, bottom=0.15)
    style_axes(ax, grid_axis="x")
    positions = [1.18 * i for i in range(5)]
    for index, (model, label) in enumerate(MODELS):
        ys = [y + (index - 1) * 0.25 for y in positions]
        values = [lookup[model, task]["dfs_minus_bfs_pp"] for task, _ in TASKS]
        ax.barh(ys, values, height=0.21, color=MODEL_COLORS[model], label=label, zorder=3)
        for y, value in zip(ys, values):
            ax.annotate(f"{value:+.2f}" if abs(value) > 1e-10 else "0.00",
                        (value, y), xytext=(5 if value >= 0 else -5, 0),
                        textcoords="offset points", va="center",
                        ha="left" if value >= 0 else "right", fontsize=10.5)
    ax.axvline(0, color=PALETTE["muted"], linewidth=1.1)
    ax.set_xlim(-2.3, 4.4)
    ax.set_xticks([-2, -1, 0, 1, 2, 3, 4])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.0f}" if x else "0"))
    ax.set_ylim(5.27, -0.55)
    ax.set_yticks(positions, [name for _, name in TASKS], fontsize=11)
    ax.set_xlabel("Mean accuracy difference, DFS − BFS (percentage points)", labelpad=12, fontsize=11)
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for y, preferred in zip(positions, ["BFS", "BFS", "BFS", "DFS", "DFS"]):
        ax.text(1.17, y, preferred, transform=transform, ha="center", va="center", fontsize=12)
    ax.text(1.17, 1.075, "GraphDO's preferred\ntraversal order", transform=ax.transAxes,
            ha="center", fontsize=10.5, color=PALETTE["muted"])
    ax.text(-0.1, 1.07, "← BFS higher", transform=ax.get_xaxis_transform(),
            ha="right", fontsize=10.5, color=PALETTE["muted"])
    ax.text(0.1, 1.07, "DFS higher →", transform=ax.get_xaxis_transform(),
            ha="left", fontsize=10.5, color=PALETTE["muted"])
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.56, 0.985),
               ncol=3, frameon=False, fontsize=11, handlelength=1.5)
    save(fig, "traversal-preferences")


def prompt_leaders(results):
    cells = {(r["model"], r["prompt"]): r for r in results["traditional_cells"]
             if r["task"] == "shortest_path"}
    averages = {r["model"]: r for r in results["traditional_task_means"] if r["task"] == "shortest_path"}
    fig, ax = plt.subplots(figsize=(12.8, 4.7))
    fig.subplots_adjust(left=0.20, right=0.98, top=0.76, bottom=0.10)
    x_positions = [0, 1, 2, 3, 4, 5.45]
    y_positions = [0, 1.18, 2.36]
    for (model, _), y in zip(MODELS, y_positions):
        values = [cells[model, prompt]["leaders"] for prompt in PROMPTS] + [averages[model]["leaders"]]
        for x, winners in zip(x_positions, values):
            height = 0.79 / len(winners)
            for j, order in enumerate(winners):
                bottom = y - 0.395 + j * height
                color = ORDER_COLORS[ORDER_LABELS[order]]
                ax.add_patch(Rectangle((x - 0.44, bottom), 0.88, height,
                                      facecolor=color, alpha=0.15, edgecolor="white", linewidth=2))
                ax.add_patch(Rectangle((x - 0.44, bottom), 0.045, height,
                                      facecolor=color, edgecolor="none"))
                ax.text(x, bottom + height / 2, ORDER_LABELS[order], ha="center", va="center",
                        fontsize=12 if len(winners) < 3 else 10.5, fontweight="bold", color=color)
    ax.set_xlim(-0.55, 6.02)
    ax.set_ylim(2.98, -0.78)
    ax.set_xticks(x_positions, ["Zero-shot", "Zero-shot\nCoT", "Few-shot", "CoT", "CoT-BAG", "Mean across\nfive prompts"], fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(y_positions, [label for _, label in MODELS], fontsize=11.5)
    ax.tick_params(axis="both", length=0, pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axvline(4.76, ymin=0.07, ymax=0.88, color=PALETTE["light_gray"], linestyle="--", linewidth=1)
    fig.text(0.59, 0.95, "Highest-accuracy order on weighted shortest path", ha="center",
             fontsize=13, fontweight="bold")
    save(fig, "shortest-path-order-leaders")


def node_families(results):
    lookup = {(r["model"], r["group"]): r for r in results["node_groups"]}
    groups = [f"{dataset}/{sampler}" for dataset in ["cora", "citeseer", "pubmed"]
              for sampler in ["ego", "forest_fire"]]
    positions = [0, 0.9, 2.65, 3.55, 5.3, 6.2]
    fig, ax = plt.subplots(figsize=(12.8, 7.0))
    fig.subplots_adjust(left=0.21, right=0.975, top=0.82, bottom=0.14)
    style_axes(ax, grid_axis="x")
    transform = blended_transform_factory(ax.transAxes, ax.transData)
    for y, label in [(-0.7, "Cora"), (1.95, "CiteSeer"), (4.6, "PubMed")]:
        ax.text(-0.21, y, label, transform=transform, clip_on=False,
                fontsize=12, fontweight="bold", va="center")
    for i, (model, label) in enumerate([MODELS[0], MODELS[2]]):
        values = [lookup[model, group]["probability_minus_traversal_pp"] for group in groups]
        ys = [y + (i - 0.5) * 0.29 for y in positions]
        ax.barh(ys, values, height=0.24, color=MODEL_COLORS[model], label=label, zorder=3)
        for y, value in zip(ys, values):
            ax.annotate(f"{value:+.2f}", (value, y), xytext=(5 if value > 0 else -5, 0),
                        textcoords="offset points", ha="left" if value > 0 else "right",
                        va="center", fontsize=10.5)
    ax.axvline(0, color=PALETTE["muted"], linewidth=1.1)
    ax.set_xlim(-20, 8.2)
    ax.set_xticks([-20, -15, -10, -5, 0, 5])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.0f}" if x else "0"))
    ax.set_ylim(6.85, -1.06)
    ax.set_yticks(positions, ["Ego sampling", "Forest Fire sampling"] * 3, fontsize=11)
    ax.set_xlabel("Accuracy difference between the best orders in each family (percentage points)",
                   labelpad=12, fontsize=11)
    ax.text(-0.3, 1.055, "← BFS/DFS higher", transform=ax.get_xaxis_transform(), ha="right",
            fontsize=10.5, color=PALETTE["muted"])
    ax.text(0.3, 1.055, "PR/PPR higher →", transform=ax.get_xaxis_transform(), ha="left",
            fontsize=10.5, color=PALETTE["muted"])
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.57, 0.98),
               ncol=2, frameon=False, fontsize=11.5, handlelength=1.5)
    save(fig, "node-order-families")


def identical_prompts(results):
    lookup = {r["model"]: r for r in results["identical_prompt_controls"]}
    fig, ax = plt.subplots(figsize=(12.8, 4.7))
    fig.subplots_adjust(left=0.235, right=0.80, top=0.75, bottom=0.20)
    style_axes(ax, grid_axis="x")
    for y, (model, _) in enumerate(MODELS):
        row = lookup[model]
        left = 0
        for key, label in [("pr_only", "PR"), ("ppr_only", "PPR")]:
            count = row["outcomes"][key]
            ax.barh(y, count, left=left, height=0.5, color=ORDER_COLORS[label],
                     edgecolor="white", linewidth=1.3, zorder=3)
            ax.text(left + count / 2, y, str(count), ha="center", va="center",
                    color="white", fontsize=12, fontweight="bold")
            left += count
        ax.text(left + 2, y, f"{row['discordant']} ({row['discordant_pct']:.1f}%)",
                va="center", fontsize=10.5)
        ax.text(1.15, y, f"{row['pr_minus_ppr_pp']:+.2f} pp", transform=ax.get_yaxis_transform(),
                ha="center", va="center", fontsize=11.5)
    ax.set_xlim(0, 114)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(2.55, -0.55)
    ax.set_yticks(range(3), [label for _, label in MODELS], fontsize=11.5)
    ax.set_xlabel("Prompt pairs with different correctness", labelpad=12, fontsize=11.5)
    ax.text(1.15, 1.08, "Accuracy gap\nPR − PPR", transform=ax.transAxes,
            ha="center", fontsize=11, color=PALETTE["muted"])
    handles = [Patch(facecolor=ORDER_COLORS[label], label=f"Only the {label} call was correct")
               for label in ["PR", "PPR"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.98),
               ncol=2, frameon=False, fontsize=11, handlelength=1.4, columnspacing=2)
    save(fig, "identical-prompt-variation")


def main():
    functions = {"traversal": traversal, "prompt": prompt_leaders,
                 "node": node_families, "control": identical_prompts}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=functions, required=True)
    args = parser.parse_args()
    apply_style()
    functions[args.figure](load_results())


if __name__ == "__main__":
    main()
