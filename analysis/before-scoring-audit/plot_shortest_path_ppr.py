"""Render Figure 4: PPR minus random accuracy on weighted shortest path.

Use final paired results in analysis/rq1_insights.json. Each comparison uses
the same 140 graph questions under the same model and prompt style.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from figure_style import MODEL_COLORS, PALETTE, apply_style, style_axes


HERE = Path(__file__).resolve().parent
results = json.loads((HERE.parent / "analysis/rq1_insights.json").read_text())
prompts = ["zero_shot", "zero_shot_cot", "few_shot", "cot", "cot_bag"]
prompt_labels = ["Zero-shot", "Zero-shot CoT", "Few-shot", "CoT", "CoT-BAG"]
models = [
    ("deepseek-v4.1-flash", "DeepSeek V4.1 Flash"),
    ("glm-5.3", "GLM-5.3"),
    ("gpt-6-luna", "GPT-6 Luna"),
]

apply_style()
fig, ax = plt.subplots(figsize=(12.2, 5.6))
centers = [1.3 * i for i in range(len(prompts))]
positive_counts = []

for i, (model, label) in enumerate(models):
    comparisons = results["traditional"][model]["ppr_shortest_path_by_prompt"]
    by_prompt = {row["prompt"]: row for row in comparisons}
    assert len(comparisons) == len(prompts) and set(by_prompt) == set(prompts)
    deltas = []
    for prompt in prompts:
        row = by_prompt[prompt]
        assert row["model"] == model and row["task"] == "shortest_path"
        assert row["order"] == "ppr" and row["n"] == 140
        delta = 100 * (row["correct"] - row["baseline_correct"]) / row["n"]
        assert math.isclose(delta, row["delta_pp"], abs_tol=1e-10)
        deltas.append(delta)
    positive_counts.append(sum(value > 0 for value in deltas))
    positions = [center + (i - 1) * 0.30 for center in centers]
    bars = ax.bar(positions, deltas, width=0.27, color=MODEL_COLORS[model], label=label, zorder=3)
    ax.bar_label(bars, labels=[f"{value:+.2f}" for value in deltas],
                 padding=5, fontsize=11, color=PALETTE["text"])
    print(label, dict(zip(prompts, [round(value, 4) for value in deltas])))

assert positive_counts == [3, 1, 5]
assert len(results["paired_comparisons"]) == 300
assert all(row["holm_p"] > 0.05 for row in results["paired_comparisons"])

ax.axhline(0, color=PALETTE["muted"], linewidth=1.2, zorder=2)
ax.set_xticks(centers, prompt_labels, fontsize=12)
ax.set_xlabel("Prompt style", fontsize=12, labelpad=12)
ax.set_ylabel("Accuracy change: PPR − random (pp)", fontsize=12, labelpad=12)
ax.set_ylim(-6.5, 17.5)
ax.set_yticks([-5, 0, 5, 10, 15])
ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.0f}" if value else "0"))
style_axes(ax)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
          frameon=False, fontsize=12, handlelength=1.5, columnspacing=2.0)
fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.18)

for extension in ["png", "svg"]:
    destination = HERE / f"graphdo-shortest-path-ppr.{extension}"
    fig.savefig(destination)
    print(destination)
plt.close(fig)
