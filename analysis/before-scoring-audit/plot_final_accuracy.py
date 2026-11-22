"""Render Figure 2 as one chart comparing original and final replication scores.

Counts verified against the server's scratch/glm-final-report-20261122.json.
Each model/order aggregates 5 tasks x 5 prompt styles x 140 questions.
Original percentages are transcribed from Ge et al. (2025), Table 1:
https://aclanthology.org/2025.acl-long.321.pdf#page=5
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figure_style import ORDER_COLORS, PALETTE, apply_style, style_axes


OUTPUT = Path(__file__).resolve().parent
MODELS = ["GPT-3.5-Turbo-0613\n(original study)", "DeepSeek V4.1 Flash", "GLM-5.3", "GPT-6 Luna"]
ORDERS = ["Random", "BFS", "DFS", "PR", "PPR"]
# Rows within each order: connectivity, cycle, Hamiltonian path, topological
# sorting, weighted shortest path. Columns: the five prompt styles in Table 1.
# Average all 25 published cell percentages rather than the rounded Avg. column.
ORIGINAL = {
    "Random": [
        [73.93, 70.71, 81.07, 83.93, 82.14],
        [51.79, 53.57, 65.36, 75.71, 76.07],
        [10.71, 15.36, 40.00, 46.07, 45.36],
        [28.93, 31.07, 58.21, 56.07, 60.36],
        [20.00, 25.00, 26.07, 38.93, 40.71],
    ],
    "BFS": [
        [82.14, 87.50, 89.29, 92.50, 95.71],
        [55.71, 56.07, 79.29, 86.07, 86.43],
        [20.00, 20.71, 57.86, 58.57, 57.14],
        [43.21, 40.36, 67.14, 61.43, 65.00],
        [35.36, 42.50, 45.36, 67.50, 65.71],
    ],
    "DFS": [
        [79.29, 82.14, 87.14, 88.21, 89.29],
        [52.14, 53.93, 73.21, 79.29, 81.07],
        [33.93, 37.50, 67.50, 63.93, 59.29],
        [42.14, 48.93, 77.86, 74.29, 72.86],
        [32.14, 34.29, 45.00, 58.57, 57.14],
    ],
    "PR": [
        [77.86, 83.57, 85.71, 84.29, 87.50],
        [55.36, 56.43, 70.36, 80.36, 83.21],
        [15.00, 19.29, 48.93, 55.00, 50.00],
        [35.36, 35.71, 71.07, 58.21, 65.36],
        [30.36, 43.93, 38.93, 43.93, 48.93],
    ],
    "PPR": [
        [76.79, 81.07, 83.93, 84.64, 86.07],
        [54.29, 55.00, 70.00, 79.29, 80.00],
        [16.43, 18.93, 50.00, 53.93, 50.36],
        [37.14, 39.64, 72.50, 58.93, 66.43],
        [32.50, 44.64, 42.14, 45.36, 49.64],
    ],
}
COUNTS = [
    [2533, 2528, 2551, 2502, 2545],
    [2439, 2399, 2399, 2427, 2399],
    [2648, 2705, 2644, 2726, 2749],
]
COLORS = [ORDER_COLORS[order] for order in ORDERS]

apply_style()

original_means = [sum(sum(row) for row in ORIGINAL[order]) / 25 for order in ORDERS]
all_scores = [original_means] + [[100 * count / 3500 for count in row] for row in COUNTS]
print("Original mean accuracies:", dict(zip(ORDERS, original_means)))

fig, ax = plt.subplots(figsize=(12.2, 5.2))
centers = [1.3 * i for i in range(len(MODELS))]
width = 0.205
for j, (order, color) in enumerate(zip(ORDERS, COLORS)):
    positions = [center + (j - 2) * (width + 0.015) for center in centers]
    values = [row[j] for row in all_scores]
    bars = ax.bar(positions, values, width, color=color, label=order, zorder=3)
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values],
                 padding=5, fontsize=10, color=PALETTE["text"])

ax.set_xticks(centers, MODELS, fontsize=12)
ax.set_ylabel("Mean accuracy (%)", fontsize=12, labelpad=12)
ax.set_ylim(0, 90)
ax.set_yticks(range(0, 81, 20))
style_axes(ax)
ax.axhline(0, color=PALETTE["muted"], linewidth=1.2)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5,
          frameon=False, fontsize=12, handlelength=1.5, columnspacing=2.0)
fig.subplots_adjust(left=0.075, right=0.99, bottom=0.18, top=0.84)

for extension in ["png", "svg"]:
    destination = OUTPUT / f"graphdo-original-and-final-accuracy.{extension}"
    fig.savefig(destination)
    print(destination)
plt.close(fig)
