"""Render the five-node example in Figure 1 using the shared report style."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from figure_style import PALETTE, apply_style


HERE = Path(__file__).resolve().parent
positions = {0: (0, 0), 1: (-1.4, 1.15), 2: (1.4, 1.15),
             3: (-2.2, -0.5), 4: (2.2, -0.5)}
edges = [(0, 1), (0, 2), (1, 3), (2, 4)]

apply_style()
fig, ax = plt.subplots(figsize=(7.4, 3.6))
for source, target in edges:
    ax.plot([positions[source][0], positions[target][0]],
            [positions[source][1], positions[target][1]],
            color=PALETTE["muted"], linewidth=2, zorder=1)
for node, (x, y) in positions.items():
    ax.add_patch(Circle((x, y), radius=0.34,
                        facecolor=PALETTE["light_gray"],
                        edgecolor=PALETTE["blue"], linewidth=2, zorder=2))
    ax.text(x, y, str(node), ha="center", va="center", fontsize=18, zorder=3)
ax.set_aspect("equal")
ax.set_xlim(-2.75, 2.75)
ax.set_ylim(-1.05, 1.7)
ax.axis("off")
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
for extension in ["png", "svg"]:
    destination = HERE / f"graphdo-edge-order-example.{extension}"
    fig.savefig(destination)
    print(destination)
plt.close(fig)
