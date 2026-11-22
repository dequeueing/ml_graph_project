"""Shared palette and typography for the report's figures.

Colors identify edge orders, models, or outcomes according to each legend.
Keep each mapping stable when reusing that variable in another figure.
"""

import matplotlib.pyplot as plt


PALETTE = {
    "blue": "#2878A8",
    "orange": "#BD5937",
    "green": "#287F6B",
    "purple": "#78549A",
    "gray": "#88939D",
    "light_gray": "#D6DDE4",
    "text": "#283442",
    "muted": "#536170",
    "grid": "#E6EBEF",
    "background": "#FFFFFF",
}
ORDER_COLORS = dict(zip(
    ["Random", "BFS", "DFS", "PR", "PPR"],
    [PALETTE[key] for key in ["gray", "blue", "orange", "green", "purple"]],
))
MODEL_COLORS = {
    "deepseek-v4.1-flash": PALETTE["blue"],
    "glm-5.3": PALETTE["orange"],
    "gpt-6-luna": PALETTE["green"],
}
OUTCOME_COLORS = {
    "higher": PALETTE["green"],
    "equal": PALETTE["light_gray"],
    "lower": PALETTE["orange"],
}


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "svg.fonttype": "none",
        "text.color": PALETTE["text"],
        "axes.labelcolor": PALETTE["text"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "figure.facecolor": PALETTE["background"],
        "axes.facecolor": PALETTE["background"],
        "savefig.facecolor": PALETTE["background"],
        "savefig.dpi": 300,
    })


def style_axes(ax, grid_axis="y"):
    ax.tick_params(axis="both", length=0, pad=10)
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
