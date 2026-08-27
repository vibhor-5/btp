import matplotlib.pyplot as plt
import numpy as np

models = {
    "Best Baseline\nLSTM (h=32)": {"F1": 0.7897, "Recall": 81.29, "MAE": 4.315, "color": "#5c85d6"},
    "Dual-Stream\nLSTM": {"F1": 0.7879, "Recall": 79.62, "MAE": 4.187, "color": "#85c1e9"},
    "Dual-Stream\n+ Shortcut": {"F1": 0.7914, "Recall": 80.94, "MAE": 4.111, "color": "#52be80"},
    "Dual-Stream\nInventory\nShortcut": {"F1": 0.7926, "Recall": 80.79, "MAE": 3.969, "color": "#27ae60"},
    "Dual-Stream\nGated\nShortcut": {"F1": 0.7913, "Recall": 80.40, "MAE": 3.991, "color": "#1a9050"},
}

names = list(models.keys())
f1_vals = [models[m]["F1"] for m in names]
recall_vals = [models[m]["Recall"] for m in names]
mae_vals = [models[m]["MAE"] for m in names]
colors = [models[m]["color"] for m in names]
x = np.arange(len(names))

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
fig.suptitle("LSTM Architecture Evolution: Baseline → Dual-Stream Variants",
             fontsize=13, fontweight="bold", y=1.02)

for ax, vals, title, ylabel, baseline, ylim, fmt in zip(
    axes,
    [f1_vals, recall_vals, mae_vals],
    ["Validation F1-Score", "Stockout Recall (%)", "Stockout MAE (hrs) ↓"],
    ["F1-Score", "Recall (%)", "MAE (hours)"],
    [0.7897, 81.29, 4.315],
    [(0.77, 0.800), (78, 84), (3.7, 4.6)],
    ["{:.4f}", "{:.1f}%", "{:.3f}"]
):
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylim(*ylim)
    ax.axhline(baseline, color="#5c85d6", linestyle="--", linewidth=1.2)
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (ylim[1]-ylim[0])*0.007,
                fmt.format(val), ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

fig.tight_layout(pad=1.5)
plt.savefig("docs/images/dualstream_evolution_comparison.png", dpi=160,
            bbox_inches="tight", facecolor="white")
print("Saved.")
