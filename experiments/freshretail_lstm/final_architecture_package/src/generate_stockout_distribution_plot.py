import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.family': 'sans-serif',
    'figure.titlesize': 13,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'savefig.dpi': 300
})

print("Loading full dataset...")
train_path = Path("data/train.parquet")
if train_path.exists():
    raw_df = pd.read_parquet(train_path)
else:
    from datasets import load_dataset
    raw_df = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train").to_pandas()

# Group strictly PER SKU (product_id) across all stores
sku_stats = raw_df.groupby("product_id").agg(
    total_stockout_hours=("stock_hour6_22_cnt", "sum"),
    total_sales=("sale_amount", "sum"),
    store_count=("store_id", "nunique")
).reset_index()

num_skus = len(sku_stats)
print(f"Total Unique SKUs (product_id): {num_skus}")

# Sort SKUs by total stockouts
sku_stats = sku_stats.sort_values("total_stockout_hours", ascending=False).reset_index(drop=True)

# Create Single Panel Figure (Diagram A Only)
fig, ax = plt.subplots(figsize=(8.5, 5.2))

# --- Single Plot: Distribution of Total Stockout Hours Per SKU (Histogram & KDE) ---
sns.histplot(
    data=sku_stats,
    x="total_stockout_hours",
    bins=35,
    kde=True,
    color="#1976d2",
    edgecolor="white",
    linewidth=0.8,
    ax=ax,
    alpha=0.6
)

mean_val = sku_stats["total_stockout_hours"].mean()
median_val = sku_stats["total_stockout_hours"].median()
max_val = sku_stats["total_stockout_hours"].max()
min_val = sku_stats["total_stockout_hours"].min()

ax.axvline(mean_val, color="#d32f2f", linestyle="--", linewidth=1.8, label=f"Mean per SKU ({mean_val:,.0f} hrs)")
ax.axvline(median_val, color="#388e3c", linestyle="-.", linewidth=1.8, label=f"Median per SKU ({median_val:,.0f} hrs)")

ax.set_title("Non-Uniform Stockout Distribution Per Product SKU\n(Extracted Across All 865 Fresh Products in FreshRetailNet-50K)", fontweight="bold", pad=12)
ax.set_xlabel("Total Stockout Hours per Product SKU", fontweight="bold", labelpad=8)
ax.set_ylabel("Number of SKUs (Products)", fontweight="bold", labelpad=8)
ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")

# Annotate skewness & justification
ax.annotate(
    f"Heavy Long-Tail Skewness\nMax: {max_val:,.0f} hrs vs Min: {min_val:,.0f} hrs\n(Justifies Statistical Filtering to Top 15 SKUs)",
    xy=(mean_val * 2.5, 45),
    xytext=(mean_val * 2.2, 70),
    fontweight="bold",
    fontsize=9.5,
    color="#0d47a1",
    arrowprops=dict(arrowstyle="->", color="#0d47a1", lw=1.3),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#e3f2fd", edgecolor="#0d47a1", lw=0.9)
)

plt.tight_layout()

out_plot_path = "docs/images/stockout_distribution_analysis.png"
plt.savefig(out_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSuccessfully saved single-panel per-SKU distribution plot at: {out_plot_path}")
