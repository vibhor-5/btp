import pandas as pd
import numpy as np

# Load datasets
train_df = pd.read_parquet('data/train.parquet')
eval_df = pd.read_parquet('data/eval.parquet')

# Combine datasets
combined_df = pd.concat([train_df, eval_df], ignore_index=True)

# Check dates
unique_dates = sorted(combined_df['dt'].unique())
print(f"Total unique days: {len(unique_dates)}")
print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")

# Find top 15 statistically important and varying SKUs
# Let's use variance of sale_amount and stock_hour6_22_cnt
sku_stats = combined_df.groupby('product_id').agg({
    'sale_amount': ['var', 'mean', 'count'],
    'stock_hour6_22_cnt': ['var', 'mean']
}).fillna(0)

# We can rank by variance of sale_amount or stock_hour6_22_cnt
# Let's use variance of sale_amount * variance of stock_hour6_22_cnt as a metric of "important and varying"
score = sku_stats[('sale_amount', 'var')] * sku_stats[('stock_hour6_22_cnt', 'var')]
top_15_skus = score.nlargest(15).index.tolist()

print(f"Top 15 SKUs: {top_15_skus}")

# Filter for top 15 SKUs
filtered_df = combined_df[combined_df['product_id'].isin(top_15_skus)].copy()

# Sort by dt
filtered_df = filtered_df.sort_values(by=['product_id', 'dt'])

# Split into first 75 days and last 15 days
# Wait, let's see how many days are there.
train_filtered = filtered_df[filtered_df['dt'].isin(unique_dates[:75])]
test_filtered = filtered_df[filtered_df['dt'].isin(unique_dates[75:90])]

print(f"Train set shape: {train_filtered.shape}")
print(f"Test set shape: {test_filtered.shape}")

# Save the datasets separately
train_filtered.to_parquet('data/top15_train.parquet', index=False)
test_filtered.to_parquet('data/top15_test.parquet', index=False)

print("Saved top 15 SKUs datasets to data/top15_train.parquet and data/top15_test.parquet")
