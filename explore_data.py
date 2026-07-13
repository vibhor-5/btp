import pandas as pd

train_df = pd.read_parquet('data/train.parquet')
eval_df = pd.read_parquet('data/eval.parquet')

print('Train shape:', train_df.shape)
print('Train columns:', train_df.columns.tolist())
print('Train sample:\n', train_df.head(2))
print('Eval shape:', eval_df.shape)
