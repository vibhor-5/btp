import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

# Hyperparameters
HISTORY_LEN = 14
FORECAST_LEN = 7
BATCH_SIZE = 256
EPOCHS = 5
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
LR = 1e-3

print("Loading data...")
train_df = pd.read_parquet('data/top15_train.parquet')
test_df = pd.read_parquet('data/top15_test.parquet')

# Combine to create continuous sequences, then we will split by time again
combined_df = pd.concat([train_df, test_df], ignore_index=True)
combined_df['dt'] = pd.to_datetime(combined_df['dt'])
combined_df = combined_df.sort_values(by=['store_id', 'product_id', 'dt']).reset_index(drop=True)

# 1. Feature Engineering & Preprocessing
num_cols = ['discount', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
cat_cols = ['city_id', 'store_id', 'management_group_id', 'first_category_id', 
            'second_category_id', 'third_category_id', 'product_id', 'holiday_flag', 'activity_flag']

# Binary Target: 1 if stockout occurred between 6 AM and 10 PM, else 0
combined_df['stockout'] = (combined_df['stock_hour6_22_cnt'] > 0).astype(np.float32)
target_col = 'stockout'

# We will also add historical sale_amount as a feature to help predict stockout
num_cols.append('sale_amount')

# Fill NAs
combined_df[num_cols] = combined_df[num_cols].fillna(0)
for col in cat_cols:
    combined_df[col] = combined_df[col].fillna(-1).astype(str)

# Scale numeric features
scaler = StandardScaler()
combined_df[num_cols] = scaler.fit_transform(combined_df[num_cols])

# Encode Categorical
cat_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])
    cat_encoders[col] = le

print("Building time-series sequences...")
grouped = combined_df.groupby(['store_id', 'product_id'])

X_hist_num, X_hist_cat, X_fut_num, X_fut_cat, Y_fut = [], [], [], [], []
train_indices = []
test_indices = []

current_idx = 0
split_date = pd.to_datetime(train_df['dt'].max())

for _, group in tqdm(grouped, total=len(grouped)):
    group = group.sort_values('dt')
    if len(group) < HISTORY_LEN + FORECAST_LEN:
        continue
    
    # Extract arrays for speed
    g_num = group[num_cols].values
    g_cat = group[cat_cols].values
    g_target = group[target_col].values
    g_dates = group['dt'].values
    
    for i in range(len(group) - HISTORY_LEN - FORECAST_LEN + 1):
        hist_end = i + HISTORY_LEN
        fut_end = hist_end + FORECAST_LEN
        
        hist_num_seq = g_num[i:hist_end]
        hist_cat_seq = g_cat[i:hist_end]
        
        # Historical target (stockout) is added as a numeric feature
        hist_target_seq = g_target[i:hist_end].reshape(-1, 1)
        hist_num_seq = np.hstack([hist_num_seq, hist_target_seq])
        
        # Future covariates (assuming known future weather/promotions, but NOT future sales!)
        # We need to drop the 'sale_amount' from the future covariates since it's unknown
        sale_amount_idx = num_cols.index('sale_amount')
        fut_num_seq = np.delete(g_num[hist_end:fut_end], sale_amount_idx, axis=1)
        fut_cat_seq = g_cat[hist_end:fut_end]
        
        fut_target_seq = g_target[hist_end:fut_end]
        
        X_hist_num.append(hist_num_seq)
        X_hist_cat.append(hist_cat_seq)
        X_fut_num.append(fut_num_seq)
        X_fut_cat.append(fut_cat_seq)
        Y_fut.append(fut_target_seq)
        
        target_start_date = pd.to_datetime(g_dates[hist_end])
        if target_start_date <= split_date:
            train_indices.append(current_idx)
        else:
            test_indices.append(current_idx)
            
        current_idx += 1

X_hist_num = np.array(X_hist_num, dtype=np.float32)
X_hist_cat = np.array(X_hist_cat, dtype=np.int64)
X_fut_num = np.array(X_fut_num, dtype=np.float32)
X_fut_cat = np.array(X_fut_cat, dtype=np.int64)
Y_fut = np.array(Y_fut, dtype=np.float32)

class TSDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        return (torch.tensor(X_hist_num[idx]), torch.tensor(X_hist_cat[idx]),
                torch.tensor(X_fut_num[idx]), torch.tensor(X_fut_cat[idx]),
                torch.tensor(Y_fut[idx]))

train_loader = DataLoader(TSDataset(train_indices), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TSDataset(test_indices), batch_size=BATCH_SIZE, shuffle=False)

print(f"Train sequences: {len(train_indices)}, Test sequences: {len(test_indices)}")

# 2. Sequence-to-Sequence Forecasting Model (TFT-inspired Transformer)
class TimeSeriesTransformer(nn.Module):
    def __init__(self, num_hist_features, num_fut_features, cat_cols_counts, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_cats, embed_dim) for num_cats in cat_cols_counts
        ])
        
        # Projection layers
        self.hist_proj = nn.Linear(num_hist_features + len(cat_cols_counts) * embed_dim, embed_dim)
        self.fut_proj = nn.Linear(num_fut_features + len(cat_cols_counts) * embed_dim, embed_dim)
        
        # Positional encodings
        self.hist_pos = nn.Parameter(torch.randn(1, HISTORY_LEN, embed_dim))
        self.fut_pos = nn.Parameter(torch.randn(1, FORECAST_LEN, embed_dim))
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, batch_first=True, dropout=0.1
        )
        
        # Output head for binary classification
        self.out_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_hist_num, x_hist_cat, x_fut_num, x_fut_cat):
        batch_size = x_hist_num.size(0)
        
        # Embed historical categorical
        hist_cat_embs = [emb(x_hist_cat[:, :, i]) for i, emb in enumerate(self.cat_embeddings)]
        hist_cat_embs = torch.cat(hist_cat_embs, dim=-1) # [B, HistLen, Cats*Embed]
        hist_feat = torch.cat([x_hist_num, hist_cat_embs], dim=-1)
        hist_emb = self.hist_proj(hist_feat) + self.hist_pos.expand(batch_size, -1, -1)
        
        # Embed future categorical
        fut_cat_embs = [emb(x_fut_cat[:, :, i]) for i, emb in enumerate(self.cat_embeddings)]
        fut_cat_embs = torch.cat(fut_cat_embs, dim=-1)
        fut_feat = torch.cat([x_fut_num, fut_cat_embs], dim=-1)
        fut_emb = self.fut_proj(fut_feat) + self.fut_pos.expand(batch_size, -1, -1)
        
        # Pass through Transformer
        out = self.transformer(src=hist_emb, tgt=fut_emb) # [B, ForecastLen, Embed]
        
        # Predict logits
        logits = self.out_head(out).squeeze(-1) # [B, ForecastLen]
        return logits

cat_cols_counts = [len(cat_encoders[c].classes_) for c in cat_cols]

# num_hist_features = len(num_cols) + 1 (for historical target)
# num_fut_features = len(num_cols) - 1 (since we dropped sale_amount for the future)
num_hist_features = len(num_cols) + 1
num_fut_features = len(num_cols) - 1

model = TimeSeriesTransformer(num_hist_features, num_fut_features, cat_cols_counts, 
                              embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"Training on {device}...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for h_num, h_cat, f_num, f_cat, target in train_loader:
        h_num, h_cat, f_num, f_cat, target = h_num.to(device), h_cat.to(device), f_num.to(device), f_cat.to(device), target.to(device)
        
        optimizer.zero_grad()
        logits = model(h_num, h_cat, f_num, f_cat)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for h_num, h_cat, f_num, f_cat, target in test_loader:
        h_num, h_cat, f_num, f_cat, target = h_num.to(device), h_cat.to(device), f_num.to(device), f_cat.to(device), target.to(device)
        logits = model(h_num, h_cat, f_num, f_cat)
        preds = torch.sigmoid(logits).cpu().numpy()
        
        all_preds.append(preds)
        all_targets.append(target.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0).flatten()
all_targets = np.concatenate(all_targets, axis=0).flatten()

auc = roc_auc_score(all_targets, all_preds)
acc = accuracy_score(all_targets, (all_preds > 0.5).astype(int))

print(f"Test AUC: {auc:.4f}")
print(f"Test Accuracy: {acc:.4f}")
