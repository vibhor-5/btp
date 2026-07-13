import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Load data
train_df = pd.read_parquet('data/top15_train.parquet')
test_df = pd.read_parquet('data/top15_test.parquet')

# 2. Preprocess
# Target: binary classification (1 if there is a stockout, 0 otherwise)
train_df['target'] = (train_df['stock_hour6_22_cnt'] > 0).astype(int)
test_df['target'] = (test_df['stock_hour6_22_cnt'] > 0).astype(int)

# Features
num_cols = ['discount', 'precpt', 'avg_temperature', 'avg_humidity', 'avg_wind_level']
cat_cols = ['city_id', 'store_id', 'management_group_id', 'first_category_id', 
            'second_category_id', 'third_category_id', 'product_id', 'holiday_flag', 'activity_flag']

# Fill NAs
for col in num_cols:
    train_df[col] = train_df[col].fillna(0)
    test_df[col] = test_df[col].fillna(0)

for col in cat_cols:
    train_df[col] = train_df[col].fillna(-1).astype(str)
    test_df[col] = test_df[col].fillna(-1).astype(str)

# Scale numeric
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_df[num_cols])
X_test_num = scaler.transform(test_df[num_cols])

# Encode categorical
cat_encoders = {}
X_train_cat = np.zeros((len(train_df), len(cat_cols)), dtype=int)
X_test_cat = np.zeros((len(test_df), len(cat_cols)), dtype=int)

for i, col in enumerate(cat_cols):
    le = LabelEncoder()
    # Fit on both to avoid unknown categories, or just handle gracefully
    combined = pd.concat([train_df[col], test_df[col]])
    le.fit(combined)
    X_train_cat[:, i] = le.transform(train_df[col])
    X_test_cat[:, i] = le.transform(test_df[col])
    cat_encoders[col] = le

y_train = train_df['target'].values
y_test = test_df['target'].values

class StockoutDataset(Dataset):
    def __init__(self, num_features, cat_features, labels):
        self.num_features = torch.tensor(num_features, dtype=torch.float32)
        self.cat_features = torch.tensor(cat_features, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.num_features[idx], self.cat_features[idx], self.labels[idx]

train_dataset = StockoutDataset(X_train_num, X_train_cat, y_train)
test_dataset = StockoutDataset(X_test_num, X_test_cat, y_test)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# 3. Model: Tabular Transformer Baseline
class TabularTransformer(nn.Module):
    def __init__(self, num_cols_count, cat_cols_counts, embed_dim=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_cats, embed_dim) for num_cats in cat_cols_counts
        ])
        # Project numeric features to embed_dim
        self.num_proj = nn.Linear(1, embed_dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x_num, x_cat):
        batch_size = x_num.shape[0]
        
        # Categorical embeddings
        cat_embs = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            cat_embs.append(emb_layer(x_cat[:, i]).unsqueeze(1))
            
        # Numeric embeddings
        num_embs = []
        for i in range(x_num.shape[1]):
            val = x_num[:, i].unsqueeze(-1)
            num_embs.append(self.num_proj(val).unsqueeze(1))
            
        # Concatenate sequence
        seq = torch.cat(cat_embs + num_embs, dim=1) # [batch_size, num_cat + num_num, embed_dim]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, seq], dim=1)
        
        # Transformer
        out = self.transformer(seq)
        
        # Use CLS token for prediction
        cls_out = out[:, 0, :]
        logits = self.mlp(cls_out).squeeze(-1)
        return logits

cat_cols_counts = [len(cat_encoders[c].classes_) for c in cat_cols]
model = TabularTransformer(len(num_cols), cat_cols_counts, embed_dim=32, num_heads=4, num_layers=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4. Training
print(f"Training on device: {device}")
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_num, x_cat, y in train_loader:
        x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# 5. Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for x_num, x_cat, y in test_loader:
        x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
        logits = model(x_num, x_cat)
        preds = torch.sigmoid(logits).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

auc = roc_auc_score(all_labels, all_preds)
acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

print(f"Test AUC: {auc:.4f}")
print(f"Test Accuracy: {acc:.4f}")
