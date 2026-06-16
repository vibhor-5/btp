import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(train_path, sample_frac=0.1):
    print(f"Loading data from {train_path}...")
    # Load dataset, drop the last row if it's just summary/NA
    df = pd.read_csv(train_path, low_memory=False)
    
    # The last row of this dataset is often entirely NA
    df.dropna(subset=['went_on_backorder'], inplace=True)
    
    # We sample the data to make baseline analysis run in a reasonable time 
    # (e.g., 10% of 1.6M rows = ~160k rows)
    df = df.sample(frac=sample_frac, random_state=42)
    
    print("Preprocessing data...")
    # Handle missing values
    # -99 is used for missing in performance metrics
    df['perf_6_month_avg'] = df['perf_6_month_avg'].replace(-99.0, np.nan)
    df['perf_12_month_avg'] = df['perf_12_month_avg'].replace(-99.0, np.nan)
    
    # Fill NAs
    df['lead_time'].fillna(df['lead_time'].median(), inplace=True)
    df['perf_6_month_avg'].fillna(df['perf_6_month_avg'].median(), inplace=True)
    df['perf_12_month_avg'].fillna(df['perf_12_month_avg'].median(), inplace=True)
    
    # Drop SKU column as it's an identifier
    df.drop(['sku'], axis=1, inplace=True)
    
    # Encoding categorical columns (Yes/No)
    categorical_cols = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop', 'went_on_backorder']
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # went_on_backorder -> 1 for Yes, 0 for No
    
    X = df.drop('went_on_backorder', axis=1)
    y = df['went_on_backorder']
    
    return X, y

def run_baseline_models(X, y):
    print("\n--- Running Baseline Models ---")
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1
    
    # 1. XGBoost Baseline
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=150,
        max_depth=6,
        subsample=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    xgb_preds = xgb_model.predict(X_test_scaled)
    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    print("XGBoost Results:")
    print(classification_report(y_test, xgb_preds))
    roc_auc_xgb = roc_auc_score(y_test, xgb_probs)
    print(f"ROC-AUC: {roc_auc_xgb:.4f}")
    
    # 2. Random Forest Baseline
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15, # Limiting depth to prevent out-of-memory and speed up
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    rf_preds = rf_model.predict(X_test_scaled)
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    print("Random Forest Results:")
    print(classification_report(y_test, rf_preds))
    roc_auc_rf = roc_auc_score(y_test, rf_probs)
    print(f"ROC-AUC: {roc_auc_rf:.4f}")
    
    return xgb_model, rf_model, X_train_scaled, X_test_scaled, y_test

def explain_model_shap(model, X_train, X_test):
    print("\n--- Running SHAP Explainability Workflow ---")
    explainer = shap.TreeExplainer(model)
    
    sample_idx = np.random.choice(X_test.shape[0], min(500, X_test.shape[0]), replace=False)
    X_test_sample = X_test.iloc[sample_idx]
    
    shap_values = explainer.shap_values(X_test_sample)
    
    plt.figure()
    plt.title("SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.savefig('shap_summary_plot.png', bbox_inches='tight')
    print("Saved SHAP summary plot to 'shap_summary_plot.png'")
    
    print("\nGenerating Explanation Card for SKU (Instance #0):")
    instance = X_test_sample.iloc[0]
    if isinstance(shap_values, list):
        instance_shap = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        instance_shap = shap_values[0]
        base_value = explainer.expected_value
        
    print(f"Base Value (Average Model Output): {base_value}")
    
    feature_contributions = list(zip(X_test_sample.columns, instance, instance_shap))
    feature_contributions.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\nTop Contributing Features for this Prediction:")
    for feat, val, contrib in feature_contributions[:5]:
        print(f"- {feat} = {val:.2f} (SHAP contribution: {contrib:+.4f})")
    
    force_plot = shap.force_plot(base_value, instance_shap, instance, matplotlib=True, show=False)
    if force_plot is not None:
        plt.savefig('shap_force_plot_instance.png', bbox_inches='tight')
        print("Saved SHAP force plot to 'shap_force_plot_instance.png'")

if __name__ == "__main__":
    train_path = "SKU_backorder_dataset/Kaggle_Training_Dataset_v2.csv"
    X, y = load_and_preprocess_data(train_path, sample_frac=0.1) # Use 10% for baseline speed
    
    xgb_model, rf_model, X_train, X_test, y_test = run_baseline_models(X, y)
    explain_model_shap(xgb_model, X_train, X_test)
