import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    print("Loading data...")
    try:
        # Load the dataset (Assuming 'latin1' encoding which is common for this dataset)
        df = pd.read_csv(file_path, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Please ensure the dataset is downloaded.")
        # Create dummy data for testing the pipeline if file is not found
        print("Generating dummy data to demonstrate the pipeline...")
        df = pd.DataFrame({
            'order date (DateOrders)': pd.date_range(start='1/1/2020', periods=1000, freq='D'),
            'Product Card Id': np.random.randint(1, 100, 1000),
            'Order Item Quantity': np.random.randint(1, 10, 1000),
            'Product Price': np.random.uniform(10, 500, 1000),
            'Category Name': np.random.choice(['A', 'B', 'C'], 1000),
            'Late_delivery_risk': np.random.choice([0, 1], 1000, p=[0.9, 0.1]),
        })
    
    print("Preprocessing data...")
    # Convert dates to datetime
    if 'order date (DateOrders)' in df.columns:
        df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'])
        df.sort_values('order date (DateOrders)', inplace=True)
    
    # Feature Engineering
    print("Feature Engineering...")
    # Aggregate to SKU-Day level
    sku_daily = df.groupby(['Product Card Id', pd.Grouper(key='order date (DateOrders)', freq='D')]).agg({
        'Order Item Quantity': 'sum',
        'Product Price': 'mean',
        'Category Name': 'first',
        'Late_delivery_risk': 'max' # Using late delivery risk as a proxy if stockout isn't explicit
    }).reset_index()
    
    # Fill missing dates for each SKU to have continuous time series
    sku_daily.set_index('order date (DateOrders)', inplace=True)
    
    # Create lag features and rolling statistics
    sku_features = []
    for sku, group in sku_daily.groupby('Product Card Id'):
        group = group.resample('D').asfreq().fillna(0) # Forward fill / zero fill for missing days
        group['Category Name'] = group['Category Name'].ffill().bfill()
        group['Product Card Id'] = sku
        
        # Demand history features
        group['sales_lag_1'] = group['Order Item Quantity'].shift(1)
        group['sales_roll_mean_7'] = group['Order Item Quantity'].shift(1).rolling(7).mean()
        group['sales_roll_std_7'] = group['Order Item Quantity'].shift(1).rolling(7).std()
        group['sales_roll_mean_30'] = group['Order Item Quantity'].shift(1).rolling(30).mean()
        
        # Define target: e.g., if demand was high but delivery was late -> proxy for stockout risk
        # Ideally, this would be a real 'Stockout' column. We create a simulated proxy here:
        # Stockout = 1 if Order Item Quantity == 0 in the next period but rolling mean > 0 (demand exists but no sales)
        # For the sake of having a reliable target for this baseline, we will create a simulated 'Stockout_Next_Period'
        group['Stockout_Next_Period'] = ((group['Order Item Quantity'].shift(-1) == 0) & (group['sales_roll_mean_7'] > 0)).astype(int)
        
        sku_features.append(group)
    
    full_df = pd.concat(sku_features).dropna()
    
    # Encoding categorical features
    le = LabelEncoder()
    full_df['Category Name Encoded'] = le.fit_transform(full_df['Category Name'].astype(str))
    
    # Select features for modeling
    features = ['sales_lag_1', 'sales_roll_mean_7', 'sales_roll_std_7', 'sales_roll_mean_30', 'Product Price', 'Category Name Encoded']
    target = 'Stockout_Next_Period'
    
    X = full_df[features]
    y = full_df[target]
    
    return X, y, full_df

def run_baseline_models(X, y):
    print("\n--- Running Baseline Models ---")
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution:\n{y.value_counts(normalize=True)}")
    
    # Time-based train-test split (80-20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Handle Imbalance with SMOTE
    print("\nApplying SMOTE for class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"Resampled class distribution:\n{y_train_res.value_counts(normalize=True)}")
    
    # 1. XGBoost Baseline
    print("\nTraining XGBoost...")
    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1) if sum(y_train == 1) > 0 else 1
    xgb_model = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=6,
        subsample=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train) # Training on original with scale_pos_weight is usually better for trees
    
    xgb_preds = xgb_model.predict(X_test_scaled)
    xgb_probs = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    print("XGBoost Results:")
    print(classification_report(y_test, xgb_preds))
    if len(np.unique(y_test)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_test, xgb_probs):.4f}")
    
    # 2. Random Forest Baseline
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    # Train RF on SMOTE resampled data
    rf_model.fit(X_train_res, y_train_res)
    
    rf_preds = rf_model.predict(X_test_scaled)
    rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    print("Random Forest Results (with SMOTE):")
    print(classification_report(y_test, rf_preds))
    if len(np.unique(y_test)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f}")
    
    return xgb_model, rf_model, X_train_scaled, X_test_scaled, y_test

def explain_model_shap(model, X_train, X_test):
    print("\n--- Running SHAP Explainability Workflow ---")
    # Use TreeExplainer for XGBoost/RF
    explainer = shap.TreeExplainer(model)
    
    # Use a sample for SHAP to speed up computation
    sample_idx = np.random.choice(X_test.shape[0], min(500, X_test.shape[0]), replace=False)
    X_test_sample = X_test.iloc[sample_idx]
    
    shap_values = explainer.shap_values(X_test_sample)
    
    # Visualizations
    plt.figure()
    plt.title("SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.savefig('shap_summary_plot.png', bbox_inches='tight')
    print("Saved SHAP summary plot to 'shap_summary_plot.png'")
    
    # Generate an "Explanation Card" for a specific test instance
    print("\nGenerating Explanation Card for SKU (Instance #0):")
    instance = X_test_sample.iloc[0]
    # For xgboost binary classification, shap_values might be a 2D array or list depending on objective
    if isinstance(shap_values, list):
        instance_shap = shap_values[1][0]
        base_value = explainer.expected_value[1]
    else:
        instance_shap = shap_values[0]
        base_value = explainer.expected_value
        
    print(f"Base Value (Average Model Output): {base_value}")
    
    # Sort features by absolute SHAP contribution
    feature_contributions = list(zip(X_test_sample.columns, instance, instance_shap))
    feature_contributions.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\nTop Contributing Features for this Prediction:")
    for feat, val, contrib in feature_contributions[:3]:
        print(f"- {feat} = {val:.2f} (SHAP contribution: {contrib:+.4f})")
    
    # Save a force plot for the instance
    shap.initjs()
    force_plot = shap.force_plot(base_value, instance_shap, instance, matplotlib=True, show=False)
    if force_plot is not None:
        plt.savefig('shap_force_plot_instance.png', bbox_inches='tight')
        print("Saved SHAP force plot to 'shap_force_plot_instance.png'")

if __name__ == "__main__":
    file_path = "DataCoSupplyChainDataset.csv"
    X, y, df = load_and_preprocess_data(file_path)
    
    if len(X) > 0:
        xgb_model, rf_model, X_train, X_test, y_test = run_baseline_models(X, y)
        explain_model_shap(xgb_model, X_train, X_test)
    else:
        print("Not enough data to run modeling.")
