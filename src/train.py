import pandas as pd
import numpy as np
import joblib
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier

# Allow Python to see preprocessing.py located in the same folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import feature_engineering

def run_training():
    # Define file paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # fraud_project folder
    DATA_PATH = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    # Ensure the models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Loading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File not found: {DATA_PATH}. Please place 'creditcard.csv' in the data folder")
        
    df = pd.read_csv(DATA_PATH)

    # Separate X (features) and y (target)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # shuffle=False is critical for time-series data to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Applying the shared logic from preprocessing.py
    X_train_processed = feature_engineering(X_train)
    X_test_processed = feature_engineering(X_test)

    # RobustScaler is used because it is robust to outliers (extreme fraud amounts)
    scaler = RobustScaler()
    
    # The scaler fits only on the training set
    # reshape(-1, 1) is required because scaler expects a 2D array
    X_train_processed['Amount'] = scaler.fit_transform(X_train_processed['Amount'].values.reshape(-1, 1))
    X_test_processed['Amount'] = scaler.transform(X_test_processed['Amount'].values.reshape(-1, 1))

    # Calculate weight for class imbalance
    pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    # Hyperparameters selected based on previous Optuna optimization
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.12106896936002161,
        max_depth=4,
        subsample=0.6727299868828402,
        colsample_bytree=0.6733618039413735,
        scale_pos_weight=pos_weight, 
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        verbosity=1
    )
    
    model.fit(X_train_processed, y_train)

    # Predict probabilities for the test set
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    
    # Calculate Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Calculate F1 Score for every possible threshold
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores) # Replace NaN with 0
    
    # Find the index of the highest F1 Score
    best_idx = np.argmax(f1_scores)
    
    # Safety check - thresholds array is 1 element shorter than precision/recall arrays
    if best_idx < len(thresholds):
        best_thresh = thresholds[best_idx]
    else:
        best_thresh = 0.5
        
    max_f1 = f1_scores[best_idx]
    print(f"Best Threshold found: {best_thresh:.4f} (Max F1-Score: {max_f1:.4f})")

    # Save the 3 essential components for production
    joblib.dump(model, os.path.join(MODELS_DIR, 'model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(best_thresh, os.path.join(MODELS_DIR, 'threshold.pkl'))

    print(f"Success. Files saved in: {MODELS_DIR}")

if __name__ == "__main__":
    run_training()