import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, confusion_matrix, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
import optuna.samplers

df = pd.read_csv('creditcard.csv')

# Checking for missing values
print(df.isnull().sum().sort_values(ascending=False))

# Imbalance analysis
fraud_count = df['Class'].sum()
total_count = len(df)
fraud_percentage = (fraud_count / total_count) * 100

print(f"Total number of transactions: {total_count}")
print(f"Number of Frauds: {fraud_count}")
print(f"Percentage of Frauds: {fraud_percentage:.4f}% (Confirmation: 1 per mille = 0.1%)")

# Imbalance plot
plt.figure(figsize=(7, 5))
sns.countplot(x='Class', data=df)
plt.title(f'Class Distribution (Imbalance: {fraud_percentage:.4f}%)', fontsize=14)
plt.xticks([0, 1], ['Normal (0)', 'Fraud (1)'])
plt.ylabel('Number of Transactions')
plt.xlabel('Class')
plt.show()

# V1-V28 columns: Already processed by PCA, no need to scale them again

# Time Feature Engineering
df['Hour_of_Day'] = (df['Time'] % 86400) / 3600

# Create Sin/Cos features to capture the cyclicality of hours
df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour_of_Day'] / 24.0)
df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour_of_Day'] / 24.0)

# Visualization (time will be a circle)
sample = df.sample(1000)
plt.figure(figsize=(6, 6))
plt.scatter(sample['Hour_Sin'], sample['Hour_Cos'], alpha=0.5)
plt.title('Cyclic Representation of Time (Random 1000 samples)')
plt.xlabel('Hour_Sin')
plt.ylabel('Hour_Cos')
plt.axis('equal')
plt.grid(True)
plt.show()


cols_to_drop_initial = ['Time', 'Hour_of_Day']
df = df.drop(columns=cols_to_drop_initial)

# Data splitting
# Chronological split
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

# Create X and y, using .copy() to avoid SettingWithCopyWarning
X_train = train_df.drop('Class', axis=1).copy()
y_train = train_df['Class']
X_test = test_df.drop('Class', axis=1).copy()
y_test = test_df['Class']

# Scaling amount (after split to avoid leakage)
print(f"Statistics for Amount (Train set):\n{X_train['Amount'].describe()}")

scaler_amount = RobustScaler()

# fit_transform on train
X_train['Amount_Scaled'] = scaler_amount.fit_transform(X_train['Amount'].values.reshape(-1, 1))

# transform on test
X_test['Amount_Scaled'] = scaler_amount.transform(X_test['Amount'].values.reshape(-1, 1))
X_train = X_train.drop(columns=['Amount'])
X_test = X_test.drop(columns=['Amount'])


# Calculate class weight
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

# Evaluation function
def time_series_cv_score(model, X, y, splits=3):
    tscv = TimeSeriesSplit(n_splits=splits)
    scores = []
    for train_index, val_index in tscv.split(X):
        X_t, X_v = X.iloc[train_index], X.iloc[val_index]
        y_t, y_v = y.iloc[train_index], y.iloc[val_index]
        
        model.fit(X_t, y_t)
        preds_proba = model.predict_proba(X_v)[:, 1]
        score = average_precision_score(y_v, preds_proba)
        scores.append(score)
    return np.mean(scores)

# Optuna

def objective_xgb(trial):
    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0
    }
    model = XGBClassifier(**params)
    return time_series_cv_score(model, X_train, y_train)

def objective_lgbm(trial):
    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "is_unbalance": True,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    model = LGBMClassifier(**params)
    return time_series_cv_score(model, X_train, y_train)

def objective_cat(trial):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "auto_class_weights": "Balanced",
        "random_state": 42,
        "verbose": 0,
        "allow_writing_files": False
    }
    model = CatBoostClassifier(**params)
    return time_series_cv_score(model, X_train, y_train)


models_config = {
    "XGBoost": {"objective": objective_xgb, "class": XGBClassifier, "fixed_params": {"n_estimators": 1000, "scale_pos_weight": scale_pos_weight, "tree_method": "hist", "n_jobs": -1, "verbosity": 0}},
    "LightGBM": {"objective": objective_lgbm, "class": LGBMClassifier, "fixed_params": {"n_estimators": 1000, "is_unbalance": True, "n_jobs": -1, "verbose": -1}},
    "CatBoost": {"objective": objective_cat, "class": CatBoostClassifier, "fixed_params": {"iterations": 1000, "auto_class_weights": "Balanced", "verbose": 0, "allow_writing_files": False}}
}

results = {}


for name, config in models_config.items():
    print(f"\nOptimization: {name}")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(config["objective"], n_trials=30)
    
    print(f"Best AUC-PR (CV): {study.best_value:.4f}")
    
    # Merging best parameters with fixed ones
    final_params = {**config["fixed_params"], **study.best_params}
    
    # Final training
    model = config["class"](**final_params)
    model.fit(X_train, y_train)
    
    # Probability prediction on test set
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        "model": model,
        "y_proba": y_proba,
        "best_cv_score": study.best_value
    }

# Finding optimal threshlold
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
plt.suptitle("Confusion Matrix Comparison (Optimal F1 Threshold)", fontsize=18, y=1.05)

comparison_table = []

# Loop
for idx, (name, res) in enumerate(results.items()):
    # Get appropriate plot axis
    ax = axes[idx] 
        
    y_proba = res["y_proba"]
        
    # Calculating precision-recall and thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        
    # Calculating F1 for all thresholds
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    f1_scores = np.nan_to_num(f1_scores) # Replace NaN with 0
        
    # Finding the best index
    best_idx = np.argmax(f1_scores)
        
    if best_idx < len(thresholds):
        best_thresh = thresholds[best_idx]
    else:
        best_thresh = thresholds[-1]
            
    best_f1 = f1_scores[best_idx]
        
    # Binary prediction (0/1) using calculated threshold
    y_pred = (y_proba >= best_thresh).astype(int)
        
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
        
    # Drawing heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                annot_kws={"size": 14}) 
        
    # Axis labels
    ax.set_title(f"{name}\nThreshold: {best_thresh:.4f} | Max F1: {best_f1:.4f}", fontsize=14)
    ax.set_xlabel("Prediction (Predicted)", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_xticklabels(['Normal', 'Fraud'], fontsize=11)
    ax.set_yticklabels(['Normal', 'Fraud'], fontsize=11)
        
    # Add to results table
    ap = average_precision_score(y_test, y_proba)
    comparison_table.append({
        "Model": name,
        "AUC-PR": ap,
        "Threshold": best_thresh,
        "F1 Score": best_f1,
        "False Negatives": cm[1, 0], 
        "False Positives": cm[0, 1]
    })

plt.tight_layout()
plt.show()


print("Results summary")

df_results = pd.DataFrame(comparison_table).set_index("Model")
print(df_results.sort_values(by="AUC-PR", ascending=False)) 

print("Cutoff Point Interpretation")
print("The threshold was selected to maximize the F1-score.")
print("A low threshold means the model is 'afraid' of missing a fraud and raises an alarm more often.")
print("A high threshold means the model reports fraud only when it is very certain.")


# Since XGBoost has the best AUC-PR, it will be chosen as a production model

# Get the trained XGBoost model object
xgb_model = results["XGBoost"]["model"]
    
# Get all parameters
params = xgb_model.get_params()
    
# Print them formatted nicely
print("Copy this block inside the run_training() function in src/train.py:\n")
    
print("model = XGBClassifier(")
print(f"n_estimators={params['n_estimators']},")
print(f"learning_rate={params['learning_rate']},")
print(f"max_depth={params['max_depth']},")
    
# Use .get() just in case these weren't tuned and are None
if params.get('subsample'):
    print(f"subsample={params['subsample']},")
if params.get('colsample_bytree'):
    print(f"colsample_bytree={params['colsample_bytree']},")
        
print(f"scale_pos_weight={params['scale_pos_weight']},")
print('tree_method="hist",')
print('n_jobs=-1,')
print('random_state=42,')
print('verbosity=1')
print("    )")