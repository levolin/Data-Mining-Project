#!/usr/bin/env python3
"""
fitbit_pipeline.py
Complete data-mining pipeline for a Fitbit-like dataset downloaded via kagglehub.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# -- Data download (kagglehub)
try:
    import kagglehub
except Exception as e:
    print("kagglehub not installed or import failed. Install/enable it or place CSV locally.")
    print("Continuing assuming dataset already available locally.")
    kagglehub = None

import pandas as pd
import numpy as np

# Visualization (optional)
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling / preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Imbalance handling
try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

# Save/Load model
import joblib

# SHAP (optional)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# -----------------------------
# Helper functions
# -----------------------------
def download_dataset_kagglehub(dataset_ref="arashnic/fitbit"):
    """
    Download dataset via kagglehub.dataset_download and return downloaded path.
    If kagglehub is not available or fails, returns None.
    """
    if kagglehub is None:
        return None
    try:
        path = kagglehub.dataset_download(dataset_ref)
        return path
    except Exception as e:
        print("kagglehub.dataset_download failed:", e)
        return None

def find_csv_in_dir(path):
    """
    Return first CSV path under the given directory. Raises if none found.
    """
    if path is None:
        return None
    for root, _, files in os.walk(path):
        for fname in files:
            if fname.lower().endswith(".csv"):
                return os.path.join(root, fname)
    return None

def smart_load_csv(path_or_dir):
    """
    If path_or_dir is file -> load directly.
    If directory -> search for csv and load first CSV found.
    """
    if os.path.isfile(path_or_dir):
        return pd.read_csv(path_or_dir)
    elif os.path.isdir(path_or_dir):
        csv_path = find_csv_in_dir(path_or_dir)
        if csv_path:
            return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No CSV found at {path_or_dir}")

def create_binary_target(df):
    """
    Heuristic target creation:
    - If 'HeartRate' (or 'heartRate', 'avg_heart_rate') exists: target = HeartRate > median
    - Else if 'Calories' and 'TotalSteps' exist: target = low activity (TotalSteps < median)
    - Else fall back to unsupervised binary label via KMeans on numeric features (2 clusters).
    Returns (df_with_target, target_column_name, note)
    """
    df_work = df.copy()
    columns_lower = {c.lower(): c for c in df_work.columns}
    note = ""
    # common HR names
    for hr_name in ["heartrate", "heart_rate", "avgheartrate", "avg_heartrate", "restingheartrate", "resting_heart_rate"]:
        if hr_name in columns_lower:
            col = columns_lower[hr_name]
            df_work['Target'] = (df_work[col] > df_work[col].median()).astype(int)
            note = f"Target created from column '{col}' (above median => 1)."
            return df_work, 'Target', note

    # steps-based fallback
    for step_name in ["totalsteps", "steps", "step_count", "stepcount"]:
        if step_name in columns_lower:
            col = columns_lower[step_name]
            df_work['Target'] = (df_work[col] < df_work[col].median()).astype(int)  # low activity => positive
            note = f"Target created from column '{col}' (below median => 1 = low activity)."
            return df_work, 'Target', note

    # calories fallback
    for cal_name in ["calories", "totalcalories", "calorie"]:
        if cal_name in columns_lower:
            col = columns_lower[cal_name]
            df_work['Target'] = (df_work[col] > df_work[col].median()).astype(int)
            note = f"Target created from column '{col}' (above median => 1)."
            return df_work, 'Target', note

    # Fallback unsupervised: KMeans binary cluster
    from sklearn.cluster import KMeans
    num_cols = df_work.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        # If too few numeric columns, create a random target (last resort)
        df_work['Target'] = np.random.binomial(1, 0.15, size=len(df_work))
        note = "Random target created (very few numeric columns)."
        return df_work, 'Target', note

    Xnum = df_work[num_cols].fillna(df_work[num_cols].median()).values
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(Xnum)
    # decide which cluster is "positive" based on mean risk-like metric (sum of standardized features)
    # cluster with higher mean sum will be labeled 1
    cluster_means = []
    for c in [0,1]:
        cluster_means.append(Xnum[clusters==c].mean())
    # label cluster with larger overall mean as 1
    cluster_mean_sums = [np.nansum(cm) for cm in cluster_means]
    positive_cluster = int(np.argmax(cluster_mean_sums))
    df_work['Target'] = (clusters == positive_cluster).astype(int)
    note = "Target created using unsupervised KMeans on numeric features (cluster with higher mean => 1)."
    return df_work, 'Target', note

def preprocess(df, target_col):
    """
    Preprocessing pipeline:
    - drop id-like columns if present
    - fill numeric missing with median
    - fill categorical missing with mode
    - encode categorical: one-hot if small cardinality else label-encode
    - scale numeric features
    Returns X, y, feature_names, raw_df
    """
    df_proc = df.copy()
    # Drop obvious ID columns
    id_like = [c for c in df_proc.columns if c.lower() in ('id', 'patient_id', 'user_id', 'patientid', 'userid')]
    df_proc.drop(columns=[c for c in id_like if c in df_proc.columns], inplace=True, errors='ignore')

    # separate y
    if target_col not in df_proc.columns:
        raise ValueError("target_col not in DataFrame")
    y = df_proc[target_col].astype(int).copy()
    df_proc.drop(columns=[target_col], inplace=True)

    # fill missing values
    num_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_proc.select_dtypes(include=['object', 'category']).columns.tolist()

    for c in num_cols:
        df_proc[c] = df_proc[c].fillna(df_proc[c].median())

    for c in cat_cols:
        df_proc[c] = df_proc[c].fillna(df_proc[c].mode().iloc[0] if not df_proc[c].mode().empty else "missing")

    # encode categories
    for c in cat_cols:
        if df_proc[c].nunique() <= 10:
            dummies = pd.get_dummies(df_proc[c], prefix=c, drop_first=True)
            df_proc = pd.concat([df_proc.drop(columns=[c]), dummies], axis=1)
        else:
            le = LabelEncoder()
            df_proc[c] = le.fit_transform(df_proc[c])

    # final numeric cols after encoding
    final_num_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()

    # scale numeric
    scaler = StandardScaler()
    df_proc[final_num_cols] = scaler.fit_transform(df_proc[final_num_cols])

    X = df_proc.copy()
    return X, y, X.columns.tolist()

def train_and_evaluate(X_train, y_train, X_test, y_test, use_smote=True):
    """
    Train multiple models and evaluate. Returns dict of fitted models and results dataframe.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42)
    }

    results = []
    fitted_models = {}

    # Optionally apply SMOTE
    if use_smote and SMOTE is not None:
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        print(f"SMOTE applied: train size {X_train.shape} -> {X_train_res.shape}")
    else:
        X_train_res, y_train_res = X_train, y_train
        if use_smote and SMOTE is None:
            print("imblearn.SMOTE not available; continuing without resampling.")

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        # probability for AUC if available
        try:
            y_prob = model.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            y_prob = None
            auc = None

        acc = accuracy_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        print(f"{name} - Accuracy: {acc:.4f}, AUC: {auc if auc is None else round(auc,4)}")
        print("Classification Report:\n", cr)
        print("Confusion Matrix:\n", cm)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "AUC": auc if auc is not None else np.nan,
            "ModelObj": model
        })
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    return fitted_models, results_df

# -----------------------------
# Main execution
# -----------------------------
def main():
    print("=== Fitbit data mining pipeline ===")

    # 1) Download dataset (best effort)
    dataset_path = None
    path = None
    try:
        dataset_path = download_dataset_kagglehub("arashnic/fitbit")
        if dataset_path:
            print("Downloaded dataset to:", dataset_path)
            csv_path = find_csv_in_dir(dataset_path)
            if csv_path:
                print("Found CSV:", csv_path)
                df = pd.read_csv(csv_path)
            else:
                print("No CSV found in downloaded folder; attempting to load direct file from path.")
                df = smart_load_csv(dataset_path)
        else:
            # try searching current directory for a CSV
            print("No kagglehub download. Searching current directory for CSV files.")
            csvs = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
            if csvs:
                print("Using CSV:", csvs[0])
                df = pd.read_csv(csvs[0])
            else:
                raise FileNotFoundError("No CSV found locally. Please place the Fitbit CSV in the working directory.")
    except Exception as e:
        print("Error while loading dataset:", e)
        print("Attempting to load CSV from current directory as fallback.")
        csvs = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
        if csvs:
            df = pd.read_csv(csvs[0])
        else:
            raise

    # 2) Basic inspection
    print("\nDataset shape:", df.shape)
    print("Columns:", df.columns.tolist()[:50])
    print("\nFirst 5 rows:\n", df.head())

    # 3) create target
    df_with_target, target_col, note = create_binary_target(df)
    print("\nTarget creation note:", note)
    print("Target distribution:\n", df_with_target[target_col].value_counts(dropna=False))

    # 4) Preprocess
    X, y, feature_names = preprocess(df_with_target, target_col)
    print("\nFeatures after preprocessing:", len(feature_names))
    print("Sample features:", feature_names[:20])

    # 5) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print("Train positive ratio:", y_train.mean(), "Test positive ratio:", y_test.mean())

    # 6) Train models & evaluate
    fitted_models, results_df = train_and_evaluate(X_train, y_train, X_test, y_test, use_smote=True if SMOTE is not None else False)

    print("\nModel comparison:\n", results_df[['Model', 'Accuracy', 'AUC']])

    # 7) Save best model
    best_row = results_df.iloc[0]
    best_model_name = best_row['Model']
    best_model_obj = fitted_models[best_model_name]
    model_file = f"best_model_{best_model_name}.joblib"
    joblib.dump(best_model_obj, model_file)
    print(f"\nBest model ({best_model_name}) saved to {model_file}")

    # 8) Feature importance (if tree-based)
    if hasattr(best_model_obj, "feature_importances_"):
        importances = pd.Series(best_model_obj.feature_importances_, index=X.columns)
        top10 = importances.sort_values(ascending=False).head(10)
        print("\nTop 10 feature importances:\n", top10)
        try:
            plt.figure(figsize=(8,5))
            top10.plot(kind='bar')
            plt.title(f"Top 10 feature importances ({best_model_name})")
            plt.tight_layout()
            plt.savefig("feature_importances_top10.png")
            print("Feature importances plot saved to feature_importances_top10.png")
        except Exception as e:
            print("Could not plot feature importances:", e)

    # 9) SHAP explainability (optional)
    if HAS_SHAP and hasattr(best_model_obj, "predict_proba"):
        try:
            print("\nRunning SHAP (this may take time)...")
            explainer = shap.Explainer(best_model_obj, X_train, feature_perturbation="interventional")
            shap_values = explainer(X_test)
            # summary plot to file
            shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig("shap_summary.png", bbox_inches='tight')
            print("SHAP summary saved to shap_summary.png")
        except Exception as e:
            print("SHAP failed or is slow in this environment:", e)
    else:
        if not HAS_SHAP:
            print("\nSHAP not installed; skipping SHAP explainability.")
        else:
            print("\nModel does not support predict_proba or is incompatible for SHAP in this run.")

    # 10) Save predictions and results
    try:
        best_preds = best_model_obj.predict(X_test)
        df_out = X_test.copy()
        df_out['y_true'] = y_test.values
        df_out['y_pred'] = best_preds
        df_out.to_csv("test_predictions_with_features.csv", index=False)
        print("Saved test predictions with features to test_predictions_with_features.csv")
    except Exception as e:
        print("Could not save predictions:", e)

    print("\nPipeline complete. Files produced (if applicable):")
    for f in ["best_model_{}.joblib".format(best_model_name), "feature_importances_top10.png", "shap_summary.png", "test_predictions_with_features.csv"]:
        if os.path.exists(f):
            print(" -", f)

if __name__ == "__main__":
    main()
