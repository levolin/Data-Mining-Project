"""
import_joblib.py
----------------
Loads a trained Logistic Regression model (saved with joblib),
makes predictions on test Fitbit data, and saves the results.
"""

import joblib
import pandas as pd
import os

# === 1. Load the trained model ===
model_path = "best_model_LogisticRegression.joblib"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"✅ Loaded model from: {model_path}")
model = joblib.load(model_path)
print(model)
print(model.get_params())
print("\n")

# === 2. Load the test dataset ===
csv_path = "test_predictions_with_features.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Test data not found: {csv_path}")

X_new = pd.read_csv(csv_path)
print(f"✅ Loaded test data: {csv_path}")
print(f"Data shape before cleanup: {X_new.shape}")

# === 3. Drop non-feature columns ===
cols_to_drop = ["Target", "Prediction", "y_true", "y_pred"]
X_new_clean = X_new.drop(columns=cols_to_drop, errors="ignore")

print(f"Data shape after cleanup: {X_new_clean.shape}")
print(f"Remaining columns: {list(X_new_clean.columns)}")
print("\n")

# === 4. Make predictions ===
predictions = model.predict(X_new_clean)
probabilities = model.predict_proba(X_new_clean)[:, 1]  # Probability of class = 1

# === 5. Append predictions to the DataFrame ===
X_new["Predicted_Label"] = predictions
X_new["Predicted_Prob_LowActivity"] = probabilities

# === 6. Save the updated file ===
output_path = "test_predictions_with_model_output.csv"
X_new.to_csv(output_path, index=False)

print("✅ Predictions complete!")
print(f"First 10 predicted labels: {predictions[:10]}")
print(f"Updated file saved to: {output_path}")
