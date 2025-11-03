# src/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
from pathlib import Path

from dataload import load_stock_data          # ✅ import your data loader
from featureengineering import prepare_features  # ✅ import your feature function

# ==============================
# 1️⃣ Load and Prepare Data
# ==============================

# Load raw stock data
df = load_stock_data(ticker="AAPL", start_date="2024-01-01", end_date="2025-01-01")

# Generate feature set (X) and target (y)
X, y = prepare_features(df)

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)


# ==============================
# 2️⃣ Train Model (XGBoost)
# ==============================

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 3️⃣ Evaluate Model
# ==============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# 4️⃣ Save Model
# ==============================

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/xgboost_model.pkl")

print("💾 Model saved successfully to models/xgboost_model.pkl")
