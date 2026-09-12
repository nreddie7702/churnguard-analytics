"""
ChurnGuard Analytics — Step 8: ML Churn Prediction Model
Uses the 4 features validated in the EDA notebook:
  - avg_csat       (customer satisfaction score)
  - tenure_days    (how long they've been a customer)
  - avg_logins     (login volume / engagement)
  - failure_rate   (payment failure rate)

Trains Logistic Regression (baseline, interpretable) and XGBoost (stronger, nonlinear),
compares them, and saves the better model + a feature importance / coefficient report.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import GradientBoostingClassifier

DATA_PATH = "data/processed/customer_features.csv"
FEATURES = ["avg_csat", "tenure_days", "avg_logins", "failure_rate"]
TARGET = "churned"
RANDOM_STATE = 42

print("=" * 70)
print("ChurnGuard — Churn Prediction Model")
print("=" * 70)

# ---------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"\nLoaded {len(df)} rows.")

n_missing_csat = df["avg_csat"].isna().sum()
n_neg_tenure = (df["tenure_days"] < 0).sum()
print(f"  - avg_csat missing: {n_missing_csat} -> median-imputed")
print(f"  - tenure_days negative: {n_neg_tenure} -> clipped to 0")

df["avg_csat"] = df["avg_csat"].fillna(df["avg_csat"].median())
df["tenure_days"] = df["tenure_days"].clip(lower=0)

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"\nClass balance: churned=1 -> {y.mean():.1%} of customers")

# ---------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

# ---------------------------------------------------------------
# 3. Model A — Logistic Regression (scaled, interpretable baseline)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
lr_pred = log_reg.predict(X_test_scaled)
lr_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------
# 4. Model B — Gradient Boosted Trees (nonlinear, usually stronger)
#    Note: xgboost isn't installable in this sandbox (no network access),
#    so sklearn's GradientBoostingClassifier is used instead — same role
#    (boosted decision trees), no external package required.
# ---------------------------------------------------------------
xgb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    random_state=RANDOM_STATE,
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]


def evaluate(name, y_true, y_pred, y_proba):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


results = [
    evaluate("Logistic Regression", y_test, lr_pred, lr_proba),
    evaluate("GradientBoosting", y_test, xgb_pred, xgb_proba),
]

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
results_df = pd.DataFrame(results).set_index("model")
print(results_df.to_string())

# ---------------------------------------------------------------
# 5. Pick the winner by ROC-AUC, save it + supporting artifacts
# ---------------------------------------------------------------
best_row = results_df["roc_auc"].idxmax()
best_model = xgb if best_row == "GradientBoosting" else log_reg
best_pred = xgb_pred if best_row == "GradientBoosting" else lr_pred
best_proba = xgb_proba if best_row == "GradientBoosting" else lr_proba

print(f"\nBest model by ROC-AUC: {best_row}")
print("\nConfusion matrix (best model):")
cm = confusion_matrix(y_test, best_pred)
print(pd.DataFrame(
    cm,
    index=["Actual: Stayed", "Actual: Churned"],
    columns=["Pred: Stayed", "Pred: Churned"],
))

print("\nClassification report (best model):")
print(classification_report(y_test, best_pred, target_names=["Stayed", "Churned"]))


lr_coefs = pd.Series(log_reg.coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
xgb_importance = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)

print("\nLogistic Regression standardized coefficients (direction + strength):")
print(lr_coefs.to_string())

print("\nXGBoost feature importances:")
print(xgb_importance.to_string())

# Save artifacts
joblib.dump(best_model, "scripts/churn_model.joblib")
joblib.dump(scaler, "scripts/feature_scaler.joblib")  # only used if best model is logistic regression

report = {
    "features_used": FEATURES,
    "data_cleaning": {
        "avg_csat_missing_imputed": int(n_missing_csat),
        "tenure_days_negative_clipped": int(n_neg_tenure),
    },
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "class_balance_churn_rate": round(float(y.mean()), 4),
    "model_comparison": results,
    "best_model": best_row,
    "logistic_regression_coefficients": lr_coefs.round(4).to_dict(),
    "xgboost_feature_importance": xgb_importance.round(4).to_dict(),
}
with open("scripts/churn_model_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nSaved: scripts/churn_model.joblib, scripts/feature_scaler.joblib, scripts/churn_model_report.json")
print("\nDone.")
