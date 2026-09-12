"""
ChurnGuard Analytics — Score all customers with churn-risk probabilities
using the model trained in build_churn_model.py.
"""

import joblib
import pandas as pd

DATA_PATH = "data/processed/customer_features.csv"
MODEL_PATH = "scripts/churn_model.joblib"
FEATURES = ["avg_csat", "tenure_days", "avg_logins", "failure_rate"]
OUT_PATH = "data/processed/customer_churn_risk_scores.csv"

df = pd.read_csv(DATA_PATH)

# Same cleaning as training
df["avg_csat"] = df["avg_csat"].fillna(df["avg_csat"].median())
df["tenure_days"] = df["tenure_days"].clip(lower=0)

model = joblib.load(MODEL_PATH)
# GradientBoostingClassifier (chosen model) takes raw features directly, no scaler needed
X = df[FEATURES]
df["churn_risk_score"] = model.predict_proba(X)[:, 1].round(4)


def risk_band(p):
    if p >= 0.7:
        return "High"
    elif p >= 0.4:
        return "Medium"
    return "Low"


df["risk_band"] = df["churn_risk_score"].apply(risk_band)

cols_out = ["customer_id", "company_size", "industry", "plan_tier", "mrr_value",
            "avg_csat", "tenure_days", "avg_logins", "failure_rate",
            "churned", "churn_risk_score", "risk_band"]
df[cols_out].sort_values("churn_risk_score", ascending=False).to_csv(OUT_PATH, index=False)

print(f"Scored {len(df)} customers -> {OUT_PATH}")
print("\nRisk band breakdown:")
print(df["risk_band"].value_counts())
print("\nTop 10 highest-risk currently-active (not-yet-churned) customers:")
active_at_risk = df[df["churned"] == 0].sort_values("churn_risk_score", ascending=False).head(10)
print(active_at_risk[["customer_id", "industry", "plan_tier", "mrr_value", "churn_risk_score"]].to_string(index=False))
