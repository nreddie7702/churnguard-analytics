# Churn Prediction Model — Results Summary

## Objective
Predict which customers are likely to churn, using the four features
validated during EDA: **CSAT score, tenure, login volume, and payment
failure rate.**

## Data
- Source: `data/processed/customer_features.csv` (1,198 customers)
- Cleaning applied before modeling:
  - `avg_csat`: 235 missing values (customers with no support tickets) — median-imputed
  - `tenure_days`: 49 rows had slightly negative values (-28 to -31 days, a date-rounding
    artifact for very recent signups) — clipped to 0
- Class balance: 47.2% of customers in the dataset have churned — a well-balanced label

## Models Trained
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 78.3% | 80.8% | 70.8% | 75.5% | 82.3% |
| **Gradient Boosted Trees** | **87.1%** | **94.6%** | 77.0% | **84.9%** | **92.7%** |

**Winner: Gradient Boosted Trees** (chosen for its higher ROC-AUC — a 0.93 score means
the model is very good at ranking customers from most to least likely to churn).

*Note: XGBoost was originally planned but wasn't installable in this environment
(no network access). Scikit-learn's `GradientBoostingClassifier` was used instead —
same underlying approach (boosted decision trees), and it produced strong results.*

### Confusion Matrix (Gradient Boosted Trees, test set of 240 customers)
| | Predicted: Stayed | Predicted: Churned |
|---|---|---|
| **Actual: Stayed** | 122 | 5 |
| **Actual: Churned** | 26 | 87 |

The model is conservative: very few false alarms (5), but it misses about 1 in 4
actual churners (26 of 113). Depending on business priorities, the decision
threshold could be lowered to catch more at-risk customers at the cost of more
false positives.

## What Drives Churn

**Logistic Regression coefficients** (direction and relative strength):
- CSAT score: −1.24 (lower satisfaction → more churn)
- Failure rate: +1.14 (more payment failures → more churn)
- Login volume: −0.84 (less engagement → more churn)
- Tenure: −0.69 (newer customers churn more)

**Gradient Boosting feature importance** (nonlinear model, ranks by predictive power):
1. Tenure (35%)
2. CSAT score (32%)
3. Login volume (21%)
4. Failure rate (12%)

Both models agree on the same four drivers; they differ slightly on ranking because
the boosted-tree model captures nonlinear effects (e.g., tenure matters most at
certain thresholds, not just linearly).

## Customer Risk Scoring
All 1,198 customers were scored with a churn-risk probability
(`scripts/score_customers.py` → `data/processed/customer_churn_risk_scores.csv`),
banded as:
- **High risk:** score ≥ 0.70
- **Medium risk:** 0.40 – 0.69
- **Low risk:** < 0.40

Among **currently active** customers (632 total, not yet churned):
- 592 Low risk, 39 Medium risk, **1 High risk**
- Total active MRR: $178,637.36 — only $56.30 of that sits with the single high-risk
  active account, so near-term revenue exposure from clearly at-risk accounts is small
- The medium-risk tier (39 accounts) is the more actionable group — early enough to
  intervene before scores climb further into high-risk territory

### Top 5 Active Accounts to Watch
| Customer ID | Industry | Plan | MRR | Churn Risk |
|---|---|---|---|---|
| 438 | Retail | Basic | $56.30 | 0.87 (High) |
| 1116 | Edtech | Basic | $63.08 | 0.66 (Medium) |
| 198 | Legal | Basic | $47.16 | 0.64 (Medium) |
| 310 | Edtech | Basic | $55.23 | 0.63 (Medium) |
| 662 | Retail | Pro | $206.05 | 0.63 (Medium) |

## Artifacts
- `scripts/build_churn_model.py` — trains and compares both models
- `scripts/score_customers.py` — scores all customers with risk probabilities
- `scripts/churn_model.joblib` — the trained (winning) model
- `scripts/churn_model_report.json` — full metrics, coefficients, and importances
- `data/processed/customer_churn_risk_scores.csv` — every customer's risk score and band
