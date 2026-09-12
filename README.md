# ChurnGuard Analytics

**An end-to-end SaaS revenue leakage & churn prevention analytics platform** — from synthetic data generation through SQL analysis, machine learning, GenAI-powered querying, and an executive-ready Power BI dashboard.

Built as a portfolio project to demonstrate the full data analyst workflow on a realistic SaaS dataset: 1,200 customers, 2 years of subscription/invoice/usage/support activity, and a central business question — **why are we losing revenue, and which customers are at risk?**

---

## Key Findings

| Metric | Value |
|---|---|
| Total Active MRR | $320.41K |
| Active Customers | 632 |
| Logo Churn Rate | 47% |
| Churned Customers | 566 |
| Average Revenue Per Customer (ARPU) | $291 |
| Churn Model Accuracy (ROC-AUC) | 0.958 |
| Customers Flagged High Churn Risk | 127 (20%) |
| Open Support Tickets | 294 |
| Failed Payments | 526 ($109.64K) |

**Headline insight:** Enterprise-tier customers drive the most revenue *and* are the most loyal (lowest churn risk, ~0.09 average), while Basic-tier customers are both the least profitable *and* the most likely to churn (~0.29 average risk) — a classic SaaS pattern that directly informs where retention effort should be focused.

---

## What's Inside

- **Synthetic data generation** — 1,200 SaaS customers, 2 years of activity, calibrated to realistic SaaS benchmarks (not random noise)
- **Data cleaning** — deduplication, inconsistent formatting fixes, referential integrity checks
- **25 SQL business questions** — window functions, CTEs, cohort analysis
- **EDA & feature engineering** — statistical testing, correlation analysis
- **Machine learning** — Random Forest churn classifier, ROC-AUC 0.958
- **GenAI layer (Claude-powered)**:
  - **AI Analyst** — natural language Q&A grounded in real SQL query results (never hallucinates numbers)
  - **AI Agent** — autonomous, multi-step investigation using both SQL and a sandboxed Python analysis tool
- **Power BI dashboard** — 5 interactive pages: Executive Overview, Business Performance, Customer Analysis, AI Predictions, Anomaly Alerts

Full technical detail is in [`docs/TECH_STACK.md`](docs/TECH_STACK.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Dashboard Pages

1. **Executive Overview** — top-line KPIs and a correctly-calculated point-in-time MRR growth trend
2. **Business Performance** — revenue by plan tier and country, ARPU, new vs. churned MRR over time
3. **Customer Analysis** — segmentation by industry, company size, risk tier, acquisition channel
4. **AI Predictions** — churn model output: risk distribution, high-risk customer list, risk by plan tier
5. **Anomaly Alerts** — open support tickets, failed payments, unresolved issues

---

## Repository Structure

```
churnguard-analytics/
├── Scripts/           # Data generation & cleaning
│   ├── generate_data.py
│   └── clean_data.py
├── Data/               # SQLite database
│   └── churnguard.db
├── SQL/                # Business question queries
│   └── business_questions.sql
├── notebooks/          # EDA and ML notebooks
│   ├── 01_eda_analysis.ipynb
│   └── 02_churn_model.ipynb
├── ML/                 # Saved model artifacts
│   ├── churn_model.pkl
│   ├── model_features.pkl
│   └── scaler.pkl
├── AI/                 # GenAI layer (Claude-powered)
│   ├── ai_analyst.py
│   └── ai_agent.py
├── Dashboard/           # Power BI dashboard
│   └── ChurnGuard_Analytics_Dashboard.pbix
└── docs/                # Write-ups and chart exports
    ├── TECH_STACK.md
    ├── ARCHITECTURE.md
    └── *.png / *.md      (EDA & model result charts)
```

---

## Tech Stack

Python (pandas) · SQLite · SQL · Jupyter · scikit-learn · Anthropic Claude API (tool-use) · Power BI · DAX

See [`docs/TECH_STACK.md`](docs/TECH_STACK.md) for the full breakdown by layer.

---

## Running This Project

**Requirements:** Python 3.9+, `pandas`, `scikit-learn`, `anthropic`, Power BI Desktop (Windows)

1. Clone the repo:
   ```
   git clone https://github.com/nreddie7702/churnguard-analytics.git
   ```
2. Generate and clean the data:
   ```
   python Scripts/generate_data.py
   python Scripts/clean_data.py
   ```
3. Explore the SQL questions in `SQL/business_questions.sql` against `Data/churnguard.db`
4. Run the EDA and ML notebooks in `notebooks/` to reproduce the churn model
5. To run the AI Analyst or AI Agent, set an `ANTHROPIC_API_KEY` environment variable, then:
   ```
   python AI/ai_analyst.py "which customers are at the highest risk of churning?"
   python AI/ai_agent.py "find our biggest churn risk segment and recommend an action"
   ```
6. Open `Dashboard/ChurnGuard_Analytics_Dashboard.pbix` in Power BI Desktop to explore the dashboard

---

## Skills Demonstrated

Data generation & cleaning · Relational database design (SQLite) · Advanced SQL (window functions, CTEs, cohort analysis) · Exploratory data analysis & statistical testing · Feature engineering · Machine learning (classification, model evaluation) · LLM tool-use / function-calling · AI safety-conscious system design (sandboxed code execution, grounded outputs) · Business intelligence (Power BI, DAX, time intelligence) · Data storytelling

---

## Author

**Narasimha Bolla**
[GitHub](https://github.com/nreddie7702)
