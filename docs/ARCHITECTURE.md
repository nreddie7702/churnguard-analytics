# ChurnGuard Analytics — Architecture

This document describes the data model, pipeline stages, and key design decisions behind ChurnGuard Analytics in more depth than the top-level tech stack summary.

---

## 1. Data Model

The database (`data/churnguard.db`, SQLite) has 5 tables:

```
customers
  customer_id (PK)
  company_name
  industry
  company_size
  country
  signup_date
  acquisition_channel
  cac

subscriptions
  subscription_id (PK)
  customer_id (FK -> customers)
  plan_tier
  mrr_value
  start_date
  end_date
  status              -- 'active' / 'canceled' / 'upgraded' / 'downgraded'
  billing_cycle

invoices
  invoice_id (PK)
  subscription_id (FK -> subscriptions)
  invoice_date
  amount
  payment_status      -- 'paid' / 'failed'
  payment_method
  retry_count

usage_events
  event_id (PK)
  customer_id (FK -> customers)
  event_date
  feature_used
  login_count
  active_users

support_tickets
  ticket_id (PK)
  customer_id (FK -> customers)
  created_date
  category
  priority
  resolved_date
  csat_score
  is_resolved
```

**Relationships:**
- One customer → many subscriptions (a customer can upgrade/downgrade, creating a new subscription record)
- One subscription → many invoices (one invoice per billing cycle)
- One customer → many usage events and many support tickets

This is a fairly standard SaaS relational model: a customer entity, a subscription entity that tracks plan/revenue over time, and two behavioral tables (usage, support) that feed the churn model's feature set.

---

## 2. Pipeline Stages

### Stage 1 — Synthetic Data Generation
`scripts/generate_data.py` generates 1,200 customers and 2 years of related activity, calibrated to realistic SaaS benchmarks rather than pure randomness — e.g., churn rates, plan-tier distribution, and MRR ranges were chosen to resemble what a real SaaS company's data would look like, rather than uniform random noise. This matters for the ML model: a churn model trained on unrealistic synthetic data would learn spurious patterns.

### Stage 2 — Data Cleaning
`scripts/clean_data.py` intentionally introduces and then fixes realistic data-quality issues:
- Duplicate customer records
- Inconsistent country name formatting (e.g., "USA" vs "United States" vs "U.S.")
- Invalid/out-of-range MRR values
- Cascading deletes to prevent orphaned rows when a customer record is removed

This stage exists specifically to demonstrate data-cleaning skill — a real interview talking point, since "the data was already clean" is rarely true in practice.

### Stage 3 — SQL Analysis
25 business questions (`sql/business_questions.sql`) using window functions, CTEs, and cohort analysis. These form the analytical foundation that the later GenAI layer also draws on (the AI Analyst effectively writes SQL in the same spirit as these hand-written queries, but generated dynamically per question).

### Stage 4 — EDA & Feature Engineering
`notebooks/01_eda_analysis.ipynb` performs statistical testing and correlation analysis, and engineers the feature set later used by the ML model — things like tenure, average logins, usage trend percentage, support ticket volume, and average CSAT score per customer.

### Stage 5 — Machine Learning
`notebooks/02_churn_model.ipynb` trains a Random Forest classifier on the engineered features (ROC-AUC 0.958) and exports two artifacts:
- `churn_risk_scores.csv` — per-customer churn probability and risk tier, imported into Power BI
- `customer_features.csv` — the full engineered feature table, later reused as one of the two DataFrames available to the AI Agent's Python tool

This is a deliberate architectural choice: **the ML layer and the GenAI layer share the same underlying feature data**, so the AI Agent's Python-based investigations ("why is this segment risky?") are grounded in the same features the model was trained on, not a separate ad hoc dataset.

### Stage 6 — GenAI Layer
Two Claude-powered tools, both grounded in real data rather than free-form generation (see `TECH_STACK.md` for full detail):
- **AI Analyst** — single-tool, single-query-per-question Q&A
- **AI Agent** — two-tool (SQL + sandboxed Python), multi-step autonomous investigation

### Stage 7 — Power BI Dashboard
A 5-page dashboard consuming the cleaned CSVs and the ML model's risk scores, described fully in `TECH_STACK.md`.

---

## 3. Key Design Decisions

### Point-in-time MRR, not start-date MRR
An early version of the Executive Overview trend chart grouped MRR by subscription `start_date`, which only shows revenue from customers who *started* in a given month — not total revenue active in that month. This understates true MRR and produces a misleadingly flat/choppy trend line.

**Fix:** a dedicated `DateTable` (built with `CALENDAR()` and marked as an official date table) plus a DAX measure that filters subscriptions to those where `start_date <= [current date] AND (end_date >= [current date] OR end_date is blank)`. This calculates true active MRR as of any given month — the correct definition of the metric — and is used throughout the dashboard (Executive Overview trend, New vs. Churned MRR, Business Performance charts).

### Structural grounding for the GenAI layer
Both AI components are designed so the model **cannot** state a number it didn't retrieve from a real tool call — this is enforced by system prompt instruction *and* by the fact that only the Python backend (not the model) ever executes a query or runs code. The model proposes what to run; the application decides whether and how to run it. This separation is what prevents hallucinated statistics from reaching the end user, and is a meaningfully different (and safer) design than simply asking an LLM to "answer using this data" in a single prompt.

### Sandboxed code execution for the AI Agent
The Agent's Python tool doesn't have unrestricted `exec()` access — it operates on a fixed, pre-loaded pair of DataFrames, with explicit blocklisting of file I/O, imports, `eval`/`exec`, `subprocess`, and dunder attribute access. This reflects a real security consideration that comes up whenever an LLM is given code-execution capability, and is worth speaking to directly in interviews as an example of thinking about AI safety in a practical, applied way. The relevant question isn't "did the AI get it right," but "was the system designed so that even if the AI got it wrong, it couldn't do damage."

### ML and BI decoupling
The churn model runs offline in a notebook and exports a CSV rather than being called live from Power BI. This mirrors how many real organizations operationalize ML: batch-scored predictions refreshed periodically, rather than a live model endpoint embedded in the BI tool. It's simpler to build, easier to explain, and avoids the added complexity/fragility of live inference inside a dashboard for a portfolio-scale project.
