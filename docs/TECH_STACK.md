# ChurnGuard Analytics — Tech Stack & Architecture

## Overview

ChurnGuard Analytics is an end-to-end SaaS revenue-leakage and churn-prevention analytics platform, built to demonstrate the full data analyst workflow: from raw data generation through cleaning, exploratory analysis, machine learning, GenAI-powered querying, and executive-level dashboarding.

The project simulates a realistic SaaS company with 1,200 customers across 2 years of subscription, invoice, usage, and support activity, and analyzes that data to answer a central business question: **why are we losing revenue, and which customers are at risk?**

---

## Architecture

```
┌─────────────────────┐
│  Data Generation     │  Python (Faker) → synthetic SaaS dataset
│  scripts/generate_data.py
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  Data Cleaning        │  Python (pandas) → dedup, standardize,
│  scripts/clean_data.py│  referential integrity fixes
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  SQLite Database      │  5 tables: customers, subscriptions,
│  data/churnguard.db   │  invoices, usage_events, support_tickets
└──────────┬───────────┘
           │
     ┌─────┴─────┬─────────────┬──────────────┐
     │            │             │              │
┌────▼────┐ ┌────▼─────┐ ┌─────▼──────┐ ┌─────▼──────┐
│ SQL      │ │ Python    │ │ ML Model   │ │ GenAI Layer│
│ Business │ │ EDA       │ │ (churn     │ │ (AI Analyst│
│ Questions│ │ Notebook  │ │ prediction)│ │ + AI Agent)│
└────┬────┘ └────┬─────┘ └─────┬──────┘ └─────┬──────┘
     │            │             │              │
     └────────────┴─────────────┴──────────────┘
                       │
              ┌────────▼─────────┐
              │  Power BI         │  5-page interactive
              │  Dashboard        │  executive dashboard
              └───────────────────┘
```

---

## Tech Stack by Layer

### 1. Data Generation & Cleaning
- **Python** — core scripting language for the entire pipeline
- **Faker / custom generation logic** — synthetic dataset of 1,200 SaaS customers, calibrated to real-world SaaS benchmarks (churn rates, MRR distribution, plan tiers)
- **pandas** — data cleaning: deduplication, standardizing inconsistent country names, correcting invalid MRR values, cascading deletes to preserve referential integrity

### 2. Data Storage
- **SQLite** — lightweight relational database (`data/churnguard.db`)
- 5 core tables: `customers`, `subscriptions`, `invoices`, `usage_events`, `support_tickets`
- Verified zero orphaned rows across all foreign key relationships

### 3. SQL Analysis
- **25 business questions** written in SQL, covering:
  - Window functions
  - CTEs (Common Table Expressions)
  - Cohort analysis
- Located in `sql/business_questions.sql`

### 4. Exploratory Data Analysis & Machine Learning
- **Jupyter Notebook** (`notebooks/01_eda_analysis.ipynb`) — statistical testing, correlation analysis, feature engineering
- **scikit-learn** — Random Forest classifier for churn prediction
  - **ROC-AUC: 0.958**
  - Exports customer-level risk scores to `churn_risk_scores.csv`, later joined into the Power BI model

### 5. GenAI & AI Agent Layer
- **Anthropic Claude (via API, model: claude-sonnet-4-6)** — powers two AI-driven components using Claude's native tool-use (function-calling) capability:
  - **AI Analyst** (`ai/ai_analyst.py`) — a SQL-grounded natural language Q&A tool. The model is given the database schema and a single `run_sql_query` tool, and is instructed via its system prompt that it must call that tool before stating any number, percentage, or ranking — it is never allowed to answer with a figure it didn't actually query. The tool itself only permits read-only `SELECT`/`WITH` statements, so the model cannot modify the database. This "plan → query → explain" loop structurally prevents hallucinated statistics: the Python code (not the model) executes every query and feeds the real result back before the model writes its final explanation.
  - **AI Agent** (`ai/ai_agent.py`) — a genuinely autonomous, multi-step agent built on the same Claude tool-use pattern, given two tools instead of one:
    - `run_sql_query` — the same read-only SQL tool as the AI Analyst
    - `run_python_analysis` — a sandboxed pandas/numpy execution tool that runs agent-written code against two pre-loaded, pre-engineered DataFrames (customer features and churn risk scores), with explicit guardrails blocking file access, imports, `exec`/`eval`, `subprocess`, and dunder access
    
    Unlike the AI Analyst (which always does one query → one answer), the Agent decides for itself how many steps it needs (up to 8), and the system prompt explicitly instructs it to investigate *why* a pattern exists — not just report a number — before making a recommendation. For example, given the goal "find our biggest churn risk segment and recommend an action," the agent might first query the risk score table to identify the highest-risk segment, then run a Python analysis to understand what's driving that risk, and only then produce a grounded recommendation referencing both results.

### 6. Business Intelligence & Visualization
- **Power BI Desktop** — 5-page interactive dashboard:
  1. **Executive Overview** — top-line KPIs (MRR, active customers, churn rate) and MRR growth trend
  2. **Business Performance** — revenue breakdowns by plan tier and country, ARPU, new vs. churned MRR
  3. **Customer Analysis** — segmentation by industry, company size, risk tier, and acquisition channel
  4. **AI Predictions** — churn model outputs: risk distribution, high-risk customer list, risk by plan tier
  5. **Anomaly Alerts** — operational risk signals: open support tickets, failed payments, unresolved issues
- **DAX (Data Analysis Expressions)** — custom measures built throughout, including:
  - A dedicated **Date table** with time-intelligence support, enabling accurate point-in-time calculations
  - A **point-in-time MRR measure** (avoids the common mistake of grouping revenue by subscription start date instead of calculating true active MRR per month)
  - Derived measures: ARPU, Logo Churn Rate, New MRR, Churned MRR, Open Tickets, Failed Payments

---

## Key Design Decisions

- **SQLite over a hosted database**: kept the project fully self-contained and portable for a portfolio piece, while still using real SQL (not a toy dataset format like flat CSVs alone).
- **Point-in-time MRR calculation**: an early version of the dashboard incorrectly calculated MRR trends by grouping on subscription start date, which understated true revenue. This was identified and corrected using a proper Date dimension table and time-intelligence DAX — a detail worth highlighting in interviews as an example of catching and fixing a real analytical error.
- **Separation of ML and dashboard layers**: the churn model runs independently in Python/scikit-learn and exports its output as a CSV, which is then imported into Power BI — mirroring how many real organizations operationalize ML outputs into BI tools without needing live model inference inside the dashboard itself.
