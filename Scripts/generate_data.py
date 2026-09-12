"""
ChurnGuard Analytics — Synthetic SaaS Dataset Generator
=========================================================
Generates a realistic (deliberately messy) subscription-billing dataset
for a fictional B2B SaaS company, calibrated to industry churn/NRR
benchmarks (ChartMogul / ProfitWell / OpenView SaaS benchmark reports):
    - Monthly logo churn: ~3-7%
    - NRR: ~95-115%
    - Involuntary churn share: ~20-40% of total churn

Output: 5 raw CSVs in data/raw/, WITH intentional data quality issues
(duplicates, inconsistent country names, negative/zero MRR, orphaned
usage rows, missing resolved_date) — these get fixed in the cleaning
stage (Step 4 of the pipeline), not here.

Run:
    python3 scripts/generate_data.py
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import date, timedelta
import random

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

NUM_CUSTOMERS = 1200
SIM_START = date(2024, 9, 1)
SIM_END = date(2026, 8, 31)          # "today" reference for the sim
TOTAL_MONTHS = (SIM_END.year - SIM_START.year) * 12 + (SIM_END.month - SIM_START.month)

INDUSTRIES = ["Retail", "Healthcare", "Fintech", "Edtech", "Logistics",
              "Manufacturing", "Media", "Real Estate", "Legal", "Hospitality"]
COMPANY_SIZES = ["Small", "Mid-Market", "Enterprise"]
COMPANY_SIZE_WEIGHTS = [0.55, 0.32, 0.13]
CHANNELS = ["Organic", "Paid Search", "Referral", "Partner", "Outbound Sales"]
CHANNEL_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

PLAN_TIERS = ["Basic", "Pro", "Enterprise"]
PLAN_WEIGHTS = [0.50, 0.35, 0.15]
PLAN_BASE_MRR = {"Basic": 49, "Pro": 199, "Enterprise": 899}

# Countries — deliberately inconsistent naming for realism (cleaned later)
COUNTRY_VARIANTS = {
    "United States": ["United States", "USA", "US", "united states"],
    "India": ["India", "IN"],
    "United Kingdom": ["United Kingdom", "UK", "U.K."],
    "Germany": ["Germany", "DE"],
    "Australia": ["Australia", "AU"],
    "Canada": ["Canada", "CA"],
}
COUNTRY_CHOICES = list(COUNTRY_VARIANTS.keys())
COUNTRY_WEIGHTS = [0.40, 0.25, 0.12, 0.10, 0.08, 0.05]

TICKET_CATEGORIES = ["Billing", "Bug", "Feature Request", "Onboarding", "General"]
TICKET_PRIORITIES = ["Low", "Medium", "High", "Critical"]

def month_add(d: date, n: int) -> date:
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    day = min(d.day, 28)
    return date(year, month, day)

def months_between(d1: date, d2: date) -> int:
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)

# ------------------------------------------------------------------
# 1. CUSTOMERS
# ------------------------------------------------------------------
customers = []
for cid in range(1, NUM_CUSTOMERS + 1):
    # bias signups toward the earlier ~20 months so accounts have runway
    signup_offset = int(np.random.triangular(0, TOTAL_MONTHS * 0.3, TOTAL_MONTHS - 1))
    signup_date = month_add(SIM_START, signup_offset)

    country_key = np.random.choice(COUNTRY_CHOICES, p=COUNTRY_WEIGHTS)
    country_raw = random.choice(COUNTRY_VARIANTS[country_key])  # messy on purpose

    channel = np.random.choice(CHANNELS, p=CHANNEL_WEIGHTS)
    cac_base = {"Organic": 50, "Paid Search": 450, "Referral": 120,
                "Partner": 300, "Outbound Sales": 900}[channel]
    cac = round(max(10, np.random.normal(cac_base, cac_base * 0.25)), 2)

    customers.append({
        "customer_id": cid,
        "company_name": fake.company(),
        "industry": random.choice(INDUSTRIES),
        "company_size": np.random.choice(COMPANY_SIZES, p=COMPANY_SIZE_WEIGHTS),
        "country": country_raw,
        "signup_date": signup_date,
        "acquisition_channel": channel,
        "cac": cac,
    })

customers_df = pd.DataFrame(customers)

# inject ~1.5% duplicate customer rows (same company, new id) — data quality issue
dupe_sample = customers_df.sample(frac=0.015, random_state=SEED).copy()
dupe_sample["customer_id"] = range(NUM_CUSTOMERS + 1, NUM_CUSTOMERS + 1 + len(dupe_sample))
customers_df = pd.concat([customers_df, dupe_sample], ignore_index=True)

# ------------------------------------------------------------------
# 2. CHURN SIMULATION ENGINE (drives subscriptions/invoices/usage/tickets)
# ------------------------------------------------------------------
# Each customer gets a "risk profile" combining plan tier, channel quality,
# and a random latent health trait — this is what later ML models will learn.
subscriptions, invoices, usage_events, support_tickets = [], [], [], []

sub_id_counter = 1
invoice_id_counter = 1
event_id_counter = 1
ticket_id_counter = 1

# baseline monthly hazard by plan tier (higher tiers churn less)
BASE_HAZARD = {"Basic": 0.055, "Pro": 0.035, "Enterprise": 0.018}

for _, cust in customers_df.iterrows():
    cid = cust["customer_id"]
    signup = cust["signup_date"]
    tenure_cap = months_between(signup, SIM_END)
    if tenure_cap < 1:
        continue

    plan = np.random.choice(PLAN_TIERS, p=PLAN_WEIGHTS)
    mrr = round(PLAN_BASE_MRR[plan] * np.random.uniform(0.85, 1.3), 2)
    latent_health = np.random.beta(2, 2)   # 0 = fragile, 1 = healthy account
    hazard = BASE_HAZARD[plan] * (1.6 - latent_health)  # unhealthy accounts churn more

    current_sub_start = signup
    current_plan, current_mrr = plan, mrr
    churn_month_idx = None
    churn_type = None

    month_idx = 0
    active = True
    while active and month_idx < tenure_cap:
        month_idx += 1
        # random small chance of upgrade/downgrade (expansion/contraction)
        if month_idx > 2 and random.random() < 0.02:
            new_plan = np.random.choice(PLAN_TIERS, p=PLAN_WEIGHTS)
            if new_plan != current_plan:
                # close old subscription record
                sub_end = month_add(current_sub_start, month_idx - 1)
                subscriptions.append({
                    "subscription_id": sub_id_counter, "customer_id": cid,
                    "plan_tier": current_plan, "mrr_value": current_mrr,
                    "start_date": current_sub_start, "end_date": sub_end,
                    "status": "upgraded" if PLAN_BASE_MRR[new_plan] > PLAN_BASE_MRR[current_plan] else "downgraded",
                    "billing_cycle": "monthly",
                })
                sub_id_counter += 1
                current_sub_start = month_add(current_sub_start, month_idx)
                current_plan = new_plan
                current_mrr = round(PLAN_BASE_MRR[new_plan] * np.random.uniform(0.85, 1.3), 2)
                month_idx = 0  # reset local counter for new sub record
                hazard = BASE_HAZARD[current_plan] * (1.6 - latent_health)

        # churn check
        if random.random() < hazard:
            active = False
            churn_month_idx = month_idx
            churn_type = "involuntary" if random.random() < 0.30 else "voluntary"

    # close final subscription record
    if churn_month_idx is not None:
        sub_end = month_add(current_sub_start, churn_month_idx)
        status = "canceled"
    else:
        sub_end = None
        status = "active"

    final_mrr = current_mrr
    # occasionally corrupt mrr_value to 0 or negative (data quality issue, ~0.8%)
    if random.random() < 0.008:
        final_mrr = round(random.choice([0, -final_mrr]), 2)

    subscriptions.append({
        "subscription_id": sub_id_counter, "customer_id": cid,
        "plan_tier": current_plan, "mrr_value": final_mrr,
        "start_date": current_sub_start, "end_date": sub_end,
        "status": status, "billing_cycle": "monthly",
    })
    sub_id_counter += 1

    # ---------------- INVOICES ----------------
    active_months = churn_month_idx if churn_month_idx else tenure_cap
    for m in range(active_months):
        inv_date = month_add(current_sub_start, m)
        # elevate failure probability in the 2 months before involuntary churn
        base_fail_p = 0.03
        if churn_type == "involuntary" and churn_month_idx and m >= churn_month_idx - 2:
            fail_p = 0.55
        else:
            fail_p = base_fail_p
        payment_status = "failed" if random.random() < fail_p else "paid"
        retry_count = np.random.poisson(1.5) if payment_status == "failed" else 0
        invoices.append({
            "invoice_id": invoice_id_counter, "subscription_id": sub_id_counter - 1,
            "invoice_date": inv_date, "amount": abs(final_mrr),
            "payment_status": payment_status,
            "payment_method": np.random.choice(["card", "bank_transfer", "upi"], p=[0.7, 0.2, 0.1]),
            "retry_count": retry_count,
        })
        invoice_id_counter += 1

    # ---------------- USAGE EVENTS (monthly aggregate per customer) --------
    base_logins = {"Basic": 8, "Pro": 20, "Enterprise": 45}[current_plan]
    for m in range(active_months):
        decline_factor = 1.0
        if churn_month_idx and m >= churn_month_idx - 3:
            # engagement decays in the 3 months leading up to churn
            decline_factor = max(0.15, 1 - 0.28 * (m - (churn_month_idx - 3) + 1))
        logins = max(0, int(np.random.normal(base_logins * decline_factor * latent_health * 1.3, 3)))
        active_users = max(1, int(logins / np.random.uniform(3, 6)))
        usage_events.append({
            "event_id": event_id_counter, "customer_id": cid,
            "event_date": month_add(current_sub_start, m),
            "feature_used": random.choice(["Dashboard", "Reports", "API", "Integrations", "Automation"]),
            "login_count": logins, "active_users": active_users,
        })
        event_id_counter += 1

    # ---------------- SUPPORT TICKETS ----------------
    for m in range(active_months):
        elevated = churn_type == "voluntary" and churn_month_idx and m >= churn_month_idx - 2
        ticket_rate = 0.55 if elevated else 0.18
        if random.random() < ticket_rate:
            created = month_add(current_sub_start, m)
            priority = np.random.choice(
                TICKET_PRIORITIES,
                p=[0.15, 0.35, 0.35, 0.15] if elevated else [0.45, 0.35, 0.15, 0.05]
            )
            resolved = random.random() > 0.12  # ~12% left unresolved
            support_tickets.append({
                "ticket_id": ticket_id_counter, "customer_id": cid,
                "created_date": created,
                "category": random.choice(TICKET_CATEGORIES),
                "priority": priority,
                "resolved_date": (created + timedelta(days=random.randint(1, 10))) if resolved else None,
                "csat_score": (np.random.randint(1, 3) if elevated else np.random.randint(3, 6)) if resolved else None,
            })
            ticket_id_counter += 1

subscriptions_df = pd.DataFrame(subscriptions)
invoices_df = pd.DataFrame(invoices)
usage_df = pd.DataFrame(usage_events)
tickets_df = pd.DataFrame(support_tickets)

# inject a handful of orphaned usage events (customer_id not in customers table)
orphan_rows = usage_df.sample(n=15, random_state=SEED).copy()
orphan_rows["customer_id"] = range(90001, 90001 + len(orphan_rows))
usage_df = pd.concat([usage_df, orphan_rows], ignore_index=True)

# ------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------
out_dir = "/home/claude/churnguard-analytics/data/raw"
customers_df.to_csv(f"{out_dir}/customers.csv", index=False)
subscriptions_df.to_csv(f"{out_dir}/subscriptions.csv", index=False)
invoices_df.to_csv(f"{out_dir}/invoices.csv", index=False)
usage_df.to_csv(f"{out_dir}/usage_events.csv", index=False)
tickets_df.to_csv(f"{out_dir}/support_tickets.csv", index=False)

# ------------------------------------------------------------------
# QUICK SANITY REPORT
# ------------------------------------------------------------------
print("=" * 55)
print("CHURNGUARD SYNTHETIC DATA — GENERATION SUMMARY")
print("=" * 55)
print(f"customers.csv        : {len(customers_df):,} rows")
print(f"subscriptions.csv    : {len(subscriptions_df):,} rows")
print(f"invoices.csv         : {len(invoices_df):,} rows")
print(f"usage_events.csv     : {len(usage_df):,} rows")
print(f"support_tickets.csv  : {len(tickets_df):,} rows")

final_subs = subscriptions_df[subscriptions_df["status"].isin(["active", "canceled"])]
n_active = (final_subs["status"] == "active").sum()
n_canceled = (final_subs["status"] == "canceled").sum()
logo_churn_rate = n_canceled / (n_active + n_canceled)
print(f"\nActive customers (final state): {n_active:,}")
print(f"Canceled customers (final state): {n_canceled:,}")
print(f"Overall logo churn rate (whole sim window): {logo_churn_rate:.1%}")
failed_pct = (invoices_df["payment_status"] == "failed").mean()
print(f"Invoice failure rate: {failed_pct:.1%}")
print(f"Unresolved ticket %: {tickets_df['resolved_date'].isna().mean():.1%}")
print("\nData quality issues injected: duplicate customers, inconsistent")
print("country naming, zero/negative MRR, orphaned usage events.")
print("=" * 55)
