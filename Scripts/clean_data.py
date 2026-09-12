"""
ChurnGuard Analytics — Data Cleaning Script
=============================================
Takes the deliberately messy raw CSVs and produces clean, analysis-ready
versions in data/processed/. Every fix is logged to the console so you
can explain exactly what was done and why (this log doubles as your
"data cleaning" portfolio talking points).

Fixes applied:
  1. customers.csv     -> standardize inconsistent country names
                        -> remove duplicate customer records (same
                           company name + signup_date)
  2. subscriptions.csv -> fix zero/negative mrr_value (flag + correct
                           using the customer's plan-tier median MRR)
  3. usage_events.csv  -> remove orphaned rows (customer_id that
                           doesn't exist in customers.csv)
  4. support_tickets.csv -> flag unresolved tickets clearly (keep as-is,
                           since "unresolved" is real business signal,
                           not a data error)
  5. invoices.csv      -> no structural issues, but recompute a
                           net_amount column to standardize sign

Run:
    python scripts/clean_data.py
"""

import pandas as pd
import numpy as np
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

log = []
def report(msg):
    log.append(msg)
    print(msg)

print("=" * 60)
print("CHURNGUARD DATA CLEANING")
print("=" * 60)

# ------------------------------------------------------------------
# 1. CUSTOMERS
# ------------------------------------------------------------------
customers = pd.read_csv(f"{RAW_DIR}/customers.csv", parse_dates=["signup_date"])
n_before = len(customers)

# Standardize country names
COUNTRY_MAP = {
    "united states": "United States", "usa": "United States", "us": "United States",
    "united states": "United States",
    "in": "India", "india": "India",
    "uk": "United Kingdom", "u.k.": "United Kingdom", "united kingdom": "United Kingdom",
    "de": "Germany", "germany": "Germany",
    "au": "Australia", "australia": "Australia",
    "ca": "Canada", "canada": "Canada",
}
customers["country"] = customers["country"].str.strip().str.lower().map(COUNTRY_MAP).fillna(customers["country"])
report(f"[customers] Standardized country names -> {customers['country'].nunique()} unique countries "
       f"(was inconsistent free text before)")

# Remove duplicate customers (same company_name + signup_date = same real account)
dupe_mask = customers.duplicated(subset=["company_name", "signup_date"], keep="first")
n_dupes = dupe_mask.sum()
removed_customer_ids = set(customers.loc[dupe_mask, "customer_id"])
customers = customers[~dupe_mask].copy()
report(f"[customers] Removed {n_dupes} duplicate customer records "
       f"(matched on company_name + signup_date)")
report(f"[customers] {n_before} -> {len(customers)} rows")

valid_customer_ids = set(customers["customer_id"])

# ------------------------------------------------------------------
# 2. SUBSCRIPTIONS
# ------------------------------------------------------------------
subs = pd.read_csv(f"{RAW_DIR}/subscriptions.csv", parse_dates=["start_date", "end_date"])
n_before = len(subs)

# Fix zero/negative MRR using the median MRR for that plan tier (excluding bad rows)
bad_mrr_mask = subs["mrr_value"] <= 0
n_bad_mrr = bad_mrr_mask.sum()
plan_medians = subs[~bad_mrr_mask].groupby("plan_tier")["mrr_value"].median()
for tier in subs.loc[bad_mrr_mask, "plan_tier"].unique():
    tier_mask = bad_mrr_mask & (subs["plan_tier"] == tier)
    subs.loc[tier_mask, "mrr_value"] = plan_medians[tier]
report(f"[subscriptions] Fixed {n_bad_mrr} rows with zero/negative mrr_value "
       f"(replaced with plan-tier median MRR)")

# Drop subscriptions belonging to customers removed as duplicates — a removed
# duplicate account means its ENTIRE history is erroneous, not just the
# customer row, so every downstream table must cascade this deletion or the
# database's foreign keys will break later.
orphan_sub_mask = subs["customer_id"].isin(removed_customer_ids)
n_orphan_subs = orphan_sub_mask.sum()
removed_subscription_ids = set(subs.loc[orphan_sub_mask, "subscription_id"])
subs = subs[~orphan_sub_mask].copy()
report(f"[subscriptions] Cascaded delete: removed {n_orphan_subs} subscription rows "
       f"belonging to duplicate customers")
report(f"[subscriptions] {n_before} -> {len(subs)} rows")

# ------------------------------------------------------------------
# 3. INVOICES
# ------------------------------------------------------------------
invoices = pd.read_csv(f"{RAW_DIR}/invoices.csv", parse_dates=["invoice_date"])
n_before = len(invoices)
invoices["amount"] = invoices["amount"].abs()  # standardize sign

# Cascade the same deletion: invoices attached to subscriptions we just removed
orphan_inv_mask = invoices["subscription_id"].isin(removed_subscription_ids)
n_orphan_inv = orphan_inv_mask.sum()
invoices = invoices[~orphan_inv_mask].copy()
report(f"[invoices] Standardized 'amount' to absolute value")
report(f"[invoices] Cascaded delete: removed {n_orphan_inv} invoice rows tied to "
       f"duplicate-customer subscriptions")
report(f"[invoices] {n_before} -> {len(invoices)} rows")

# ------------------------------------------------------------------
# 4. USAGE EVENTS — remove orphaned rows
# ------------------------------------------------------------------
usage = pd.read_csv(f"{RAW_DIR}/usage_events.csv", parse_dates=["event_date"])
n_before = len(usage)
# Removes both the originally-injected orphan rows AND rows cascaded from
# duplicate-customer removal — both are "customer_id not in the clean table"
orphan_mask = ~usage["customer_id"].isin(valid_customer_ids)
n_orphans = orphan_mask.sum()
usage = usage[~orphan_mask].copy()
report(f"[usage_events] Removed {n_orphans} orphaned rows (customer_id not in "
       f"customers table — includes injected orphans + cascaded duplicate removals)")
report(f"[usage_events] {n_before} -> {len(usage)} rows")

# ------------------------------------------------------------------
# 5. SUPPORT TICKETS — flag unresolved, no rows dropped
# ------------------------------------------------------------------
tickets = pd.read_csv(f"{RAW_DIR}/support_tickets.csv", parse_dates=["created_date", "resolved_date"])
n_before = len(tickets)
orphan_ticket_mask = tickets["customer_id"].isin(removed_customer_ids)
n_orphan_tickets = orphan_ticket_mask.sum()
tickets = tickets[~orphan_ticket_mask].copy()
tickets["is_resolved"] = tickets["resolved_date"].notna()
n_unresolved = (~tickets["is_resolved"]).sum()
report(f"[support_tickets] Cascaded delete: removed {n_orphan_tickets} ticket rows "
       f"belonging to duplicate customers ({n_before} -> {len(tickets)} rows)")
report(f"[support_tickets] Flagged {n_unresolved} unresolved tickets "
       f"(kept — this is real signal, not a data error)")

# ------------------------------------------------------------------
# SAVE CLEANED FILES
# ------------------------------------------------------------------
customers.to_csv(f"{PROCESSED_DIR}/customers_clean.csv", index=False)
subs.to_csv(f"{PROCESSED_DIR}/subscriptions_clean.csv", index=False)
invoices.to_csv(f"{PROCESSED_DIR}/invoices_clean.csv", index=False)
usage.to_csv(f"{PROCESSED_DIR}/usage_events_clean.csv", index=False)
tickets.to_csv(f"{PROCESSED_DIR}/support_tickets_clean.csv", index=False)

# Save the cleaning log as a text file for your documentation/README
with open(f"{PROCESSED_DIR}/cleaning_log.txt", "w") as f:
    f.write("\n".join(log))

print("\n" + "=" * 60)
print("CLEANING COMPLETE — files saved to data/processed/")
print("Cleaning log saved to data/processed/cleaning_log.txt")
print("=" * 60)
