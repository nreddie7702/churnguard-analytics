"""
ChurnGuard Analytics — Database Loader
=========================================
Creates a SQLite database with a proper relational schema (primary keys,
foreign keys, sensible data types) and loads the cleaned CSVs into it.

Why SQLite: zero setup (no server, no credentials), the whole database
is a single portable file, and it's fully compatible with standard SQL
(window functions, CTEs, etc.) — everything we need for the SQL
analysis stage. This also mirrors a real ETL pattern: raw -> clean ->
load into a queryable store.

Run:
    python scripts/load_to_db.py
"""

import sqlite3
import pandas as pd
import os

PROCESSED_DIR = "data/processed"
DB_PATH = "data/churnguard.db"

# Start fresh each time this script runs (idempotent — safe to re-run)
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("PRAGMA foreign_keys = ON;")

# ------------------------------------------------------------------
# SCHEMA — explicit types + keys, not just pandas' auto-inferred schema
# ------------------------------------------------------------------
SCHEMA = """
CREATE TABLE customers (
    customer_id         INTEGER PRIMARY KEY,
    company_name         TEXT NOT NULL,
    industry             TEXT,
    company_size         TEXT,
    country              TEXT,
    signup_date          DATE NOT NULL,
    acquisition_channel   TEXT,
    cac                  REAL
);

CREATE TABLE subscriptions (
    subscription_id      INTEGER PRIMARY KEY,
    customer_id          INTEGER NOT NULL,
    plan_tier            TEXT NOT NULL,
    mrr_value            REAL NOT NULL,
    start_date           DATE NOT NULL,
    end_date             DATE,
    status                TEXT NOT NULL,
    billing_cycle         TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE invoices (
    invoice_id           INTEGER PRIMARY KEY,
    subscription_id      INTEGER NOT NULL,
    invoice_date          DATE NOT NULL,
    amount                REAL NOT NULL,
    payment_status        TEXT NOT NULL,
    payment_method        TEXT,
    retry_count           INTEGER,
    FOREIGN KEY (subscription_id) REFERENCES subscriptions(subscription_id)
);

CREATE TABLE usage_events (
    event_id             INTEGER PRIMARY KEY,
    customer_id          INTEGER NOT NULL,
    event_date            DATE NOT NULL,
    feature_used          TEXT,
    login_count           INTEGER,
    active_users          INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE support_tickets (
    ticket_id            INTEGER PRIMARY KEY,
    customer_id          INTEGER NOT NULL,
    created_date          DATE NOT NULL,
    category              TEXT,
    priority              TEXT,
    resolved_date         DATE,
    csat_score            INTEGER,
    is_resolved           INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
"""

cur.executescript(SCHEMA)
conn.commit()
print("Schema created: customers, subscriptions, invoices, usage_events, support_tickets")

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
tables = {
    "customers": "customers_clean.csv",
    "subscriptions": "subscriptions_clean.csv",
    "invoices": "invoices_clean.csv",
    "usage_events": "usage_events_clean.csv",
    "support_tickets": "support_tickets_clean.csv",
}

for table_name, filename in tables.items():
    df = pd.read_csv(f"{PROCESSED_DIR}/{filename}")
    if "is_resolved" in df.columns:
        df["is_resolved"] = df["is_resolved"].astype(int)
    df.to_sql(table_name, conn, if_exists="append", index=False)
    print(f"Loaded {len(df):,} rows into '{table_name}'")

conn.commit()

# ------------------------------------------------------------------
# VERIFICATION QUERIES
# ------------------------------------------------------------------
print("\n" + "=" * 55)
print("DATABASE VERIFICATION")
print("=" * 55)

checks = [
    ("Total customers", "SELECT COUNT(*) FROM customers"),
    ("Total subscriptions", "SELECT COUNT(*) FROM subscriptions"),
    ("Total invoices", "SELECT COUNT(*) FROM invoices"),
    ("Total usage events", "SELECT COUNT(*) FROM usage_events"),
    ("Total support tickets", "SELECT COUNT(*) FROM support_tickets"),
    ("Active customers (current MRR)", """
        SELECT COUNT(*) FROM subscriptions WHERE status = 'active'
    """),
    ("Orphaned subscriptions (should be 0)", """
        SELECT COUNT(*) FROM subscriptions s
        LEFT JOIN customers c ON s.customer_id = c.customer_id
        WHERE c.customer_id IS NULL
    """),
    ("Orphaned invoices (should be 0)", """
        SELECT COUNT(*) FROM invoices i
        LEFT JOIN subscriptions s ON i.subscription_id = s.subscription_id
        WHERE s.subscription_id IS NULL
    """),
]

for label, query in checks:
    result = cur.execute(query).fetchone()[0]
    print(f"{label}: {result:,}")

conn.close()
print("\nDatabase saved to data/churnguard.db")
print("=" * 55)
