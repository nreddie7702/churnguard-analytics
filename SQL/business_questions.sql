-- ChurnGuard Analytics — 25 SQL Business Questions
-- Run against data/churnguard.db (SQLite)
-- Every query below has been executed and verified against the live database.

-- Q1: What is our current total MRR (active subscriptions only)?
-- Techniques: Aggregation, WHERE
SELECT ROUND(SUM(mrr_value), 2) AS current_mrr
FROM subscriptions
WHERE status = 'active';

-- Q2: How does current MRR break down by plan tier?
-- Techniques: GROUP BY, window function (SUM OVER)
SELECT plan_tier,
       COUNT(*) AS active_subscriptions,
       ROUND(SUM(mrr_value), 2) AS mrr,
       ROUND(100.0 * SUM(mrr_value) / SUM(SUM(mrr_value)) OVER (), 1) AS pct_of_total_mrr
FROM subscriptions
WHERE status = 'active'
GROUP BY plan_tier
ORDER BY mrr DESC;

-- Q3: What has new MRR looked like month by month for the last 12 months?
-- Techniques: Date functions, GROUP BY, subquery
SELECT strftime('%Y-%m', start_date) AS signup_month,
       COUNT(*) AS new_subscriptions,
       ROUND(SUM(mrr_value), 2) AS new_mrr
FROM subscriptions
WHERE start_date >= date((SELECT MAX(start_date) FROM subscriptions), '-12 months')
GROUP BY signup_month
ORDER BY signup_month;

-- Q4: What has churned MRR looked like month by month?
-- Techniques: Date functions, GROUP BY, CASE-free filtering
SELECT strftime('%Y-%m', end_date) AS churn_month,
       COUNT(*) AS churned_subscriptions,
       ROUND(SUM(mrr_value), 2) AS churned_mrr
FROM subscriptions
WHERE status = 'canceled' AND end_date IS NOT NULL
GROUP BY churn_month
ORDER BY churn_month;

-- Q5: What is the net new MRR movement (new minus churned) by month?
-- Techniques: CTE, LEFT JOIN, COALESCE
WITH new_mrr AS (
    SELECT strftime('%Y-%m', start_date) AS month, SUM(mrr_value) AS amount
    FROM subscriptions GROUP BY month
),
churned_mrr AS (
    SELECT strftime('%Y-%m', end_date) AS month, SUM(mrr_value) AS amount
    FROM subscriptions WHERE status = 'canceled' AND end_date IS NOT NULL
    GROUP BY month
)
SELECT COALESCE(n.month, c.month) AS month,
       ROUND(COALESCE(n.amount, 0), 2) AS new_mrr,
       ROUND(COALESCE(c.amount, 0), 2) AS churned_mrr,
       ROUND(COALESCE(n.amount, 0) - COALESCE(c.amount, 0), 2) AS net_mrr_change
FROM new_mrr n
LEFT JOIN churned_mrr c ON n.month = c.month
ORDER BY month;

-- Q6: What is our monthly logo churn rate (% of active customers who canceled)?
-- Techniques: CTE, LEFT JOIN, NULLIF
WITH monthly_status AS (
    SELECT strftime('%Y-%m', end_date) AS churn_month, COUNT(*) AS churned
    FROM subscriptions WHERE status = 'canceled' AND end_date IS NOT NULL
    GROUP BY churn_month
),
active_base AS (
    SELECT strftime('%Y-%m', start_date) AS active_month, COUNT(*) AS active_count
    FROM subscriptions GROUP BY active_month
)
SELECT m.churn_month,
       m.churned,
       a.active_count AS base_customers,
       ROUND(100.0 * m.churned / NULLIF(a.active_count, 0), 2) AS logo_churn_pct
FROM monthly_status m
LEFT JOIN active_base a ON m.churn_month = a.active_month
ORDER BY m.churn_month;

-- Q7: Which industries have the highest revenue churn rate?
-- Techniques: JOIN, CASE WHEN, HAVING, aggregation
SELECT c.industry,
       ROUND(SUM(CASE WHEN s.status = 'canceled' THEN s.mrr_value ELSE 0 END), 2) AS churned_mrr,
       ROUND(SUM(s.mrr_value), 2) AS total_mrr_ever,
       ROUND(100.0 * SUM(CASE WHEN s.status = 'canceled' THEN s.mrr_value ELSE 0 END)
             / NULLIF(SUM(s.mrr_value), 0), 1) AS revenue_churn_pct
FROM subscriptions s
JOIN customers c ON s.customer_id = c.customer_id
GROUP BY c.industry
HAVING COUNT(*) >= 20
ORDER BY revenue_churn_pct DESC;

-- Q8: Rank customers by current MRR within their industry.
-- Techniques: Window function RANK, PARTITION BY, JOIN
SELECT c.industry, c.company_name, s.mrr_value,
       RANK() OVER (PARTITION BY c.industry ORDER BY s.mrr_value DESC) AS mrr_rank_in_industry
FROM subscriptions s
JOIN customers c ON s.customer_id = c.customer_id
WHERE s.status = 'active'
ORDER BY c.industry, mrr_rank_in_industry
LIMIT 30;

-- Q9: Who are our top 10 customers by total lifetime revenue actually paid?
-- Techniques: Multi-table JOIN, GROUP BY, aggregation
SELECT c.customer_id, c.company_name, c.industry,
       ROUND(SUM(i.amount), 2) AS lifetime_revenue_paid
FROM invoices i
JOIN subscriptions s ON i.subscription_id = s.subscription_id
JOIN customers c ON s.customer_id = c.customer_id
WHERE i.payment_status = 'paid'
GROUP BY c.customer_id, c.company_name, c.industry
ORDER BY lifetime_revenue_paid DESC
LIMIT 10;

-- Q10: Dense-rank plan tiers by their churn rate (ties get the same rank).
-- Techniques: CTE, DENSE_RANK
WITH tier_churn AS (
    SELECT plan_tier,
           ROUND(100.0 * SUM(CASE WHEN status='canceled' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_pct
    FROM subscriptions
    GROUP BY plan_tier
)
SELECT plan_tier, churn_pct,
       DENSE_RANK() OVER (ORDER BY churn_pct DESC) AS churn_rank
FROM tier_churn;

-- Q11: Build a signup-month cohort retention table: what % of each cohort is still active N months later?
-- Techniques: CTE, JOIN, date math (julianday), CASE WHEN
WITH cohorts AS (
    SELECT customer_id, strftime('%Y-%m', signup_date) AS cohort_month
    FROM customers
),
subs_with_cohort AS (
    SELECT s.customer_id, c.cohort_month, s.status,
           CAST((julianday(COALESCE(s.end_date, (SELECT MAX(invoice_date) FROM invoices))) 
                 - julianday(s.start_date)) / 30 AS INT) AS months_active
    FROM subscriptions s
    JOIN cohorts c ON s.customer_id = c.customer_id
)
SELECT cohort_month,
       COUNT(*) AS cohort_size,
       SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS still_active,
       ROUND(100.0 * SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_retained
FROM subs_with_cohort
GROUP BY cohort_month
ORDER BY cohort_month;

-- Q12: What is the average tenure (in days) before churn, by plan tier?
-- Techniques: Date math, GROUP BY
SELECT plan_tier,
       ROUND(AVG(julianday(end_date) - julianday(start_date)), 0) AS avg_days_to_churn
FROM subscriptions
WHERE status = 'canceled' AND end_date IS NOT NULL
GROUP BY plan_tier
ORDER BY avg_days_to_churn;

-- Q13: For each customer, show login_count this month vs. last month (to spot engagement drops).
-- Techniques: Window function LAG, PARTITION BY
SELECT customer_id, event_date, login_count,
       LAG(login_count) OVER (PARTITION BY customer_id ORDER BY event_date) AS prev_month_logins,
       login_count - LAG(login_count) OVER (PARTITION BY customer_id ORDER BY event_date) AS change_in_logins
FROM usage_events
ORDER BY customer_id, event_date
LIMIT 30;

-- Q14: Which active customers had a >50% drop in logins month-over-month (early churn warning)?
-- Techniques: CTE, LAG, multi-JOIN, at-risk filtering
WITH usage_with_lag AS (
    SELECT customer_id, event_date, login_count,
           LAG(login_count) OVER (PARTITION BY customer_id ORDER BY event_date) AS prev_logins
    FROM usage_events
)
SELECT u.customer_id, c.company_name, u.event_date, u.prev_logins, u.login_count,
       ROUND(100.0 * (u.login_count - u.prev_logins) / NULLIF(u.prev_logins, 0), 1) AS pct_change
FROM usage_with_lag u
JOIN customers c ON u.customer_id = c.customer_id
JOIN subscriptions s ON s.customer_id = u.customer_id AND s.status = 'active'
WHERE u.prev_logins > 5
  AND (u.login_count - u.prev_logins) < -0.5 * u.prev_logins
ORDER BY pct_change
LIMIT 20;

-- Q15: What is our monthly payment failure rate?
-- Techniques: Date functions, CASE WHEN, aggregation
SELECT strftime('%Y-%m', invoice_date) AS month,
       COUNT(*) AS total_invoices,
       SUM(CASE WHEN payment_status='failed' THEN 1 ELSE 0 END) AS failed_invoices,
       ROUND(100.0 * SUM(CASE WHEN payment_status='failed' THEN 1 ELSE 0 END) / COUNT(*), 2) AS failure_pct
FROM invoices
GROUP BY month
ORDER BY month;

-- Q16: Which currently-active customers have had 2+ failed payments in their last 3 invoices (at-risk list)?
-- Techniques: ROW_NUMBER, CTE, HAVING
WITH ranked_invoices AS (
    SELECT i.*, s.customer_id,
           ROW_NUMBER() OVER (PARTITION BY s.customer_id ORDER BY i.invoice_date DESC) AS rn
    FROM invoices i
    JOIN subscriptions s ON i.subscription_id = s.subscription_id
    WHERE s.status = 'active'
)
SELECT customer_id, SUM(CASE WHEN payment_status='failed' THEN 1 ELSE 0 END) AS recent_failures
FROM ranked_invoices
WHERE rn <= 3
GROUP BY customer_id
HAVING recent_failures >= 2
ORDER BY recent_failures DESC;

-- Q17: Which support ticket categories generate the highest volume, and what's their average resolution time?
-- Techniques: GROUP BY, HAVING, date math
SELECT category,
       COUNT(*) AS ticket_count,
       ROUND(AVG(julianday(resolved_date) - julianday(created_date)), 1) AS avg_resolution_days,
       SUM(CASE WHEN is_resolved = 0 THEN 1 ELSE 0 END) AS unresolved_count
FROM support_tickets
GROUP BY category
HAVING ticket_count > 50
ORDER BY ticket_count DESC;

-- Q18: Do churned customers have lower average CSAT scores than retained customers?
-- Techniques: JOIN, GROUP BY, filtering NULLs
SELECT s.status,
       ROUND(AVG(t.csat_score), 2) AS avg_csat,
       COUNT(t.ticket_id) AS ticket_count
FROM support_tickets t
JOIN subscriptions s ON t.customer_id = s.customer_id
WHERE t.csat_score IS NOT NULL
GROUP BY s.status;

-- Q19: Show each customer's most recent support ticket only.
-- Techniques: ROW_NUMBER, CTE
WITH latest_ticket AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY created_date DESC) AS rn
    FROM support_tickets
)
SELECT customer_id, created_date, category, priority, is_resolved
FROM latest_ticket
WHERE rn = 1
ORDER BY created_date DESC
LIMIT 20;

-- Q20: How many days has it been since each active customer last logged in (engagement recency)?
-- Techniques: Subquery, JOIN, date math, GROUP BY
SELECT c.customer_id, c.company_name,
       MAX(u.event_date) AS last_active_date,
       CAST(julianday((SELECT MAX(event_date) FROM usage_events)) - julianday(MAX(u.event_date)) AS INT) AS days_since_last_active
FROM customers c
JOIN usage_events u ON c.customer_id = u.customer_id
JOIN subscriptions s ON s.customer_id = c.customer_id AND s.status = 'active'
GROUP BY c.customer_id, c.company_name
ORDER BY days_since_last_active DESC
LIMIT 20;

-- Q21: Compare acquisition channels on CAC vs. average revenue generated per customer.
-- Techniques: Multi-JOIN, aggregation, ratio calculation
SELECT c.acquisition_channel,
       COUNT(DISTINCT c.customer_id) AS customers,
       ROUND(AVG(c.cac), 2) AS avg_cac,
       ROUND(SUM(i.amount) / COUNT(DISTINCT c.customer_id), 2) AS avg_revenue_per_customer,
       ROUND((SUM(i.amount) / COUNT(DISTINCT c.customer_id)) / NULLIF(AVG(c.cac), 0), 2) AS revenue_to_cac_ratio
FROM customers c
JOIN subscriptions s ON c.customer_id = s.customer_id
JOIN invoices i ON i.subscription_id = s.subscription_id AND i.payment_status = 'paid'
GROUP BY c.acquisition_channel
ORDER BY revenue_to_cac_ratio DESC;

-- Q22: What's the running total of new MRR added, month over month, across the whole timeline?
-- Techniques: CTE, window function SUM OVER with ORDER BY (running total)
WITH monthly_new AS (
    SELECT strftime('%Y-%m', start_date) AS month, SUM(mrr_value) AS new_mrr
    FROM subscriptions GROUP BY month
)
SELECT month, ROUND(new_mrr, 2) AS new_mrr,
       ROUND(SUM(new_mrr) OVER (ORDER BY month), 2) AS running_total_mrr
FROM monthly_new
ORDER BY month;

-- Q23: Find customers who upgraded and then later downgraded (a red flag pattern).
-- Techniques: LAG, pattern detection across rows
WITH sub_sequence AS (
    SELECT customer_id, status, start_date,
           LAG(status) OVER (PARTITION BY customer_id ORDER BY start_date) AS previous_status
    FROM subscriptions
)
SELECT customer_id
FROM sub_sequence
WHERE status = 'downgraded' AND previous_status = 'upgraded';

-- Q24: What % of total company MRR comes from our top 10% of customers (revenue concentration risk)?
-- Techniques: NTILE, CASE WHEN, concentration analysis
WITH ranked AS (
    SELECT customer_id, mrr_value,
           NTILE(10) OVER (ORDER BY mrr_value DESC) AS decile
    FROM subscriptions
    WHERE status = 'active'
)
SELECT ROUND(100.0 * SUM(CASE WHEN decile = 1 THEN mrr_value ELSE 0 END) / SUM(mrr_value), 1) AS pct_mrr_from_top_decile
FROM ranked;

-- Q25: Give a full customer risk snapshot: plan, MRR, days since last login, recent payment failures, open tickets.
-- Techniques: Multi-table LEFT JOIN, GROUP BY, HAVING — composite risk view
SELECT c.customer_id, c.company_name, s.plan_tier, s.mrr_value,
       CAST(julianday((SELECT MAX(event_date) FROM usage_events)) - julianday(MAX(u.event_date)) AS INT) AS days_since_login,
       SUM(CASE WHEN i.payment_status = 'failed' THEN 1 ELSE 0 END) AS failed_payments,
       SUM(CASE WHEN t.is_resolved = 0 THEN 1 ELSE 0 END) AS open_tickets
FROM customers c
JOIN subscriptions s ON c.customer_id = s.customer_id AND s.status = 'active'
LEFT JOIN usage_events u ON u.customer_id = c.customer_id
LEFT JOIN invoices i ON i.subscription_id = s.subscription_id
LEFT JOIN support_tickets t ON t.customer_id = c.customer_id
GROUP BY c.customer_id, c.company_name, s.plan_tier, s.mrr_value
HAVING failed_payments >= 1 OR open_tickets >= 1
ORDER BY failed_payments DESC, open_tickets DESC
LIMIT 20;
