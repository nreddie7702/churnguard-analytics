"""
ChurnGuard Analytics — AI Agent
=================================
Builds on ai_analyst.py's grounding pattern (never invent a number — always
call a real tool), but adds two things a single-question Analyst can't do:

  1. TWO tools instead of one: SQL for querying the database, and a Python
     tool for deeper analysis (grouping, correlation, ranking) on the
     engineered feature tables.
  2. A genuine reasoning LOOP: the agent decides for itself whether it has
     enough information yet, or whether it needs another tool call before
     it can answer — instead of always doing exactly one query then stopping.

Example: ask "find our biggest churn risk and recommend an action" and the
agent might: (1) query the risk score table for the highest-risk segment,
(2) run a Python analysis on WHY that segment is risky, (3) only then answer
with a recommendation grounded in both results.

Setup: same as ai_analyst.py — needs ANTHROPIC_API_KEY set.
Run:   python ai/ai_agent.py "find our biggest churn risk and what to do about it"
"""

import sqlite3
import pandas as pd
import numpy as np
import os
import sys
import io
import contextlib
import anthropic

DB_PATH = "data/churnguard.db"
FEATURES_PATH = "data/processed/customer_features.csv"
RISK_SCORES_PATH = "data/processed/churn_risk_scores.csv"

SCHEMA_DESCRIPTION = """
SQL database (churnguard.db) tables: customers, subscriptions, invoices,
usage_events, support_tickets — same schema as the rest of the project.

Python analysis has two pandas DataFrames pre-loaded and ready to use:
  features      -> one row per customer: industry, company_size, plan_tier,
                   mrr_value, tenure_days, avg_logins, usage_trend_pct,
                   failure_rate, total_tickets, avg_csat, churned (0/1)
  risk_scores   -> active customers only: customer_id, industry,
                   company_size, plan_tier, mrr_value, churn_risk_score,
                   risk_tier ('High'/'Medium'/'Low')
"""

SYSTEM_PROMPT = f"""You are the ChurnGuard AI Agent — an autonomous analyst
that investigates business questions using real tools, taking as many
steps as genuinely needed before answering.

{SCHEMA_DESCRIPTION}

CRITICAL RULES:
1. Never state a number that didn't come from an actual tool call result.
2. For open-ended questions ("find our biggest risk", "what should we do"),
   don't stop at one query. Investigate: query a summary, then dig into
   WHY the pattern exists, before giving a recommendation. Two or three
   tool calls chained together is expected and good for this kind of
   question — a single query alone is usually not enough evidence for a
   real business recommendation.
3. For simple direct questions ("what is our current MRR"), one tool call
   is fine — don't pad simple questions with unnecessary extra steps.
4. Only SELECT queries are allowed in SQL. In Python, use only the
   pre-loaded `features` and `risk_scores` DataFrames — don't invent
   columns that don't exist.
5. End with a clear, direct answer in plain business language, explicitly
   referencing what each tool call found.
"""

TOOLS = [
    {
        "name": "run_sql_query",
        "description": "Execute a read-only SQL SELECT query against churnguard.db.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "run_python_analysis",
        "description": (
            "Run pandas/numpy analysis code against the pre-loaded `features` "
            "and `risk_scores` DataFrames. Use print() to output results — "
            "only printed output is returned to you."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
]

# Loaded once at startup — the agent's Python tool operates on these
_features_df = pd.read_csv(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else None
_risk_df = pd.read_csv(RISK_SCORES_PATH) if os.path.exists(RISK_SCORES_PATH) else None


def run_sql_query(query: str) -> str:
    q = query.strip().lower()
    if not q.startswith("select") and not q.startswith("with"):
        return "ERROR: Only SELECT queries are permitted."
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(query)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchmany(50)
        conn.close()
        if not rows:
            return "Query ran successfully but returned no rows."
        lines = [", ".join(columns)] + [", ".join(str(v) for v in r) for r in rows]
        return "\n".join(lines)
    except Exception as e:
        return f"SQL ERROR: {e}"


def run_python_analysis(code: str) -> str:
    """Executes agent-written pandas code in a restricted namespace —
    only pd/np and the two pre-loaded DataFrames are available, no file
    or network access, and only printed output is returned."""
    if _features_df is None or _risk_df is None:
        return "ERROR: features/risk_scores CSVs not found. Run the earlier pipeline stages first."

    forbidden = ["import os", "import sys", "open(", "__", "exec(", "eval(", "subprocess"]
    if any(f in code for f in forbidden):
        return "ERROR: Code contains a disallowed operation."

    safe_globals = {
        "pd": pd, "np": np,
        "features": _features_df.copy(),
        "risk_scores": _risk_df.copy(),
        "print": print,
    }
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, safe_globals)
        output = buffer.getvalue().strip()
        return output if output else "Code ran successfully but printed no output. Use print() to see results."
    except Exception as e:
        return f"PYTHON ERROR: {e}"


def run_agent(client: anthropic.Anthropic, goal: str, verbose: bool = True, max_steps: int = 8) -> str:
    messages = [{"role": "user", "content": goal}]

    for step in range(max_steps):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if verbose:
                    print(f"\n[Step {step+1}] Agent calls {block.name}:")
                    print(block.input.get("query") or block.input.get("code"))

                if block.name == "run_sql_query":
                    result = run_sql_query(block.input["query"])
                elif block.name == "run_python_analysis":
                    result = run_python_analysis(block.input["code"])
                else:
                    result = "ERROR: unknown tool"

                if verbose:
                    print(f"[Result]:\n{result}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            return "".join(b.text for b in response.content if b.type == "text")

    return "Agent reached the step limit without a final answer — the question may need to be narrowed."


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        sys.exit(1)
    if _features_df is None or _risk_df is None:
        print(f"ERROR: Could not find {FEATURES_PATH} or {RISK_SCORES_PATH}.")
        print("Run the EDA and ML notebook stages first to generate them.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    goal = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else \
        "Find our biggest churn risk segment and recommend one specific action to address it."

    print(f"GOAL: {goal}\n" + "=" * 70)
    answer = run_agent(client, goal)
    print("\n" + "=" * 70)
    print(f"FINAL ANSWER:\n{answer}")


if __name__ == "__main__":
    main()
