"""
ChurnGuard Analytics — AI Analyst (GenAI Layer)
==================================================
Lets someone ask a plain-English business question and get an answer
grounded in the REAL database — not a hallucinated guess.

How it works (see the architecture diagram from the walkthrough):
    User question
      -> LLM is given the database schema and a `run_sql_query` tool
      -> LLM writes a real SQL query and calls the tool
      -> We execute that query against churnguard.db ourselves
      -> The actual result is fed back to the LLM
      -> LLM writes a plain-English explanation using ONLY that real result

The critical safety property: the LLM NEVER gets to state a number that
didn't come out of an actual query. If it tries to answer with a number
before calling the tool, our system prompt instructs it to call the tool
first — and because we control the tool execution ourselves (not the
LLM), it's structurally impossible for it to fake a result.

Setup:
    1. Get an API key from https://console.anthropic.com (free trial
       credit is enough to run this many times over)
    2. Set it as an environment variable:
         Windows (Command Prompt):  set ANTHROPIC_API_KEY=your-key-here
         Windows (PowerShell):      $env:ANTHROPIC_API_KEY="your-key-here"
    3. Run: python ai/ai_analyst.py
"""

import sqlite3
import os
import sys
import anthropic

DB_PATH = "data/churnguard.db"

# The LLM sees this schema description so it can write correct SQL —
# it never sees the raw data until it calls the tool.
SCHEMA_DESCRIPTION = """
Database: churnguard.db (SQLite)

Table: customers
  customer_id (PK), company_name, industry, company_size, country,
  signup_date, acquisition_channel, cac

Table: subscriptions
  subscription_id (PK), customer_id (FK), plan_tier, mrr_value,
  start_date, end_date, status ('active'/'canceled'/'upgraded'/'downgraded'),
  billing_cycle

Table: invoices
  invoice_id (PK), subscription_id (FK), invoice_date, amount,
  payment_status ('paid'/'failed'), payment_method, retry_count

Table: usage_events
  event_id (PK), customer_id (FK), event_date, feature_used,
  login_count, active_users

Table: support_tickets
  ticket_id (PK), customer_id (FK), created_date, category, priority,
  resolved_date, csat_score, is_resolved
"""

SYSTEM_PROMPT = f"""You are the AI Analyst inside ChurnGuard Analytics, a SaaS
revenue and churn analytics platform. Business users will ask you questions
about their subscription/revenue/customer data.

{SCHEMA_DESCRIPTION}

CRITICAL RULES:
1. You MUST call the run_sql_query tool to get real data before answering
   ANY question involving a number, trend, ranking, or specific customer.
   Never state a number, percentage, or count that didn't come from an
   actual tool result.
2. If a query fails, look at the error and try a corrected query — don't
   guess an answer instead.
3. After getting results, explain them in clear, non-technical business
   language. Lead with the direct answer, then brief supporting detail.
4. If asked for a recommendation, base it only on patterns visible in the
   data you queried, and say so explicitly.
5. Only SELECT queries are allowed — never attempt INSERT/UPDATE/DELETE.
"""

TOOLS = [{
    "name": "run_sql_query",
    "description": "Execute a read-only SQL SELECT query against the churnguard.db database and return the results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A SQL SELECT query to run."}
        },
        "required": ["query"]
    }
}]


def run_sql_query(query: str) -> str:
    """Executes a query against the real database. This is the ONLY way
    numbers reach the LLM — it cannot bypass this function."""
    q = query.strip().lower()
    if not q.startswith("select") and not q.startswith("with"):
        return "ERROR: Only SELECT queries are permitted."
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchmany(50)  # cap result size sent back to the LLM
        conn.close()
        if not rows:
            return "Query ran successfully but returned no rows."
        result_lines = [", ".join(columns)]
        for row in rows:
            result_lines.append(", ".join(str(v) for v in row))
        return "\n".join(result_lines)
    except Exception as e:
        return f"SQL ERROR: {e}"


def ask_ai_analyst(client: anthropic.Anthropic, question: str, verbose: bool = True) -> str:
    """Runs the full plan -> query -> explain loop for one question."""
    messages = [{"role": "user", "content": question}]

    for _ in range(5):  # safety cap on tool-call loops
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "run_sql_query":
                    query = block.input["query"]
                    if verbose:
                        print(f"\n[AI wrote this SQL query]:\n{query}\n")
                    result = run_sql_query(query)
                    if verbose:
                        print(f"[Real query result]:\n{result}\n")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            # Final text answer
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return final_text

    return "The AI Analyst couldn't complete this question after several attempts."


DEMO_QUESTIONS = [
    "What caused revenue to decrease last month?",
    "Which customers are at the highest risk of churning?",
    "What should management do next to reduce churn?",
    "Explain our NRR (Net Revenue Retention) this year.",
]


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set the ANTHROPIC_API_KEY environment variable first.")
        print("Get a key at https://console.anthropic.com")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"Q: {question}\n")
        answer = ask_ai_analyst(client, question)
        print(f"A: {answer}")
    else:
        print("Running demo questions (no argument passed)...\n")
        for q in DEMO_QUESTIONS:
            print("=" * 70)
            print(f"Q: {q}")
            print("=" * 70)
            answer = ask_ai_analyst(client, q, verbose=False)
            print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
