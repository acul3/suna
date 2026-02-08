#!/usr/bin/env python3
"""
End-to-end agent simulation test.

Simulates a user sending a chat message and monitors the full agent lifecycle:
- Thread/project creation
- Agent execution
- Tool calls and results
- Error detection
- Database persistence verification

Usage:
    uv run python tests/e2e_agent_test.py
    uv run python tests/e2e_agent_test.py --prompt "your custom prompt"
    uv run python tests/e2e_agent_test.py --timeout 300
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

import httpx
import jwt
import psycopg


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_PROMPT = "build beautiful slide about last news about what happen in indonesia,in bahasa indonesia"
DEFAULT_TIMEOUT = 300  # 5 minutes


def get_config():
    from dotenv import load_dotenv
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_file):
        load_dotenv(env_file)

    required = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_JWT_SECRET"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"ERROR: Missing env vars: {missing}")
        print("Make sure backend/.env exists and has these values.")
        sys.exit(1)

    return {
        "supabase_url": os.getenv("SUPABASE_URL"),
        "service_role_key": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        "jwt_secret": os.getenv("SUPABASE_JWT_SECRET"),
        "database_url": os.getenv("DATABASE_URL"),
    }


def get_first_user(config: dict) -> dict:
    resp = httpx.get(
        f"{config['supabase_url']}/auth/v1/admin/users?per_page=1",
        headers={
            "Authorization": f"Bearer {config['service_role_key']}",
            "apikey": config["service_role_key"],
        },
    )
    resp.raise_for_status()
    users = resp.json().get("users", [])
    if not users:
        print("ERROR: No users found in Supabase. Sign up first.")
        sys.exit(1)
    user = users[0]
    print(f"  User: {user['email']} (id={user['id'][:8]}...)")
    return user


def generate_jwt(user_id: str, jwt_secret: str) -> str:
    payload = {
        "sub": user_id,
        "aud": "authenticated",
        "role": "authenticated",
        "iss": "supabase",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
    }
    return jwt.encode(payload, jwt_secret, algorithm="HS256")


def ensure_credits(db_url: str, account_id: str, min_balance: float = 50.0):
    conn = psycopg.connect(db_url)
    cur = conn.cursor()
    cur.execute(
        "SELECT balance, tier FROM credit_accounts WHERE account_id = %s",
        (account_id,),
    )
    row = cur.fetchone()
    if not row:
        print(f"  WARNING: No credit_accounts row for {account_id}")
        conn.close()
        return

    balance, tier = float(row[0]), row[1]
    print(f"  Credits: balance={balance}, tier={tier}")

    if balance < min_balance or tier in ("none", None):
        cur.execute(
            """
            UPDATE credit_accounts
            SET balance = GREATEST(balance, %s), tier = COALESCE(NULLIF(tier, 'none'), 'free'),
                daily_credits_balance = GREATEST(daily_credits_balance, %s)
            WHERE account_id = %s
            """,
            (min_balance, min_balance, account_id),
        )
        conn.commit()
        print(f"  Credits topped up to {min_balance}, tier set to 'free'")

    conn.close()


async def start_agent(backend_url: str, token: str, prompt: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{backend_url}/v1/agent/start",
            headers={"Authorization": f"Bearer {token}"},
            data={"prompt": prompt},
        )
        if resp.status_code != 200:
            print(f"  ERROR: Agent start failed: {resp.status_code} {resp.text}")
            sys.exit(1)
        return resp.json()


async def stream_agent_output(backend_url: str, token: str, agent_run_id: str, timeout: int) -> list:
    events = []
    url = f"{backend_url}/v1/agent-run/{agent_run_id}/stream?token={token}"

    start = time.time()
    buffer = ""

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout + 10, connect=10)) as client:
        try:
            async with client.stream("GET", url) as resp:
                async for chunk in resp.aiter_text():
                    if time.time() - start > timeout:
                        print(f"  Stream timeout after {timeout}s")
                        break

                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line.startswith("data: ") and line != "data: ":
                            try:
                                data = json.loads(line[6:])
                                events.append(data)

                                # Check for completion
                                if data.get("type") == "status" and data.get("status") in ("completed", "failed"):
                                    return events
                            except json.JSONDecodeError:
                                pass
        except httpx.ReadTimeout:
            print(f"  Stream read timeout after {timeout}s")
        except Exception as e:
            print(f"  Stream error: {e}")

    return events


def analyze_events(events: list) -> dict:
    report = {
        "total_events": len(events),
        "event_types": Counter(),
        "tool_calls": [],
        "tool_results": [],
        "errors": [],
        "thread_runs": set(),
        "completion_status": None,
        "timing": {},
    }

    current_tools = {}

    for d in events:
        etype = d.get("type", "unknown")
        report["event_types"][etype] += 1

        # Extract tool calls from metadata
        md_str = d.get("metadata", "{}")
        try:
            md = json.loads(md_str) if isinstance(md_str, str) else md_str
        except (json.JSONDecodeError, TypeError):
            md = {}

        for tc in md.get("tool_calls", []):
            fn = tc.get("function_name", "")
            tc_id = tc.get("tool_call_id", "")
            if fn and tc_id not in current_tools:
                current_tools[tc_id] = fn
                report["tool_calls"].append({"name": fn, "id": tc_id})

        # Track thread runs
        tr_id = md.get("thread_run_id", "")
        if tr_id:
            report["thread_runs"].add(tr_id)

        # Tool results
        if etype == "tool":
            content_str = d.get("content", "{}")
            try:
                content = json.loads(content_str) if isinstance(content_str, str) else content_str
            except (json.JSONDecodeError, TypeError):
                content = {"raw": content_str}

            tool_name = md.get("tool_name", content.get("name", "unknown"))
            result_str = str(content.get("content", content.get("result", content)))[:500]
            is_error = any(kw in result_str.lower() for kw in ["error", "traceback", "exception", "❌"])
            is_empty = len(result_str.strip()) < 5

            report["tool_results"].append({
                "name": tool_name,
                "result_len": len(result_str),
                "is_error": is_error,
                "is_empty": is_empty,
                "preview": result_str[:200],
            })

        # Errors
        if etype == "error":
            report["errors"].append(d.get("error", "unknown"))

        # Completion
        if etype == "status" and d.get("status") in ("completed", "failed"):
            report["completion_status"] = d.get("status")

        # Timing
        if etype == "timing":
            report["timing"]["first_response_ms"] = d.get("first_response_ms")
        if etype == "llm_ttft":
            report["timing"].setdefault("ttft_list", []).append(d.get("ttft_seconds"))

    return report


def verify_db_persistence(db_url: str, thread_id: str, agent_run_id: str) -> dict:
    results = {"thread_exists": False, "agent_run_exists": False, "message_count": 0, "agent_run_status": None}

    conn = psycopg.connect(db_url)
    cur = conn.cursor()

    cur.execute("SELECT thread_id, name FROM threads WHERE thread_id = %s", (thread_id,))
    row = cur.fetchone()
    results["thread_exists"] = row is not None
    if row:
        results["thread_name"] = row[1]

    cur.execute("SELECT id, status, error FROM agent_runs WHERE id = %s", (agent_run_id,))
    row = cur.fetchone()
    results["agent_run_exists"] = row is not None
    if row:
        results["agent_run_status"] = row[1]
        results["agent_run_error"] = row[2]

    cur.execute("SELECT count(*) FROM messages WHERE thread_id = %s", (thread_id,))
    results["message_count"] = cur.fetchone()[0]

    conn.close()
    return results


def print_report(report: dict, db_results: dict, elapsed: float):
    print("\n" + "=" * 60)
    print("  E2E AGENT TEST REPORT")
    print("=" * 60)

    print(f"\n--- Execution ---")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  SSE events: {report['total_events']}")
    print(f"  Agent steps: {len(report['thread_runs'])}")
    print(f"  Status: {report['completion_status'] or 'UNKNOWN'}")

    if report["timing"]:
        print(f"\n--- Timing ---")
        if "first_response_ms" in report["timing"]:
            print(f"  First response: {report['timing']['first_response_ms']:.0f}ms")
        for i, ttft in enumerate(report["timing"].get("ttft_list", [])):
            print(f"  LLM TTFT #{i+1}: {ttft:.2f}s")

    print(f"\n--- Tool Calls ({len(report['tool_calls'])}) ---")
    tool_counts = Counter(tc["name"] for tc in report["tool_calls"])
    for name, count in tool_counts.most_common():
        print(f"  {name}: {count}x")

    print(f"\n--- Tool Results ---")
    for tr in report["tool_results"]:
        status = "ERROR" if tr["is_error"] else ("EMPTY" if tr["is_empty"] else "OK")
        icon = {"ERROR": "❌", "EMPTY": "⚠️", "OK": "✅"}[status]
        print(f"  {icon} [{status}] {tr['name']}: {tr['result_len']} chars")
        if tr["is_error"]:
            print(f"      {tr['preview'][:150]}")

    if report["errors"]:
        print(f"\n--- Errors ({len(report['errors'])}) ---")
        for e in report["errors"]:
            print(f"  ❌ {e}")
    else:
        print(f"\n--- Errors: None ---")

    print(f"\n--- DB Persistence ---")
    print(f"  Thread exists:    {'✅' if db_results['thread_exists'] else '❌'} {db_results.get('thread_name', '')}")
    print(f"  Agent run exists: {'✅' if db_results['agent_run_exists'] else '❌'} status={db_results.get('agent_run_status', 'N/A')}")
    print(f"  Messages saved:   {'✅' if db_results['message_count'] > 0 else '❌'} count={db_results['message_count']}")

    # Overall verdict
    print(f"\n--- VERDICT ---")
    issues = []
    if not db_results["thread_exists"]:
        issues.append("Thread NOT persisted to DB")
    if not db_results["agent_run_exists"]:
        issues.append("Agent run NOT persisted to DB")
    if db_results["message_count"] == 0:
        issues.append("No messages saved to DB")
    if report["completion_status"] != "completed":
        issues.append(f"Agent did not complete (status={report['completion_status']})")
    if report["errors"]:
        issues.append(f"{len(report['errors'])} error(s) in stream")
    for tr in report["tool_results"]:
        if tr["is_error"]:
            issues.append(f"Tool '{tr['name']}' returned error")

    if issues:
        print(f"  ❌ ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  ✅ ALL CHECKS PASSED")

    print("=" * 60)
    return len(issues) == 0


async def main():
    parser = argparse.ArgumentParser(description="E2E agent simulation test")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Chat prompt to send")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Max seconds to wait")
    parser.add_argument("--backend-url", default=BACKEND_URL, help="Backend API URL")
    args = parser.parse_args()

    backend_url = args.backend_url

    print(f"🔧 Loading config (backend: {backend_url})...")
    config = get_config()

    print("👤 Finding test user...")
    user = get_first_user(config)
    user_id = user["id"]

    print("🔑 Generating JWT...")
    token = generate_jwt(user_id, config["jwt_secret"])
    print(f"  Token: {token[:30]}...")

    print("💰 Ensuring credits...")
    ensure_credits(config["database_url"], user_id, min_balance=50.0)

    print(f"🚀 Starting agent...")
    print(f"  Prompt: {args.prompt[:80]}...")
    start_time = time.time()
    result = await start_agent(backend_url, token, args.prompt)

    thread_id = result["thread_id"]
    agent_run_id = result["agent_run_id"]
    project_id = result["project_id"]
    print(f"  Thread:  {thread_id}")
    print(f"  Run:     {agent_run_id}")
    print(f"  Project: {project_id}")

    print(f"📡 Streaming agent output (timeout={args.timeout}s)...")
    events = await stream_agent_output(backend_url, token, agent_run_id, args.timeout)
    elapsed = time.time() - start_time
    print(f"  Received {len(events)} events in {elapsed:.1f}s")

    print("📊 Analyzing results...")
    report = analyze_events(events)

    print("🔍 Verifying DB persistence...")
    # Wait briefly for async DB writes to complete
    await asyncio.sleep(3)
    db_results = verify_db_persistence(config["database_url"], thread_id, agent_run_id)

    passed = print_report(report, db_results, elapsed)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
