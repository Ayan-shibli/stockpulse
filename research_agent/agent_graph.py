"""LangGraph-powered Research Agent.

Graph structure:
  START → ticker_resolver → researcher (tool loop) → synthesizer → reflector → END
                                  ↑                                      │
                                  └──────────── retry (if score < 7) ───┘

Nodes:
  ticker_resolver  : Validates the ticker. If query looks like a company name,
                     calls search_tickers to resolve it first.
  researcher       : Groq LLM with 5 tools in a ReAct-style loop. Decides which
                     tools to call and in what order.
  synthesizer      : Formats the accumulated tool results into a structured JSON report.
  reflector        : Self-critiques the report (1-10). Routes to END or back to researcher.
"""

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from groq import Groq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from typing import Literal

from tools import (
    get_company_news,
    get_stock_price,
    get_technical_info,
    get_earnings_info,
    search_tickers,
)

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_TOOL_ITERATIONS = 8  # max tool calls in researcher node
MAX_RETRIES = 2  # max reflection-driven retries

# ─── Tool Registry ────────────────────────────────────────────────────────────
TOOLS_MAP = {
    "get_company_news": get_company_news,
    "get_stock_price": get_stock_price,
    "get_technical_info": get_technical_info,
    "get_earnings_info": get_earnings_info,
    "search_tickers": search_tickers,
}

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_company_news",
            "description": "Fetch recent news articles and per-ticker sentiment scores from Alpha Vantage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. AAPL",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max articles to fetch, default 8",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get real-time stock price, market cap, P/E ratio, analyst targets, and dividend yield.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. AAPL",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_info",
            "description": "Get technical analysis: 50/200-day moving averages, volume, beta, trend signal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. AAPL",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_earnings_info",
            "description": "Get earnings history, next earnings date, EPS estimates, revenue/profit growth.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. AAPL",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tickers",
            "description": (
                "Search for stock ticker symbols by company name or keyword. "
                "Use this when you have a company name but not the exact ticker, "
                "e.g. 'Apple' → AAPL, 'electric vehicles' → TSLA, RIVN, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Company name or keyword",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results, default 8",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

# ─── System Prompts ───────────────────────────────────────────────────────────
RESEARCHER_SYSTEM = """You are an expert financial research agent with access to 5 tools:
  1. get_company_news     — news articles + sentiment scores
  2. get_stock_price      — real-time price, market cap, P/E, analyst targets
  3. get_technical_info   — moving averages, volume, beta, trend signal
  4. get_earnings_info    — EPS history, revenue growth, profit margins
  5. search_tickers       — resolve a company name to its ticker symbol

Strategy:
- Always call get_stock_price and get_company_news first.
- Then call get_technical_info and get_earnings_info for deeper analysis.
- If you don't know the exact ticker, call search_tickers first.
- Call tools in parallel when possible (call multiple in one turn).
- After collecting data, summarize your findings. Do NOT produce the final JSON yet.
"""

SYNTHESIZER_SYSTEM = """You are a financial report formatter.

Given accumulated research data, produce a single JSON object with EXACTLY these fields:
{
  "ticker": "string",
  "company_name": "string",
  "sector": "string or null",
  "current_price": number or null,
  "change_percent_today": number or null,
  "market_cap": number or null,
  "pe_ratio": number or null,
  "analyst_target": number or null,
  "analyst_recommendation": "string or null",
  "week_52_low": number or null,
  "week_52_high": number or null,
  "trend_signal": "string or null",
  "beta": number or null,
  "revenue_growth": number or null,
  "profit_margin": number or null,
  "next_earnings_date": "string or null",
  "summary": "3-5 sentence synthesis combining news, price, technicals and earnings",
  "sentiment": "bullish" | "bearish" | "neutral",
  "sentiment_score": float between -1.0 and 1.0,
  "key_events": ["bullet 1", "bullet 2", ...],
  "sources": ["url or source name", ...],
  "tools_used": ["list of tool names that were called"],
  "confidence_score": integer 1-10
}

Rules:
- Only include facts from the tool results. Never invent data.
- Output ONLY the JSON object. No markdown fences, no commentary.
"""

REFLECTOR_SYSTEM = """You are a quality-control agent reviewing a stock research report.

Evaluate the report on:
  1. Data completeness (are price, news, technicals, earnings all present?)
  2. Factual accuracy (are claims supported by tool data?)
  3. Synthesis quality (does the summary integrate all data points?)
  4. Confidence (is the sentiment backed by evidence?)

Respond ONLY with a JSON object:
{
  "score": <integer 1-10>,
  "passed": <true if score >= 7>,
  "critique": "1-2 sentence critique",
  "missing_data": ["list of any data gaps"]
}
"""


# ─── Agent State ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    ticker: str  # User-provided ticker or query
    resolved_ticker: str  # Validated ticker symbol
    messages: Annotated[list, add_messages]  # Full conversation history
    tool_results: dict  # Accumulated tool outputs keyed by tool name
    report: dict  # Final structured report
    reflection: dict  # Reflector's critique
    retry_count: int  # Number of reflection-driven retries
    steps: list  # Human-readable log of agent steps


# ─── Groq Client ─────────────────────────────────────────────────────────────
def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    return Groq(api_key=api_key)


# ─── Node: Ticker Resolver ────────────────────────────────────────────────────
def ticker_resolver_node(state: AgentState) -> dict:
    """Validate the ticker. If it looks like a company name, resolve it."""
    raw = state["ticker"].strip().upper()
    steps = list(state.get("steps", []))

    # Simple heuristic: if the query has spaces or is longer than 6 chars,
    # it's probably a company name — search for it
    if " " in state["ticker"].strip() or len(state["ticker"].strip()) > 6:
        steps.append(f"🔍 Resolving '{state['ticker']}' to a ticker symbol…")
        result = search_tickers(state["ticker"].strip(), limit=1)
        if result.get("results"):
            resolved = result["results"][0]["ticker"]
            steps.append(f"✅ Resolved to {resolved}")
            return {"resolved_ticker": resolved, "steps": steps}

    steps.append(f"✅ Using ticker {raw}")
    return {"resolved_ticker": raw, "steps": steps}


# ─── Node: Researcher (ReAct tool loop) ──────────────────────────────────────
def researcher_node(state: AgentState) -> dict:
    """Groq LLM + tools in a ReAct loop. Collects data from all tools."""
    client = get_client()
    ticker = state.get("resolved_ticker") or state["ticker"]
    steps = list(state.get("steps", []))
    tool_results = dict(state.get("tool_results", {}))

    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM},
        {
            "role": "user",
            "content": f"Research the stock {ticker}. Call all relevant tools to gather comprehensive data.",
        },
    ]

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                tools=TOOL_DECLARATIONS,
                tool_choice="auto",
            )
        except Exception as api_err:
            # Groq can return a 400 tool_use_failed when the model generates a
            # malformed tool call (e.g. XML-style <function=...> instead of JSON).
            # Retry without tools so the model can produce a plain text summary
            # of whatever data was already gathered.
            steps.append(
                f"⚠️  Groq tool-call error on iteration {iteration + 1}: {api_err}. Falling back to no-tools call."
            )
            try:
                fallback = client.chat.completions.create(
                    model=DEFAULT_MODEL,
                    messages=messages,
                )
                message = fallback.choices[0].message
                steps.append("🧠 Researcher completed via fallback (no tools)")
            except Exception:
                steps.append(
                    "❌ Fallback call also failed — using data collected so far."
                )
            break

        message = response.choices[0].message

        # Convert Groq's ChatCompletionMessage to a plain dict so it can be
        # re-sent to the Groq API and won't trigger LangGraph's message
        # coercion check when returned to state.
        msg_dict: dict = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(msg_dict)

        tool_calls = message.tool_calls or []
        if not tool_calls:
            # No more tool calls — researcher is done gathering data
            steps.append(f"🧠 Researcher completed after {iteration + 1} iterations")
            break

        # Execute each tool call
        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            steps.append(
                f"🛠️  Calling {fn_name}({', '.join(f'{k}={v}' for k,v in args.items())})"
            )

            func = TOOLS_MAP.get(fn_name)
            result = func(**args) if func else {"error": f"Unknown tool: {fn_name}"}

            # Store tool result
            tool_results[fn_name] = result

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    return {
        # Note: we intentionally do NOT return "messages" here.
        # The local `messages` list uses plain dicts in Groq format, which are
        # incompatible with LangGraph's add_messages reducer (it expects
        # LangChain message objects). Downstream nodes (synthesizer, reflector)
        # build their own message lists from tool_results anyway.
        "tool_results": tool_results,
        "steps": steps,
    }


# ─── Node: Synthesizer ────────────────────────────────────────────────────────
def synthesizer_node(state: AgentState) -> dict:
    """Format all collected tool results into the final structured JSON report."""
    client = get_client()
    ticker = state.get("resolved_ticker") or state["ticker"]
    steps = list(state.get("steps", []))
    steps.append("📊 Synthesizing report from all tool data…")

    tool_summary = json.dumps(state.get("tool_results", {}), indent=2)
    tools_used = list(state.get("tool_results", {}).keys())

    messages = [
        {"role": "system", "content": SYNTHESIZER_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Ticker: {ticker}\n\n"
                f"Accumulated tool results:\n{tool_summary}\n\n"
                f"Produce the final JSON report."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
    )

    raw = response.choices[0].message.content or ""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        cleaned = "\n".join(inner).strip()

    try:
        report = json.loads(cleaned)
    except json.JSONDecodeError:
        report = {"error": "Could not parse synthesizer output", "raw": raw}

    report["tools_used"] = tools_used
    steps.append("✅ Report synthesized")

    return {"report": report, "steps": steps}


# ─── Node: Reflector ─────────────────────────────────────────────────────────
def reflector_node(state: AgentState) -> dict:
    """Self-critique the report. Routes to END if quality is sufficient."""
    client = get_client()
    steps = list(state.get("steps", []))
    retry_count = state.get("retry_count", 0)
    steps.append("🔎 Reflecting on report quality…")

    report_str = json.dumps(state.get("report", {}), indent=2)
    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM},
        {"role": "user", "content": f"Evaluate this research report:\n{report_str}"},
    ]

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
    )

    raw = response.choices[0].message.content or ""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        cleaned = "\n".join(inner).strip()

    try:
        reflection = json.loads(cleaned)
    except json.JSONDecodeError:
        reflection = {
            "score": 8,
            "passed": True,
            "critique": "Unable to parse reflection.",
        }

    score = reflection.get("score", 8)
    passed = reflection.get("passed", True) or retry_count >= MAX_RETRIES

    steps.append(
        f"{'✅' if passed else '🔄'} Reflection score: {score}/10 — {'PASSED' if passed else f'retry {retry_count + 1}'}"
    )
    if reflection.get("critique"):
        steps.append(f"   Critique: {reflection['critique']}")

    return {
        "reflection": reflection,
        "retry_count": retry_count + (0 if passed else 1),
        "steps": steps,
    }


# ─── Conditional Router ───────────────────────────────────────────────────────
def should_retry(state: AgentState) -> Literal["researcher", "__end__"]:
    """Route: retry researcher if reflection failed and retries remain."""
    reflection = state.get("reflection", {})
    retry_count = state.get("retry_count", 0)
    passed = reflection.get("passed", True)

    if not passed and retry_count < MAX_RETRIES:
        return "researcher"
    return "__end__"


# ─── Build Graph ──────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("ticker_resolver", ticker_resolver_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("reflector", reflector_node)

    graph.add_edge(START, "ticker_resolver")
    graph.add_edge("ticker_resolver", "researcher")
    graph.add_edge("researcher", "synthesizer")
    graph.add_edge("synthesizer", "reflector")

    graph.add_conditional_edges(
        "reflector",
        should_retry,
        {"researcher": "researcher", "__end__": END},
    )

    return graph.compile()


# ─── Public Interface ─────────────────────────────────────────────────────────
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_research(ticker: str, verbose: bool = True) -> dict:
    """Run the LangGraph research agent for a given ticker or company name.

    Args:
        ticker: Ticker symbol (e.g. "AAPL") or company name (e.g. "Apple").
        verbose: If True, print step-by-step progress.

    Returns:
        The final structured research report dict.
    """
    graph = get_graph()

    initial_state: AgentState = {
        "ticker": ticker,
        "resolved_ticker": "",
        "messages": [],
        "tool_results": {},
        "report": {},
        "reflection": {},
        "retry_count": 0,
        "steps": [],
    }

    final_state = graph.invoke(initial_state)

    if verbose:
        for step in final_state.get("steps", []):
            print(step)

    report = final_state.get("report", {})
    report["_steps"] = final_state.get("steps", [])
    report["_reflection"] = final_state.get("reflection", {})
    return report


def stream_research(ticker: str):
    """Generator that yields research progress as WebSocket-ready dicts.

    Yields:
        {"type": "step", "data": "step text"} for each agent step
        {"type": "result", "data": {report}} as the final message

    This uses graph.stream() to get per-node state updates, then diffs
    the steps list to find newly added steps.
    """
    graph = get_graph()

    initial_state: AgentState = {
        "ticker": ticker,
        "resolved_ticker": "",
        "messages": [],
        "tool_results": {},
        "report": {},
        "reflection": {},
        "retry_count": 0,
        "steps": [],
    }

    seen_steps = 0
    final_state = None

    try:
        for state_update in graph.stream(initial_state):
            # state_update is a dict of {node_name: partial_state}
            for node_name, partial in state_update.items():
                steps = partial.get("steps", [])
                # Yield any new steps
                for step in steps[seen_steps:]:
                    yield {"type": "step", "data": step}
                if len(steps) > seen_steps:
                    seen_steps = len(steps)

                # Track the latest state for final report
                if "report" in partial and partial["report"]:
                    final_state = partial

        # Build final report
        if final_state and "report" in final_state:
            report = final_state["report"]
            report["_steps"] = final_state.get("steps", [])
            report["_reflection"] = final_state.get("reflection", {})
            yield {"type": "result", "data": report}
        else:
            # Fallback: run synchronously if streaming didn't produce report
            result = run_research(ticker, verbose=False)
            yield {"type": "result", "data": result}

    except Exception as e:
        yield {"type": "error", "data": str(e)}
