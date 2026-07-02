"""Research agent: a minimal agentic loop with one tool.

This loop is written manually (not using an SDK's automatic function
calling) on purpose, so every step of the agent's reasoning is visible:
call the model -> check for a tool call -> run it -> feed the result back
-> repeat until the model gives a final answer.

Now powered by Groq (llama-3.3-70b-versatile) via its OpenAI-compatible API.
"""
import json
import os

from dotenv import load_dotenv
from groq import Groq

# Load .env BEFORE importing tools so ALPHA_VANTAGE_API_KEY is available.
load_dotenv()

from tools import get_company_news  # noqa: E402  (import after load_dotenv)

SYSTEM_PROMPT = """You are a market research agent. When asked about a stock,
call the get_company_news tool to fetch recent news, then summarize your
findings.

Always give your final answer as a single JSON object with exactly these
fields, and nothing else (no markdown fences, no extra commentary):
{
  "ticker": "string",
  "summary": "2-4 sentence summary of what's happening",
  "sentiment": "bullish" | "bearish" | "neutral",
  "sentiment_score": float between -1.0 and 1.0,
  "key_events": ["short bullet", "short bullet", ...],
  "sources": ["url1", "url2", ...]
}

Only state facts that appear in the tool results. If the tool returns no
news or an error, say so honestly in the summary instead of inventing
information. Never invent a price target, event, or quote that isn't in
the articles you were given.
"""

# Groq tool definition (OpenAI-compatible JSON Schema format)
GET_NEWS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_company_news",
        "description": "Fetch recent news articles and sentiment data for a stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of articles to fetch, default 8",
                },
            },
            "required": ["ticker"],
        },
    },
}

AVAILABLE_FUNCTIONS = {"get_company_news": get_company_news}

MAX_ITERATIONS = 5
# Groq model — llama-3.3-70b-versatile supports tool/function calling
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class ResearchAgent:
    def __init__(self, model: str = DEFAULT_MODEL):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is not set in your environment.")
        self.client = Groq(api_key=api_key)
        self.model = model

    def run(self, ticker: str, verbose: bool = True) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Research the stock {ticker}."},
        ]

        for step in range(MAX_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[GET_NEWS_TOOL],
                tool_choice="auto",
            )

            message = response.choices[0].message

            # Convert Groq's ChatCompletionMessage to a plain dict before
            # appending. Re-sending a typed Groq object to the API on the next
            # iteration triggers LangChain's MESSAGE_COERCION_FAILURE because
            # it expects plain dicts, not groq.types.chat.ChatCompletionMessage.
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
                # No tool call -> model produced its final answer
                if verbose:
                    print(f"[step {step}] final answer received")
                return self._parse_final_answer(message.content or "")

            # Execute each requested tool call
            for tc in tool_calls:
                if verbose:
                    print(f"[step {step}] calling {tc.function.name}({tc.function.arguments})")

                func = AVAILABLE_FUNCTIONS.get(tc.function.name)
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                if func:
                    result = func(**args)
                else:
                    result = {"error": f"Unknown tool: {tc.function.name}"}

                # Feed the tool result back into the conversation
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )

        return {"error": f"Agent did not produce a final answer within {MAX_ITERATIONS} steps."}

    @staticmethod
    def _parse_final_answer(text: str) -> dict:
        """Parse the model's final text output as JSON.

        BUG FIX: the original used str.strip('`') which removes individual
        backtick characters rather than stripping the full ``` fence block.
        We now split by lines and strip the fence properly.
        """
        cleaned = text.strip()

        # Strip markdown code fences if present (e.g. ```json ... ```)
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove first line (``` or ```json) and last line (```)
            inner_lines = lines[1:]
            if inner_lines and inner_lines[-1].strip() == "```":
                inner_lines = inner_lines[:-1]
            cleaned = "\n".join(inner_lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": "Could not parse model output as JSON", "raw_text": text}
