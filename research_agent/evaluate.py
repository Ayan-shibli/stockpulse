"""Run the research agent over a labeled test set and print a comparison.

This does NOT auto-score sentiment or fact correctness for you, that's a
judgment call only a human can make well. It runs the agent, prints its
output next to your own expected answer, and saves everything to
eval_results.json so you can go through it and mark matches/mismatches.
"""
import json
import sys

from agent import ResearchAgent


def load_eval_set(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_eval(path: str = "eval_set.json"):
    cases = load_eval_set(path)
    agent = ResearchAgent()

    results = []
    for case in cases:
        ticker = case["ticker"]
        print(f"\n{'=' * 60}")
        print(f"Ticker: {ticker}")
        print("-" * 60)

        output = agent.run(ticker, verbose=False)

        print("YOUR EXPECTED SENTIMENT :", case.get("expected_sentiment"))
        print("AGENT SENTIMENT         :", output.get("sentiment"))
        print("YOUR EXPECTED FACTS:")
        for fact in case.get("expected_key_facts", []):
            print(f"   - {fact}")
        print("AGENT KEY EVENTS:")
        for event in output.get("key_events", []):
            print(f"   - {event}")
        print("AGENT SUMMARY           :", output.get("summary"))
        print("SOURCES CITED           :", output.get("sources"))

        results.append({"ticker": ticker, "expected": case, "actual": output})

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to eval_results.json")
    print("Now go through each one: did sentiment match, were the facts")
    print("accurate, and did anything get hallucinated that wasn't in any source?")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "eval_set.json"
    run_eval(target)
