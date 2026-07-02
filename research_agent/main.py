"""Quick manual test: run the research agent on a single ticker.

Usage:
    uv run main.py AAPL
"""

import json
import sys

from agent import ResearchAgent


def main():
    """Run the research agent on a single ticker passed as a CLI argument."""
    if len(sys.argv) < 2:
        print("Usage: uv run main.py <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1]
    agent = ResearchAgent()
    result = agent.run(ticker)

    print("\nFinal output:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
