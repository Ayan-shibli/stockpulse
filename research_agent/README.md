# Research agent (v1)

A minimal agentic loop with one tool: fetch recent news + sentiment for a
stock ticker from Alpha Vantage, then have an LLM turn that into a
structured JSON verdict (sentiment, key events, sources).

## Setup

1. Get a free Gemini API key: https://aistudio.google.com
2. Get a free Alpha Vantage API key: https://www.alphavantage.co/support/#api-key
3. Copy `.env.example` to `.env` and fill in both keys:
   cp .env.example .env
4. From inside your uv project folder, add the dependencies:
   uv add google-genai requests python-dotenv

## Usage

Quick single-ticker test:
   uv run main.py AAPL

Run your full evaluation set:
   uv run evaluate.py

## Testing accuracy

Open `eval_set.json` and replace the placeholder entries with 8-10 real
tickers. For each one:

1. Read the actual recent news for that ticker yourself first.
2. Write down what sentiment and key facts you'd expect a good summary to
   contain, before you ever run the agent (this keeps you honest, you're
   not unconsciously matching your expectations to whatever it outputs).
3. Run `uv run evaluate.py` and compare the agent's output to your notes,
   side by side.

Look specifically for:
- Facts in the agent's output that aren't supported by any source article
  (hallucination, the most important failure to catch)
- Sentiment calls that don't match what you'd conclude reading the same
  articles yourself
- Important stories from the news feed that got left out entirely
- The Alpha Vantage `overall_sentiment_label` field already gives you a
  second, independent sentiment opinion per article. Worth comparing the
  agent's final sentiment call against it too.

Results from each run are saved to `eval_results.json` so you can track
how accuracy changes as you tweak the system prompt or model.

## If something breaks

The Gemini SDK's function-calling API has shifted slightly across recent
versions. If you get an error about an unknown argument or attribute,
paste the exact error and we'll patch it, this is the kind of small API
drift that's normal to hit and easy to fix once you see the message.
