# AstraQuant — AI Stock Research Agent

> Real-time stock intelligence powered by a LangGraph multi-agent pipeline, Groq LLM, and a PyTorch LSTM ensemble forecaster.

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![React](https://img.shields.io/badge/react-19-61dafb)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)
![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen)

---

## What it does

Type any stock ticker or company name. AstraQuant spins up a **4-node LangGraph agent** that fans out across five data tools, synthesises a structured research report, then self-critiques it with a reflector node before presenting results. Alongside the research, a **PyTorch LSTM ensemble** trains on 6 months of price history and forecasts the next 7 trading days with online learning via BPTT.

---

## Architecture
User query

│

▼

┌─────────────────────────────────────────────────────┐

│                  LangGraph Agent                    │

│                                                     │

│  ticker_resolver → researcher → synthesizer         │

│                         │              │            │

│                   (tool loop)          ▼            │

│                  ┌──────────┐     reflector         │

│                  │  Tools   │     score ≥ 7?        │

│                  │ price    │      yes / no          │

│                  │ news     │      END / retry       │

│                  │ technical│                        │

│                  │ earnings │                        │

│                  │ search   │                        │

│                  └──────────┘                        │

└─────────────────────────────────────────────────────┘

│

▼

FastAPI backend  ←→  React frontend (Vite)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 19, Vite 8, Framer Motion, Recharts |
| Backend | FastAPI, Uvicorn, Python 3.11+ |
| Agent | LangGraph, ReAct pattern |
| LLM | Groq — llama-3.3-70b-versatile |
| ML | PyTorch LSTM ensemble, BPTT fine-tuning |
| Market data | yfinance |
| News | Alpha Vantage NEWS_SENTIMENT |
| Deploy | Vercel |

---

## Local Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- [Groq API key](https://console.groq.com)
- [Alpha Vantage key](https://www.alphavantage.co/support/#api-key)

### 1. Configure
```bash
git clone https://github.com/your-username/stock_research.git
cd stock_research
cp research_agent/.env.example research_agent/.env
# Add your API keys to .env
```

### 2. Backend (Terminal 1)
```powershell
cd research_agent
pip install fastapi uvicorn pydantic requests yfinance python-dotenv groq langgraph torch numpy
uvicorn server:app --host 127.0.0.1 --port 8001 --reload
```

### 3. Frontend (Terminal 2)
```powershell
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

---

## Running Tests

```bash
cd research_agent
pytest tests/ -v
```

36 tests — feature engineering, LSTM architecture, rollout realism, all 4 tools with mocked network calls.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/research` | POST | Full LangGraph research report |
| `/api/predict/{ticker}` | GET | LSTM 7-day forecast |
| `/api/predict/{ticker}/backtest` | GET | Walk-forward accuracy test |
| `/api/prices` | GET | Live prices for ticker strip |
| `/api/search` | GET | Search ticker by company name |
| `/api/trending` | GET | Trending ticker list |

---

## Environment Variables

```env
GROQ_API_KEY=your_groq_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

---

## License
MIT
