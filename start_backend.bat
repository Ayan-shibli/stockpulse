@echo off
title AstraQuant - Backend (FastAPI)
cd /d "%~dp0research_agent"
echo.
echo  ==========================================
echo   AstraQuant Backend - FastAPI + LangGraph
echo   Running on http://localhost:8001
echo  ==========================================
echo.
python -m uvicorn server:app --host 0.0.0.0 --port 8001
pause
