# ============================================================
# StockPulse — Start Backend + Frontend
# Double-click or run from PowerShell:  .\start.ps1
# ============================================================

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  StockPulse - Starting dev environment"           -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Launch backend in a new cmd window
Write-Host "[1/2] Starting FastAPI backend on http://localhost:8001 ..." -ForegroundColor Yellow
Start-Process cmd -ArgumentList "/k", "`"$root\start_backend.bat`""

Start-Sleep -Seconds 2

# Launch frontend in a new cmd window
Write-Host "[2/2] Starting Vite frontend on http://localhost:5173 ..."   -ForegroundColor Yellow
Start-Process cmd -ArgumentList "/k", "`"$root\start_frontend.bat`""

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "  Both servers launched!" -ForegroundColor Green
Write-Host ""
Write-Host "  Backend  ->  http://localhost:8001"              -ForegroundColor White
Write-Host "  Frontend ->  http://localhost:5173"              -ForegroundColor White
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
