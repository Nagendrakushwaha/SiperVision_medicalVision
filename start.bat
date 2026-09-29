@echo off
title SmartMed Vision Launcher
echo ========================================================
echo Starting SmartMed Vision (Backend + Frontend)
echo ========================================================

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "SmartMed Vision - Backend" cmd /k "python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Starting Vite Frontend on http://localhost:5173 ...
npm --prefix frontend run dev
