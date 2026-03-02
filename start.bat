@echo off
REM CSAO Demo Application - Windows Startup Script
REM Starts both backend (FastAPI) and frontend (React/Vite)

echo ========================================
echo   CSAO Demo Application
echo   Cart Super Add-On Recommendations
echo ========================================
echo.

cd /d "%~dp0"

REM Check for venv
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

REM Install backend deps
echo [1/4] Checking backend dependencies...
%PYTHON% -c "import fastapi" 2>NUL || (
    echo   Installing FastAPI + uvicorn...
    where uv >NUL 2>NUL && (
        uv pip install fastapi uvicorn --python %PYTHON% -q
    ) || (
        %PYTHON% -m pip install fastapi uvicorn -q
    )
)

REM Install frontend deps
echo [2/4] Installing frontend dependencies...
cd frontend
call npm install --silent 2>NUL
cd ..

REM Start backend
echo [3/4] Starting backend on http://localhost:8000 ...
start "CSAO-Backend" /B %PYTHON% backend\server.py

REM Wait for backend
echo   Waiting for backend to load data...
timeout /t 30 /nobreak >NUL

REM Start frontend
echo [4/4] Starting frontend on http://localhost:5173 ...
cd frontend
start "CSAO-Frontend" /B npm run dev
cd ..

echo.
echo ========================================
echo   CSAO Demo Running!
echo   Frontend: http://localhost:5173
echo   Backend:  http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo ========================================
echo.
echo Press any key to stop both servers...
pause >NUL

REM Kill processes
taskkill /FI "WINDOWTITLE eq CSAO-Backend" /F >NUL 2>NUL
taskkill /FI "WINDOWTITLE eq CSAO-Frontend" /F >NUL 2>NUL
