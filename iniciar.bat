@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo No se encontro Python. Instalalo desde https://www.python.org/downloads/ y marca "Add Python to PATH".
  pause & exit /b
)
where ollama >nul 2>nul
if errorlevel 1 (
  echo No se encontro Ollama. Descargalo desde https://ollama.com/download/windows
  pause & exit /b
)
start "" http://localhost:8000
python server.py
pause
