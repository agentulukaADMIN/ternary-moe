@echo off
setlocal
title Ternary MoE - starting...
cd /d "%~dp0"

set "SERVER_EXE=%~dp0fabric\build-cpu\bin\Release\llama-server.exe"
set "MODEL=%~dp0models\bitnet-2b-tq1_0.gguf"

if not exist "%SERVER_EXE%" echo ERROR: llama-server.exe not found. See README-run.md. & pause & exit /b 1
if not exist "%MODEL%" echo ERROR: model file not found. See README-run.md. & pause & exit /b 1

curl -s -m 2 http://127.0.0.1:8080/health >nul 2>&1
if not errorlevel 1 goto ready

echo Starting the AI server... first load takes about 30 seconds.
start "Ternary MoE server - keep me open" /min "%SERVER_EXE%" -m "%MODEL%" --lora "%~dp0adapters\mult-f16.gguf" --lora "%~dp0adapters\roman-f16.gguf" --lora-init-without-apply -c 4096 --port 8080 -t 10

:wait
timeout /t 2 /nobreak >nul
curl -s -m 2 http://127.0.0.1:8080/health >nul 2>&1
if errorlevel 1 goto wait

:ready
title Ternary MoE - chat
echo.
echo Server is up. Now loading the assistant - wait for the green "Ready!" below...
echo.
python "%~dp0moe_driver.py"
echo.
echo Chat ended. Press any key to close.
pause >nul
