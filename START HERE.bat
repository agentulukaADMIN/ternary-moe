@echo off
title Ternary MoE - starting...
cd /d "%~dp0"

rem If the server isn't already running, start it minimized
curl -s -m 2 http://127.0.0.1:8080/health >nul 2>&1
if errorlevel 1 (
    echo Starting the AI server (first load takes ~30 seconds)...
    start "Ternary MoE server - keep this window open" /min ^
        "%~dp0fabric\build-cpu\bin\Release\llama-server.exe" ^
        -m "%~dp0models\bitnet-2b-tq1_0.gguf" ^
        --lora "%~dp0adapters\mult-f16.gguf" ^
        --lora "%~dp0adapters\roman-f16.gguf" ^
        --lora-init-without-apply -c 4096 --port 8080 -t 10
)

rem Wait until the server answers
:wait
curl -s -m 2 http://127.0.0.1:8080/health >nul 2>&1
if errorlevel 1 (
    timeout /t 2 /nobreak >nul
    goto wait
)

title Ternary MoE - chat
echo.
echo Ready! Ask math questions, for example:
echo    What is 47 times 62?
echo    Convert 1994 to Roman numerals
echo (Press Ctrl+C or close this window to quit)
echo.
python "%~dp0moe_driver.py"
pause
