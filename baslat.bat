@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
    echo Once kurulum.bat dosyasini calistirin.
    pause
    exit /b 1
)

rem Gemini anahtariniz varsa asagidaki satirin basindaki "rem" kelimesini silip anahtari yazin:
rem set GOOGLE_API_KEY=buraya_anahtar

set HF_HUB_DISABLE_SYMLINKS_WARNING=1
.venv\Scripts\python app.py %*
pause
