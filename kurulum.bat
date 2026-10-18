@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   Adli Fotorobot - Windows kurulumu
echo ============================================
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo [HATA] Python bulunamadi. https://www.python.org/downloads/release/python-3119/
    echo        adresinden Python 3.11 kurun ve tekrar deneyin.
    pause
    exit /b 1
)

py -3.11 -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [HATA] Python 3.11 bulunamadi. PyTorch CUDA paketleri icin Python 3.11 gerekir.
    echo        https://www.python.org/downloads/release/python-3119/
    pause
    exit /b 1
)

if not exist .venv\Scripts\python.exe (
    echo [1/4] Sanal ortam olusturuluyor...
    py -3.11 -m venv .venv || goto hata
)

echo [2/4] PyTorch (CUDA 12.4) kuruluyor... (~2.5 GB)
.venv\Scripts\python -m pip install --upgrade pip -q || goto hata
.venv\Scripts\python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 || goto hata

echo [3/4] Diger kutuphaneler kuruluyor...
.venv\Scripts\python -m pip install -r requirements.txt || goto hata

echo [4/4] InsightFace kuruluyor...
.venv\Scripts\python -m pip install --no-deps insightface==1.0.1 || goto hata

echo.
echo Kurulum tamam! Uygulamayi baslatmak icin baslat.bat dosyasini calistirin.
echo (Ilk calistirmada modeller indirilir, ~7 GB.)
pause
exit /b 0

:hata
echo.
echo [HATA] Kurulum basarisiz oldu. Yukaridaki hata mesajini kontrol edin.
pause
exit /b 1
