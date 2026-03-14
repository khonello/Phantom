@echo off
REM Setup script for roop-cam on Windows
REM Creates virtual environment and installs dependencies

setlocal enabledelayedexpansion

echo.
echo roop-cam Setup Script
echo ====================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Install from python.org
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python 3.9 or higher required
    exit /b 1
)
echo ^[OK^] Python version 3.9+
echo.

REM Check FFmpeg
echo Checking FFmpeg...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] FFmpeg not found
    echo Install from: https://ffmpeg.org/download.html
    echo Or via Chocolatey: choco install ffmpeg
    echo Or via winget: winget install FFmpeg
    echo.
    set /p CONTINUE="Continue without FFmpeg? (y/n): "
    if /i not "!CONTINUE!"=="y" exit /b 1
) else (
    echo [OK] FFmpeg found
)
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Error: Failed to create virtual environment
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: Failed to upgrade pip
)
echo [OK] pip upgraded
echo.

REM Install dependencies
echo Installing dependencies...
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    set REQUIREMENTS=requirements-pipeline-gpu.txt
    echo CUDA detected -- using GPU requirements
) else (
    set REQUIREMENTS=requirements-pipeline-cpu.txt
    echo No CUDA detected -- using CPU requirements
)

pip install -r "%REQUIREMENTS%"
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    exit /b 1
)
echo [OK] Dependencies installed from %REQUIREMENTS%
echo.

REM Verify installation
echo Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import insightface; print('InsightFace: OK')"
echo [OK] Key dependencies verified
echo.

REM Report CUDA status
echo Checking CUDA support...
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo.

echo Setup complete!
echo.
echo Quick start:
echo   venv\Scripts\activate.bat   - Activate virtual environment
echo   python pipeline.py               - Launch GUI
echo   python pipeline.py --help        - See CLI options
echo.
