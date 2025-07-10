@echo off
echo ======================================================
echo LLAMA GUARD 3 CONDA SETUP INSTALLER
echo ======================================================
echo.

REM Create project directory
echo Creating project directory...
if not exist "C:\LlamaGuard" mkdir "C:\LlamaGuard"
cd /d "C:\LlamaGuard"

REM Check if conda is available
echo Checking conda installation...
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Conda not found!
    echo Please install Anaconda or Miniconda first:
    echo https://www.anaconda.com/products/distribution
    pause
    exit /b 1
)

echo ✅ Conda found!
echo.

REM Create environment from yml file
echo Creating conda environment 'LlamaGuard'...
echo This may take 5-10 minutes...
conda env create -f environment.yml

if %errorlevel% neq 0 (
    echo WARNING: Environment creation failed or already exists
    echo Updating existing environment...
    conda env update -f environment.yml
)

echo.
echo ======================================================
echo NEXT STEPS:
echo ======================================================
echo 1. Install Ollama from: https://ollama.ai/download/windows
echo 2. Open Anaconda Prompt
echo 3. Run: cd C:\LlamaGuard
echo 4. Run: conda activate LlamaGuard
echo 5. Run: python llama_guard_conda_setup.py
echo.
echo After Ollama is installed, run the setup script to complete installation.
echo ======================================================
pause