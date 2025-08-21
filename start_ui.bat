@echo off
echo Starting Legal Document Review AI Agent UI...
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again
    pause
    exit /b 1
)

:: Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Check for .env file
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Copying .env.example to .env...
    copy .env.example .env
    echo.
    echo IMPORTANT: Please edit .env file and add your API keys:
    echo - OPENAI_API_KEY or ANTHROPIC_API_KEY
    echo.
    pause
)

:: Start the Streamlit UI
echo.
echo Starting Streamlit UI...
echo The web interface will open in your browser at http://localhost:8501
echo.
streamlit run src/ui/app.py

pause
