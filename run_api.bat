@echo off
REM ============================================================
REM Banking Compliance API Runner for Windows
REM This script sets up and runs the Banking Compliance API
REM ============================================================

echo.
echo ========================================
echo   Banking Compliance API Setup (Windows)
echo ========================================
echo.

REM Check if Python is installed
where python > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.7+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check Python version (need 3.7+)
python -c "import sys; sys.exit(0) if sys.version_info >= (3, 7) else sys.exit(1)" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.7 or higher is required.
    echo Your current Python version is:
    python --version
    pause
    exit /b 1
)

echo [OK] Python check passed.

REM Check if pip is installed
python -m pip --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] pip is not installed.
    echo Please install pip for Python.
    pause
    exit /b 1
)

echo [OK] pip is installed.

REM Setup virtual environment if it doesn't exist
if not exist "venv\" (
    echo Creating new virtual environment...
    python -m venv venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        echo Try installing venv: pip install virtualenv
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

echo [OK] Virtual environment activated.

REM Check for requirements.txt and install dependencies
if exist "requirements.txt" (
    echo Installing packages from requirements.txt...
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Some requirements may not have installed properly.
    ) else (
        echo [OK] Requirements installed.
    )
) else (
    echo [WARNING] requirements.txt not found. Will install only API dependencies.
)

REM Install required API packages
echo Installing API specific packages...
pip install fastapi uvicorn pydantic python-multipart httpx
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install API dependencies.
    pause
    exit /b 1
)

echo [OK] API dependencies installed.

REM Check if api.py exists
if not exist "api.py" (
    echo [ERROR] api.py file not found in the current directory.
    echo Make sure the API file exists in the current folder.
    pause
    exit /b 1
)

echo [OK] API file found.

REM Check for ODBC Driver for SQL Server
echo Checking for ODBC Driver...
powershell -Command "Get-OdbcDriver | Where-Object { $_.Name -like 'ODBC Driver * for SQL Server' } | Format-Table -AutoSize" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Could not verify ODBC Driver for SQL Server.
    echo If you encounter database connection issues, please install ODBC Driver.
    echo Download from: https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
)

REM Check for API credentials
echo Checking API credentials...
if not defined APP_USERNAME (
    echo [INFO] Using default username: admin
    set APP_USERNAME=admin
)

if not defined APP_PASSWORD (
    echo [INFO] Using default password: pass123
    set APP_PASSWORD=pass123
)

REM Optional: Check for database config
if exist ".streamlit\secrets.toml" (
    echo [OK] Streamlit secrets file found. Will use database credentials from secrets.
) else (
    echo [INFO] No .streamlit\secrets.toml file found.
    echo You'll need to provide database credentials when using database endpoints.
)

echo.
echo ========================================
echo   Starting Banking Compliance API Server
echo ========================================
echo.
echo API will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server
echo.

REM Start the API
python api.py

REM Deactivate virtual environment when done
call venv\Scripts\deactivate.bat

pause