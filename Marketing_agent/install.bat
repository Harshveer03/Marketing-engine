@echo off
echo ========================================
echo   Marketing Engine Installation
echo ========================================
echo.

echo Installing Python dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies!
    echo Please make sure Python and pip are installed.
    pause
    exit /b 1
)

echo.
echo Creating necessary directories...
if not exist "data" mkdir data
if not exist "config" mkdir config
if not exist "generated" mkdir generated
if not exist "generated\content" mkdir generated\content
if not exist "generated\content\blogs" mkdir generated\content\blogs
if not exist "generated\content\social" mkdir generated\content\social
if not exist "generated\analytics" mkdir generated\analytics
if not exist "generated\news" mkdir generated\news
if not exist "generated\topics" mkdir generated\topics

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Add your .env file with API keys
echo 2. Place your business document in the 'data' folder
echo 3. Run: python setup.py (for CLI setup)
echo    OR
echo    Run: python app.py (for web dashboard)
echo.
echo Web Dashboard: http://localhost:5000
echo.
pause