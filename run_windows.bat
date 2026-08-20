@echo off
cd /d "%~dp0"
echo Installing/updating required packages...
py -m pip install -r requirements.txt
if errorlevel 1 (
  echo Trying python instead...
  python -m pip install -r requirements.txt
)
echo.
echo Starting Momentum + Froth Scanner V6.1...
py -m streamlit run app.py
if errorlevel 1 python -m streamlit run app.py
pause
