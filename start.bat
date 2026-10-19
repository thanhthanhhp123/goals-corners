@echo off
rem Double-click to start the Goals & Corners Predictor at http://127.0.0.1:8000
cd /d "%~dp0"
if not exist web\data\meta.json (
  echo First run: downloading data and fitting the models. This takes a few minutes...
  python -m predictor update || goto :error
)
start "" http://127.0.0.1:8000
python -m predictor serve --port 8000
goto :eof
:error
echo Build failed. Check that Python 3.10+ and the packages in requirements.txt are installed.
pause
