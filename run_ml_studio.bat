@echo off
title SkyFlow - Machine Learning Studio
echo =====================================================================
echo  SkyFlow: Machine Learning Academic Laboratory & Studio Console
echo =====================================================================
echo.
echo Launching Streamlit ML Studio on http://localhost:8502 ...
echo.
cd /d "%~dp0"
streamlit run app_ml_studio.py --server.port 8502
pause
