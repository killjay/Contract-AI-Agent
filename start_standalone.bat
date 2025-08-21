@echo off
title Legal Document Review AI - Standalone UI

echo.
echo  ========================================
echo   Legal Document Review AI - Standalone
echo  ========================================
echo.
echo Starting the web interface...
echo.

streamlit run standalone_ui.py --server.port 8501

echo.
echo Application stopped.
pause
