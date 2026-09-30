@echo off
title Cloudflare Tunnel - Public Grafana Access
color 0B
cls
echo ======================================================================
echo    CLOUDFLARE TUNNEL - PUBLIC GRAFANA INDUSTRIAL DASHBOARDS
echo ======================================================================
echo.
echo  Starting secure public HTTPS tunnel for Grafana (Port 3000)...
echo.
cd /d "%~dp0"
.\cloudflared.exe tunnel --url http://localhost:3000
pause
