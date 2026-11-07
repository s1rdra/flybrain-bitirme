@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Sanal Meyve Sinegi Beyni ===
docker info >nul 2>&1
if errorlevel 1 (
  echo Docker Desktop calismiyor. Once Docker Desktop'i acip "Engine running" yazmasini bekle.
  pause
  exit /b 1
)
echo Imaj hazirlaniyor (ilk seferde 5-10 dk surer)...
docker compose build || (pause & exit /b 1)
echo Arayuz aciliyor: http://localhost:8000  (kapatmak icin bu pencerede Ctrl+C)
start "" cmd /c "timeout /t 15 >nul & start http://localhost:8000"
docker compose up
pause
