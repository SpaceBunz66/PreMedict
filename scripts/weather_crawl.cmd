@echo off
rem Job tai ERA5-Land hang ngay (Task Scheduler "forecast_v2 weather-crawl", xem README). Log noi vao runs\weather_crawl.log
cd /d "%~dp0.."
if not exist runs mkdir runs
echo ===== %date% %time% >> runs\weather_crawl.log
.venv-ml\Scripts\python.exe -m forecast_v2.cli weather-crawl >> runs\weather_crawl.log 2>&1
