@echo off
echo =========================================
echo Pump Motor IIoT Monitoring System
echo Simplified Version
echo =========================================
echo.

echo Starting sensor simulator...
start "Sensor Simulator" python sensor\sensor_simple.py
timeout /t 2

echo Starting edge processor...
start "Edge Processor" python edge\processor_simple.py
timeout /t 2

echo Starting digital twin...
start "Digital Twin" python digital_twin\twin_simple.py
timeout /t 2

echo Starting visualization uploader...
start "Visualization" python viz\uploader_simple.py

echo.
echo All components started!
echo Close the windows to stop
pause
