#!/bin/bash

echo "========================================="
echo "Pump Motor IIoT Monitoring System"
echo "Simplified Version"
echo "========================================="
echo ""

echo "Starting sensor simulator in 2 seconds..."
python sensor/sensor_simple.py &
SENSOR_PID=$!
sleep 2

echo "Starting edge processor in 2 seconds..."
python edge/processor_simple.py &
EDGE_PID=$!
sleep 2

echo "Starting digital twin in 2 seconds..."
python digital_twin/twin_simple.py &
TWIN_PID=$!
sleep 2

echo "Starting visualization uploader..."
python viz/uploader_simple.py &
VIZ_PID=$!

echo ""
echo "All components started!"
echo "Sensor PID: $SENSOR_PID"
echo "Edge PID: $EDGE_PID"
echo "Twin PID: $TWIN_PID"
echo "Viz PID: $VIZ_PID"
echo ""
echo "Press Ctrl+C to stop all components"
echo ""

wait
