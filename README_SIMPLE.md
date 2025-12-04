# Simplified Pump Motor IIoT System

## Quick Start

### Installation
```bash
pip install -r requirements_simple.txt
```

### Run (Windows)
```bash
run_simple.bat
```

### Run (Linux/Mac)
```bash
bash run_simple.sh
```

### Run Manually (Recommended for Testing)

**Terminal 1 - Sensor Simulator:**
```bash
python sensor/sensor_simple.py
```

**Terminal 2 - Edge Processor (after ~2 seconds):**
```bash
python edge/processor_simple.py
```

**Terminal 3 - Digital Twin (after ~2 seconds):**
```bash
python digital_twin/twin_simple.py
```

**Terminal 4 - Visualization/ThingSpeak (after ~2 seconds):**
```bash
python viz/uploader_simple.py
```

## System Architecture

1. **sensor_simple.py** - Generates realistic sensor readings
   - Publishes to: `pump/motor/sensor`
   - Data: vibration, temperature, current, flow, pressure

2. **processor_simple.py** - Analyzes sensor data in real-time
   - Subscribes from: `pump/motor/sensor`
   - Publishes to: `pump/motor/twin/state`
   - Calculates: Health Index, Alerts

3. **twin_simple.py** - Virtual equipment replica
   - Subscribes from: `pump/motor/twin/state`
   - Publishes to: `pump/motor/twin/state`
   - Calculates: RUL prediction

4. **uploader_simple.py** - Cloud visualization
   - Subscribes from: `pump/motor/twin/state`
   - Uploads to: ThingSpeak Channel 3170500

## Key Differences from Full Version

✅ **Removed:**
- Complex fault injection (now simple random variations)
- Advanced signal processing (FFT, rolling stats)
- ML model training
- Complex recommendation engine
- Telegram notifications
- Docstrings and verbose comments

✅ **Kept:**
- Core MQTT communication
- Real-time health calculation
- RUL prediction (simple heuristic)
- ThingSpeak integration
- Structured data flow

## File Sizes

```
Original:  ~50KB of code
Simplified: ~8KB of code
```

## Troubleshooting

If you see "Waiting for sensor data..." but no data arrives:
1. Make sure sensor simulator started first
2. Wait 3-5 seconds before starting edge processor
3. Check MQTT broker connectivity

