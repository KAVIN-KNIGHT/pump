# Simplified Project Structure

## What to Keep (Essential Files Only)

```
pump/
├── sensor/
│   └── sensor_simple.py          (simplified sensor simulator)
├── edge/
│   └── processor_simple.py        (simplified edge processor)
├── digital_twin/
│   └── twin_simple.py             (simplified digital twin)
├── viz/
│   └── uploader_simple.py         (simplified visualization)
├── utils/
│   └── mqtt_client.py             (keep as-is, it's good)
├── requirements_simple.txt        (only essential packages)
└── run.sh or run.bat             (startup script)
```

## What to Remove

- `models/train_rul_model.py` - ML training (not essential for live system)
- `utils/signal_processing.py` - Complex signal analysis
- All complex fault injection logic
- Docstrings and verbose comments
- Redundant error handling
- Complex configuration files

## Essential Dependencies Only

- paho-mqtt (MQTT communication)
- requests (HTTP for ThingSpeak)
- numpy (basic math)

