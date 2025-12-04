# 🏭 Industrial IoT Predictive Maintenance System

A complete **Industry 4.0 solution** for real-time pump motor monitoring using edge computing and digital twin technology. Predict equipment failures before they happen and optimize maintenance schedules.

## 📋 Overview

This project demonstrates a production-ready **Predictive Maintenance System** that combines:
- **Real-time sensor monitoring** with local edge analytics
- **Digital Twin technology** for virtual equipment modeling
- **Cloud integration** with ThingSpeak for remote visualization
- **Predictive algorithms** for Remaining Useful Life (RUL) estimation

### Architecture


Sensor Layer (Raw Data)
    ↓
Edge Processor (Real-time Analytics)
    ↓
Digital Twin (Predictive Intelligence)
    ↓
Cloud Layer (ThingSpeak Visualization)


## 🎯 Key Features

✅ **Real-time Equipment Monitoring**
- Vibration analysis (FFT & RMS calculations)
- Temperature monitoring (bearing & motor windings)
- Current analysis (overload detection)
- Flow rate and pressure monitoring

✅ **Edge Computing**
- Instant fault detection (no cloud latency)
- Health Index calculation (0-100%)
- Local anomaly detection
- Works offline if internet disconnects

✅ **Digital Twin**
- Virtual equipment state synchronization
- Remaining Useful Life (RUL) prediction
- Maintenance recommendations
- Trend analysis and pattern recognition

✅ **Cloud Integration**
- Real-time ThingSpeak dashboards
- Historical data storage
- Remote monitoring from anywhere
- Mobile-friendly visualization

✅ **Alert System**
- Critical threshold alerts
- Multi-level severity (INFO, WARNING, CRITICAL, EMERGENCY)
- Telegram notifications
- Structured alert messages

## 📊 Monitoring Parameters

| Parameter | Range | Optimal | Unit |
|-----------|-------|---------|------|
| Vibration RMS | 0-100 | <10 | m/s² |
| Bearing Temperature | 25-200 | <80 | °C |
| Motor Current | 10-100 | 15-45 | A |
| Flow Rate | 50-500 | 200-400 | L/min |
| Pressure | 1-10 | 3-7 | bar |
| Winding Temperature | 25-250 | <120 | °C |

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Internet connection (for MQTT & ThingSpeak)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pump-predictive-maintenance.git
cd pump-predictive-maintenance
```

2. **Install dependencies**
```bash
pip install -r requirements_simple.txt
```

Required packages:
- `paho-mqtt` - MQTT communication
- `requests` - HTTP requests for cloud APIs
- `numpy` - Numerical computing

3. **Configure credentials** (Optional)

Edit `viz/uploader_simple.py` to add your ThingSpeak credentials:
```python
config = {
    'thingspeak_api_key': 'YOUR_API_KEY',
    'thingspeak_channel_id': 'YOUR_CHANNEL_ID',
    'telegram_bot_token': 'YOUR_BOT_TOKEN',
    'telegram_chat_id': 'YOUR_CHAT_ID'
}
```

### Running the System

#### **Windows (Recommended)**

```bash
run_simple.bat
```

This will start all 4 components in separate terminals.

#### **Manual Start** (Linux/Mac/Windows)

Open 4 separate terminals and run:

**Terminal 1 - Sensor Simulator**
```bash
python sensor/sensor_simple.py
```

**Terminal 2 - Edge Processor** (start after sensor is running)
```bash
python edge/processor_simple.py
```

**Terminal 3 - Digital Twin** (start after edge is running)
```bash
python digital_twin/twin_simple.py
```

**Terminal 4 - Cloud Uploader** (start after twin is running)
```bash
python viz/uploader_simple.py
```

⚠️ **Important:** Start components in order with 2-3 second delays between each.

## 📈 Expected Output

### Sensor Simulator Output
```
INFO - Generated reading #1: vibration=2.5 m/s², temp=78.5°C, current=35.2A
INFO - Generated reading #2: vibration=3.1 m/s², temp=79.2°C, current=36.1A
```

### Edge Processor Output
```
Processing #1 | HI=100.0% | Vibration=2.5 m/s² | Bearing=78.5°C
Processing #2 | HI=99.8% | Vibration=3.1 m/s² | Bearing=79.2°C
```

### Digital Twin Output
```
Twin Update #1 | Health=100.0% | RUL=8760h | Status=NORMAL
Twin Update #2 | Health=99.8% | RUL=8759h | Status=NORMAL
```

### ThingSpeak Integration
```
INFO - Data uploaded to ThingSpeak (Entry ID: 119)
INFO - Successfully published 8 fields to cloud
```

## 🎨 Project Structure

```
pump/
├── sensor/
│   └── sensor_simple.py           # Generates realistic pump motor data
├── edge/
│   └── processor_simple.py        # Real-time analytics & alerts
├── digital_twin/
│   └── twin_simple.py             # Predictive intelligence
├── viz/
│   └── uploader_simple.py         # Cloud integration
├── utils/
│   ├── mqtt_client.py             # MQTT communication library
│   └── signal_processing.py       # Signal analysis utilities
├── requirements_simple.txt        # Python dependencies
├── run_simple.bat                 # Windows startup script
├── run_simple.sh                  # Linux/Mac startup script
├── README.md                      # This file
└── LICENSE                        # Project license
```

## 🔄 System Workflow

### 1️⃣ Sensor Data Generation (Every 3 seconds)

Generates realistic pump motor readings:
```json
{
  "ts": "2025-11-21 07:35:45",
  "vibration_rms": 8.5,
  "bearing_temp": 92.3,
  "winding_temp": 115.6,
  "current_a": 45.2,
  "flow_lpm": 320,
  "pressure_bar": 4.8
}
```

### 2️⃣ Edge Processing (Instant Analysis)

Calculates **Health Index** using weighted formula:
```
Health = 100.0
- (Vibration penalty: 25% weight)
- (Temperature penalty: 30% weight)
- (Current penalty: 20% weight)
Result: Health Index = 67.9% (for example data above)
```

Detects alerts:
```
CRITICAL: Vibration 8.5 m/s² (exceeds 5.0 threshold)
CRITICAL: Bearing temp 92.3°C (exceeds 85.0 threshold)
```

### 3️⃣ Digital Twin Processing

**RUL Prediction:**
- Health 80-100% → 8760 hours (healthy)
- Health 60-80% → 2160 hours (degrading)
- Health 40-60% → 720 hours (poor)
- Health 20-40% → 168 hours (critical)
- Health 0-20% → 24 hours (emergency)

**Recommendations:**
```
Health 67.9% → Status: WARNING, Action: "Monitor closely"
```

### 4️⃣ Cloud Visualization

Published to ThingSpeak with 8 fields:
- Field 1: Health Index (%)
- Field 2: Vibration RMS (m/s²)
- Field 3: Bearing Temperature (°C)
- Field 4: Motor Current (A)
- Field 5: Flow Rate (L/min)
- Field 6: Pressure (bar)
- Field 7: RUL Hours
- Field 8: Alert Count

**Live Dashboard:** https://thingspeak.com/channels/3170500

## 📊 Health Index Calculation

The Health Index combines multiple sensor parameters with weighted penalties:

python
health = 100.0

# Vibration Analysis (25% weight)
if vibration > 5.0 m/s²:
    health -= (vibration - 5.0) × 5

# Temperature Analysis (30% weight)
if bearing_temp > 85°C:
    health -= (bearing_temp - 85) × 2
if winding_temp > 130°C:
    health -= (winding_temp - 130) × 1.5

# Current Analysis (20% weight)
if current > 72A:
    health -= (current - 72) × 3

# Final Health Index
return max(0, min(100, health))


### Health Index Interpretation

| Range | Status | Action |
|-------|--------|--------|
| 80-100% | ✅ Healthy | Normal operation |
| 60-79% | 🟡 Fair | Monitor closely |
| 40-59% | 🟠 Poor | Schedule maintenance |
| 20-39% | 🔴 Critical | Immediate maintenance |
| 0-19% | ⚫ Emergency | **SHUTDOWN REQUIRED** |

