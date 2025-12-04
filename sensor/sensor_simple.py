import json
import time
import random
import logging
from datetime import datetime, timezone
from typing import Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mqtt_client import MQTTClient, Topics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSensorSimulator:
    def __init__(self, mqtt_client: MQTTClient):
        self.mqtt_client = mqtt_client
        self.sampling_interval = 3.0
        
    def simulate_reading(self) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        
        reading = {
            'ts': timestamp,
            'vibration_rms': round(random.uniform(0.5, 3.0), 2),
            'bearing_temp': round(random.uniform(40, 80), 1),
            'winding_temp': round(random.uniform(45, 85), 1),
            'current_a': round(random.uniform(45, 65), 1),
            'voltage_v': round(380 + random.uniform(-5, 5), 1),
            'flow_lpm': round(random.uniform(150, 250), 0),
            'pressure_bar': round(random.uniform(3, 7), 1),
            'rpm': round(random.uniform(1700, 1900), 0),
            'oil_level_pct': round(random.uniform(70, 100), 1),
            'seal_leak': 0
        }
        return reading
    
    def run_simulation(self):
        logger.info("Starting Pump Motor Sensor Simulator...")
        logger.info(f"Publishing to topic: {Topics.SENSOR_DATA}")
        
        try:
            iteration = 0
            while True:
                sensor_data = self.simulate_reading()
                
                success = self.mqtt_client.publish(Topics.SENSOR_DATA, sensor_data)
                
                if success:
                    logger.info(f"Reading #{iteration}: Vibration={sensor_data['vibration_rms']:.2f} m/s² | "
                              f"Bearing_temp={sensor_data['bearing_temp']:.1f}°C | "
                              f"Current={sensor_data['current_a']:.1f}A")
                else:
                    logger.error(f"Failed to publish reading #{iteration}")
                
                iteration += 1
                time.sleep(self.sampling_interval)
                
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}")

def main():
    mqtt_client = MQTTClient(
        broker_host="broker.hivemq.com",
        broker_port=1883,
        client_id="pump_sensor_simple"
    )
    
    if not mqtt_client.connect():
        logger.error("Failed to connect to MQTT broker")
        return
    
    logger.info("Connected to MQTT broker")
    
    simulator = SimpleSensorSimulator(mqtt_client)
    simulator.run_simulation()

if __name__ == "__main__":
    main()
