import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mqtt_client import MQTTClient, Topics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleEdgeProcessor:
    def __init__(self, mqtt_client: MQTTClient):
        self.mqtt_client = mqtt_client
        self.processing_count = 0
        
        self.thresholds = {
            'vibration_rms': 5.0,
            'bearing_temp': 85.0,
            'winding_temp': 130.0,
            'current_a': 72.0,
        }
    
    def calculate_health_index(self, data: Dict[str, Any]) -> float:
        health = 100.0
        
        if data['vibration_rms'] > self.thresholds['vibration_rms']:
            health -= (data['vibration_rms'] - self.thresholds['vibration_rms']) * 5
        
        if data['bearing_temp'] > self.thresholds['bearing_temp']:
            health -= (data['bearing_temp'] - self.thresholds['bearing_temp']) * 2
        
        if data['winding_temp'] > self.thresholds['winding_temp']:
            health -= (data['winding_temp'] - self.thresholds['winding_temp']) * 1.5
        
        if data['current_a'] > self.thresholds['current_a']:
            health -= (data['current_a'] - self.thresholds['current_a']) * 3
        
        return max(0, min(100, health))
    
    def check_alerts(self, data: Dict[str, Any]) -> List[str]:
        alerts = []
        
        if data['vibration_rms'] > self.thresholds['vibration_rms']:
            alerts.append(f"CRITICAL: Vibration {data['vibration_rms']:.2f} m/s²")
        
        if data['bearing_temp'] > self.thresholds['bearing_temp']:
            alerts.append(f"CRITICAL: Bearing temp {data['bearing_temp']:.1f}°C")
        
        if data['current_a'] > self.thresholds['current_a']:
            alerts.append(f"WARNING: Current {data['current_a']:.1f}A")
        
        return alerts
    
    def process_sensor_data(self, topic: str, data: Dict[str, Any]):
        try:
            self.processing_count += 1
            
            health_index = self.calculate_health_index(data)
            alerts = self.check_alerts(data)
            
            processed_data = {
                'timestamp': data.get('ts'),
                'health_index': round(health_index, 1),
                'vibration_rms': data['vibration_rms'],
                'bearing_temp': data['bearing_temp'],
                'current_a': data['current_a'],
                'alerts': alerts
            }
            
            self.mqtt_client.publish(Topics.TWIN_STATE, processed_data)
            
            if alerts:
                for alert in alerts:
                    logger.warning(alert)
            
            logger.info(f"Processing #{self.processing_count} | HI={health_index:.1f}% | "
                       f"Vibration={data['vibration_rms']:.2f} m/s² | Bearing={data['bearing_temp']:.1f}°C")
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
    
    def start_processing(self):
        logger.info("Starting Edge Processor...")
        logger.info(f"Subscribing to topic: {Topics.SENSOR_DATA}")
        
        success = self.mqtt_client.subscribe(Topics.SENSOR_DATA, self.process_sensor_data)
        
        if success:
            logger.info("Edge Processor started successfully")
            logger.info("Waiting for sensor data...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Edge Processor stopped by user")
        else:
            logger.error("Failed to subscribe to sensor data topic")

def main():
    mqtt_client = MQTTClient(
        broker_host="broker.hivemq.com",
        broker_port=1883,
        client_id="pump_edge_simple"
    )
    
    if not mqtt_client.connect():
        logger.error("Failed to connect to MQTT broker")
        return
    
    logger.info("Connected to MQTT broker")
    
    processor = SimpleEdgeProcessor(mqtt_client)
    processor.start_processing()

if __name__ == "__main__":
    main()
