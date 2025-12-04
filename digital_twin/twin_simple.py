import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mqtt_client import MQTTClient, Topics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleDigitalTwin:
    def __init__(self, mqtt_client: MQTTClient):
        self.mqtt_client = mqtt_client
        self.update_count = 0
        self.current_state = None
    
    def predict_rul(self, health_index: float) -> float:
        if health_index > 80:
            rul = 8760
        elif health_index > 60:
            rul = 2000
        elif health_index > 40:
            rul = 500
        elif health_index > 20:
            rul = 100
        else:
            rul = 10
        
        return rul
    
    def process_edge_data(self, topic: str, data: Dict[str, Any]):
        try:
            self.update_count += 1
            
            health_index = data.get('health_index', 100)
            rul_hours = self.predict_rul(health_index)
            
            twin_state = {
                'timestamp': data.get('timestamp'),
                'health_index': health_index,
                'rul_hours': round(rul_hours, 1),
                'vibration_rms': data.get('vibration_rms'),
                'bearing_temp': data.get('bearing_temp'),
                'current_a': data.get('current_a'),
                'alerts': data.get('alerts', [])
            }
            
            self.current_state = twin_state
            self.mqtt_client.publish(Topics.TWIN_STATE, twin_state)
            
            logger.info(f"Twin Update #{self.update_count} | HI={health_index:.1f}% | RUL={rul_hours:.1f}h")
            
        except Exception as e:
            logger.error(f"Error processing twin data: {e}")
    
    def start_twin(self):
        logger.info("Starting Digital Twin...")
        logger.info(f"Subscribing to topic: {Topics.TWIN_STATE}")
        
        success = self.mqtt_client.subscribe(Topics.TWIN_STATE, self.process_edge_data)
        
        if success:
            logger.info("Digital Twin started successfully")
            logger.info("Waiting for edge data...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Digital Twin stopped by user")
        else:
            logger.error("Failed to subscribe to edge data topic")

def main():
    mqtt_client = MQTTClient(
        broker_host="broker.hivemq.com",
        broker_port=1883,
        client_id="pump_twin_simple"
    )
    
    if not mqtt_client.connect():
        logger.error("Failed to connect to MQTT broker")
        return
    
    logger.info("Connected to MQTT broker")
    
    twin = SimpleDigitalTwin(mqtt_client)
    twin.start_twin()

if __name__ == "__main__":
    main()
