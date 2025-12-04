import requests
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

class SimpleThingSpeakUploader:
    def __init__(self, api_key: str, channel_id: str):
        self.api_key = api_key
        self.channel_id = channel_id
        self.base_url = "https://api.thingspeak.com/update"
        self.last_upload_time = 0
        self.min_interval = 16
        self.data_count = 0
    
    def can_upload(self) -> bool:
        return (time.time() - self.last_upload_time) >= self.min_interval
    
    def upload_data(self, data: Dict[str, Any]) -> bool:
        if not self.can_upload():
            return False
        
        try:
            payload = {
                'api_key': self.api_key,
                'field1': data.get('health_index', 0),
                'field2': data.get('vibration_rms', 0),
                'field3': data.get('bearing_temp', 0),
                'field4': data.get('current_a', 0),
                'field5': data.get('rul_hours', 0),
            }
            
            response = requests.post(self.base_url, data=payload, timeout=10)
            
            if response.status_code == 200 and response.text.strip() != '0':
                self.last_upload_time = time.time()
                logger.info(f"Data uploaded to ThingSpeak (Entry: {response.text.strip()})")
                return True
        except Exception as e:
            logger.error(f"ThingSpeak upload error: {e}")
        
        return False

class SimpleVisualizationManager:
    def __init__(self, mqtt_client: MQTTClient, thingspeak_uploader: SimpleThingSpeakUploader):
        self.mqtt_client = mqtt_client
        self.uploader = thingspeak_uploader
        self.data_count = 0
    
    def process_twin_data(self, topic: str, data: Dict[str, Any]):
        try:
            self.data_count += 1
            
            self.uploader.upload_data(data)
            
            logger.info(f"Processed data #{self.data_count} | "
                       f"HI={data.get('health_index')}% | "
                       f"RUL={data.get('rul_hours')}h")
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
    
    def start_manager(self):
        logger.info("Starting Visualization Manager...")
        logger.info(f"Subscribing to topic: {Topics.TWIN_STATE}")
        
        success = self.mqtt_client.subscribe(Topics.TWIN_STATE, self.process_twin_data)
        
        if success:
            logger.info("Visualization Manager started successfully")
            logger.info("Uploading data to ThingSpeak...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Visualization Manager stopped by user")
        else:
            logger.error("Failed to subscribe to twin data topic")

def main():
    mqtt_client = MQTTClient(
        broker_host="broker.hivemq.com",
        broker_port=1883,
        client_id="pump_viz_simple"
    )
    
    if not mqtt_client.connect():
        logger.error("Failed to connect to MQTT broker")
        return
    
    logger.info("Connected to MQTT broker")
    
    thingspeak = SimpleThingSpeakUploader(
        api_key="FJ0EVUV443VH771F",
        channel_id="3170500"
    )
    
    manager = SimpleVisualizationManager(mqtt_client, thingspeak)
    manager.start_manager()

if __name__ == "__main__":
    main()
