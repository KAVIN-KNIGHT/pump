"""
MQTT Client Utility for IIoT Pump Motor System
Provides reusable MQTT connection and messaging functionality
"""

import paho.mqtt.client as mqtt
import json
import logging
import time
from typing import Callable, Optional, Dict, Any

class MQTTClient:
    def __init__(self, broker_host: str = "broker.hivemq.com", broker_port: int = 1883, 
                 client_id: str = "pump_motor_client"):
        """
        Initialize MQTT Client for pump motor system
        
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port 
            client_id: Unique client identifier
        """
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        # Create MQTT client with proper callback API version for paho-mqtt 2.x
        try:
            self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
        except (TypeError, AttributeError):
            # Fallback for older versions
            self.client = mqtt.Client(client_id=client_id)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Connection callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # Store message callbacks for different topics
        self.message_callbacks: Dict[str, Callable] = {}
        
        # Connection status
        self.connected = False
        
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when client connects to broker"""
        if rc == 0:
            self.connected = True
            self.logger.info(f"Connected to MQTT broker {self.broker_host}:{self.broker_port}")
        else:
            self.connected = False
            self.logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for when client disconnects from broker"""
        self.connected = False
        self.logger.info("Disconnected from MQTT broker")
        
    def _on_message(self, client, userdata, msg):
        """Generic message handler that routes to topic-specific callbacks"""
        topic = msg.topic
        payload = msg.payload.decode('utf-8')
        
        self.logger.debug(f"Received message on topic '{topic}': {payload}")
        
        # Route to specific callback if registered
        if topic in self.message_callbacks:
            try:
                # Try to parse as JSON, fallback to raw string
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    data = payload
                    
                self.message_callbacks[topic](topic, data)
            except Exception as e:
                self.logger.error(f"Error processing message on topic '{topic}': {e}")
        else:
            self.logger.warning(f"No callback registered for topic: {topic}")
    
    def connect(self, timeout: int = 15, max_retries: int = 3) -> bool:
        """
        Connect to MQTT broker with retry logic
        
        Args:
            timeout: Connection timeout in seconds
            max_retries: Maximum number of connection attempts
            
        Returns:
            True if connected successfully, False otherwise
        """
        # List of alternative MQTT brokers to try
        brokers = [
            ("broker.hivemq.com", 1883),
            ("test.mosquitto.org", 1883),
            ("broker.emqx.io", 1883),
            ("mqtt.eclipseprojects.io", 1883)
        ]
        
        for retry_count in range(max_retries):
            for broker_host, broker_port in brokers:
                try:
                    self.logger.info(f"Attempting to connect to {broker_host}:{broker_port} (attempt {retry_count + 1})")
                    
                    # Update broker info if different from default
                    if broker_host != self.broker_host:
                        self.broker_host = broker_host
                        self.broker_port = broker_port
                    
                    self.client.connect(self.broker_host, self.broker_port, timeout)
                    self.client.loop_start()
                    
                    # Wait for connection to be established
                    start_time = time.time()
                    while not self.connected and (time.time() - start_time) < timeout:
                        time.sleep(0.2)
                    
                    if self.connected:
                        self.logger.info(f"Successfully connected to {self.broker_host}:{self.broker_port}")
                        return True
                    else:
                        self.logger.warning(f"Connection timeout to {broker_host}:{broker_port}")
                        self.client.loop_stop()
                        
                except Exception as e:
                    self.logger.warning(f"Failed to connect to {broker_host}:{broker_port}: {e}")
                    try:
                        self.client.loop_stop()
                    except:
                        pass
                    
                # Wait before trying next broker
                time.sleep(2)
            
            # Wait before retry
            if retry_count < max_retries - 1:
                self.logger.info(f"Retrying connection in 3 seconds...")
                time.sleep(3)
        
        self.logger.error("Failed to connect to any MQTT broker after all attempts")
        return False
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
    
    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0) -> bool:
        """
        Publish message to MQTT topic
        
        Args:
            topic: MQTT topic to publish to
            payload: Data to publish (will be JSON-encoded)
            qos: Quality of Service level (0, 1, or 2)
            
        Returns:
            True if published successfully, False otherwise
        """
        if not self.connected:
            self.logger.error("Cannot publish - not connected to MQTT broker")
            return False
        
        try:
            json_payload = json.dumps(payload)
            result = self.client.publish(topic, json_payload, qos)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.debug(f"Published to topic '{topic}': {json_payload}")
                return True
            else:
                self.logger.error(f"Failed to publish to topic '{topic}'. Return code: {result.rc}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error publishing to topic '{topic}': {e}")
            return False
    
    def subscribe(self, topic: str, callback: Callable, qos: int = 0) -> bool:
        """
        Subscribe to MQTT topic with callback
        
        Args:
            topic: MQTT topic to subscribe to
            callback: Function to call when message received (topic, data)
            qos: Quality of Service level
            
        Returns:
            True if subscribed successfully, False otherwise
        """
        if not self.connected:
            self.logger.error("Cannot subscribe - not connected to MQTT broker")
            return False
        
        try:
            # Store callback for this topic
            self.message_callbacks[topic] = callback
            
            # Subscribe to topic
            result = self.client.subscribe(topic, qos)
            
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                self.logger.info(f"Subscribed to topic '{topic}' with QoS {qos}")
                return True
            else:
                self.logger.error(f"Failed to subscribe to topic '{topic}'. Return code: {result[0]}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error subscribing to topic '{topic}': {e}")
            return False
    
    def unsubscribe(self, topic: str) -> bool:
        """
        Unsubscribe from MQTT topic
        
        Args:
            topic: MQTT topic to unsubscribe from
            
        Returns:
            True if unsubscribed successfully, False otherwise
        """
        if not self.connected:
            return False
            
        try:
            result = self.client.unsubscribe(topic)
            
            if result[0] == mqtt.MQTT_ERR_SUCCESS:
                # Remove callback
                if topic in self.message_callbacks:
                    del self.message_callbacks[topic]
                    
                self.logger.info(f"Unsubscribed from topic '{topic}'")
                return True
            else:
                self.logger.error(f"Failed to unsubscribe from topic '{topic}'. Return code: {result[0]}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing from topic '{topic}': {e}")
            return False

# MQTT Topics Configuration
class Topics:
    """MQTT topic definitions for the pump motor system"""
    SENSOR_DATA = "pump/motor/sensor"
    EDGE_ALERTS = "pump/motor/edge/alerts"  
    TWIN_STATE = "pump/motor/twin/state"
    TWIN_COMMANDS = "pump/motor/twin/commands"
    CONTROL_COMMANDS = "pump/motor/control/commands"