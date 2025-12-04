"""
ThingSpeak Data Uploader for IIoT Pump Motor System
Uploads processed metrics to ThingSpeak for cloud visualization
"""

import requests
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mqtt_client import MQTTClient, Topics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ThingSpeakUploader:
    """Upload data to ThingSpeak cloud platform"""
    
    def __init__(self, api_key: str, channel_id: str):
        """
        Initialize ThingSpeak uploader
        
        Args:
            api_key: ThingSpeak Write API Key
            channel_id: ThingSpeak Channel ID
        """
        self.api_key = api_key
        self.channel_id = channel_id
        self.base_url = "https://api.thingspeak.com/update"
        
        # Rate limiting (ThingSpeak allows 1 update per 15 seconds for free accounts)
        self.min_upload_interval = 16  # seconds
        self.last_upload_time = 0
        
        # Data buffer for aggregation
        self.data_buffer = []
        self.buffer_size = 5  # Average over 5 readings before upload
        
    def can_upload(self) -> bool:
        """Check if enough time has passed since last upload"""
        return (time.time() - self.last_upload_time) >= self.min_upload_interval
    
    def upload_data(self, data: Dict[str, Any]) -> bool:
        """
        Upload data to ThingSpeak
        
        Args:
            data: Dictionary containing sensor/analytics data
            
        Returns:
            True if upload successful, False otherwise
        """
        if not self.can_upload():
            logger.debug("Upload rate limited, skipping...")
            return False
        
        try:
            # Prepare ThingSpeak payload
            # ThingSpeak supports up to 8 fields per channel
            payload = {
                'api_key': self.api_key,
                'field1': data.get('health_index', 0),          # Overall Health Index
                'field2': data.get('vibration_rms', 0),         # Vibration RMS
                'field3': data.get('bearing_temp', 0),          # Bearing Temperature  
                'field4': data.get('current_a', 0),             # Motor Current
                'field5': data.get('flow_lpm', 0),              # Flow Rate
                'field6': data.get('pressure_bar', 0),          # Pressure
                'field7': data.get('rul_hours', 0),             # Remaining Useful Life
                'field8': data.get('alert_count', 0)            # Number of active alerts
            }
            
            # Send HTTP POST request
            response = requests.post(self.base_url, data=payload, timeout=10)
            
            if response.status_code == 200:
                entry_id = response.text.strip()
                if entry_id != '0':  # ThingSpeak returns 0 on error
                    logger.info(f"Data uploaded to ThingSpeak (Entry ID: {entry_id})")
                    self.last_upload_time = time.time()
                    return True
                else:
                    logger.error("ThingSpeak rejected data (returned 0)")
                    return False
            else:
                logger.error(f"ThingSpeak upload failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error uploading to ThingSpeak: {e}")
            return False
        except Exception as e:
            logger.error(f"Error uploading to ThingSpeak: {e}")
            return False
    
    def buffer_and_upload(self, raw_data: Dict[str, Any]) -> bool:
        """
        Buffer data and upload averaged values to reduce API calls
        
        Args:
            raw_data: Raw sensor/analytics data
            
        Returns:
            True if data was uploaded, False if buffering
        """
        # Add to buffer
        self.data_buffer.append(raw_data)
        
        # Check if buffer is full or enough time has passed
        if len(self.data_buffer) >= self.buffer_size and self.can_upload():
            # Calculate averages
            avg_data = self._calculate_averages()
            
            # Upload averaged data
            success = self.upload_data(avg_data)
            
            if success:
                # Clear buffer on successful upload
                self.data_buffer.clear()
                return True
            else:
                # Keep buffer if upload failed, but limit size
                if len(self.data_buffer) > self.buffer_size * 2:
                    self.data_buffer = self.data_buffer[-self.buffer_size:]
        
        return False
    
    def _calculate_averages(self) -> Dict[str, float]:
        """Calculate average values from buffered data"""
        if not self.data_buffer:
            return {}
        
        # Initialize sums
        sums = {}
        counts = {}
        
        # Sum all numeric values
        for data_point in self.data_buffer:
            for key, value in data_point.items():
                if isinstance(value, (int, float)):
                    sums[key] = sums.get(key, 0) + value
                    counts[key] = counts.get(key, 0) + 1
        
        # Calculate averages
        averages = {}
        for key, total in sums.items():
            if counts[key] > 0:
                averages[key] = round(total / counts[key], 2)
        
        # Add metadata
        averages['buffer_size'] = len(self.data_buffer)
        averages['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        return averages

class TelegramNotifier:
    """Send alerts and notifications via Telegram bot"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram Bot Token from BotFather
            chat_id: Telegram Chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Rate limiting
        self.min_notification_interval = 300  # 5 minutes between similar alerts
        self.last_notifications = {}
        
    def can_send_notification(self, alert_type: str) -> bool:
        """Check if enough time has passed since last notification of same type"""
        last_time = self.last_notifications.get(alert_type, 0)
        return (time.time() - last_time) >= self.min_notification_interval
    
    def send_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        Send alert notification via Telegram
        
        Args:
            alert_data: Alert information dictionary
            
        Returns:
            True if sent successfully, False otherwise
        """
        alert_type = alert_data.get('type', 'unknown')
        
        # Check rate limiting
        if not self.can_send_notification(alert_type):
            logger.debug(f"Telegram notification rate limited for {alert_type}")
            return False
        
        try:
            # Format alert message
            message = self._format_alert_message(alert_data)
            
            # Prepare Telegram payload
            payload = {
                'chat_id': self.chat_id,
                'text': message
                # Removed parse_mode to avoid Markdown formatting issues
            }
            
            # Send HTTP POST request
            response = requests.post(self.base_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Alert sent to Telegram: {alert_type}")
                self.last_notifications[alert_type] = time.time()
                return True
            else:
                logger.error(f"Telegram send failed: HTTP {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error sending to Telegram: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending to Telegram: {e}")
            return False
    
    def _format_alert_message(self, alert_data: Dict[str, Any]) -> str:
        """Format alert data into readable Telegram message"""
        
        alert = alert_data.get('alert', {})
        timestamp = alert_data.get('timestamp', datetime.now(timezone.utc).isoformat())
        
        # Parse timestamp for display
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            time_str = timestamp
        
        # Determine emoji based on severity
        severity = alert.get('severity', 'info')
        emoji_map = {
            'critical': '🔴',
            'warning': '⚠️',
            'info': 'ℹ️',
            'high': '🟠',
            'medium': '🟡'
        }
        emoji = emoji_map.get(severity, 'ℹ️')
        
        # Build message
        message_parts = [
            f"{emoji} Pump Motor Alert",
            f"",
            f"Severity: {severity.upper()}",
            f"Type: {alert.get('type', 'Unknown')}",
            f"Time: {time_str}",
            f"",
            f"Message: {alert.get('message', 'No details available')}"
        ]
        
        # Add sensor-specific details
        if 'sensor' in alert:
            message_parts.extend([
                f"Sensor: {alert['sensor']}",
                f"Value: {alert.get('value', 'N/A')}"
            ])

        if 'threshold' in alert:
            message_parts.append(f"Threshold: {alert['threshold']}")        # Add fault-specific details
        if alert.get('type') == 'vibration_fault':
            fault_type = alert.get('fault_type', 'unknown')
            message_parts.append(f"*Fault Type:* {fault_type}")
        
        # Add health degradation details
        if 'health_index' in alert:
            message_parts.append(f"*Health Index:* {alert['health_index']:.1f}%")
        
        # Add recommendations if available
        if 'actions' in alert and alert['actions']:
            message_parts.extend([
                f"",
                f"*Recommended Actions:*"
            ])
            for action in alert['actions']:
                message_parts.append(f"• {action}")
        
        return '\n'.join(message_parts)
    
    def send_status_update(self, status_data: Dict[str, Any]) -> bool:
        """
        Send periodic status update
        
        Args:
            status_data: System status information
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            message = self._format_status_message(status_data)
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(self.base_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info("Status update sent to Telegram")
                return True
            else:
                logger.error(f"Telegram status send failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending status to Telegram: {e}")
            return False
    
    def _format_status_message(self, status_data: Dict[str, Any]) -> str:
        """Format status data into readable message"""
        
        timestamp = status_data.get('timestamp', datetime.now(timezone.utc).isoformat())
        
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime('%H:%M:%S UTC')
        except:
            time_str = timestamp
        
        health_index = status_data.get('health_index', 0)
        rul_hours = status_data.get('rul_hours', 0)
        
        # Determine health emoji
        if health_index >= 80:
            health_emoji = '✅'
        elif health_index >= 60:
            health_emoji = '🟡'
        else:
            health_emoji = '🔴'
        
        message_parts = [
            f"{health_emoji} *Pump Motor Status*",
            f"",
            f"*Time:* {time_str}",
            f"*Health Index:* {health_index:.1f}%",
            f"*RUL:* {rul_hours:.0f} hours ({rul_hours/24:.1f} days)",
            f"",
            f"*Sensors:*",
            f"• Vibration: {status_data.get('vibration_rms', 0):.2f} m/s²",
            f"• Bearing Temp: {status_data.get('bearing_temp', 0):.1f}°C",
            f"• Current: {status_data.get('current_a', 0):.1f}A",
            f"• Flow: {status_data.get('flow_lpm', 0):.0f} L/min"
        ]
        
        return '\n'.join(message_parts)

class VisualizationManager:
    """Manage all visualization and notification services"""
    
    def __init__(self, mqtt_client: MQTTClient, config: Dict[str, str]):
        """
        Initialize visualization manager
        
        Args:
            mqtt_client: MQTT client for receiving data
            config: Configuration with API keys and settings
        """
        self.mqtt_client = mqtt_client
        
        # Initialize services
        self.thingspeak = None
        self.telegram = None
        
        # Setup ThingSpeak if configured
        if config.get('thingspeak_api_key') and config.get('thingspeak_channel_id'):
            self.thingspeak = ThingSpeakUploader(
                config['thingspeak_api_key'],
                config['thingspeak_channel_id']
            )
            logger.info("ThingSpeak uploader initialized")
        
        # Setup Telegram if configured
        if config.get('telegram_bot_token') and config.get('telegram_chat_id'):
            self.telegram = TelegramNotifier(
                config['telegram_bot_token'],
                config['telegram_chat_id']
            )
            logger.info("Telegram notifier initialized")
        
        # Processing counters
        self.data_count = 0
        self.alert_count = 0
        
        # Status update interval
        self.status_update_interval = 3600  # 1 hour
        self.last_status_update = 0
    
    def process_twin_data(self, topic: str, data: Dict[str, Any]):
        """Process data from digital twin"""
        try:
            self.data_count += 1
            
            # Extract relevant data for visualization
            if 'current_state' in data:
                state = data['current_state']
                
                # Prepare data for ThingSpeak
                if self.thingspeak:
                    thingspeak_data = {
                        'health_index': state.get('health_index', 0),
                        'vibration_rms': state.get('vibration_rms', 0),
                        'bearing_temp': state.get('bearing_temp', 0),
                        'current_a': state.get('current_a', 0),
                        'flow_lpm': state.get('flow_lpm', 0),
                        'pressure_bar': state.get('pressure_bar', 0),
                        'rul_hours': state.get('rul_hours', 0),
                        'alert_count': len(data.get('recommendations', []))
                    }
                    
                    # Upload to ThingSpeak (with buffering)
                    self.thingspeak.buffer_and_upload(thingspeak_data)
                
                # Send periodic status updates via Telegram
                if (self.telegram and 
                    time.time() - self.last_status_update > self.status_update_interval):
                    
                    status_data = {
                        'timestamp': state.get('timestamp'),
                        'health_index': state.get('health_index', 0),
                        'rul_hours': state.get('rul_hours', 0),
                        'vibration_rms': state.get('vibration_rms', 0),
                        'bearing_temp': state.get('bearing_temp', 0),
                        'current_a': state.get('current_a', 0),
                        'flow_lpm': state.get('flow_lpm', 0)
                    }
                    
                    self.telegram.send_status_update(status_data)
                    self.last_status_update = time.time()
            
            logger.debug(f"Processed twin data #{self.data_count}")
            
        except Exception as e:
            logger.error(f"Error processing twin data: {e}")
    
    def process_alert_data(self, topic: str, data: Dict[str, Any]):
        """Process alert data from edge processor"""
        try:
            self.alert_count += 1
            
            # Send alert via Telegram if configured
            if self.telegram:
                self.telegram.send_alert(data)
            
            logger.info(f"Processed alert #{self.alert_count}: {data.get('alert', {}).get('message', 'Unknown alert')}")
            
        except Exception as e:
            logger.error(f"Error processing alert data: {e}")
    
    def start_visualization_manager(self):
        """Start the visualization manager"""
        logger.info("Starting Visualization Manager...")
        
        # Subscribe to twin data
        success1 = self.mqtt_client.subscribe(Topics.TWIN_STATE, self.process_twin_data)
        
        # Subscribe to alerts
        success2 = self.mqtt_client.subscribe(Topics.EDGE_ALERTS, self.process_alert_data)
        
        if success1 and success2:
            logger.info("Visualization Manager started successfully")
            logger.info("Subscribed to twin data and alerts")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Visualization Manager stopped by user")
            finally:
                if hasattr(self, 'mqtt_client') and self.mqtt_client:
                    self.mqtt_client.disconnect()
        else:
            logger.error("Failed to subscribe to required topics")

def main():
    """Main entry point"""
    
    # Configuration (replace with actual values)
    config = {
        # ThingSpeak configuration (replace with your actual values)
        'thingspeak_api_key': 'FJ0EVUV443VH771F',  # Get from thingspeak.com
        'thingspeak_channel_id': '3170500',
        
        # Telegram configuration (replace with your actual values)  
        'telegram_bot_token': '8158126239:AAGREZX0lRjuBxCbgA6HsuG6n6nxnf45SnU',  # Get from BotFather
        'telegram_chat_id': '5874140142'  # Your actual chat ID
    }
    
    # Configuration validation - ensure Telegram credentials are provided
    if not config['telegram_chat_id'] or not config['telegram_bot_token']:
        logger.warning("Telegram credentials not configured - Telegram notifications will be disabled")
        config['telegram_bot_token'] = ''  # Disable Telegram
        config['telegram_chat_id'] = ''
    
    # Initialize MQTT client
    mqtt_client = MQTTClient(
        broker_host="broker.hivemq.com",
        broker_port=1883,
        client_id="pump_visualization"
    )
    
    # Connect to MQTT broker
    if not mqtt_client.connect():
        logger.error("Failed to connect to MQTT broker")
        return
    
    logger.info("Connected to MQTT broker")
    
    # Create and start visualization manager
    viz_manager = VisualizationManager(mqtt_client, config)
    viz_manager.start_visualization_manager()

if __name__ == "__main__":
    main()