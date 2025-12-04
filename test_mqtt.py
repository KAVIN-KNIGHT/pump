"""
MQTT Connection Test for IIoT Pump Motor System
Tests MQTT connectivity and basic pub/sub functionality
"""

import time
import json
import logging
from datetime import datetime, timezone
import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.mqtt_client import MQTTClient, Topics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MQTTTester:
    """Test MQTT connectivity and messaging functionality"""
    
    def __init__(self):
        self.messages_received = []
        self.test_results = {}
    
    def test_message_callback(self, topic: str, data):
        """Callback for received test messages"""
        logger.info(f"Received test message on topic '{topic}': {data}")
        self.messages_received.append({
            'topic': topic,
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    def run_connectivity_test(self) -> bool:
        """Test basic MQTT broker connectivity"""
        logger.info("=== Testing MQTT Broker Connectivity ===")
        
        try:
            # Test connection to public broker
            client = MQTTClient(
                broker_host="broker.hivemq.com",
                broker_port=1883,
                client_id="mqtt_tester"
            )
            
            logger.info("Attempting to connect to broker.hivemq.com...")
            success = client.connect(timeout=15)
            
            if success:
                logger.info("✅ Successfully connected to MQTT broker")
                client.disconnect()
                self.test_results['connectivity'] = True
                return True
            else:
                logger.error("❌ Failed to connect to MQTT broker")
                self.test_results['connectivity'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection test failed with error: {e}")
            self.test_results['connectivity'] = False
            return False
    
    def run_publish_test(self) -> bool:
        """Test MQTT publishing functionality"""
        logger.info("=== Testing MQTT Publishing ===")
        
        try:
            client = MQTTClient(
                broker_host="broker.hivemq.com",
                broker_port=1883,
                client_id="mqtt_publisher_test"
            )
            
            if not client.connect():
                logger.error("❌ Failed to connect for publish test")
                return False
            
            # Test publishing to each topic
            test_topics = [
                Topics.SENSOR_DATA,
                Topics.EDGE_ALERTS,
                Topics.TWIN_STATE,
                Topics.TWIN_COMMANDS
            ]
            
            publish_success = True
            
            for topic in test_topics:
                test_message = {
                    'test': True,
                    'topic': topic,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'message': f'Test message for {topic}'
                }
                
                logger.info(f"Publishing test message to '{topic}'...")
                success = client.publish(topic, test_message)
                
                if success:
                    logger.info(f"✅ Successfully published to '{topic}'")
                else:
                    logger.error(f"❌ Failed to publish to '{topic}'")
                    publish_success = False
                
                time.sleep(1)  # Small delay between publishes
            
            client.disconnect()
            self.test_results['publish'] = publish_success
            return publish_success
            
        except Exception as e:
            logger.error(f"❌ Publish test failed with error: {e}")
            self.test_results['publish'] = False
            return False
    
    def run_subscribe_test(self) -> bool:
        """Test MQTT subscription functionality"""
        logger.info("=== Testing MQTT Subscription ===")
        
        try:
            # Publisher client
            pub_client = MQTTClient(
                broker_host="broker.hivemq.com",
                broker_port=1883,
                client_id="mqtt_pub_test"
            )
            
            # Subscriber client
            sub_client = MQTTClient(
                broker_host="broker.hivemq.com",
                broker_port=1883,
                client_id="mqtt_sub_test"
            )
            
            if not pub_client.connect() or not sub_client.connect():
                logger.error("❌ Failed to connect clients for subscribe test")
                return False
            
            # Subscribe to test topic
            test_topic = "pump/motor/test"
            logger.info(f"Subscribing to test topic '{test_topic}'...")
            
            sub_success = sub_client.subscribe(test_topic, self.test_message_callback)
            
            if not sub_success:
                logger.error(f"❌ Failed to subscribe to '{test_topic}'")
                return False
            
            logger.info("✅ Successfully subscribed")
            
            # Wait a moment for subscription to be established
            time.sleep(2)
            
            # Publish test message
            test_message = {
                'test_type': 'subscribe_test',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'This is a subscription test message'
            }
            
            logger.info("Publishing test message...")
            pub_success = pub_client.publish(test_topic, test_message)
            
            if not pub_success:
                logger.error("❌ Failed to publish test message")
                return False
            
            # Wait for message to be received
            logger.info("Waiting for message to be received...")
            time.sleep(3)
            
            # Check if message was received
            if self.messages_received:
                logger.info("✅ Successfully received subscribed message")
                sub_client.disconnect()
                pub_client.disconnect()
                self.test_results['subscribe'] = True
                return True
            else:
                logger.error("❌ No message received (subscription may have failed)")
                sub_client.disconnect()
                pub_client.disconnect()
                self.test_results['subscribe'] = False
                return False
                
        except Exception as e:
            logger.error(f"❌ Subscribe test failed with error: {e}")
            self.test_results['subscribe'] = False
            return False
    
    def run_system_topics_test(self) -> bool:
        """Test all system topics used by the pump motor system"""
        logger.info("=== Testing System Topics ===")
        
        try:
            client = MQTTClient(
                broker_host="broker.hivemq.com",
                broker_port=1883,
                client_id="system_topic_test"
            )
            
            if not client.connect():
                logger.error("❌ Failed to connect for system topics test")
                return False
            
            # Test all system topics
            system_topics = {
                Topics.SENSOR_DATA: "Sensor simulator → Edge processor",
                Topics.EDGE_ALERTS: "Edge processor → Notification systems",
                Topics.TWIN_STATE: "Edge processor → Digital twin",
                Topics.TWIN_COMMANDS: "Digital twin → Control systems"
            }
            
            all_success = True
            
            for topic, description in system_topics.items():
                logger.info(f"Testing topic '{topic}' ({description})")
                
                test_payload = {
                    'test': True,
                    'topic_name': topic,
                    'description': description,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                success = client.publish(topic, test_payload)
                
                if success:
                    logger.info(f"✅ Topic '{topic}' - OK")
                else:
                    logger.error(f"❌ Topic '{topic}' - FAILED")
                    all_success = False
                
                time.sleep(0.5)
            
            client.disconnect()
            self.test_results['system_topics'] = all_success
            return all_success
            
        except Exception as e:
            logger.error(f"❌ System topics test failed: {e}")
            self.test_results['system_topics'] = False
            return False
    
    def run_performance_test(self) -> bool:
        """Test MQTT performance with multiple messages"""
        logger.info("=== Testing MQTT Performance ===")
        
        try:
            client = MQTTClient(
                broker_host="broker.hivemq.com",
                broker_port=1883,
                client_id="perf_test"
            )
            
            if not client.connect():
                logger.error("❌ Failed to connect for performance test")
                return False
            
            # Send multiple messages quickly
            num_messages = 10
            topic = "pump/motor/perf_test"
            
            logger.info(f"Sending {num_messages} messages for performance test...")
            
            start_time = time.time()
            success_count = 0
            
            for i in range(num_messages):
                message = {
                    'test': 'performance',
                    'message_id': i + 1,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                success = client.publish(topic, message)
                if success:
                    success_count += 1
                
                time.sleep(0.1)  # Small delay to avoid overwhelming
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"Performance test completed:")
            logger.info(f"  Messages sent: {success_count}/{num_messages}")
            logger.info(f"  Success rate: {(success_count/num_messages)*100:.1f}%")
            logger.info(f"  Duration: {duration:.2f} seconds")
            logger.info(f"  Messages/second: {success_count/duration:.1f}")
            
            client.disconnect()
            
            # Consider test successful if >90% messages sent
            performance_success = (success_count / num_messages) >= 0.9
            self.test_results['performance'] = performance_success
            
            if performance_success:
                logger.info("✅ Performance test - PASSED")
            else:
                logger.error("❌ Performance test - FAILED (low success rate)")
            
            return performance_success
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            self.test_results['performance'] = False
            return False
    
    def print_test_summary(self):
        """Print summary of all test results"""
        logger.info("=" * 50)
        logger.info("MQTT TEST SUMMARY")
        logger.info("=" * 50)
        
        all_passed = True
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            emoji = "✅" if result else "❌"
            logger.info(f"{emoji} {test_name.upper()}: {status}")
            
            if not result:
                all_passed = False
        
        logger.info("=" * 50)
        
        if all_passed:
            logger.info("🎉 ALL TESTS PASSED - MQTT system ready!")
        else:
            logger.error("⚠️  Some tests failed - check configuration and network connectivity")
        
        logger.info("=" * 50)
        
        return all_passed

def main():
    """Run all MQTT tests"""
    logger.info("Starting MQTT System Tests for Pump Motor IIoT System")
    logger.info("=" * 60)
    
    tester = MQTTTester()
    
    # Run all tests
    tests = [
        ("Connectivity Test", tester.run_connectivity_test),
        ("Publish Test", tester.run_publish_test),
        ("Subscribe Test", tester.run_subscribe_test),
        ("System Topics Test", tester.run_system_topics_test),
        ("Performance Test", tester.run_performance_test)
    ]
    
    for test_name, test_func in tests:
        logger.info("")
        try:
            test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            tester.test_results[test_name.lower().replace(' ', '_')] = False
        
        time.sleep(1)  # Brief pause between tests
    
    # Print final summary
    logger.info("")
    all_passed = tester.print_test_summary()
    
    if all_passed:
        logger.info("")
        logger.info("System is ready! You can now run:")
        logger.info("1. python sensor/pump_motor_sensor_sim.py")
        logger.info("2. python edge/edge_processor.py") 
        logger.info("3. python digital_twin/twin_model.py")
        logger.info("4. python viz/thingspeak_uploader.py")

if __name__ == "__main__":
    main()