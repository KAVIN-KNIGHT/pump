"""
Edge Processor for IIoT Pump Motor System
Provides real-time data processing, threshold monitoring, and anomaly detection
"""

import json
import time
import math
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mqtt_client import MQTTClient, Topics
from utils.signal_processing import RollingStatistics, AnomalyDetector, VibrationAnalyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EdgeProcessor:
    """Main edge processing class for pump motor monitoring"""
    
    def __init__(self, mqtt_client: MQTTClient):
        self.mqtt_client = mqtt_client
        
        # Processing components
        self.rolling_stats = RollingStatistics(window_size=10)
        self.anomaly_detector = AnomalyDetector(threshold_multiplier=3.0)
        self.vibration_analyzer = VibrationAnalyzer(sampling_rate=1000.0)
        
        # Alert thresholds
        self.thresholds = {
            'vibration_rms': {'warning': 3.0, 'critical': 5.0},
            'bearing_temp': {'warning': 75.0, 'critical': 85.0},
            'winding_temp': {'warning': 100.0, 'critical': 130.0},
            'current_a': {'warning': 66.0, 'critical': 72.0},  # 110% and 120% of rated
            'flow_lpm': {'warning': 800.0, 'critical': 600.0},  # Low flow thresholds
            'pressure_bar': {'warning': 4.0, 'critical': 3.0},  # Low pressure thresholds
            'oil_level_pct': {'warning': 40.0, 'critical': 25.0}  # Low oil thresholds
        }
        
        # Processing state
        self.last_reading_time = None
        self.processing_count = 0
        self.alert_count = 0
        
        # Vibration analysis buffer
        self.vibration_history = []
        self.vibration_buffer_size = 50
        
        # Health calculation weights
        self.health_weights = {
            'vibration': 0.35,
            'bearing_temp': 0.25,
            'current': 0.15,
            'flow_pressure': 0.15,
            'oil_rpm': 0.10
        }
        
    def process_sensor_data(self, topic: str, data: Dict[str, Any]):
        """Main processing function for incoming sensor data"""
        try:
            self.processing_count += 1
            processing_start = time.time()
            
            # Validate and parse sensor data
            if not self._validate_sensor_data(data):
                logger.error(f"Invalid sensor data received: {data}")
                return
            
            logger.debug(f"Processing reading #{self.processing_count}")
            
            # Update rolling statistics for all sensors
            sensor_stats = self._update_rolling_statistics(data)
            
            # Perform threshold checks
            threshold_alerts = self._check_thresholds(data, sensor_stats)
            
            # Anomaly detection
            anomaly_results = self._detect_anomalies(data)
            
            # Vibration spectral analysis
            vibration_analysis = self._analyze_vibration(data)
            
            # Calculate health metrics
            health_metrics = self._calculate_health_metrics(data, sensor_stats, vibration_analysis)
            
            # Generate processed data package
            processed_data = {
                'timestamp': data.get('ts', datetime.now(timezone.utc).isoformat()),
                'raw_sensors': data,
                'rolling_stats': sensor_stats,
                'threshold_alerts': threshold_alerts,
                'anomalies': anomaly_results,
                'vibration_analysis': vibration_analysis,
                'health_metrics': health_metrics,
                'processing_time_ms': round((time.time() - processing_start) * 1000, 2)
            }
            
            # Check for alerts and notifications
            alerts = self._generate_alerts(processed_data)
            
            if alerts:
                # Display formatted alert if critical conditions detected
                health_index = processed_data['health_metrics']['overall_health_index']
                if health_index < 60 or any(a.get('severity') == 'critical' for a in alerts):
                    self._display_formatted_alert(data, health_index, alerts)
                
                self._send_alerts(alerts)
            
            # Send processed data to digital twin
            self._send_to_digital_twin(processed_data)
            
            # Log processing results
            self._log_processing_results(processed_data)
            
            # Update timing
            self.last_reading_time = time.time()
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
    
    def _validate_sensor_data(self, data: Dict[str, Any]) -> bool:
        """Validate incoming sensor data format and ranges"""
        required_fields = [
            'ts', 'vibration_rms', 'bearing_temp', 'winding_temp', 
            'current_a', 'voltage_v', 'flow_lpm', 'pressure_bar',
            'rpm', 'oil_level_pct', 'seal_leak'
        ]
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Basic range validation
        if data['vibration_rms'] < 0 or data['vibration_rms'] > 50:
            logger.warning(f"Vibration RMS out of expected range: {data['vibration_rms']}")
        
        if data['bearing_temp'] < 0 or data['bearing_temp'] > 200:
            logger.warning(f"Bearing temperature out of range: {data['bearing_temp']}")
        
        if data['current_a'] < 0 or data['current_a'] > 300:
            logger.warning(f"Current out of range: {data['current_a']}")
        
        return True
    
    def _update_rolling_statistics(self, data: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Update rolling statistics for all sensors"""
        sensor_stats = {}
        
        numeric_sensors = [
            'vibration_rms', 'bearing_temp', 'winding_temp', 'current_a',
            'flow_lpm', 'pressure_bar', 'rpm', 'oil_level_pct'
        ]
        
        for sensor in numeric_sensors:
            if sensor in data:
                stats = self.rolling_stats.update(sensor, data[sensor])
                sensor_stats[sensor] = stats
        
        return sensor_stats
    
    def _check_thresholds(self, data: Dict[str, Any], 
                         stats: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Check sensor values against defined thresholds"""
        alerts = []
        
        for sensor, thresholds in self.thresholds.items():
            if sensor in data:
                value = data[sensor]
                
                # For low-value alerts (flow, pressure, oil), check if below threshold
                if sensor in ['flow_lpm', 'pressure_bar', 'oil_level_pct']:
                    if value <= thresholds['critical']:
                        alerts.append({
                            'type': 'threshold',
                            'severity': 'critical',
                            'sensor': sensor,
                            'value': value,
                            'threshold': thresholds['critical'],
                            'message': f'{sensor} critically low: {value}'
                        })
                    elif value <= thresholds['warning']:
                        alerts.append({
                            'type': 'threshold',
                            'severity': 'warning',
                            'sensor': sensor,
                            'value': value,
                            'threshold': thresholds['warning'],
                            'message': f'{sensor} low: {value}'
                        })
                
                # For high-value alerts, check if above threshold
                else:
                    if value >= thresholds['critical']:
                        alerts.append({
                            'type': 'threshold',
                            'severity': 'critical',
                            'sensor': sensor,
                            'value': value,
                            'threshold': thresholds['critical'],
                            'message': f'{sensor} critically high: {value}'
                        })
                    elif value >= thresholds['warning']:
                        alerts.append({
                            'type': 'threshold',
                            'severity': 'warning',
                            'sensor': sensor,
                            'value': value,
                            'threshold': thresholds['warning'],
                            'message': f'{sensor} high: {value}'
                        })
        
        return alerts
    
    def _detect_anomalies(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Perform statistical anomaly detection on sensor readings"""
        anomaly_results = {}
        
        numeric_sensors = [
            'vibration_rms', 'bearing_temp', 'winding_temp', 'current_a',
            'flow_lpm', 'pressure_bar', 'rpm', 'oil_level_pct'
        ]
        
        for sensor in numeric_sensors:
            if sensor in data:
                result = self.anomaly_detector.detect_anomaly(sensor, data[sensor])
                if result['is_anomaly']:
                    anomaly_results[sensor] = result
        
        return anomaly_results
    
    def _analyze_vibration(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vibration data for spectral characteristics"""
        if 'vibration_rms' not in data or 'rpm' not in data:
            return {}
        
        # Update vibration history
        self.vibration_history.append(data['vibration_rms'])
        if len(self.vibration_history) > self.vibration_buffer_size:
            self.vibration_history.pop(0)
        
        # Need sufficient data for meaningful analysis
        if len(self.vibration_history) < 20:
            return {'status': 'insufficient_data', 'samples': len(self.vibration_history)}
        
        try:
            # Update analyzer RPM
            self.vibration_analyzer.rpm = data['rpm']
            self.vibration_analyzer.shaft_freq = data['rpm'] / 60.0
            
            # Analyze spectrum
            analysis_result = self.vibration_analyzer.analyze_spectrum(self.vibration_history)
            
            # Add fault detection logic based on spectral features
            fault_indicators = self._detect_vibration_faults(analysis_result)
            analysis_result['fault_indicators'] = fault_indicators
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in vibration analysis: {e}")
            return {'status': 'analysis_error', 'error': str(e)}
    
    def _detect_vibration_faults(self, vibration_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect specific faults from vibration spectral analysis"""
        if not vibration_analysis or 'shaft_harmonics' not in vibration_analysis:
            return {}
        
        fault_indicators = {}
        shaft_harmonics = vibration_analysis['shaft_harmonics']
        bearing_freqs = vibration_analysis.get('bearing_frequencies', {})
        
        # Imbalance detection (high 1x shaft frequency)
        if '1x_shaft' in shaft_harmonics:
            if shaft_harmonics['1x_shaft'] > 2.0:  # Threshold for imbalance
                fault_indicators['imbalance'] = {
                    'detected': True,
                    'severity': 'high' if shaft_harmonics['1x_shaft'] > 4.0 else 'medium',
                    'amplitude': shaft_harmonics['1x_shaft']
                }
        
        # Misalignment detection (high 2x shaft frequency)
        if '2x_shaft' in shaft_harmonics:
            if shaft_harmonics['2x_shaft'] > 1.5:
                fault_indicators['misalignment'] = {
                    'detected': True,
                    'severity': 'high' if shaft_harmonics['2x_shaft'] > 3.0 else 'medium',
                    'amplitude': shaft_harmonics['2x_shaft']
                }
        
        # Bearing fault detection
        bearing_fault_detected = False
        for freq_type, amplitude in bearing_freqs.items():
            if amplitude > 1.0:  # Threshold for bearing defect
                bearing_fault_detected = True
                break
        
        if bearing_fault_detected:
            fault_indicators['bearing_defect'] = {
                'detected': True,
                'severity': 'high' if max(bearing_freqs.values()) > 2.0 else 'medium',
                'frequencies': bearing_freqs
            }
        
        return fault_indicators
    
    def _calculate_health_metrics(self, data: Dict[str, Any], stats: Dict[str, Dict[str, float]],
                                vibration_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive health metrics"""
        
        # Individual component health scores (0-100)
        component_health = {}
        
        # Vibration health (based on RMS and spectral analysis)
        vib_rms = data.get('vibration_rms', 0)
        vib_health = max(0, min(100, 100 - (vib_rms - 0.5) * 20))  # Scale from 0.5-5.0 range
        
        # Add penalty for spectral faults
        if vibration_analysis.get('fault_indicators'):
            fault_penalty = len(vibration_analysis['fault_indicators']) * 15
            vib_health = max(0, vib_health - fault_penalty)
        
        component_health['vibration'] = vib_health
        
        # Bearing temperature health
        bearing_temp = data.get('bearing_temp', 0)
        bearing_health = max(0, min(100, 100 - (bearing_temp - 40) * 1.5))  # Scale from 40-100°C
        component_health['bearing_temp'] = bearing_health
        
        # Winding temperature health  
        winding_temp = data.get('winding_temp', 0)
        winding_health = max(0, min(100, 100 - (winding_temp - 45) * 1.1))  # Scale from 45-135°C
        component_health['winding_temp'] = winding_health
        
        # Current health (deviation from nominal)
        current = data.get('current_a', 60)
        current_deviation = abs(current - 60) / 60  # Nominal 60A
        current_health = max(0, 100 - current_deviation * 200)  # 50% deviation = 0 health
        component_health['current'] = current_health
        
        # Flow/pressure consistency health
        flow = data.get('flow_lpm', 1200)
        pressure = data.get('pressure_bar', 6.2)
        
        flow_health = max(0, min(100, flow / 12))  # Scale to percentage of nominal
        pressure_health = max(0, min(100, pressure / 6.2 * 100))
        fp_health = (flow_health + pressure_health) / 2
        component_health['flow_pressure'] = fp_health
        
        # Oil and RPM health
        oil_level = data.get('oil_level_pct', 80)
        rpm = data.get('rpm', 1800)
        
        oil_health = max(0, oil_level)  # Direct percentage
        rpm_deviation = abs(rpm - 1800) / 1800
        rpm_health = max(0, 100 - rpm_deviation * 200)
        oil_rpm_health = (oil_health + rpm_health) / 2
        component_health['oil_rpm'] = oil_rpm_health
        
        # Calculate overall health index using weights
        overall_health = (
            component_health['vibration'] * self.health_weights['vibration'] +
            component_health['bearing_temp'] * self.health_weights['bearing_temp'] +
            current_health * self.health_weights['current'] +
            component_health['flow_pressure'] * self.health_weights['flow_pressure'] +
            component_health['oil_rpm'] * self.health_weights['oil_rpm']
        )
        
        # Calculate trend from rolling statistics
        trends = {}
        for sensor in ['vibration_rms', 'bearing_temp', 'current_a']:
            if sensor in stats:
                trend = self.rolling_stats.get_trend(sensor)
                trends[sensor] = trend if trend is not None else 0.0
        
        return {
            'overall_health_index': round(overall_health, 1),
            'component_health': component_health,
            'trends': trends,
            'calculation_time': datetime.now(timezone.utc).isoformat()
        }
    
    def _generate_alerts(self, processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on processed data"""
        alerts = []
        
        # Threshold alerts
        if processed_data['threshold_alerts']:
            alerts.extend(processed_data['threshold_alerts'])
        
        # Anomaly alerts
        for sensor, anomaly_data in processed_data['anomalies'].items():
            if anomaly_data['is_anomaly'] and anomaly_data['z_score'] > 4.0:
                alerts.append({
                    'type': 'anomaly',
                    'severity': 'warning',
                    'sensor': sensor,
                    'z_score': anomaly_data['z_score'],
                    'message': f'Statistical anomaly detected in {sensor}: Z-score {anomaly_data["z_score"]:.2f}'
                })
        
        # Vibration fault alerts
        fault_indicators = processed_data['vibration_analysis'].get('fault_indicators', {})
        for fault_type, fault_data in fault_indicators.items():
            if fault_data.get('detected'):
                alerts.append({
                    'type': 'vibration_fault',
                    'severity': fault_data.get('severity', 'medium'),
                    'fault_type': fault_type,
                    'message': f'Vibration fault detected: {fault_type} ({fault_data["severity"]} severity)'
                })
        
        # Health index alerts
        health_index = processed_data['health_metrics']['overall_health_index']
        if health_index < 50:
            alerts.append({
                'type': 'health_degradation',
                'severity': 'critical',
                'health_index': health_index,
                'message': f'Critical health degradation: HI = {health_index:.1f}%'
            })
        elif health_index < 70:
            alerts.append({
                'type': 'health_degradation',
                'severity': 'warning',
                'health_index': health_index,
                'message': f'Health degradation detected: HI = {health_index:.1f}%'
            })
        
        return alerts
    
    def _display_formatted_alert(self, sensor_data: Dict[str, Any], health_index: float, alerts: List[Dict[str, Any]]):
        """Display formatted alert message in terminal"""
        if not alerts:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Determine equipment state based on alerts
        critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
        if critical_alerts:
            if any('bearing' in str(a) for a in critical_alerts):
                equipment_state = "bearing_overheating"
            elif any('vibration' in str(a) for a in critical_alerts):
                equipment_state = "critical_vibration"
            elif any('current' in str(a) for a in critical_alerts):
                equipment_state = "motor_overload"
            elif any('flow' in str(a) for a in critical_alerts):
                equipment_state = "pump_cavitation"
            else:
                equipment_state = "multiple_critical_faults"
        else:
            equipment_state = "degraded_operation"
        
        # Get issues and recommendations
        issues = []
        for alert in alerts:
            if alert.get('type') == 'threshold':
                issues.append(f"{alert['sensor']}_fault")
            elif alert.get('type') == 'vibration_fault':
                issues.append(alert.get('fault_type', 'vibration_fault'))
            elif alert.get('type') == 'health_degradation':
                issues.append("health_degradation")
        
        recommendations = self._get_maintenance_recommendation(issues)
        
        # Create formatted alert message
        alert_message = f"""
🚨 PUMP MOTOR EQUIPMENT ALERT 🚨
Time: {timestamp}
Equipment State: {equipment_state}
Health Index: {health_index:.1f}%

Current Readings:
- Vibration: {sensor_data.get('vibration_rms', 0):.1f} m/s²
- Bearing Temp: {sensor_data.get('bearing_temp', 0):.0f}°C
- Motor Current: {sensor_data.get('current_a', 0):.0f}A
- Flow Rate: {sensor_data.get('flow_lpm', 0):.0f} L/min
- Pressure: {sensor_data.get('pressure_bar', 0):.1f} bar
- Winding Temp: {sensor_data.get('winding_temp', 0):.0f}°C

Issues: {', '.join(issues) if issues else 'none'}
Recommendation: {recommendations}
"""
        
        print(alert_message)
        return alert_message

    def _get_maintenance_recommendation(self, faults: List[str]) -> str:
        """Generate maintenance recommendations based on faults"""
        recommendations = []
        
        for fault in faults:
            if "bearing" in fault:
                recommendations.append("bearing_replacement")
            elif "vibration" in fault or "imbalance" in fault:
                recommendations.append("alignment_check")
            elif "current" in fault or "overload" in fault:
                recommendations.append("reduce_load")
            elif "flow" in fault or "cavitation" in fault:
                recommendations.append("check_suction_line")
            elif "misalignment" in fault:
                recommendations.append("shaft_alignment")
            elif "health_degradation" in fault:
                recommendations.append("comprehensive_inspection")
        
        if len(faults) >= 3:
            recommendations.append("emergency_shutdown")
        
        return ", ".join(recommendations) if recommendations else "monitor_closely"

    def _send_alerts(self, alerts: List[Dict[str, Any]]):
        """Send alerts via MQTT"""
        for alert in alerts:
            self.alert_count += 1
            alert_payload = {
                'alert_id': self.alert_count,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'source': 'edge_processor',
                'alert': alert
            }
            
            success = self.mqtt_client.publish(Topics.EDGE_ALERTS, alert_payload, qos=1)
            if success:
                logger.warning(f"ALERT #{self.alert_count}: {alert['message']}")
            else:
                logger.error(f"Failed to send alert: {alert['message']}")
    
    def _send_to_digital_twin(self, processed_data: Dict[str, Any]):
        """Send processed data to digital twin"""
        twin_payload = {
            'timestamp': processed_data['timestamp'],
            'sensor_readings': processed_data['raw_sensors'],
            'health_metrics': processed_data['health_metrics'],
            'vibration_analysis': processed_data['vibration_analysis'],
            'processing_metadata': {
                'processing_count': self.processing_count,
                'processing_time_ms': processed_data['processing_time_ms'],
                'edge_id': 'edge_001'
            }
        }
        
        success = self.mqtt_client.publish(Topics.TWIN_STATE, twin_payload)
        if not success:
            logger.error("Failed to send data to digital twin")
    
    def _log_processing_results(self, processed_data: Dict[str, Any]):
        """Log processing results"""
        health_index = processed_data['health_metrics']['overall_health_index']
        num_alerts = len(processed_data['threshold_alerts'])
        num_anomalies = len(processed_data['anomalies'])
        processing_time = processed_data['processing_time_ms']
        
        vibration_rms = processed_data['raw_sensors']['vibration_rms']
        bearing_temp = processed_data['raw_sensors']['bearing_temp']
        current = processed_data['raw_sensors']['current_a']
        
        logger.info(f"Processing #{self.processing_count} | "
                   f"HI={health_index:.1f}% | "
                   f"Vibration={vibration_rms:.2f} m/s² | "
                   f"Bearing_temp={bearing_temp:.1f}°C | "
                   f"Current={current:.1f}A | "
                   f"Alerts={num_alerts} | "
                   f"Anomalies={num_anomalies} | "
                   f"Processing={processing_time:.1f}ms")
    
    def start_processing(self):
        """Start the edge processor"""
        logger.info("Starting Edge Processor...")
        logger.info(f"Subscribing to topic: {Topics.SENSOR_DATA}")
        
        # Subscribe to sensor data
        success = self.mqtt_client.subscribe(Topics.SENSOR_DATA, self.process_sensor_data)
        
        if success:
            logger.info("Edge Processor started successfully")
            logger.info("Waiting for sensor data...")
            
            try:
                # Keep the processor running
                while True:
                    time.sleep(1)
                    
                    # Check for processing timeout
                    if self.last_reading_time and (time.time() - self.last_reading_time) > 30:
                        logger.warning("No sensor data received for 30 seconds")
                        
            except KeyboardInterrupt:
                logger.info("Edge Processor stopped by user")
            except Exception as e:
                logger.error(f"Edge Processor error: {e}")
        else:
            logger.error("Failed to subscribe to sensor data topic")
        
        

def main():
    """Main entry point"""
    # Initialize MQTT client
    mqtt_client = MQTTClient(
        broker_host="broker.hivemq.com",
        broker_port=1883,
        client_id="pump_edge_processor"
    )
    
    # Connect to MQTT broker
    if not mqtt_client.connect():
        logger.error("Failed to connect to MQTT broker")
        return
    
    logger.info("Connected to MQTT broker")
    
    # Create and start edge processor
    processor = EdgeProcessor(mqtt_client)
    processor.start_processing()

if __name__ == "__main__":
    main()