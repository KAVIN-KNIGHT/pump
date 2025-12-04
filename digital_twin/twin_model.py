"""
Digital Twin Model for IIoT Pump Motor System
Maintains virtual state, calculates health metrics, and provides predictive analytics
"""

import json
import time
import pickle
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import deque
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mqtt_client import MQTTClient, Topics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EquipmentState:
    """Current state of pump motor equipment"""
    timestamp: str
    vibration_rms: float
    bearing_temp: float
    winding_temp: float
    current_a: float
    voltage_v: float
    flow_lpm: float
    pressure_bar: float
    rpm: float
    oil_level_pct: float
    seal_leak: int
    health_index: float = 0.0
    rul_hours: float = 0.0
    
class HealthCalculator:
    """Calculate equipment health metrics"""
    
    def __init__(self):
        # Component weights for overall health calculation
        self.weights = {
            'vibration': 0.35,
            'thermal': 0.25,
            'electrical': 0.15,
            'hydraulic': 0.15,
            'mechanical': 0.10
        }
        
        # Nominal operating ranges
        self.nominal_ranges = {
            'vibration_rms': {'min': 0.5, 'max': 2.0, 'critical': 5.0},
            'bearing_temp': {'min': 40, 'max': 70, 'critical': 85},
            'winding_temp': {'min': 45, 'max': 80, 'critical': 130},
            'current_a': {'nominal': 60, 'tolerance': 0.1, 'critical': 72},
            'flow_lpm': {'nominal': 1200, 'tolerance': 0.2, 'critical': 600},
            'pressure_bar': {'nominal': 6.2, 'tolerance': 0.15, 'critical': 3.0},
            'rpm': {'nominal': 1800, 'tolerance': 0.05, 'critical_low': 1600},
            'oil_level_pct': {'min': 70, 'critical': 25}
        }
    
    def calculate_component_health(self, state: EquipmentState) -> Dict[str, float]:
        """Calculate health scores for individual components"""
        component_scores = {}
        
        # Vibration health (0-100)
        vib_rms = state.vibration_rms
        ranges = self.nominal_ranges['vibration_rms']
        if vib_rms <= ranges['max']:
            vib_health = 100 - ((vib_rms - ranges['min']) / (ranges['max'] - ranges['min'])) * 20
        else:
            vib_health = max(0, 80 - ((vib_rms - ranges['max']) / (ranges['critical'] - ranges['max'])) * 80)
        component_scores['vibration'] = max(0, min(100, vib_health))
        
        # Thermal health (bearing temperature)
        bearing_temp = state.bearing_temp
        temp_ranges = self.nominal_ranges['bearing_temp']
        if bearing_temp <= temp_ranges['max']:
            thermal_health = 100 - ((bearing_temp - temp_ranges['min']) / (temp_ranges['max'] - temp_ranges['min'])) * 20
        else:
            thermal_health = max(0, 80 - ((bearing_temp - temp_ranges['max']) / (temp_ranges['critical'] - temp_ranges['max'])) * 80)
        
        # Include winding temperature
        winding_temp = state.winding_temp
        winding_ranges = self.nominal_ranges['winding_temp']
        if winding_temp <= winding_ranges['max']:
            winding_health = 100 - ((winding_temp - winding_ranges['min']) / (winding_ranges['max'] - winding_ranges['min'])) * 20
        else:
            winding_health = max(0, 80 - ((winding_temp - winding_ranges['max']) / (winding_ranges['critical'] - winding_ranges['max'])) * 80)
        
        component_scores['thermal'] = (thermal_health + winding_health) / 2
        
        # Electrical health (current)
        current = state.current_a
        current_ranges = self.nominal_ranges['current_a']
        current_deviation = abs(current - current_ranges['nominal']) / current_ranges['nominal']
        if current_deviation <= current_ranges['tolerance']:
            electrical_health = 100 - (current_deviation / current_ranges['tolerance']) * 10
        else:
            electrical_health = max(0, 90 - ((current_deviation - current_ranges['tolerance']) / 0.2) * 90)
        component_scores['electrical'] = max(0, min(100, electrical_health))
        
        # Hydraulic health (flow + pressure)
        flow = state.flow_lpm
        flow_ranges = self.nominal_ranges['flow_lpm']
        flow_deviation = abs(flow - flow_ranges['nominal']) / flow_ranges['nominal']
        flow_health = max(0, 100 - (flow_deviation / flow_ranges['tolerance']) * 100)
        
        pressure = state.pressure_bar
        pressure_ranges = self.nominal_ranges['pressure_bar']
        pressure_deviation = abs(pressure - pressure_ranges['nominal']) / pressure_ranges['nominal']
        pressure_health = max(0, 100 - (pressure_deviation / pressure_ranges['tolerance']) * 100)
        
        component_scores['hydraulic'] = (flow_health + pressure_health) / 2
        
        # Mechanical health (RPM + oil level)
        rpm = state.rpm
        rpm_ranges = self.nominal_ranges['rpm']
        rpm_deviation = abs(rpm - rpm_ranges['nominal']) / rpm_ranges['nominal']
        rpm_health = max(0, 100 - (rpm_deviation / rpm_ranges['tolerance']) * 100)
        
        oil_level = state.oil_level_pct
        oil_ranges = self.nominal_ranges['oil_level_pct']
        oil_health = max(0, min(100, oil_level))
        
        component_scores['mechanical'] = (rpm_health + oil_health) / 2
        
        return component_scores
    
    def calculate_overall_health(self, component_scores: Dict[str, float]) -> float:
        """Calculate weighted overall health index"""
        overall_health = sum(
            component_scores[component] * self.weights[component]
            for component in self.weights.keys()
        )
        return round(overall_health, 1)

class RULPredictor:
    """Predict Remaining Useful Life using simple heuristic model"""
    
    def __init__(self):
        # Load ML model if it exists, otherwise use heuristic
        self.model = None
        self.model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rul_model.pkl')
        
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Loaded RUL prediction model")
            else:
                logger.info("No ML model found, using heuristic RUL prediction")
        except Exception as e:
            logger.error(f"Error loading RUL model: {e}")
        
        # Heuristic parameters
        self.base_life_hours = 8760  # 1 year baseline
        self.degradation_factors = {
            'vibration': 2.0,
            'thermal': 1.8,
            'electrical': 1.2,
            'hydraulic': 1.5,
            'mechanical': 1.3
        }
    
    def predict_rul(self, health_index: float, component_scores: Dict[str, float],
                   trend_data: Optional[Dict[str, float]] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Predict remaining useful life in hours
        
        Args:
            health_index: Overall health index (0-100)
            component_scores: Individual component health scores
            trend_data: Historical trend information
            
        Returns:
            Tuple of (RUL in hours, prediction metadata)
        """
        try:
            if self.model is not None:
                # Use ML model if available
                return self._predict_with_model(health_index, component_scores, trend_data)
            else:
                # Use heuristic model
                return self._predict_with_heuristic(health_index, component_scores, trend_data)
        except Exception as e:
            logger.error(f"Error in RUL prediction: {e}")
            return 0.0, {'error': str(e), 'method': 'error'}
    
    def _predict_with_heuristic(self, health_index: float, component_scores: Dict[str, float],
                               trend_data: Optional[Dict[str, float]]) -> Tuple[float, Dict[str, Any]]:
        """Heuristic RUL prediction based on health degradation"""
        
        # Base RUL from health index
        if health_index > 90:
            base_rul = self.base_life_hours * 0.9  # Near full life
        elif health_index > 70:
            base_rul = self.base_life_hours * (health_index / 100) * 0.8
        elif health_index > 50:
            base_rul = self.base_life_hours * (health_index / 100) * 0.4
        elif health_index > 30:
            base_rul = self.base_life_hours * (health_index / 100) * 0.2
        else:
            base_rul = min(168, self.base_life_hours * (health_index / 100) * 0.1)  # Max 1 week
        
        # Apply component-specific degradation factors
        degradation_multiplier = 1.0
        for component, score in component_scores.items():
            if score < 60:  # Component in poor health
                degradation = (60 - score) / 60  # 0 to 1
                degradation_multiplier *= (1 - degradation * (self.degradation_factors.get(component, 1.0) - 1))
        
        # Apply trend acceleration if available
        trend_multiplier = 1.0
        if trend_data:
            # If sensors are trending worse, reduce RUL
            negative_trends = sum(1 for trend in trend_data.values() if trend and trend > 0.1)
            if negative_trends > 0:
                trend_multiplier = max(0.5, 1.0 - (negative_trends * 0.15))
        
        final_rul = base_rul * degradation_multiplier * trend_multiplier
        
        # Ensure minimum bounds
        final_rul = max(1.0, final_rul)  # At least 1 hour
        
        metadata = {
            'method': 'heuristic',
            'base_rul': round(base_rul, 1),
            'degradation_multiplier': round(degradation_multiplier, 3),
            'trend_multiplier': round(trend_multiplier, 3),
            'component_factors': {k: v for k, v in self.degradation_factors.items() if component_scores.get(k, 100) < 60}
        }
        
        return round(final_rul, 1), metadata
    
    def _predict_with_model(self, health_index: float, component_scores: Dict[str, float],
                          trend_data: Optional[Dict[str, float]]) -> Tuple[float, Dict[str, Any]]:
        """ML-based RUL prediction (placeholder for trained model)"""
        # This would use the actual trained model
        # For now, fall back to heuristic
        return self._predict_with_heuristic(health_index, component_scores, trend_data)

class RecommendationEngine:
    """Generate maintenance and operational recommendations"""
    
    def __init__(self):
        self.recommendation_rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[str, Any]:
        """Initialize recommendation rules"""
        return {
            'vibration_high': {
                'threshold': 3.0,
                'actions': ['Check alignment', 'Inspect bearings', 'Verify balancing'],
                'urgency': 'high'
            },
            'temperature_high': {
                'bearing_threshold': 80,
                'winding_threshold': 120,
                'actions': ['Check lubrication', 'Inspect cooling', 'Reduce load'],
                'urgency': 'high'
            },
            'current_anomaly': {
                'threshold': 0.15,  # 15% deviation
                'actions': ['Check electrical connections', 'Inspect motor windings', 'Verify load conditions'],
                'urgency': 'medium'
            },
            'flow_low': {
                'threshold': 800,
                'actions': ['Check for blockages', 'Inspect suction line', 'Verify pump priming'],
                'urgency': 'high'
            },
            'oil_level_low': {
                'threshold': 40,
                'actions': ['Add lubrication', 'Check for leaks', 'Inspect seals'],
                'urgency': 'medium'
            },
            'rul_critical': {
                'threshold': 72,  # 3 days
                'actions': ['Schedule immediate maintenance', 'Prepare replacement parts', 'Consider shutdown'],
                'urgency': 'critical'
            }
        }
    
    def generate_recommendations(self, state: EquipmentState, health_metrics: Dict[str, Any],
                               rul_hours: float) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on current state"""
        recommendations = []
        
        # Vibration-based recommendations
        if state.vibration_rms > self.recommendation_rules['vibration_high']['threshold']:
            recommendations.append({
                'category': 'vibration',
                'urgency': 'high',
                'actions': self.recommendation_rules['vibration_high']['actions'],
                'reason': f'High vibration detected: {state.vibration_rms:.2f} m/s²',
                'target_completion': 'within 24 hours'
            })
        
        # Temperature-based recommendations
        temp_rules = self.recommendation_rules['temperature_high']
        if (state.bearing_temp > temp_rules['bearing_threshold'] or 
            state.winding_temp > temp_rules['winding_threshold']):
            recommendations.append({
                'category': 'thermal',
                'urgency': 'high',
                'actions': temp_rules['actions'],
                'reason': f'High temperature: Bearing {state.bearing_temp:.1f}°C, Winding {state.winding_temp:.1f}°C',
                'target_completion': 'within 12 hours'
            })
        
        # Current anomaly recommendations
        current_deviation = abs(state.current_a - 60) / 60
        if current_deviation > self.recommendation_rules['current_anomaly']['threshold']:
            recommendations.append({
                'category': 'electrical',
                'urgency': 'medium',
                'actions': self.recommendation_rules['current_anomaly']['actions'],
                'reason': f'Current anomaly: {state.current_a:.1f}A ({current_deviation*100:.1f}% deviation)',
                'target_completion': 'within 48 hours'
            })
        
        # Flow-based recommendations
        if state.flow_lpm < self.recommendation_rules['flow_low']['threshold']:
            recommendations.append({
                'category': 'hydraulic',
                'urgency': 'high',
                'actions': self.recommendation_rules['flow_low']['actions'],
                'reason': f'Low flow rate: {state.flow_lpm:.0f} L/min',
                'target_completion': 'within 8 hours'
            })
        
        # Oil level recommendations
        if state.oil_level_pct < self.recommendation_rules['oil_level_low']['threshold']:
            recommendations.append({
                'category': 'lubrication',
                'urgency': 'medium',
                'actions': self.recommendation_rules['oil_level_low']['actions'],
                'reason': f'Low oil level: {state.oil_level_pct:.1f}%',
                'target_completion': 'within 24 hours'
            })
        
        # RUL-based recommendations
        if rul_hours < self.recommendation_rules['rul_critical']['threshold']:
            recommendations.append({
                'category': 'predictive_maintenance',
                'urgency': 'critical',
                'actions': self.recommendation_rules['rul_critical']['actions'],
                'reason': f'Critical RUL: {rul_hours:.1f} hours remaining',
                'target_completion': 'immediate'
            })
        elif rul_hours < 168:  # 1 week
            recommendations.append({
                'category': 'predictive_maintenance',
                'urgency': 'high',
                'actions': ['Schedule maintenance window', 'Order replacement parts', 'Plan downtime'],
                'reason': f'Low RUL: {rul_hours:.1f} hours remaining',
                'target_completion': 'within 72 hours'
            })
        
        # Health-based recommendations
        health_index = health_metrics.get('overall_health_index', 100)
        if health_index < 50:
            recommendations.append({
                'category': 'general_health',
                'urgency': 'critical',
                'actions': ['Comprehensive inspection', 'Consider equipment shutdown', 'Emergency maintenance'],
                'reason': f'Critical health degradation: HI = {health_index:.1f}%',
                'target_completion': 'immediate'
            })
        elif health_index < 70:
            recommendations.append({
                'category': 'general_health',
                'urgency': 'high',
                'actions': ['Detailed condition assessment', 'Increase monitoring frequency', 'Schedule maintenance'],
                'reason': f'Health degradation detected: HI = {health_index:.1f}%',
                'target_completion': 'within 48 hours'
            })
        
        return recommendations

class DigitalTwin:
    """Main Digital Twin class for pump motor system"""
    
    def __init__(self, mqtt_client: MQTTClient):
        self.mqtt_client = mqtt_client
        
        # Twin components
        self.health_calculator = HealthCalculator()
        self.rul_predictor = RULPredictor()
        self.recommendation_engine = RecommendationEngine()
        
        # Twin state
        self.current_state: Optional[EquipmentState] = None
        self.state_history: deque = deque(maxlen=1000)  # Keep last 1000 states
        
        # Analytics
        self.last_update_time = None
        self.update_count = 0
        
        # Trend analysis window
        self.trend_window_size = 30
        
    def update_state(self, sensor_data: Dict[str, Any], edge_metrics: Dict[str, Any]):
        """Update twin state with new sensor data and edge analytics"""
        try:
            self.update_count += 1
            
            # Create new equipment state
            new_state = EquipmentState(
                timestamp=sensor_data.get('timestamp', datetime.now(timezone.utc).isoformat()),
                vibration_rms=sensor_data.get('vibration_rms', 0.0),
                bearing_temp=sensor_data.get('bearing_temp', 25.0),
                winding_temp=sensor_data.get('winding_temp', 25.0),
                current_a=sensor_data.get('current_a', 0.0),
                voltage_v=sensor_data.get('voltage_v', 0.0),
                flow_lpm=sensor_data.get('flow_lpm', 0.0),
                pressure_bar=sensor_data.get('pressure_bar', 0.0),
                rpm=sensor_data.get('rpm', 0.0),
                oil_level_pct=sensor_data.get('oil_level_pct', 100.0),
                seal_leak=sensor_data.get('seal_leak', False)
            )
            
            # Calculate health metrics
            component_scores = self.health_calculator.calculate_component_health(new_state)
            overall_health = self.health_calculator.calculate_overall_health(component_scores)
            
            # Get trend data for RUL prediction
            trend_data = self._calculate_trends()
            
            # Predict RUL
            rul_hours, rul_metadata = self.rul_predictor.predict_rul(
                overall_health, component_scores, trend_data
            )
            
            # Update state with calculated values
            new_state.health_index = overall_health
            new_state.rul_hours = rul_hours
            
            # Generate recommendations
            health_metrics = {
                'overall_health_index': overall_health,
                'component_scores': component_scores,
                'rul_prediction': rul_metadata
            }
            
            recommendations = self.recommendation_engine.generate_recommendations(
                new_state, health_metrics, rul_hours
            )
            
            # Store state in history
            self.state_history.append(new_state)
            self.current_state = new_state
            self.last_update_time = time.time()
            
            # Create response
            twin_response = {
                'twin_id': 'pump_motor_twin_001',
                'update_timestamp': new_state.timestamp,
                'update_count': self.update_count,
                'current_state': asdict(new_state),
                'health_metrics': health_metrics,
                'recommendations': recommendations,
                'trend_analysis': trend_data or {},
                'sync_status': 'synchronized'
            }
            
            # Log twin update
            self._log_twin_update(twin_response)
            
            # Publish twin state for visualization
            self.mqtt_client.publish(Topics.TWIN_STATE, twin_response, qos=1)
            
            # Send any critical recommendations as commands
            self._handle_critical_recommendations(recommendations)
            
            return twin_response
            
        except Exception as e:
            logger.error(f"Error updating twin state: {e}")
            return {
                'twin_id': 'pump_motor_twin_001',
                'update_timestamp': datetime.now(timezone.utc).isoformat(),
                'sync_status': 'error',
                'error': str(e)
            }
    
    def _calculate_trends(self) -> Optional[Dict[str, float]]:
        """Calculate trends from historical state data"""
        if len(self.state_history) < self.trend_window_size:
            return None
        
        try:
            # Get recent states for trend analysis
            recent_states = list(self.state_history)[-self.trend_window_size:]
            
            trends = {}
            
            # Calculate trends for key sensors
            sensors = ['vibration_rms', 'bearing_temp', 'winding_temp', 'current_a']
            
            for sensor in sensors:
                values = [getattr(state, sensor) for state in recent_states]
                
                # Simple linear trend (slope)
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                slope = coeffs[0]
                
                # Normalize slope by mean value for relative trend
                mean_value = np.mean(values)
                if mean_value > 0:
                    normalized_slope = slope / mean_value
                else:
                    normalized_slope = 0
                
                trends[sensor] = round(normalized_slope, 6)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
            return None
    
    def _display_twin_status(self, twin_response: Dict[str, Any]):
        """Display formatted digital twin status"""
        state = twin_response['current_state']
        health_index = state['health_index']
        rul_hours = state['rul_hours']
        recommendations = twin_response['recommendations']
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Determine severity and icon
        if health_index < 30 or rul_hours < 24:
            severity = "CRITICAL"
            icon = "🔴"
        elif health_index < 60 or rul_hours < 168:
            severity = "WARNING"
            icon = "🟡"
        else:
            severity = "NORMAL"
            icon = "🟢"
        
        # Get prediction confidence
        confidence = self._get_prediction_confidence(health_index)
        
        # Format recommendations for display
        rec_list = [rec['category'] for rec in recommendations[:3]] if recommendations else ['none_required']
        
        status_message = f"""
{icon} DIGITAL TWIN STATUS UPDATE {icon}
Time: {timestamp}
Equipment Health: {health_index:.1f}%
Remaining Life: {rul_hours:.1f} hours
Severity: {severity}

Virtual Sensor State:

Maintenance Actions: {', '.join(rec_list)}
Prediction Confidence: {confidence:.1f}%
"""
        
        print(status_message)
        return status_message

    def _get_prediction_confidence(self, health_index: float) -> float:
        """Calculate prediction confidence based on health index"""
        if health_index > 80:
            return 95.0
        elif health_index > 60:
            return 88.0
        elif health_index > 40:
            return 82.0
        elif health_index > 20:
            return 75.0
        else:
            return 68.0

    def _log_twin_update(self, twin_response: Dict[str, Any]):
        """Log twin update results"""
        state = twin_response['current_state']
        health_index = state['health_index']
        rul_hours = state['rul_hours']
        num_recommendations = len(twin_response['recommendations'])
        
        # Display formatted status for critical conditions
        if health_index < 60 or rul_hours < 168 or num_recommendations > 0:
            self._display_twin_status(twin_response)
        else:
            # Regular log for normal conditions
            logger.info(f"Twin Update #{self.update_count} | "
                       f"HI={health_index:.1f}% | "
                       f"RUL={rul_hours:.1f}h | "
                       f"Recommendations={num_recommendations} | "
                       f"Vibration={state['vibration_rms']:.2f} m/s² | "
                       f"Bearing_temp={state['bearing_temp']:.1f}°C")
    
    def _handle_critical_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Handle critical recommendations by sending control commands"""
        for rec in recommendations:
            if rec['urgency'] == 'critical':
                # Send critical alert
                command = {
                    'command_type': 'critical_alert',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'urgency': 'critical',
                    'category': rec['category'],
                    'reason': rec['reason'],
                    'actions': rec['actions'],
                    'source': 'digital_twin'
                }
                
                # Publish command (could trigger automatic responses)
                self.mqtt_client.publish(Topics.TWIN_COMMANDS, command, qos=1)
                logger.warning(f"CRITICAL RECOMMENDATION: {rec['reason']}")
    
    def process_edge_data(self, topic: str, data: Dict[str, Any]):
        """Process incoming data from edge processor"""
        try:
            logger.debug(f"Received edge data on topic: {topic}")
            
            # Update twin state
            twin_response = self.update_state(data, data.get('health_metrics', {}))
            
            # Optional: Send response back (for debugging or confirmation)
            # This could be used for twin synchronization confirmation
            
        except Exception as e:
            logger.error(f"Error processing edge data: {e}")
    
    def start_twin(self):
        """Start the digital twin"""
        logger.info("Starting Digital Twin...")
        logger.info(f"Subscribing to topic: {Topics.SENSOR_DATA}")
        
        # Subscribe to sensor data from edge processor
        success = self.mqtt_client.subscribe(Topics.SENSOR_DATA, self.process_edge_data)
        
        if success:
            logger.info("Digital Twin started successfully")
            logger.info("Waiting for edge data...")
            
            try:
                # Keep the twin running
                while True:
                    time.sleep(1)
                    
                    # Check for data timeout
                    if self.last_update_time and (time.time() - self.last_update_time) > 30:
                        logger.warning("No edge data received for 30 seconds")
                        
            except KeyboardInterrupt:
                logger.info("Digital Twin stopped by user")
            except Exception as e:
                logger.error(f"Digital Twin error: {e}")
            finally:
                if hasattr(self, 'mqtt_client') and self.mqtt_client:
                    self.mqtt_client.disconnect()
        else:
            logger.error("Failed to subscribe to edge data topic")
        

def main():
    """Main entry point"""
    # Initialize MQTT client
    mqtt_client = MQTTClient(
        broker_host="broker.hivemq.com",
        broker_port=1883,
        client_id="pump_digital_twin"
    )
    
    # Connect to MQTT broker
    if not mqtt_client.connect():
        logger.error("Failed to connect to MQTT broker")
        return
    
    logger.info("Connected to MQTT broker")
    
    # Create and start digital twin
    twin = DigitalTwin(mqtt_client)
    twin.start_twin()

if __name__ == "__main__":
    main()