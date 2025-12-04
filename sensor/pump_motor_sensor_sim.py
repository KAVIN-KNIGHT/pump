"""
Pump Motor Sensor Simulator for IIoT System
Simulates realistic sensor data with noise and fault injection capabilities
"""

import json
import time
import random
import math
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import sys
import os
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mqtt_client import MQTTClient, Topics
from utils.signal_processing import generate_synthetic_vibration

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SensorRanges:
    """Define normal operating ranges for all sensors"""
    # Motor sensors
    vibration_rms_min: float = 0.5      # m/s² - normal range
    vibration_rms_max: float = 2.0
    vibration_rms_critical: float = 5.0  # Alert threshold
    
    bearing_temp_min: float = 40         # °C - normal range
    bearing_temp_max: float = 70
    bearing_temp_critical: float = 80    # Alert threshold
    
    winding_temp_min: float = 45         # °C
    winding_temp_max: float = 80
    winding_temp_critical: float = 120
    
    current_min: float = 45              # A - depends on load
    current_max: float = 65
    current_rated: float = 60            # Rated current
    current_critical: float = 66        # 110% of rated
    
    voltage_nominal: float = 380         # V - three-phase
    voltage_tolerance: float = 0.1       # ±10%
    
    # Pump sensors
    flow_min: float = 1000               # L/min
    flow_max: float = 1500
    flow_nominal: float = 1200
    
    pressure_min: float = 5.5            # bar
    pressure_max: float = 7.0
    pressure_nominal: float = 6.2
    
    # Mechanical
    rpm_nominal: float = 1800            # rpm
    rpm_tolerance: float = 0.05          # ±5%
    
    oil_level_min: float = 70            # %
    oil_level_max: float = 90

class FaultInjector:
    """Inject realistic faults into sensor readings"""
    
    def __init__(self):
        self.active_faults = set()
        self.fault_start_times = {}
        
        # Fault injection probabilities (per 1000 readings)
        self.fault_probabilities = {
            'bearing_fault': 0.010,      # 1%
            'rotor_imbalance': 0.008,    # 0.8%
            'misalignment': 0.006,       # 0.6%
            'cavitation': 0.005,         # 0.5%
            'seal_leak': 0.004,          # 0.4%
            'overload': 0.003,           # 0.3%
            'insulation_degradation': 0.002  # 0.2%
        }
        
        # Fault duration ranges (in seconds)
        self.fault_durations = {
            'bearing_fault': (300, 1800),        # 5-30 minutes
            'rotor_imbalance': (600, 3600),      # 10-60 minutes
            'misalignment': (1200, 7200),        # 20-120 minutes
            'cavitation': (60, 300),             # 1-5 minutes
            'seal_leak': (1800, 7200),          # 30-120 minutes
            'overload': (30, 180),               # 30 seconds - 3 minutes
            'insulation_degradation': (3600, 14400)  # 1-4 hours
        }
    
    def check_new_faults(self) -> List[str]:
        """Check if new faults should be injected"""
        new_faults = []
        
        for fault_type, probability in self.fault_probabilities.items():
            if fault_type not in self.active_faults:
                if random.random() < probability:
                    new_faults.append(fault_type)
                    self.active_faults.add(fault_type)
                    self.fault_start_times[fault_type] = time.time()
                    logger.warning(f"Injecting fault: {fault_type}")
        
        return new_faults
    
    def check_fault_expiry(self):
        """Check if any active faults should expire"""
        current_time = time.time()
        expired_faults = []
        
        for fault_type in list(self.active_faults):
            start_time = self.fault_start_times[fault_type]
            duration_range = self.fault_durations[fault_type]
            max_duration = duration_range[1]
            
            if current_time - start_time > max_duration:
                expired_faults.append(fault_type)
                self.active_faults.remove(fault_type)
                del self.fault_start_times[fault_type]
                logger.info(f"Fault expired: {fault_type}")
        
        return expired_faults
    
    def get_active_faults(self) -> List[str]:
        """Get list of currently active faults"""
        return list(self.active_faults)

class PumpMotorSensorSimulator:
    """Main sensor simulator class"""
    
    def __init__(self, mqtt_client: MQTTClient):
        self.mqtt_client = mqtt_client
        self.sensor_ranges = SensorRanges()
        self.fault_injector = FaultInjector()
        
        # Current sensor values
        self.current_values = {}
        
        # Simulation parameters
        self.sampling_interval = 3.0  # seconds
        self.noise_factor = 0.01      # Reduced to 1% noise for cleaner readings
        
        # Data smoothing parameters for cleaner ThingSpeak visualization
        self.smoothing_factor = 0.2   # Exponential smoothing factor (0-1, lower = smoother)
        self.trend_amplitude = 0.05   # Maximum trending variation (5%)
        self.previous_values = {}     # Store previous values for smoothing
        
        # Initialize with nominal values
        self._initialize_sensors()
        
        # Vibration buffer for FFT analysis
        self.vibration_buffer = []
        self.vibration_buffer_size = 100
        
    def _initialize_sensors(self):
        """Initialize sensor values to nominal operating conditions"""
        ranges = self.sensor_ranges
        
        self.current_values = {
            'vibration_rms': (ranges.vibration_rms_min + ranges.vibration_rms_max) / 2,
            'bearing_temp': (ranges.bearing_temp_min + ranges.bearing_temp_max) / 2,
            'winding_temp': (ranges.winding_temp_min + ranges.winding_temp_max) / 2,
            'current_a': ranges.current_rated * 0.8,  # 80% load initially
            'voltage_v': ranges.voltage_nominal,
            'flow_lpm': ranges.flow_nominal,
            'pressure_bar': ranges.pressure_nominal,
            'rpm': ranges.rpm_nominal,
            'oil_level_pct': (ranges.oil_level_min + ranges.oil_level_max) / 2,
            'seal_leak': 0  # Binary sensor
        }
        
        # Initialize previous values for smoothing
        self.previous_values = self.current_values.copy()
    
    def _add_smooth_variation(self, param_name: str, target_value: float) -> float:
        """Add smooth variation with exponential smoothing for cleaner ThingSpeak display"""
        if param_name not in self.previous_values:
            self.previous_values[param_name] = target_value
        
        # Generate small trending variation instead of random noise
        trend = random.uniform(-self.trend_amplitude, self.trend_amplitude)
        varied_value = target_value * (1 + trend)
        
        # Apply exponential smoothing to reduce jumpiness
        smoothed_value = (self.smoothing_factor * varied_value + 
                         (1 - self.smoothing_factor) * self.previous_values[param_name])
        
        # Store for next iteration
        self.previous_values[param_name] = smoothed_value
        
        return smoothed_value
    
    def _apply_fault_effects(self, readings: Dict[str, float]) -> Dict[str, float]:
        """Apply fault effects to sensor readings with smooth transitions"""
        active_faults = self.fault_injector.get_active_faults()
        
        for fault in active_faults:
            if fault == 'bearing_fault':
                # Gradual increase in vibration and bearing temperature
                readings['vibration_rms'] += self._add_smooth_variation('bearing_fault_vib', 5.5)
                readings['bearing_temp'] += self._add_smooth_variation('bearing_fault_temp', 25.0) - 25.0
                
            elif fault == 'rotor_imbalance':
                # Smooth periodic vibration at 1x shaft frequency
                phase = (time.time() * 2 * math.pi * (readings['rpm'] / 60)) % (2 * math.pi)
                imbalance_amplitude = 2.75 + 1.0 * math.sin(phase)  # Smoother amplitude
                readings['vibration_rms'] += self._add_smooth_variation('imbalance_vib', imbalance_amplitude) - 2.75
                
            elif fault == 'misalignment':
                # Gradual increase in vibration and temperatures
                readings['vibration_rms'] += self._add_smooth_variation('misalign_vib', 2.75) - 2.75
                readings['bearing_temp'] += self._add_smooth_variation('misalign_temp', 14.0) - 14.0
                readings['winding_temp'] += self._add_smooth_variation('misalign_wind', 10.0) - 10.0
                
            elif fault == 'cavitation':
                # Smooth flow reduction and pressure variations
                flow_reduction = self._add_smooth_variation('cavitation_flow', 0.7)  # 30% reduction
                readings['flow_lpm'] *= flow_reduction
                pressure_variation = self._add_smooth_variation('cavitation_press', 0.0)  # Centered around 0
                readings['pressure_bar'] += pressure_variation * 1.0  # ±1 bar variation
                readings['vibration_rms'] += self._add_smooth_variation('cavitation_vib', 2.0) - 2.0
                
            elif fault == 'seal_leak':
                # Very gradual oil level drop
                oil_reduction = self._add_smooth_variation('seal_oil', 0.3)  # 0.3% per reading
                readings['oil_level_pct'] -= oil_reduction
                readings['pressure_bar'] -= self._add_smooth_variation('seal_press', 0.5) - 0.5
                if readings['oil_level_pct'] < 30:
                    readings['seal_leak'] = 1
                    
            elif fault == 'overload':
                # Smooth current increase and RPM reduction
                current_multiplier = self._add_smooth_variation('overload_curr', 1.225)  # 22.5% increase
                readings['current_a'] *= current_multiplier
                rpm_multiplier = self._add_smooth_variation('overload_rpm', 0.9)  # 10% reduction
                readings['rpm'] *= rpm_multiplier
                readings['winding_temp'] += self._add_smooth_variation('overload_temp', 17.5) - 17.5
                
            elif fault == 'insulation_degradation':
                # Very gradual temperature increase
                temp_increase = self._add_smooth_variation('insulation_temp', 1.25) - 1.25
                readings['winding_temp'] += temp_increase
        
        return readings
    
    def _generate_vibration_sample(self, base_rms: float, active_faults: List[str]) -> float:
        """Generate single vibration sample with fault characteristics"""
        # Determine fault type for vibration generation
        fault_type = 'none'
        if 'rotor_imbalance' in active_faults:
            fault_type = 'imbalance'
        elif 'bearing_fault' in active_faults:
            fault_type = 'bearing'
        elif 'misalignment' in active_faults:
            fault_type = 'misalignment'
        
        # Generate short vibration signal
        duration = 0.1  # 100ms sample
        sampling_rate = 1000  # 1kHz
        rpm = self.current_values.get('rpm', 1800)
        
        vibration_signal = generate_synthetic_vibration(
            duration=duration,
            sampling_rate=sampling_rate,
            base_amplitude=base_rms,
            rpm=rpm,
            fault_type=fault_type
        )
        
        # Return RMS of the signal
        return np.sqrt(np.mean(np.array(vibration_signal)**2))
    
    def simulate_reading(self) -> Dict[str, Any]:
        """Generate one complete sensor reading with smooth variations for clean ThingSpeak display"""
        # Check for new faults and fault expiry
        new_faults = self.fault_injector.check_new_faults()
        self.fault_injector.check_fault_expiry()
        
        # Generate smooth base readings using exponential smoothing
        readings = {}
        ranges = self.sensor_ranges
        
        # Generate target values based on nominal ranges with smooth variations
        target_values = {
            'vibration_rms': (ranges.vibration_rms_min + ranges.vibration_rms_max) / 2,
            'bearing_temp': (ranges.bearing_temp_min + ranges.bearing_temp_max) / 2,
            'winding_temp': (ranges.winding_temp_min + ranges.winding_temp_max) / 2,
            'current_a': ranges.current_rated * 0.8,  # 80% load
            'voltage_v': ranges.voltage_nominal,
            'flow_lpm': ranges.flow_nominal,
            'pressure_bar': ranges.pressure_nominal,
            'rpm': ranges.rpm_nominal,
            'oil_level_pct': (ranges.oil_level_min + ranges.oil_level_max) / 2,
            'seal_leak': 0
        }
        
        # Apply smooth variations to each parameter
        for param_name, target_value in target_values.items():
            if param_name != 'seal_leak':  # Binary sensor handled separately
                readings[param_name] = self._add_smooth_variation(param_name, target_value)
            else:
                readings[param_name] = target_value
        
        # Apply fault effects (these will be more dramatic changes)
        readings = self._apply_fault_effects(readings)
        
        # Enhanced vibration with spectral characteristics (smoothed)
        active_faults = self.fault_injector.get_active_faults()
        if active_faults:
            # For faults, add controlled vibration increase
            fault_vibration = self._generate_vibration_sample(readings['vibration_rms'], active_faults)
            readings['vibration_rms'] = self._add_smooth_variation('vibration_fault', fault_vibration)
        
        # Clamp values to realistic ranges
        readings = self._clamp_values(readings)
        
        # Update current values for next iteration
        self.current_values.update(readings)
        
        # Create complete sensor payload
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Create clean payload with optimized precision for ThingSpeak visualization
        payload = {
            'ts': timestamp,
            'vibration_rms': round(readings['vibration_rms'], 1),     # 1 decimal for cleaner charts
            'bearing_temp': round(readings['bearing_temp'], 0),       # Whole degrees for temperature
            'winding_temp': round(readings['winding_temp'], 0),       # Whole degrees for temperature
            'current_a': round(readings['current_a'], 0),             # Whole amperes for cleaner display
            'voltage_v': round(readings['voltage_v'], 0),             # Whole volts
            'flow_lpm': round(readings['flow_lpm'], 0),               # Whole liters per minute
            'pressure_bar': round(readings['pressure_bar'], 1),       # 1 decimal for pressure
            'rpm': round(readings['rpm'], 0),                         # Whole RPM
            'oil_level_pct': round(readings['oil_level_pct'], 0),     # Whole percentage
            'seal_leak': int(readings['seal_leak']),
            'sim_fault': ','.join(active_faults) if active_faults else 'none'
        }
        
        return payload
    
    def _clamp_values(self, readings: Dict[str, float]) -> Dict[str, float]:
        """Clamp sensor values to realistic physical limits"""
        ranges = self.sensor_ranges
        
        # Vibration - no upper clamp (can spike very high during faults)
        readings['vibration_rms'] = max(0.0, readings['vibration_rms'])
        
        # Temperatures - physical limits
        readings['bearing_temp'] = max(20, min(150, readings['bearing_temp']))
        readings['winding_temp'] = max(20, min(200, readings['winding_temp']))
        
        # Current - can't be negative, limited by protection
        readings['current_a'] = max(0, min(200, readings['current_a']))
        
        # Voltage - grid stability limits
        voltage_min = ranges.voltage_nominal * (1 - ranges.voltage_tolerance)
        voltage_max = ranges.voltage_nominal * (1 + ranges.voltage_tolerance)
        readings['voltage_v'] = max(voltage_min, min(voltage_max, readings['voltage_v']))
        
        # Flow - can't be negative
        readings['flow_lpm'] = max(0, min(2500, readings['flow_lpm']))
        
        # Pressure - can't be negative  
        readings['pressure_bar'] = max(0, min(15, readings['pressure_bar']))
        
        # RPM - can't be negative
        readings['rpm'] = max(0, min(4000, readings['rpm']))
        
        # Oil level - percentage
        readings['oil_level_pct'] = max(0, min(100, readings['oil_level_pct']))
        
        return readings
    
    def run_simulation_offline(self):
        """Run simulation in offline mode (no MQTT publishing)"""
        logger.info("Starting Pump Motor Sensor Simulator in OFFLINE MODE...")
        logger.info(f"Sampling interval: {self.sampling_interval} seconds")
        logger.info("Sensor data will be displayed but not published to MQTT")
        
        try:
            iteration = 0
            while True:
                # Generate sensor reading
                sensor_data = self.simulate_reading()
                
                # Display data locally
                active_faults = sensor_data['sim_fault']
                logger.info(f"Reading #{iteration}: HI_estimate={self._estimate_health_index(sensor_data):.1f}% | "
                          f"Vibration={sensor_data['vibration_rms']:.1f} m/s² | "
                          f"Bearing_temp={sensor_data['bearing_temp']:.0f}°C | "
                          f"Current={sensor_data['current_a']:.0f}A | "
                          f"Faults={active_faults}")
                
                iteration += 1
                time.sleep(self.sampling_interval)
                
        except KeyboardInterrupt:
            logger.info("Offline simulation stopped by user")
        except Exception as e:
            logger.error(f"Offline simulation error: {e}")
    
    def run_simulation(self):
        """Main simulation loop with MQTT publishing"""
        if not self.mqtt_client:
            logger.warning("No MQTT client available, switching to offline mode")
            return self.run_simulation_offline()
            
        logger.info("Starting Pump Motor Sensor Simulator...")
        logger.info(f"Publishing to topic: {Topics.SENSOR_DATA}")
        logger.info(f"Sampling interval: {self.sampling_interval} seconds")
        
        try:
            # Test first sensor reading generation
            logger.info("Generating first sensor reading...")
            test_data = self.simulate_reading()
            logger.info("First sensor reading generated successfully")
            
            iteration = 0
            while True:
                # Generate sensor reading
                sensor_data = self.simulate_reading()
                
                # Publish to MQTT
                if self.mqtt_client and self.mqtt_client.connected:
                    success = self.mqtt_client.publish(Topics.SENSOR_DATA, sensor_data)
                    
                    if success:
                        active_faults = sensor_data['sim_fault']
                        logger.info(f"Reading #{iteration}: HI_estimate={self._estimate_health_index(sensor_data):.1f}% | "
                                  f"Vibration={sensor_data['vibration_rms']:.1f} m/s² | "
                                  f"Bearing_temp={sensor_data['bearing_temp']:.0f}°C | "
                                  f"Current={sensor_data['current_a']:.0f}A | "
                                  f"Faults={active_faults}")
                    else:
                        logger.warning(f"Failed to publish reading #{iteration} - continuing in offline mode")
                        active_faults = sensor_data['sim_fault']
                        logger.info(f"OFFLINE Reading #{iteration}: HI_estimate={self._estimate_health_index(sensor_data):.1f}% | "
                                  f"Vibration={sensor_data['vibration_rms']:.1f} m/s² | "
                                  f"Bearing_temp={sensor_data['bearing_temp']:.0f}°C | "
                                  f"Current={sensor_data['current_a']:.0f}A | "
                                  f"Faults={active_faults}")
                else:
                    # MQTT not available, run offline
                    active_faults = sensor_data['sim_fault']
                    logger.info(f"OFFLINE Reading #{iteration}: HI_estimate={self._estimate_health_index(sensor_data):.1f}% | "
                              f"Vibration={sensor_data['vibration_rms']:.1f} m/s² | "
                              f"Bearing_temp={sensor_data['bearing_temp']:.0f}°C | "
                              f"Current={sensor_data['current_a']:.0f}A | "
                              f"Faults={active_faults}")
                
                iteration += 1
                time.sleep(self.sampling_interval)
                
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise  # Re-raise to see the full error
    
    def _estimate_health_index(self, reading: Dict[str, Any]) -> float:
        """Quick health index estimate for logging (simplified version)"""
        ranges = self.sensor_ranges
        
        # Vibration score (35% weight)
        vib_score = min(100, (ranges.vibration_rms_critical - reading['vibration_rms']) / 
                       ranges.vibration_rms_critical * 100)
        
        # Temperature score (25% weight)
        temp_score = min(100, (ranges.bearing_temp_critical - reading['bearing_temp']) / 
                        ranges.bearing_temp_critical * 100)
        
        # Current score (15% weight)
        current_score = min(100, (ranges.current_critical - reading['current_a']) / 
                           ranges.current_critical * 100)
        
        # Flow/Pressure score (15% weight)
        flow_deviation = abs(reading['flow_lpm'] - ranges.flow_nominal) / ranges.flow_nominal
        flow_score = max(0, 100 - flow_deviation * 200)  # 50% deviation = 0 score
        
        # Oil level score (10% weight)
        oil_score = max(0, reading['oil_level_pct'])
        
        # Weighted average
        health_index = (vib_score * 0.35 + temp_score * 0.25 + current_score * 0.15 + 
                       flow_score * 0.15 + oil_score * 0.10)
        
        return max(0, min(100, health_index))

def main():
    """Main entry point"""
    logger.info("Starting Pump Motor Sensor Simulator...")
    
    # Initialize MQTT client with unique ID
    mqtt_client = MQTTClient(
        broker_host="broker.hivemq.com",
        broker_port=1883,
        client_id=f"pump_sensor_sim_{random.randint(1000, 9999)}"
    )
    
    try:
        # Connect to MQTT broker with retry logic
        logger.info("Connecting to MQTT broker...")
        if not mqtt_client.connect(timeout=20, max_retries=2):
            logger.error("Failed to connect to any MQTT broker")
            logger.info("Running in offline mode - sensor data will be generated but not published")
            
            # Create simulator without MQTT
            simulator = PumpMotorSensorSimulator(None)
            simulator.run_simulation_offline()
            return
        
        logger.info("MQTT connection established successfully")
        
        # Give connection a moment to stabilize
        time.sleep(3)
        
        # Create and run simulator
        simulator = PumpMotorSensorSimulator(mqtt_client)
        simulator.run_simulation()
        
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Ensure cleanup
        try:
            if mqtt_client and mqtt_client.connected:
                mqtt_client.disconnect()
                logger.info("Disconnected from MQTT broker")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()