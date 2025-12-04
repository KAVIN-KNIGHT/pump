"""
Signal Processing Utilities for IIoT Pump Motor System
Provides FFT analysis, statistical processing, and anomaly detection functions
"""

import numpy as np
from scipy import fft
from scipy.signal import welch, find_peaks
from typing import List, Tuple, Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VibrationAnalyzer:
    """FFT-based vibration analysis for pump motor fault detection"""
    
    def __init__(self, sampling_rate: float = 1000.0, rpm: float = 1800.0):
        """
        Initialize vibration analyzer
        
        Args:
            sampling_rate: Sampling rate in Hz
            rpm: Motor rotational speed in RPM
        """
        self.sampling_rate = sampling_rate
        self.rpm = rpm
        self.shaft_freq = rpm / 60.0  # Convert RPM to Hz
        
        # Bearing defect frequency multipliers (typical values)
        self.bearing_multipliers = {
            'bpfo': 0.6,   # Ball Pass Frequency Outer race
            'bpfi': 1.4,   # Ball Pass Frequency Inner race  
            'bsf': 0.4,    # Ball Spin Frequency
            'ftf': 0.4     # Fundamental Train Frequency
        }
    
    def compute_fft(self, signal: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of vibration signal
        
        Args:
            signal: Time domain vibration signal
            
        Returns:
            Tuple of (frequencies, magnitudes)
        """
        try:
            signal_array = np.array(signal)
            
            # Remove DC component
            signal_array = signal_array - np.mean(signal_array)
            
            # Apply window to reduce spectral leakage
            windowed_signal = signal_array * np.hanning(len(signal_array))
            
            # Compute FFT
            fft_result = fft.fft(windowed_signal)
            frequencies = fft.fftfreq(len(signal_array), 1/self.sampling_rate)
            
            # Take only positive frequencies
            positive_freq_idx = frequencies >= 0
            frequencies = frequencies[positive_freq_idx]
            magnitudes = np.abs(fft_result[positive_freq_idx])
            
            return frequencies, magnitudes
            
        except Exception as e:
            logger.error(f"Error computing FFT: {e}")
            return np.array([]), np.array([])
    
    def detect_shaft_harmonics(self, frequencies: np.ndarray, magnitudes: np.ndarray, 
                             num_harmonics: int = 5) -> Dict[str, float]:
        """
        Detect shaft speed harmonics in frequency spectrum
        
        Args:
            frequencies: Frequency array from FFT
            magnitudes: Magnitude array from FFT
            num_harmonics: Number of harmonics to analyze
            
        Returns:
            Dictionary with harmonic amplitudes
        """
        results = {}
        
        try:
            for harmonic in range(1, num_harmonics + 1):
                target_freq = self.shaft_freq * harmonic
                
                # Find closest frequency bin
                freq_idx = np.argmin(np.abs(frequencies - target_freq))
                
                # Check if we're close enough to the target frequency
                if abs(frequencies[freq_idx] - target_freq) < (self.shaft_freq * 0.1):
                    results[f'{harmonic}x_shaft'] = magnitudes[freq_idx]
                else:
                    results[f'{harmonic}x_shaft'] = 0.0
                    
            return results
            
        except Exception as e:
            logger.error(f"Error detecting shaft harmonics: {e}")
            return {}
    
    def detect_bearing_frequencies(self, frequencies: np.ndarray, 
                                 magnitudes: np.ndarray) -> Dict[str, float]:
        """
        Detect bearing defect frequencies
        
        Args:
            frequencies: Frequency array from FFT
            magnitudes: Magnitude array from FFT
            
        Returns:
            Dictionary with bearing defect frequency amplitudes
        """
        results = {}
        
        try:
            for defect_type, multiplier in self.bearing_multipliers.items():
                target_freq = self.shaft_freq * multiplier
                
                # Find closest frequency bin
                freq_idx = np.argmin(np.abs(frequencies - target_freq))
                
                # Check if we're close enough to the target frequency
                if abs(frequencies[freq_idx] - target_freq) < (self.shaft_freq * 0.05):
                    results[defect_type] = magnitudes[freq_idx]
                else:
                    results[defect_type] = 0.0
                    
            return results
            
        except Exception as e:
            logger.error(f"Error detecting bearing frequencies: {e}")
            return {}
    
    def analyze_spectrum(self, signal: List[float]) -> Dict[str, any]:
        """
        Complete spectral analysis of vibration signal
        
        Args:
            signal: Time domain vibration signal
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Compute FFT
            frequencies, magnitudes = self.compute_fft(signal)
            
            if len(frequencies) == 0:
                return {}
            
            # Overall RMS
            rms = np.sqrt(np.mean(np.array(signal)**2))
            
            # Peak frequency
            peak_idx = np.argmax(magnitudes)
            peak_frequency = frequencies[peak_idx]
            peak_amplitude = magnitudes[peak_idx]
            
            # Shaft harmonics
            shaft_harmonics = self.detect_shaft_harmonics(frequencies, magnitudes)
            
            # Bearing frequencies  
            bearing_frequencies = self.detect_bearing_frequencies(frequencies, magnitudes)
            
            # Spectral energy distribution
            total_energy = np.sum(magnitudes**2)
            low_freq_energy = np.sum(magnitudes[frequencies < 100]**2) / total_energy
            mid_freq_energy = np.sum(magnitudes[(frequencies >= 100) & (frequencies < 1000)]**2) / total_energy
            high_freq_energy = np.sum(magnitudes[frequencies >= 1000]**2) / total_energy
            
            return {
                'rms': rms,
                'peak_frequency': peak_frequency,
                'peak_amplitude': peak_amplitude,
                'shaft_harmonics': shaft_harmonics,
                'bearing_frequencies': bearing_frequencies,
                'energy_distribution': {
                    'low_freq': low_freq_energy,
                    'mid_freq': mid_freq_energy, 
                    'high_freq': high_freq_energy
                },
                'total_energy': total_energy
            }
            
        except Exception as e:
            logger.error(f"Error in spectrum analysis: {e}")
            return {}

class RollingStatistics:
    """Calculate rolling statistics for sensor data"""
    
    def __init__(self, window_size: int = 10):
        """
        Initialize rolling statistics calculator
        
        Args:
            window_size: Number of samples for rolling window
        """
        self.window_size = window_size
        self.data_buffers = {}
    
    def update(self, sensor_name: str, value: float) -> Dict[str, float]:
        """
        Update rolling statistics for a sensor
        
        Args:
            sensor_name: Name of the sensor
            value: New sensor value
            
        Returns:
            Dictionary with rolling statistics
        """
        # Initialize buffer if it doesn't exist
        if sensor_name not in self.data_buffers:
            self.data_buffers[sensor_name] = []
        
        # Add new value
        self.data_buffers[sensor_name].append(value)
        
        # Keep only the latest window_size values
        if len(self.data_buffers[sensor_name]) > self.window_size:
            self.data_buffers[sensor_name].pop(0)
        
        # Calculate statistics
        data = np.array(self.data_buffers[sensor_name])
        
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'latest': value,
            'samples': len(data)
        }
    
    def get_trend(self, sensor_name: str) -> Optional[float]:
        """
        Calculate trend (slope) for sensor data
        
        Args:
            sensor_name: Name of the sensor
            
        Returns:
            Trend slope or None if insufficient data
        """
        if sensor_name not in self.data_buffers:
            return None
            
        data = self.data_buffers[sensor_name]
        if len(data) < 3:
            return None
        
        # Simple linear trend calculation
        x = np.arange(len(data))
        y = np.array(data)
        
        # Linear regression
        slope = np.polyfit(x, y, 1)[0]
        return slope

class AnomalyDetector:
    """Statistical anomaly detection for sensor readings"""
    
    def __init__(self, threshold_multiplier: float = 3.0):
        """
        Initialize anomaly detector
        
        Args:
            threshold_multiplier: Number of standard deviations for anomaly threshold
        """
        self.threshold_multiplier = threshold_multiplier
        self.rolling_stats = RollingStatistics(window_size=30)
    
    def detect_anomaly(self, sensor_name: str, value: float) -> Dict[str, any]:
        """
        Detect if current value is anomalous
        
        Args:
            sensor_name: Name of the sensor
            value: Current sensor value
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Update rolling statistics
        stats = self.rolling_stats.update(sensor_name, value)
        
        # Need at least 5 samples for meaningful detection
        if stats['samples'] < 5:
            return {
                'is_anomaly': False,
                'deviation': 0.0,
                'threshold': 0.0,
                'stats': stats
            }
        
        # Calculate deviation from mean
        deviation = abs(value - stats['mean'])
        threshold = self.threshold_multiplier * stats['std']
        
        is_anomaly = deviation > threshold and stats['std'] > 0
        
        return {
            'is_anomaly': is_anomaly,
            'deviation': deviation,
            'threshold': threshold,
            'z_score': deviation / stats['std'] if stats['std'] > 0 else 0,
            'stats': stats
        }

def generate_synthetic_vibration(duration: float, sampling_rate: float, 
                               base_amplitude: float = 1.0, rpm: float = 1800.0,
                               fault_type: str = 'none') -> List[float]:
    """
    Generate synthetic vibration signal with optional fault injection
    
    Args:
        duration: Signal duration in seconds
        sampling_rate: Sampling rate in Hz
        base_amplitude: Base vibration amplitude
        rpm: Motor RPM for shaft frequency
        fault_type: Type of fault to inject ('none', 'imbalance', 'bearing', 'misalignment')
        
    Returns:
        List of vibration samples
    """
    num_samples = int(duration * sampling_rate)
    time_vector = np.linspace(0, duration, num_samples)
    
    shaft_freq = rpm / 60.0  # Convert to Hz
    
    # Base signal with some random noise
    signal = base_amplitude * (0.1 * np.random.randn(num_samples))
    
    # Add shaft frequency component
    signal += 0.2 * base_amplitude * np.sin(2 * np.pi * shaft_freq * time_vector)
    
    # Add fault-specific components
    if fault_type == 'imbalance':
        # Strong 1x shaft component with some 2x
        signal += 2.0 * base_amplitude * np.sin(2 * np.pi * shaft_freq * time_vector)
        signal += 0.5 * base_amplitude * np.sin(2 * np.pi * 2 * shaft_freq * time_vector)
        
    elif fault_type == 'bearing':
        # Add bearing defect frequencies
        bpfo_freq = shaft_freq * 0.6
        signal += 1.5 * base_amplitude * np.sin(2 * np.pi * bpfo_freq * time_vector)
        # Add high frequency noise
        signal += 0.8 * base_amplitude * np.random.randn(num_samples)
        
    elif fault_type == 'misalignment':
        # Strong 2x shaft component with 3x
        signal += 1.5 * base_amplitude * np.sin(2 * np.pi * 2 * shaft_freq * time_vector)
        signal += 0.8 * base_amplitude * np.sin(2 * np.pi * 3 * shaft_freq * time_vector)
    
    return signal.tolist()