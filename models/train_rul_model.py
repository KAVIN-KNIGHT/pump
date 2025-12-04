"""
ML Model Training for RUL Prediction
Generates a simple RandomForest model for remaining useful life prediction
"""

import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Tuple, Dict
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RULModelTrainer:
    """Train a simple RUL prediction model using synthetic data"""
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            'health_index', 'vibration_score', 'thermal_score', 
            'electrical_score', 'hydraulic_score', 'mechanical_score',
            'vibration_trend', 'temperature_trend', 'current_trend'
        ]
        
    def generate_synthetic_training_data(self, num_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for RUL prediction"""
        logger.info(f"Generating {num_samples} synthetic samples for training...")
        
        # Generate random component health scores
        np.random.seed(42)  # For reproducibility
        
        # Component scores (0-100)
        vibration_scores = np.random.uniform(10, 100, num_samples)
        thermal_scores = np.random.uniform(20, 100, num_samples)
        electrical_scores = np.random.uniform(30, 100, num_samples)
        hydraulic_scores = np.random.uniform(25, 100, num_samples)
        mechanical_scores = np.random.uniform(20, 100, num_samples)
        
        # Calculate weighted health index
        health_indices = (
            vibration_scores * 0.35 +
            thermal_scores * 0.25 +
            electrical_scores * 0.15 +
            hydraulic_scores * 0.15 +
            mechanical_scores * 0.10
        )
        
        # Add some noise to health index
        health_indices += np.random.normal(0, 2, num_samples)
        health_indices = np.clip(health_indices, 0, 100)
        
        # Generate trend data (normalized slopes)
        vibration_trends = np.random.uniform(-0.01, 0.05, num_samples)  # Mostly positive (degrading)
        temperature_trends = np.random.uniform(-0.005, 0.03, num_samples)
        current_trends = np.random.uniform(-0.008, 0.02, num_samples)
        
        # Create feature matrix
        X = np.column_stack([
            health_indices,
            vibration_scores,
            thermal_scores,
            electrical_scores,
            hydraulic_scores,
            mechanical_scores,
            vibration_trends,
            temperature_trends,
            current_trends
        ])
        
        # Generate RUL targets based on realistic degradation patterns
        y = self._calculate_synthetic_rul(
            health_indices, vibration_scores, thermal_scores, 
            vibration_trends, temperature_trends
        )
        
        logger.info(f"Generated training data: X shape {X.shape}, y range [{y.min():.1f}, {y.max():.1f}] hours")
        
        return X, y
    
    def _calculate_synthetic_rul(self, health_indices: np.ndarray, vibration_scores: np.ndarray,
                                thermal_scores: np.ndarray, vibration_trends: np.ndarray,
                                temperature_trends: np.ndarray) -> np.ndarray:
        """Calculate synthetic RUL based on health patterns"""
        
        base_life = 8760  # 1 year in hours
        
        # Base RUL from health index
        rul_base = base_life * (health_indices / 100) ** 2
        
        # Apply component-specific effects
        vibration_effect = np.where(vibration_scores < 50, 0.3, 0.8)
        thermal_effect = np.where(thermal_scores < 60, 0.4, 0.9)
        
        # Apply trend effects (negative trends reduce RUL faster)
        trend_effect = 1.0 - (vibration_trends + temperature_trends) * 10
        trend_effect = np.clip(trend_effect, 0.1, 2.0)
        
        # Combine effects
        rul = rul_base * vibration_effect * thermal_effect * trend_effect
        
        # Add some realistic noise and constraints
        rul += np.random.normal(0, rul * 0.1)  # 10% noise
        
        # Apply realistic bounds
        rul = np.clip(rul, 1.0, base_life * 1.2)  # 1 hour to 1.2 years
        
        # Create some failure cases (very low RUL for poor health)
        critical_mask = health_indices < 30
        rul[critical_mask] = np.random.uniform(1, 168, np.sum(critical_mask))  # 1 hour to 1 week
        
        return rul
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train the RUL prediction model"""
        logger.info("Training RUL prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        logger.info(f"Model training completed:")
        logger.info(f"  Train MAE: {metrics['train_mae']:.1f} hours")
        logger.info(f"  Test MAE: {metrics['test_mae']:.1f} hours")
        logger.info(f"  Train R²: {metrics['train_r2']:.3f}")
        logger.info(f"  Test R²: {metrics['test_r2']:.3f}")
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        logger.info("Feature importance:")
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {feature}: {importance:.3f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model to file"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model with metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_date': datetime.now().isoformat(),
            'model_type': 'RandomForestRegressor',
            'version': '1.0'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from: {filepath}")
        logger.info(f"Model type: {model_data.get('model_type', 'Unknown')}")
        logger.info(f"Training date: {model_data.get('training_date', 'Unknown')}")
    
    def predict_rul(self, health_index: float, component_scores: Dict[str, float],
                   trend_data: Dict[str, float] = None) -> Tuple[float, Dict[str, any]]:
        """Predict RUL for given input"""
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        # Prepare input features
        if trend_data is None:
            trend_data = {'vibration_rms': 0.0, 'bearing_temp': 0.0, 'current_a': 0.0}
        
        features = np.array([[
            health_index,
            component_scores.get('vibration', 50),
            component_scores.get('thermal', 50),
            component_scores.get('electrical', 50),
            component_scores.get('hydraulic', 50),
            component_scores.get('mechanical', 50),
            trend_data.get('vibration_rms', 0.0),
            trend_data.get('bearing_temp', 0.0),
            trend_data.get('current_a', 0.0)
        ]])
        
        # Make prediction
        rul_prediction = self.model.predict(features)[0]
        
        # Get prediction confidence (using tree variance)
        tree_predictions = np.array([tree.predict(features)[0] for tree in self.model.estimators_])
        confidence = 1.0 - (np.std(tree_predictions) / np.mean(tree_predictions))
        confidence = max(0.0, min(1.0, confidence))
        
        metadata = {
            'method': 'random_forest',
            'confidence': round(confidence, 3),
            'prediction_std': round(np.std(tree_predictions), 1),
            'feature_vector': features.flatten().tolist()
        }
        
        return round(rul_prediction, 1), metadata

def create_rul_model():
    """Main function to create and save RUL model"""
    logger.info("=== RUL Model Training Started ===")
    
    # Initialize trainer
    trainer = RULModelTrainer()
    
    # Generate training data
    X, y = trainer.generate_synthetic_training_data(num_samples=3000)
    
    # Train model
    metrics = trainer.train_model(X, y)
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), 'rul_model.pkl')
    trainer.save_model(model_path)
    
    # Create a simple test
    logger.info("Testing model with sample prediction...")
    
    test_health_index = 75.0
    test_component_scores = {
        'vibration': 80.0,
        'thermal': 70.0,
        'electrical': 85.0,
        'hydraulic': 75.0,
        'mechanical': 80.0
    }
    test_trends = {
        'vibration_rms': 0.001,
        'bearing_temp': 0.002,
        'current_a': 0.0
    }
    
    rul, metadata = trainer.predict_rul(test_health_index, test_component_scores, test_trends)
    logger.info(f"Sample prediction: RUL = {rul:.1f} hours, Confidence = {metadata['confidence']:.3f}")
    
    logger.info("=== RUL Model Training Completed ===")
    return model_path

if __name__ == "__main__":
    create_rul_model()