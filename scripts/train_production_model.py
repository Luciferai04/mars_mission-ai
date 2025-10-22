#!/usr/bin/env python3
"""
Production Model Training for Mars Hazard Detection

Trains a machine learning model on DEM data to predict terrain hazards,
slopes, and traversability for autonomous rover planning.
"""

import sys
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class HazardDetectionTrainer:
    """Trains ML models for Mars terrain hazard detection."""
    
    def __init__(self, dem_path: str, output_dir: str = "models"):
        self.dem_path = Path(dem_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dem_data = None
        self.features = None
        self.labels = None
        
        self.model = None
        
    def load_dem(self):
        """Load DEM data."""
        logger.info(f"Loading DEM from {self.dem_path}")
        
        if self.dem_path.suffix == '.npy':
            self.dem_data = np.load(self.dem_path)
        else:
            raise ValueError(f"Unsupported DEM format: {self.dem_path.suffix}")
        
        logger.info(f"  DEM shape: {self.dem_data.shape}")
        logger.info(f"  Elevation range: {self.dem_data.min():.1f}m to {self.dem_data.max():.1f}m")
    
    def extract_features(self):
        """Extract terrain features from DEM."""
        logger.info("Extracting terrain features...")
        
        height, width = self.dem_data.shape
        
        # Compute gradients (slopes)
        grad_y, grad_x = np.gradient(self.dem_data)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        slope_deg = np.degrees(np.arctan(slope))
        
        # Compute curvature (roughness)
        grad2_y, grad2_x = np.gradient(slope)
        curvature = np.sqrt(grad2_x**2 + grad2_y**2)
        
        # Compute local elevation variance (rocks)
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(self.dem_data, size=5)
        local_var = uniform_filter((self.dem_data - local_mean)**2, size=5)
        roughness = np.sqrt(local_var)
        
        # Create feature matrix
        features = []
        labels = []
        
        # Sample grid points (avoid edges)
        margin = 5
        sample_points = []
        
        for i in range(margin, height - margin, 2):  # Sample every 2 pixels
            for j in range(margin, width - margin, 2):
                sample_points.append((i, j))
        
        for i, j in sample_points:
            # Feature vector for this location
            feature_vec = [
                self.dem_data[i, j],          # Elevation
                slope_deg[i, j],              # Slope
                curvature[i, j],              # Curvature
                roughness[i, j],              # Roughness
                slope_deg[i-2:i+3, j-2:j+3].mean(),  # Local mean slope
                slope_deg[i-2:i+3, j-2:j+3].std(),   # Local slope variance
                roughness[i-2:i+3, j-2:j+3].mean(),  # Local mean roughness
            ]
            
            features.append(feature_vec)
            
            # Label: SAFE (0), CAUTION (1), HAZARD (2)
            label = self._compute_label(slope_deg[i, j], roughness[i, j], curvature[i, j])
            labels.append(label)
        
        self.features = np.array(features)
        self.labels = np.array(labels)
        
        logger.info(f"  Extracted {len(features)} samples")
        logger.info(f"  Feature dimensions: {self.features.shape[1]}")
        
        # Distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        label_names = ['SAFE', 'CAUTION', 'HAZARD']
        for label, count in zip(unique, counts):
            logger.info(f"  {label_names[label]}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    def _compute_label(self, slope: float, roughness: float, curvature: float) -> int:
        """Compute ground truth label from terrain metrics.
        
        NASA Perseverance constraints:
        - Slope >30° = HAZARD (cannot traverse)
        - Slope 15-30° = CAUTION (reduced speed)
        - Slope <15° = SAFE
        - High roughness (>2m variance) = CAUTION/HAZARD
        """
        # Slope-based classification
        if slope > 30:
            return 2  # HAZARD
        elif slope > 15:
            slope_class = 1  # CAUTION
        else:
            slope_class = 0  # SAFE
        
        # Roughness-based adjustment
        if roughness > 3.0:
            return 2  # HAZARD (very rough)
        elif roughness > 1.5:
            roughness_class = 1  # CAUTION
        else:
            roughness_class = 0  # SAFE
        
        # Curvature-based adjustment
        if curvature > 0.5:
            curvature_class = 1  # CAUTION (highly curved = unstable)
        else:
            curvature_class = 0
        
        # Combined label (most conservative)
        return max(slope_class, roughness_class, curvature_class)
    
    def train_model(self, test_size: float = 0.2):
        """Train Random Forest hazard classifier."""
        logger.info("Training Random Forest classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_size,
            random_state=42,
            stratify=self.labels
        )
        
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Test samples: {len(X_test)}")
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.model.fit(X_train, y_train)
        
        logger.info("   Model trained")
        
        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        logger.info(f"  Training accuracy: {train_acc:.3f}")
        logger.info(f"  Test accuracy: {test_acc:.3f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        
        logger.info("\n  Classification Report:")
        report = classification_report(
            y_test, y_pred,
            target_names=['SAFE', 'CAUTION', 'HAZARD'],
            zero_division=0
        )
        print(report)
        
        logger.info("  Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Feature importance
        feature_names = [
            'elevation', 'slope', 'curvature', 'roughness',
            'local_mean_slope', 'local_slope_std', 'local_mean_roughness'
        ]
        
        importances = self.model.feature_importances_
        logger.info("\n  Feature Importances:")
        for name, importance in sorted(zip(feature_names, importances), 
                                      key=lambda x: x[1], reverse=True):
            logger.info(f"    {name}: {importance:.3f}")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importances': dict(zip(feature_names, importances.tolist()))
        }
    
    def save_model(self, name: str = "hazard_detector"):
        """Save trained model."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.output_dir / f"{name}_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info(f"\n Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': name,
            'trained_at': datetime.utcnow().isoformat(),
            'dem_path': str(self.dem_path),
            'num_samples': len(self.features),
            'num_features': self.features.shape[1],
            'model_type': 'RandomForestClassifier',
            'model_params': self.model.get_params(),
            'feature_names': [
                'elevation', 'slope', 'curvature', 'roughness',
                'local_mean_slope', 'local_slope_std', 'local_mean_roughness'
            ]
        }
        
        metadata_path = self.output_dir / f"{name}_{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f" Metadata saved to: {metadata_path}")
        
        # Create "latest" symlinks
        latest_model = self.output_dir / f"{name}_latest.pkl"
        latest_meta = self.output_dir / f"{name}_latest_metadata.json"
        
        if latest_model.exists():
            latest_model.unlink()
        if latest_meta.exists():
            latest_meta.unlink()
        
        latest_model.symlink_to(model_path.name)
        latest_meta.symlink_to(metadata_path.name)
        
        logger.info(f" Latest model linked: {latest_model}")
        
        return model_path, metadata_path


def main():
    """Train production hazard detection model."""
    
    print("\n" + "="*80)
    print("MARS HAZARD DETECTION MODEL TRAINING")
    print("="*80 + "\n")
    
    # Configuration
    dem_path = "data/dem/jezero_demo.npy"
    output_dir = "models"
    
    # Check DEM exists
    if not Path(dem_path).exists():
        logger.error(f"DEM not found: {dem_path}")
        logger.error("Please generate DEM first or provide path to real DEM")
        return 1
    
    # Initialize trainer
    trainer = HazardDetectionTrainer(dem_path, output_dir)
    
    # Training pipeline
    try:
        # 1. Load DEM
        trainer.load_dem()
        
        # 2. Extract features
        trainer.extract_features()
        
        # 3. Train model
        metrics = trainer.train_model(test_size=0.2)
        
        # 4. Save model
        model_path, meta_path = trainer.save_model("hazard_detector")
        
        print("\n" + "="*80)
        print(" TRAINING COMPLETE")
        print("="*80)
        print(f"\nModel: {model_path}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.1%}")
        print(f"\nThe model can now predict terrain safety:")
        print("  - SAFE: Slopes <15°, smooth terrain")
        print("  - CAUTION: Slopes 15-30°, moderate roughness")
        print("  - HAZARD: Slopes >30°, very rough terrain")
        print("\nUse in production system:")
        print("  from src.core.production_mission_system import ProductionMissionSystem")
        print("  system = ProductionMissionSystem()")
        print("  # Model automatically loaded for terrain analysis")
        print()
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
