#!/usr/bin/env python3
"""
Train Hazard Detection Model from Official GeoTIFF DEMs

- Loads Mars DEM GeoTIFFs (MOLA, HRSC/MOLA, Gale) using DEMProcessor
- Extracts a Jezero Crater regional subset (configurable)
- Computes slope and roughness metrics
- Derives labels from NASA slope thresholds (SAFE<15°, CAUTION 15-30°, HAZARD>30°)
- Trains a RandomForest classifier and saves the model
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
from typing import List, Tuple
import pickle
from scipy.ndimage import uniform_filter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.dem_processor import DEMProcessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


DEFAULT_JEZERO_BOUNDS = {
    'lat_min': 18.2,
    'lat_max': 18.7,
    'lon_min': 77.2,
    'lon_max': 77.7
}


def extract_features(elevation: np.ndarray, pixel_size: Tuple[float, float]):
    # Slope in degrees
    processor = DEMProcessor()
    slope_deg = processor.compute_slope(elevation, pixel_size)

    # Roughness via local elevation variance
    local_mean = uniform_filter(elevation, size=5)
    local_var = uniform_filter((elevation - local_mean) ** 2, size=5)
    roughness = np.sqrt(local_var)

    # Curvature via gradient of slope
    grad2_y, grad2_x = np.gradient(slope_deg)
    curvature = np.sqrt(grad2_x ** 2 + grad2_y ** 2)

    return slope_deg, roughness, curvature


def build_dataset(elevation: np.ndarray,
                  pixel_size: Tuple[float, float],
                  sample_stride: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and label vector y from elevation tile."""
    slope_deg, roughness, curvature = extract_features(elevation, pixel_size)

    H, W = elevation.shape
    X, y = [], []

    for i in range(5, H - 5, sample_stride):
        for j in range(5, W - 5, sample_stride):
            feat = [
                elevation[i, j],
                slope_deg[i, j],
                curvature[i, j],
                roughness[i, j],
                slope_deg[i - 2:i + 3, j - 2:j + 3].mean(),
                slope_deg[i - 2:i + 3, j - 2:j + 3].std(),
                roughness[i - 2:i + 3, j - 2:j + 3].mean(),
            ]
            X.append(feat)

            # NASA slope thresholds
            if slope_deg[i, j] > 30:
                label = 2  # HAZARD
            elif slope_deg[i, j] > 15:
                label = 1  # CAUTION
            else:
                label = 0  # SAFE
            y.append(label)

    return np.array(X), np.array(y)


def train_on_dems(dem_paths: List[str], bounds: dict, output: str) -> dict:
    processor = DEMProcessor()

    X_all, y_all = [], []

    for dem_path in dem_paths:
        dem_path = str(dem_path)
        print(f"\nProcessing DEM: {dem_path}")
        try:
            elev, meta = processor.extract_region(
                dem_path,
                bounds['lat_min'], bounds['lat_max'],
                bounds['lon_min'], bounds['lon_max']
            )
            print(f"  Region shape: {elev.shape}")
            X, y = build_dataset(elev, meta['pixel_size'], sample_stride=4)
            print(f"  Samples: {len(X)}")
            X_all.append(X)
            y_all.append(y)
        except Exception as e:
            print(f"    Skipping {dem_path}: {e}")
            continue

    if not X_all:
        raise RuntimeError("No samples built from provided DEMs.")

    X = np.vstack(X_all)
    y = np.hstack(y_all)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Classifier
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=18, min_samples_split=8, min_samples_leaf=4,
        class_weight='balanced', n_jobs=-1, random_state=42
    )

    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['SAFE', 'CAUTION', 'HAZARD'], zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Save model
    model_path = Path(output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    meta = {
        'trained_on': dem_paths,
        'bounds': bounds,
        'num_samples': int(len(X)),
        'class_distribution': {
            'SAFE': int((y == 0).sum()),
            'CAUTION': int((y == 1).sum()),
            'HAZARD': int((y == 2).sum()),
        },
        'confusion_matrix': cm,
        'classification_report': report,
        'feature_names': [
            'elevation', 'slope', 'curvature', 'roughness',
            'local_mean_slope', 'local_slope_std', 'local_mean_roughness'
        ]
    }

    with open(model_path.with_suffix('.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n Training complete")
    print(f"  Model: {model_path}")
    print("\nClassification Report:\n" + report)

    return meta


def parse_args():
    ap = argparse.ArgumentParser(description='Train hazard model from official DEM GeoTIFFs')
    ap.add_argument('--dem', action='append', required=True, help='Path to DEM GeoTIFF (repeatable)')
    ap.add_argument('--lat-min', type=float, default=DEFAULT_JEZERO_BOUNDS['lat_min'])
    ap.add_argument('--lat-max', type=float, default=DEFAULT_JEZERO_BOUNDS['lat_max'])
    ap.add_argument('--lon-min', type=float, default=DEFAULT_JEZERO_BOUNDS['lon_min'])
    ap.add_argument('--lon-max', type=float, default=DEFAULT_JEZERO_BOUNDS['lon_max'])
    ap.add_argument('--output', default='models/hazard_detector_mars_dems.pkl')
    return ap.parse_args()


def main():
    args = parse_args()
    bounds = {
        'lat_min': args.lat_min,
        'lat_max': args.lat_max,
        'lon_min': args.lon_min,
        'lon_max': args.lon_max
    }
    return 0 if train_on_dems(args.dem, bounds, args.output) else 1


if __name__ == '__main__':
    sys.exit(main())
