#!/usr/bin/env python3
"""
Build hazard training dataset from DEM.

Extracts terrain features and labels from DEM for model training.
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from scipy.ndimage import generic_filter

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.dem_processor import DEMProcessor


def compute_advanced_features(elevation, pixel_size):
    """Compute advanced terrain features."""
    
    # Slope
    dx = np.gradient(elevation, pixel_size[0], axis=1)
    dy = np.gradient(elevation, pixel_size[1], axis=0)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # Curvature
    dxx = np.gradient(dx, pixel_size[0], axis=1)
    dyy = np.gradient(dy, pixel_size[1], axis=0)
    curvature = dxx + dyy
    
    # Roughness (std dev in 3x3 window)
    def local_std(values):
        return np.std(values) if len(values) > 0 else 0.0
    roughness = generic_filter(elevation, local_std, size=3, mode='constant')
    
    # Slope variance
    def slope_var(values):
        return np.var(values) if len(values) > 0 else 0.0
    slope_variance = generic_filter(slope, slope_var, size=3, mode='constant')
    
    return slope, curvature, roughness, slope_variance


def label_pixel(slope):
    """Label pixel based on slope."""
    if slope <= 15.0:
        return 0, "SAFE"
    elif slope <= 30.0:
        return 1, "CAUTION"
    else:
        return 2, "HAZARD"


def extract_samples(elevation, features, metadata, num_samples=10000):
    """Extract random balanced samples."""
    h, w = elevation.shape
    slope, curvature, roughness, slope_variance = features
    
    samples = []
    
    # Target samples per class (balanced)
    samples_per_class = {0: num_samples // 3, 1: num_samples // 3, 2: num_samples // 3}
    class_counts = {0: 0, 1: 0, 2: 0}
    
    # Random sampling with replacement until we get enough per class
    max_attempts = num_samples * 10
    attempts = 0
    
    while (min(class_counts.values()) < num_samples // 3) and attempts < max_attempts:
        # Random position
        i = np.random.randint(5, h-5)
        j = np.random.randint(5, w-5)
        
        # Extract features
        slope_val = float(slope[i, j])
        curv_val = float(curvature[i, j])
        rough_val = float(roughness[i, j])
        slope_var_val = float(slope_variance[i, j])
        elev_val = float(elevation[i, j])
        
        # Skip NaN or invalid
        if not all(np.isfinite([slope_val, curv_val, rough_val, slope_var_val])):
            attempts += 1
            continue
        
        # Get label
        label, label_name = label_pixel(slope_val)
        
        # Check if we need more of this class
        if class_counts[label] >= samples_per_class[label]:
            attempts += 1
            continue
        
        # Add sample
        sample = {
            'features': {
                'slope_deg': slope_val,
                'elevation_m': elev_val,
                'curvature': curv_val,
                'roughness': rough_val,
                'local_slope_variance': slope_var_val
            },
            'label': int(label),
            'label_name': label_name
        }
        
        samples.append(sample)
        class_counts[label] += 1
        attempts += 1
    
    return samples, class_counts


def main():
    parser = argparse.ArgumentParser(
        description='Build hazard training dataset from DEM'
    )
    parser.add_argument('--dem', required=True, help='Path to DEM GeoTIFF')
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of samples (default: 10000)')
    parser.add_argument('--lat', type=float, default=18.4,
                       help='Approximate latitude for metadata')
    
    args = parser.parse_args()
    
    print(f"Building hazard dataset from {args.dem}")
    
    # Load DEM
    processor = DEMProcessor()
    
    try:
        elevation, metadata = processor.load_dem(args.dem)
        print(f"Loaded DEM: {metadata['width']}x{metadata['height']}")
    except Exception as e:
        print(f"Error loading DEM: {e}")
        sys.exit(1)
    
    # Compute features
    print("Computing terrain features...")
    features = compute_advanced_features(elevation, metadata['pixel_size'])
    
    # Extract samples
    print(f"Extracting {args.samples} samples...")
    samples, class_counts = extract_samples(elevation, features, metadata, args.samples)
    
    # Build dataset
    dataset = {
        'samples': samples,
        'metadata': {
            'dem_source': args.dem,
            'total_samples': len(samples),
            'class_distribution': {
                'SAFE': class_counts[0],
                'CAUTION': class_counts[1],
                'HAZARD': class_counts[2]
            },
            'latitude': args.lat,
            'pixel_size_m': metadata['pixel_size'][0]
        }
    }
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nDataset saved to {output_path}")
    print(f"Total samples: {len(samples)}")
    print(f"Class distribution:")
    print(f"  SAFE: {class_counts[0]}")
    print(f"  CAUTION: {class_counts[1]}")
    print(f"  HAZARD: {class_counts[2]}")


if __name__ == '__main__':
    main()
