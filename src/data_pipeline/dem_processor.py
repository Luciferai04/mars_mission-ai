#!/usr/bin/env python3
"""
DEM Processing Pipeline for Mars Mission Planning

Handles GeoTIFF DEM loading, slope calculation, and hazard mapping
for NASA Mars MOLA, Gale Crater, and HRSC/MOLA blended DEMs.
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import generic_filter
import json
import logging


class DEMProcessor:
    """Process Mars DEM GeoTIFF files for mission planning."""
    
    def __init__(self, cache_dir: str = "./data/cache/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def load_dem(self, dem_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load DEM GeoTIFF and extract metadata.
        
        Args:
            dem_path: Path to DEM GeoTIFF file
            
        Returns:
            Tuple of (elevation_array, metadata_dict)
        """
        if not os.path.exists(dem_path):
            raise FileNotFoundError(f"DEM file not found: {dem_path}")
        
        self.logger.info(f"Loading DEM from {dem_path}")
        
        with rasterio.open(dem_path) as dataset:
            elevation = dataset.read(1)
            
            metadata = {
                'width': dataset.width,
                'height': dataset.height,
                'bounds': dataset.bounds,
                'transform': dataset.transform,
                'crs': str(dataset.crs),
                'pixel_size': (dataset.transform[0], abs(dataset.transform[4])),
                'nodata': dataset.nodata
            }
            
        self.logger.info(f"Loaded DEM: {metadata['width']}x{metadata['height']} pixels")
        return elevation, metadata
    
    def compute_slope(self, elevation: np.ndarray, pixel_size: Tuple[float, float]) -> np.ndarray:
        """Compute slope in degrees from elevation data.
        
        Args:
            elevation: 2D elevation array in meters
            pixel_size: Tuple of (pixel_width_m, pixel_height_m)
            
        Returns:
            2D slope array in degrees
        """
        self.logger.info("Computing slope from elevation data")
        
        # Compute gradients using Sobel-like kernel
        dx = np.gradient(elevation, pixel_size[0], axis=1)
        dy = np.gradient(elevation, pixel_size[1], axis=0)
        
        # Calculate slope magnitude in radians, then convert to degrees
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        
        # Handle NaN and invalid values
        slope_deg = np.nan_to_num(slope_deg, nan=0.0, posinf=90.0, neginf=0.0)
        
        self.logger.info(f"Slope computed: min={slope_deg.min():.2f}°, max={slope_deg.max():.2f}°, mean={slope_deg.mean():.2f}°")
        return slope_deg
    
    def identify_hazards(self, slope_deg: np.ndarray, 
                        max_safe_slope: float = 30.0,
                        preferred_slope: float = 15.0) -> Dict[str, Any]:
        """Identify hazardous terrain based on slope thresholds.
        
        Args:
            slope_deg: 2D slope array in degrees
            max_safe_slope: Maximum safe traversable slope (NASA limit)
            preferred_slope: Preferred operational slope limit
            
        Returns:
            Dictionary with hazard classifications and statistics
        """
        self.logger.info("Identifying terrain hazards")
        
        # Create hazard masks
        safe_mask = slope_deg <= preferred_slope
        caution_mask = (slope_deg > preferred_slope) & (slope_deg <= max_safe_slope)
        hazard_mask = slope_deg > max_safe_slope
        
        total_pixels = slope_deg.size
        
        hazards = {
            'safe_percentage': (safe_mask.sum() / total_pixels) * 100,
            'caution_percentage': (caution_mask.sum() / total_pixels) * 100,
            'hazard_percentage': (hazard_mask.sum() / total_pixels) * 100,
            'safe_mask': safe_mask.astype(np.uint8),
            'caution_mask': caution_mask.astype(np.uint8),
            'hazard_mask': hazard_mask.astype(np.uint8),
            'max_safe_slope': max_safe_slope,
            'preferred_slope': preferred_slope
        }
        
        self.logger.info(f"Hazard analysis: {hazards['safe_percentage']:.1f}% safe, "
                        f"{hazards['caution_percentage']:.1f}% caution, "
                        f"{hazards['hazard_percentage']:.1f}% hazardous")
        
        return hazards
    
    def compute_and_cache_slope(self, dem_path: str, 
                                max_safe_slope: float = 30.0) -> Dict[str, Any]:
        """Compute slope and hazards, cache results for fast retrieval.
        
        Args:
            dem_path: Path to DEM GeoTIFF
            max_safe_slope: Maximum safe slope threshold
            
        Returns:
            Dictionary with slope data path and statistics
        """
        dem_name = Path(dem_path).stem
        cache_path = self.cache_dir / f"{dem_name}_slope.npz"
        cache_meta_path = self.cache_dir / f"{dem_name}_slope_meta.json"
        
        if cache_path.exists() and cache_meta_path.exists():
            self.logger.info(f"Loading cached slope data from {cache_path}")
            with np.load(cache_path) as data:
                slope_deg = data['slope_deg']
            with open(cache_meta_path, 'r') as f:
                metadata = json.load(f)
            return {
                'slope_cache_path': str(cache_path),
                'metadata': metadata,
                'cache_hit': True
            }
        
        # Compute from scratch
        elevation, metadata = self.load_dem(dem_path)
        slope_deg = self.compute_slope(elevation, metadata['pixel_size'])
        hazards = self.identify_hazards(slope_deg, max_safe_slope)
        
        # Cache results
        np.savez_compressed(cache_path, slope_deg=slope_deg)
        
        cache_metadata = {
            'dem_path': dem_path,
            'dem_name': dem_name,
            'max_safe_slope': max_safe_slope,
            'safe_percentage': hazards['safe_percentage'],
            'caution_percentage': hazards['caution_percentage'],
            'hazard_percentage': hazards['hazard_percentage'],
            'min_slope': float(slope_deg.min()),
            'max_slope': float(slope_deg.max()),
            'mean_slope': float(slope_deg.mean()),
            'std_slope': float(slope_deg.std())
        }
        
        with open(cache_meta_path, 'w') as f:
            json.dump(cache_metadata, f, indent=2)
        
        self.logger.info(f"Cached slope data to {cache_path}")
        
        return {
            'slope_cache_path': str(cache_path),
            'metadata': cache_metadata,
            'cache_hit': False
        }
    
    def extract_region(self, dem_path: str, 
                      lat_min: float, lat_max: float,
                      lon_min: float, lon_max: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract a regional subset from DEM for focused planning.
        
        Args:
            dem_path: Path to DEM GeoTIFF
            lat_min, lat_max: Latitude bounds (degrees)
            lon_min, lon_max: Longitude bounds (degrees, 0-360)
            
        Returns:
            Tuple of (regional_elevation, metadata)
        """
        with rasterio.open(dem_path) as dataset:
            # Convert lat/lon to pixel coordinates
            from rasterio.windows import from_bounds
            window = from_bounds(lon_min, lat_min, lon_max, lat_max, dataset.transform)
            
            # Read subset
            elevation = dataset.read(1, window=window)
            
            # Create transform for subset
            transform = dataset.window_transform(window)
            
            metadata = {
                'width': elevation.shape[1],
                'height': elevation.shape[0],
                'bounds': (lon_min, lat_min, lon_max, lat_max),
                'transform': transform,
                'crs': str(dataset.crs),
                'pixel_size': (dataset.transform[0], abs(dataset.transform[4]))
            }
            
        return elevation, metadata
    
    def get_elevation_at_point(self, dem_path: str, lat: float, lon: float) -> float:
        """Get elevation at specific lat/lon coordinate.
        
        Args:
            dem_path: Path to DEM GeoTIFF
            lat: Latitude (degrees)
            lon: Longitude (degrees, 0-360)
            
        Returns:
            Elevation in meters
        """
        with rasterio.open(dem_path) as dataset:
            # Sample elevation at point
            for val in dataset.sample([(lon, lat)]):
                return float(val[0])
        return 0.0
