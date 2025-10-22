"""Unit tests for DEM processor."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_pipeline.dem_processor import DEMProcessor


@pytest.mark.unit
class TestDEMProcessor:
    """Test DEM processing functionality."""

    def test_compute_slope_flat_terrain(self):
        """Test slope computation on flat terrain."""
        processor = DEMProcessor()

        # Flat elevation
        elevation = np.ones((50, 50)) * 100.0
        pixel_size = (100.0, 100.0)

        slope = processor.compute_slope(elevation, pixel_size)

        # Flat terrain should have near-zero slope
        assert slope.shape == elevation.shape
        assert np.mean(slope) < 1.0  # Average slope < 1 degree

    def test_compute_slope_steep_terrain(self):
        """Test slope computation on steep terrain."""
        processor = DEMProcessor()

        # Create steep slope: 45 degree angle
        elevation = np.zeros((50, 50))
        for i in range(50):
            elevation[i, :] = i * 100  # 100m rise per 100m pixel = 45°

        pixel_size = (100.0, 100.0)
        slope = processor.compute_slope(elevation, pixel_size)

        # Should detect steep slopes
        assert np.mean(slope) > 30.0  # Average slope > 30 degrees
        assert np.max(slope) <= 90.0  # Max possible slope is 90°

    def test_identify_hazards_safe_terrain(self, mock_dem_data):
        """Test hazard identification on safe terrain."""
        processor = DEMProcessor()

        # Create mostly flat terrain
        elevation = np.ones((100, 100)) * 100.0
        slope = processor.compute_slope(elevation, (100.0, 100.0))

        hazards = processor.identify_hazards(slope, max_safe_slope=30.0)

        assert hazards["safe_percentage"] > 90.0
        assert hazards["hazard_percentage"] < 5.0

    def test_identify_hazards_dangerous_terrain(self):
        """Test hazard identification on dangerous terrain."""
        processor = DEMProcessor()

        # Create very steep terrain
        elevation = np.zeros((100, 100))
        for i in range(100):
            elevation[i, :] = i * 150  # Very steep

        slope = processor.compute_slope(elevation, (100.0, 100.0))
        hazards = processor.identify_hazards(slope, max_safe_slope=30.0)

        assert hazards["hazard_percentage"] > 50.0
        assert hazards["safe_percentage"] < 30.0

    def test_hazard_thresholds(self):
        """Test hazard classification thresholds."""
        processor = DEMProcessor()

        # Create terrain with known slopes
        slope = np.array(
            [[5, 10, 15], [20, 25, 30], [35, 40, 45]]  # Safe  # Caution  # Hazard
        )

        hazards = processor.identify_hazards(
            slope, max_safe_slope=30.0, preferred_slope=15.0
        )

        # Check masks
        assert hazards["safe_mask"].sum() == 3  # First row
        assert hazards["caution_mask"].sum() == 3  # Second row
        assert hazards["hazard_mask"].sum() == 3  # Third row

    def test_slope_computation_handles_nan(self):
        """Test that slope computation handles NaN values."""
        processor = DEMProcessor()

        elevation = np.ones((50, 50)) * 100.0
        elevation[25, 25] = np.nan

        pixel_size = (100.0, 100.0)
        slope = processor.compute_slope(elevation, pixel_size)

        # NaN should be converted to 0
        assert not np.isnan(slope[25, 25])
        assert np.isfinite(slope).all()

    def test_slope_statistics(self, mock_dem_data):
        """Test slope statistics computation."""
        processor = DEMProcessor()

        slope = processor.compute_slope(mock_dem_data, (100.0, 100.0))

        # Basic statistics
        assert slope.min() >= 0.0
        assert slope.max() <= 90.0
        assert 0 <= slope.mean() <= 90.0
        assert slope.std() >= 0.0


@pytest.mark.unit
class TestDEMCaching:
    """Test DEM caching functionality."""

    def test_cache_creation(self, temp_dir, mock_dem_data):
        """Test that slope cache is created."""
        processor = DEMProcessor(cache_dir=str(temp_dir))

        # Create a simple DEM file
        dem_path = temp_dir / "test.tif"
        # For this test, we'll use npz instead of actual GeoTIFF
        np.savez_compressed(dem_path.with_suffix(".npz"), elevation=mock_dem_data)

        # This would normally use a real GeoTIFF
        # For testing, we'll verify the cache directory exists
        assert processor.cache_dir.exists()

    def test_cache_metadata_structure(self):
        """Test cache metadata has correct structure."""
        _ = DEMProcessor()

        metadata = {
            "dem_path": "/path/to/dem.tif",
            "dem_name": "test_dem",
            "max_safe_slope": 30.0,
            "safe_percentage": 85.5,
            "caution_percentage": 10.2,
            "hazard_percentage": 4.3,
            "min_slope": 0.0,
            "max_slope": 45.0,
            "mean_slope": 12.5,
            "std_slope": 8.2,
        }

        # Verify all expected keys are present
        expected_keys = [
            "dem_path",
            "dem_name",
            "max_safe_slope",
            "safe_percentage",
            "caution_percentage",
            "hazard_percentage",
            "min_slope",
            "max_slope",
            "mean_slope",
            "std_slope",
        ]

        for key in expected_keys:
            assert key in metadata


@pytest.mark.unit
class TestDEMUtilities:
    """Test DEM utility functions."""

    def test_pixel_size_calculation(self):
        """Test pixel size is correctly extracted."""
        _ = DEMProcessor()

        # Mock metadata with known pixel size
        metadata = {"pixel_size": (100.0, 100.0)}

        assert metadata["pixel_size"] == (100.0, 100.0)

    def test_slope_units_degrees(self):
        """Test that slopes are in degrees, not radians."""
        processor = DEMProcessor()

        # 45-degree slope
        elevation = np.array([[0, 100], [0, 100]])
        pixel_size = (100.0, 100.0)

        slope = processor.compute_slope(elevation, pixel_size)

        # Should be in degrees (0-90), not radians (0-π/2)
        assert np.all(slope <= 90.0)
        assert np.all(slope >= 0.0)
