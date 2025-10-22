"""Pytest configuration and shared fixtures."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def mock_dem_data():
    """Create mock DEM elevation data."""
    # Create a simple 100x100 elevation grid
    np.random.seed(42)
    elevation = np.random.uniform(-1000, 2000, (100, 100))
    
    # Add some features
    # Flat plain in center
    elevation[40:60, 40:60] = 100
    
    # Steep slope on left
    for i in range(100):
        elevation[i, 0:20] = i * 10
    
    # Crater on right
    y, x = np.ogrid[:100, :100]
    crater_center = (50, 80)
    distance = np.sqrt((x - crater_center[1])**2 + (y - crater_center[0])**2)
    crater_mask = distance < 15
    elevation[crater_mask] = -500 + distance[crater_mask] * 20
    
    return elevation.astype(np.float32)


@pytest.fixture
def mock_dem_metadata():
    """Mock DEM metadata."""
    return {
        'width': 100,
        'height': 100,
        'bounds': (77.0, 18.0, 78.0, 19.0),
        'transform': None,  # Would be affine.Affine in real use
        'crs': 'GEOGCS["Mars 2000"]',
        'pixel_size': (100.0, 100.0),  # 100m per pixel
        'nodata': -9999
    }


@pytest.fixture
def mock_mission_constraints():
    """Mock mission constraints."""
    return {
        'time_budget_min': 480,
        'power_budget_wh': 1200,
        'data_budget_mb': 250,
        'max_drive_m': 200,
        'comm_windows': [
            {'start_min': 420, 'end_min': 450}
        ]
    }


@pytest.fixture
def mock_rover_state():
    """Mock rover state."""
    return {
        'location': {
            'lat': 18.4447,
            'lon': 77.4508
        },
        'sol': 1400,
        'battery_soc': 0.85,
        'temperature_c': -60,
        'systems_status': 'nominal'
    }


@pytest.fixture
def mock_activities():
    """Mock activity list."""
    return [
        {
            'id': 'pre_checks',
            'type': 'system_checks',
            'duration_min': 10,
            'power_w': 50
        },
        {
            'id': 'drive_segment',
            'type': 'drive',
            'duration_min': 30,
            'distance_m': 50,
            'power_w': 200
        },
        {
            'id': 'science_op',
            'type': 'imaging',
            'duration_min': 20,
            'instrument': 'Mastcam-Z',
            'power_w': 120
        },
        {
            'id': 'post_tx',
            'type': 'data_tx',
            'duration_min': 15,
            'power_w': 180
        }
    ]


@pytest.fixture
def mock_mission_goals():
    """Mock mission goals."""
    return [
        "Perform daily system health checks",
        "Conduct Mastcam-Z panoramic imaging",
        "Analyze rock sample with SuperCam",
        "Complete data transmission to Earth"
    ]


@pytest.fixture
def mock_nasa_traverse_response():
    """Mock NASA traverse API response."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [77.4508, 18.4447]
                },
                "properties": {
                    "sol": 1400,
                    "site": 142,
                    "drive": 2056
                }
            }
        ]
    }


@pytest.fixture
def mock_pds_image_metadata():
    """Mock PDS image metadata response."""
    return {
        "sol": 1400,
        "instrument": "MASTCAM_Z_LEFT",
        "product_id": "ZL0_1400_0123456789_000ECM_N0123456ZCAM08000_1100LMJ01",
        "image_time": "2025-01-15T12:30:45.123Z",
        "rover_elevation_m": -2575.5,
        "sun_elevation_deg": 45.2
    }


@pytest.fixture
def sample_dem_file(temp_dir, mock_dem_data, mock_dem_metadata):
    """Create a sample DEM file for testing."""
    dem_path = temp_dir / "test_dem.npz"
    np.savez_compressed(
        dem_path,
        elevation=mock_dem_data,
        metadata=mock_dem_metadata
    )
    return dem_path


@pytest.fixture
def mock_environment_data():
    """Mock MEDA environmental data."""
    return {
        "sol": 1400,
        "temperature_c": -62.5,
        "pressure_pa": 820,
        "wind_speed_ms": 7.2,
        "wind_direction_deg": 135,
        "humidity_percent": 0.5,
        "uv_index": 0.3,
        "dust_opacity": 0.45
    }


@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.interfaces.web_interface import app
    return TestClient(app)
