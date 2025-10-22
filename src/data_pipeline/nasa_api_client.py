#!/usr/bin/env python3
"""
NASA API Client for Mars Mission Data

Provides access to:
- NASA PDS (Planetary Data System) imagery and metadata
- ODE (Orbital Data Explorer) REST API
- Mars 2020 rover telemetry
- Mastcam-Z image metadata
- MEDA environmental sensor data
"""

import os
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json


class NASAAPIClient:
    """Client for NASA Mars 2020 mission data APIs."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NASA_API_KEY", "DEMO_KEY")
        self.pds_base_url = os.getenv(
            "NASA_PDS_BASE_URL", "https://pds-imaging.jpl.nasa.gov/data/"
        )
        self.ode_base_url = os.getenv(
            "NASA_ODE_API_URL", "https://oderest.rsl.wustl.edu/live2/"
        )
        self.mars2020_url = "https://mars.nasa.gov/mars2020/"

        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(os.getenv("CACHE_DIR", "./data/cache/"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_rover_telemetry(self, sol: int) -> Dict[str, Any]:
        """Get rover telemetry for specific sol.

        Args:
            sol: Mars sol number

        Returns:
            Dictionary with rover state, location, and system status
        """
        self.logger.info(f"Fetching rover telemetry for Sol {sol}")

        try:
            # Mars 2020 mission API endpoint
            url = f"{self.mars2020_url}mission/mars-rover-api/"

            response = requests.get(url, params={"sol": sol}, timeout=15)

            if response.status_code == 200:
                data = response.json()
                return self._parse_telemetry(data, sol)
            else:
                self.logger.warning(f"Telemetry API returned {response.status_code}")
                return self._get_fallback_telemetry(sol)

        except Exception as e:
            self.logger.error(f"Failed to fetch telemetry: {e}")
            return self._get_fallback_telemetry(sol)

    def get_mastcamz_metadata(self, sol: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get Mastcam-Z image metadata for specific sol.

        Args:
            sol: Mars sol number
            limit: Maximum number of images to return

        Returns:
            List of image metadata dictionaries
        """
        self.logger.info(f"Fetching Mastcam-Z metadata for Sol {sol}")

        cache_file = self.cache_dir / f"mastcamz_sol{sol}.json"

        # Check cache first
        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < timedelta(days=7):
                self.logger.info("Using cached Mastcam-Z metadata")
                with open(cache_file, "r") as f:
                    return json.load(f)

        try:
            # PDS Imaging API
            url = f"{self.pds_base_url}mars2020/mastcamz/"

            response = requests.get(
                url,
                params={"sol": sol, "limit": limit, "api_key": self.api_key},
                timeout=30,
            )

            if response.status_code == 200:
                images = self._parse_mastcamz_response(response.json())

                # Cache results
                with open(cache_file, "w") as f:
                    json.dump(images, f, indent=2)

                return images
            else:
                self.logger.warning(f"Mastcam-Z API returned {response.status_code}")
                return []

        except Exception as e:
            self.logger.error(f"Failed to fetch Mastcam-Z metadata: {e}")
            return []

    def get_meda_environment_data(self, sol: int) -> Dict[str, Any]:
        """Get MEDA environmental sensor data.

        MEDA (Mars Environmental Dynamics Analyzer) provides:
        - Temperature
        - Pressure
        - Wind speed and direction
        - Humidity
        - Dust opacity

        Args:
            sol: Mars sol number

        Returns:
            Environmental data dictionary
        """
        self.logger.info(f"Fetching MEDA data for Sol {sol}")

        cache_file = self.cache_dir / f"meda_sol{sol}.json"

        # Check cache
        if cache_file.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if age < timedelta(hours=24):
                with open(cache_file, "r") as f:
                    return json.load(f)

        try:
            # Mars 2020 weather API
            url = f"{self.mars2020_url}weather/api/"

            response = requests.get(url, params={"sol": sol}, timeout=15)

            if response.status_code == 200:
                data = self._parse_meda_response(response.json(), sol)

                # Cache results
                with open(cache_file, "w") as f:
                    json.dump(data, f, indent=2)

                return data
            else:
                return self._get_fallback_environment(sol)

        except Exception as e:
            self.logger.error(f"Failed to fetch MEDA data: {e}")
            return self._get_fallback_environment(sol)

    def get_traverse_data(self) -> Dict[str, Any]:
        """Get latest rover traverse data.

        Returns:
            GeoJSON FeatureCollection with rover path
        """
        self.logger.info("Fetching traverse data")

        try:
            url = "https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_traverse.json"

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            self.logger.error(f"Failed to fetch traverse data: {e}")
            return {"type": "FeatureCollection", "features": []}

    def get_waypoints(self) -> Dict[str, Any]:
        """Get rover waypoints.

        Returns:
            GeoJSON FeatureCollection with waypoints
        """
        self.logger.info("Fetching waypoints")

        try:
            url = "https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_waypoints.json"

            response = requests.get(url, timeout=10)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            self.logger.error(f"Failed to fetch waypoints: {e}")
            return {"type": "FeatureCollection", "features": []}

    def search_ode_products(
        self,
        target: str = "mars",
        instrument: str = "mastcamz",
        product_type: str = "rdr",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search ODE for Mars data products.

        Args:
            target: Target body (mars, phobos, deimos)
            instrument: Instrument name
            product_type: Product type (edr, rdr, etc.)
            limit: Maximum results

        Returns:
            List of product metadata
        """
        self.logger.info(f"Searching ODE for {instrument} products")

        try:
            url = f"{self.ode_base_url}products"

            response = requests.get(
                url,
                params={
                    "target": target,
                    "instrument": instrument,
                    "pt": product_type,
                    "limit": limit,
                    "output": "json",
                },
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("ODEResults", {}).get("Products", {}).get("Product", [])
            else:
                self.logger.warning(f"ODE search returned {response.status_code}")
                return []

        except Exception as e:
            self.logger.error(f"ODE search failed: {e}")
            return []

    # Helper methods

    def _parse_telemetry(self, data: Dict[str, Any], sol: int) -> Dict[str, Any]:
        """Parse raw telemetry response."""
        return {
            "sol": sol,
            "location": {
                "lat": data.get("latitude", 18.4447),
                "lon": data.get("longitude", 77.4508),
                "elevation_m": data.get("elevation", -2575.5),
            },
            "systems": {
                "battery_soc": data.get("battery_state_of_charge", 0.85),
                "power_w": data.get("power_generation", 110.0),
                "temperature_c": data.get("temperature", -60),
                "status": data.get("status", "nominal"),
            },
            "odometry": {
                "total_distance_km": data.get("total_distance", 0.0),
                "drive_distance_m": data.get("drive_distance", 0.0),
            },
            "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
        }

    def _get_fallback_telemetry(self, sol: int) -> Dict[str, Any]:
        """Fallback telemetry when API unavailable."""
        return {
            "sol": sol,
            "location": {
                "lat": 18.4447,  # Jezero Crater
                "lon": 77.4508,
                "elevation_m": -2575.5,
            },
            "systems": {
                "battery_soc": 0.85,
                "power_w": 110.0,
                "temperature_c": -60,
                "status": "nominal",
            },
            "odometry": {
                "total_distance_km": sol * 0.1,  # Rough estimate
                "drive_distance_m": 0.0,
            },
            "timestamp": datetime.utcnow().isoformat(),
            "source": "fallback",
        }

    def _parse_mastcamz_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Mastcam-Z API response."""
        images = []

        for item in data.get("images", []):
            images.append(
                {
                    "product_id": item.get("imageid"),
                    "sol": item.get("sol"),
                    "instrument": item.get("instrument", "MASTCAM_Z"),
                    "camera": item.get("camera"),
                    "image_time": item.get("date_taken"),
                    "url": item.get("image_url"),
                    "thumbnail_url": item.get("thumbnail_url"),
                    "sample_type": item.get("sample_type"),
                    "attitude": {
                        "elevation": item.get("sun_elevation"),
                        "azimuth": item.get("sun_azimuth"),
                    },
                }
            )

        return images

    def _parse_meda_response(self, data: Dict[str, Any], sol: int) -> Dict[str, Any]:
        """Parse MEDA environmental data response."""
        return {
            "sol": sol,
            "temperature_c": data.get("temperature", -62.5),
            "temperature_min_c": data.get("min_temp", -80),
            "temperature_max_c": data.get("max_temp", -40),
            "pressure_pa": data.get("pressure", 820),
            "wind_speed_ms": data.get("wind_speed", 7.2),
            "wind_direction_deg": data.get("wind_direction", 135),
            "humidity_percent": data.get("humidity", 0.5),
            "uv_index": data.get("uv", 0.3),
            "dust_opacity": data.get("opacity", 0.45),
            "local_time": data.get("local_time", "12:00"),
            "season": data.get("season", "northern_spring"),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _get_fallback_environment(self, sol: int) -> Dict[str, Any]:
        """Fallback environmental data."""
        return {
            "sol": sol,
            "temperature_c": -62.5,
            "temperature_min_c": -80,
            "temperature_max_c": -40,
            "pressure_pa": 820,
            "wind_speed_ms": 7.2,
            "wind_direction_deg": 135,
            "humidity_percent": 0.5,
            "uv_index": 0.3,
            "dust_opacity": 0.45,
            "local_time": "12:00",
            "season": "northern_spring",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "fallback",
        }
