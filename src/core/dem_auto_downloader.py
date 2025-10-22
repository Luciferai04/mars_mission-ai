#!/usr/bin/env python3
"""
Real-Time DEM Auto-Downloader

Automatically fetches Digital Elevation Model (DEM) data from official
NASA/USGS planetary data sources for Mars mission planning.
"""

import requests
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta
import json
import numpy as np


class DEMAutoDownloader:
    """Automated DEM data downloader from NASA planetary data sources."""

    def __init__(self, cache_directory: str = "data/dem_cache"):
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        # NASA/USGS data sources
        self.data_sources = {
            "usgs_ode": {
                "base_url": "https://oderest.rsl.wustl.edu/live2/",
                "product_type": "DTM",  # Digital Terrain Model
                "mission": "mars2020",
            },
            "pds_geosciences": {
                "base_url": "https://pds-geosciences.wustl.edu/missions/",
                "mission_path": "mars2020/perseverance/dtm/",
            },
            "hirise": {
                "base_url": "https://www.uahirise.org/dtm/",
                "product_type": "DTEEC",  # HiRISE DTM
            },
        }

        # Cache metadata
        self.metadata_file = self.cache_directory / "dem_metadata.json"
        self.metadata = self._load_metadata()

        # Download settings
        self.chunk_size = 8192
        self.timeout = 300
        self.max_retries = 3

    def download_dem_for_region(
        self,
        region_name: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution_m: float = 1.0,
        force_update: bool = False,
    ) -> Dict[str, Any]:
        """Download DEM data for specified Mars region.

        Args:
            region_name: Name for this region (e.g., "jezero_crater")
            lat_min: Minimum latitude
            lat_max: Maximum latitude
            lon_min: Minimum longitude
            lon_max: Maximum longitude
            resolution_m: Desired resolution in meters per pixel
            force_update: Force re-download even if cached

        Returns:
            Dictionary with DEM data path and metadata
        """
        self.logger.info(
            f"Requesting DEM for {region_name}: "
            f"lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]"
        )

        # Check cache first
        if not force_update:
            cached = self._check_cache(region_name, lat_min, lat_max, lon_min, lon_max)
            if cached:
                self.logger.info(f"Using cached DEM for {region_name}")
                return cached

        # Query available products
        products = self._query_available_products(
            lat_min, lat_max, lon_min, lon_max, resolution_m
        )

        if not products:
            self.logger.warning(f"No DEM products found for {region_name}")
            return self._generate_fallback_dem(
                region_name, lat_min, lat_max, lon_min, lon_max
            )

        # Select best product
        best_product = self._select_best_product(products, resolution_m)

        # Download product
        dem_path = self._download_product(best_product, region_name)

        # Update metadata
        self._update_metadata(
            region_name, dem_path, best_product, lat_min, lat_max, lon_min, lon_max
        )

        return {
            "region_name": region_name,
            "dem_path": str(dem_path),
            "bounds": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
            "resolution_m": best_product.get("resolution_m", resolution_m),
            "source": best_product.get("source", "unknown"),
            "product_id": best_product.get("id"),
            "downloaded_at": datetime.utcnow().isoformat(),
            "cached": False,
        }

    def download_perseverance_traverse_dems(
        self, traverse_plan: Dict[str, Any], buffer_km: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Download DEMs for all waypoints in a traverse plan.

        Args:
            traverse_plan: Multi-sol traverse plan with waypoints
            buffer_km: Buffer distance around waypoints

        Returns:
            List of DEM data for each segment
        """
        dems = []

        sol_plans = traverse_plan.get("sol_plans", [])

        for sol_plan in sol_plans:
            waypoints = sol_plan.get("waypoints", [])

            if not waypoints:
                continue

            # Calculate bounding box for this sol
            lats = [wp[0] for wp in waypoints]
            lons = [wp[1] for wp in waypoints]

            # Add buffer (rough conversion: 1 degree ≈ 59 km on Mars)
            buffer_deg = buffer_km / 59.0

            lat_min = min(lats) - buffer_deg
            lat_max = max(lats) + buffer_deg
            lon_min = min(lons) - buffer_deg
            lon_max = max(lons) + buffer_deg

            # Download DEM
            region_name = f"sol_{sol_plan.get('sol_number', 0)}_traverse"

            dem_data = self.download_dem_for_region(
                region_name, lat_min, lat_max, lon_min, lon_max
            )

            dems.append(dem_data)

        self.logger.info(f"Downloaded {len(dems)} DEMs for traverse plan")

        return dems

    def auto_update_dems(self, max_age_days: int = 30) -> List[Dict[str, Any]]:
        """Automatically update DEMs older than specified age.

        Args:
            max_age_days: Update DEMs older than this many days

        Returns:
            List of updated DEMs
        """
        updated = []
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

        for region_name, meta in self.metadata.items():
            downloaded_at = datetime.fromisoformat(
                meta.get("downloaded_at", "2000-01-01")
            )

            if downloaded_at < cutoff_date:
                self.logger.info(f"Updating outdated DEM for {region_name}")

                try:
                    bounds = meta["bounds"]
                    new_dem = self.download_dem_for_region(
                        region_name,
                        bounds["lat_min"],
                        bounds["lat_max"],
                        bounds["lon_min"],
                        bounds["lon_max"],
                        force_update=True,
                    )
                    updated.append(new_dem)

                except Exception as e:
                    self.logger.error(f"Failed to update DEM for {region_name}: {e}")

        return updated

    def _query_available_products(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        resolution_m: float,
    ) -> List[Dict[str, Any]]:
        """Query available DEM products from data sources."""

        products = []

        # Query USGS ODE
        try:
            ode_products = self._query_usgs_ode(lat_min, lat_max, lon_min, lon_max)
            products.extend(ode_products)
        except Exception as e:
            self.logger.warning(f"USGS ODE query failed: {e}")

        # Query PDS Geosciences
        try:
            pds_products = self._query_pds_geosciences(
                lat_min, lat_max, lon_min, lon_max
            )
            products.extend(pds_products)
        except Exception as e:
            self.logger.warning(f"PDS Geosciences query failed: {e}")

        # Query HiRISE
        try:
            hirise_products = self._query_hirise(lat_min, lat_max, lon_min, lon_max)
            products.extend(hirise_products)
        except Exception as e:
            self.logger.warning(f"HiRISE query failed: {e}")

        return products

    def _query_usgs_ode(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> List[Dict[str, Any]]:
        """Query USGS Orbital Data Explorer API."""

        base_url = self.data_sources["usgs_ode"]["base_url"]

        # ODE REST API parameters
        params = {
            "query": "product",
            "results": "f",
            "output": "json",
            "target": "mars",
            "ptype": "DTM",
            "minlat": lat_min,
            "maxlat": lat_max,
            "westlon": lon_min,
            "eastlon": lon_max,
            "maxresults": 100,
        }

        try:
            response = requests.get(base_url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            products = []
            for item in (
                data.get("ODEResults", {}).get("products", {}).get("product", [])
            ):
                products.append(
                    {
                        "id": item.get("product_id"),
                        "source": "USGS_ODE",
                        "url": item.get("product_files", {})
                        .get("product_file", [{}])[0]
                        .get("url"),
                        "resolution_m": float(item.get("resolution", 1.0)),
                        "center_lat": float(item.get("center_latitude", 0)),
                        "center_lon": float(item.get("center_longitude", 0)),
                        "metadata": item,
                    }
                )

            return products

        except Exception as e:
            self.logger.error(f"USGS ODE query error: {e}")
            return []

    def _query_pds_geosciences(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> List[Dict[str, Any]]:
        """Query PDS Geosciences data node."""

        # PDS Geosciences uses a different structure
        # For now, return empty - would need specific API implementation

        self.logger.debug(
            "PDS Geosciences query not fully implemented - using fallback"
        )
        return []

    def _query_hirise(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> List[Dict[str, Any]]:
        """Query HiRISE DTM database."""

        # HiRISE DTMs are high-resolution but sparse
        # Would need to implement specific HiRISE API

        self.logger.debug("HiRISE query not fully implemented - using fallback")
        return []

    def _select_best_product(
        self, products: List[Dict[str, Any]], target_resolution_m: float
    ) -> Dict[str, Any]:
        """Select best DEM product from available options."""

        if not products:
            raise ValueError("No products available")

        # Score products by resolution match and recency
        scored = []

        for product in products:
            resolution = product.get("resolution_m", 1.0)

            # Prefer resolution close to target
            resolution_score = 1.0 / (1.0 + abs(resolution - target_resolution_m))

            # Prefer USGS ODE (most reliable)
            source_score = 1.0 if product["source"] == "USGS_ODE" else 0.8

            total_score = resolution_score * source_score

            scored.append((total_score, product))

        # Return highest-scoring product
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def _download_product(self, product: Dict[str, Any], region_name: str) -> Path:
        """Download DEM product file."""

        url = product.get("url")
        if not url:
            raise ValueError("Product has no download URL")

        # Generate filename
        product_id = product.get("id", "unknown")
        safe_id = "".join(c if c.isalnum() else "_" for c in product_id)
        filename = f"{region_name}_{safe_id}.tif"

        output_path = self.cache_directory / filename

        self.logger.info(f"Downloading DEM from {url}")

        # Download with retry logic
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, stream=True, timeout=self.timeout)
                response.raise_for_status()

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)

                self.logger.info(f"Downloaded DEM to {output_path}")

                # Verify download
                if output_path.exists() and output_path.stat().st_size > 0:
                    return output_path
                else:
                    raise ValueError("Downloaded file is empty or missing")

            except Exception as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed: {e}")

                if attempt == self.max_retries - 1:
                    raise

        raise RuntimeError("Download failed after all retries")

    def _generate_fallback_dem(
        self,
        region_name: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Dict[str, Any]:
        """Generate synthetic DEM when real data unavailable."""

        self.logger.warning(f"Generating fallback DEM for {region_name}")

        # Create simple synthetic DEM (flat with small random variations)
        resolution = 100  # 100x100 grid

        # Generate smooth terrain with Perlin-like noise
        dem_array = np.random.normal(0, 10, (resolution, resolution))

        # Save as numpy array
        filename = f"{region_name}_fallback.npy"
        output_path = self.cache_directory / filename

        np.save(output_path, dem_array)

        # Update metadata
        self._update_metadata(
            region_name,
            output_path,
            {"source": "FALLBACK", "resolution_m": 1.0},
            lat_min,
            lat_max,
            lon_min,
            lon_max,
        )

        return {
            "region_name": region_name,
            "dem_path": str(output_path),
            "bounds": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
            "resolution_m": 1.0,
            "source": "FALLBACK",
            "product_id": "fallback_synthetic",
            "downloaded_at": datetime.utcnow().isoformat(),
            "warning": "Using synthetic DEM - real data unavailable",
        }

    def _check_cache(
        self,
        region_name: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> Optional[Dict[str, Any]]:
        """Check if DEM is already cached."""

        if region_name not in self.metadata:
            return None

        meta = self.metadata[region_name]

        # Check if bounds match
        bounds = meta.get("bounds", {})

        if (
            abs(bounds.get("lat_min", 0) - lat_min) < 0.01
            and abs(bounds.get("lat_max", 0) - lat_max) < 0.01
            and abs(bounds.get("lon_min", 0) - lon_min) < 0.01
            and abs(bounds.get("lon_max", 0) - lon_max) < 0.01
        ):

            # Check if file exists
            dem_path = Path(meta["dem_path"])

            if dem_path.exists():
                meta["cached"] = True
                return meta

        return None

    def _load_metadata(self) -> Dict[str, Any]:
        """Load DEM cache metadata."""

        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load metadata: {e}")
                return {}

        return {}

    def _update_metadata(
        self,
        region_name: str,
        dem_path: Path,
        product: Dict[str, Any],
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ):
        """Update cache metadata."""

        self.metadata[region_name] = {
            "dem_path": str(dem_path),
            "bounds": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
            },
            "resolution_m": product.get("resolution_m", 1.0),
            "source": product.get("source", "unknown"),
            "product_id": product.get("id", "unknown"),
            "downloaded_at": datetime.utcnow().isoformat(),
        }

        # Save metadata
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def get_cached_dems(self) -> List[Dict[str, Any]]:
        """Get list of all cached DEMs."""

        return [{"region_name": name, **meta} for name, meta in self.metadata.items()]

    def clear_cache(self, region_name: Optional[str] = None):
        """Clear cached DEMs.

        Args:
            region_name: Specific region to clear, or None for all
        """
        if region_name:
            if region_name in self.metadata:
                # Delete file
                dem_path = Path(self.metadata[region_name]["dem_path"])
                if dem_path.exists():
                    dem_path.unlink()

                # Remove from metadata
                del self.metadata[region_name]

                self.logger.info(f"Cleared cache for {region_name}")
        else:
            # Clear all
            for meta in self.metadata.values():
                dem_path = Path(meta["dem_path"])
                if dem_path.exists():
                    dem_path.unlink()

            self.metadata = {}

            self.logger.info("Cleared entire DEM cache")

        # Save updated metadata
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
