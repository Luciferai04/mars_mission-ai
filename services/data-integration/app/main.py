#!/usr/bin/env python3
"""
Data Integration Microservice
Aggregates data from multiple Mars mission data sources
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import httpx
import logging
from datetime import datetime
import asyncio

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Data Integration Service",
    description="Mars mission data aggregation and integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class MarsEnvironment(BaseModel):
    sol: int
    temp_c: float
    wind_speed: float
    pressure: float
    uv_index: float
    dust_opacity: float
    timestamp: str


class TerrainData(BaseModel):
    lat: float
    lon: float
    elevation: float
    slope: float
    roughness: float
    rock_abundance: float


class IntegratedData(BaseModel):
    environment: MarsEnvironment
    terrain: Optional[TerrainData]
    science_targets: List[Dict[str, Any]]
    rover_status: Dict[str, Any]
    timestamp: str


# External data sources (simulated/mock for demonstration)
# In production, these would connect to real Mars data APIs

async def fetch_environment_data(sol: int) -> Dict[str, Any]:
    """Fetch environmental data for a given sol"""
    # Mock data - replace with real API calls
    # e.g., Mars Weather API, REMS data
    return {
        "sol": sol,
        "temp_c": -63.0,
        "wind_speed": 2.1,
        "pressure": 750.0,
        "uv_index": 2.5,
        "dust_opacity": 0.4,
        "timestamp": datetime.utcnow().isoformat()
    }


async def fetch_terrain_data(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch terrain data for coordinates"""
    # Mock data - replace with Mars DEM/terrain APIs
    # e.g., HiRISE, CTX, MOLA data
    return {
        "lat": lat,
        "lon": lon,
        "elevation": 1250.5,
        "slope": 5.2,
        "roughness": 0.3,
        "rock_abundance": 0.15
    }


async def fetch_science_targets(lat: float, lon: float, radius_km: float) -> List[Dict[str, Any]]:
    """Fetch science targets in area"""
    # Mock data - replace with science target database
    return [
        {
            "id": "SCI_001",
            "name": "Layered Rock Formation",
            "lat": lat + 0.001,
            "lon": lon + 0.002,
            "priority": 8,
            "instruments": ["mastcam", "chemcam"]
        },
        {
            "id": "SCI_002",
            "name": "Ancient Streambed",
            "lat": lat - 0.002,
            "lon": lon + 0.001,
            "priority": 9,
            "instruments": ["apxs", "mahli"]
        }
    ]


async def fetch_rover_status() -> Dict[str, Any]:
    """Fetch current rover status"""
    # Mock data - replace with rover telemetry API
    return {
        "battery_soc": 85.0,
        "power_generation": 150.0,
        "power_consumption": 120.0,
        "mobility_ok": True,
        "instruments_ok": True,
        "communication_ok": True
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "data-integration-service",
        "version": "1.0.0",
        "data_sources": ["environment", "terrain", "science", "rover"]
    }


@app.get("/environment/{sol}", response_model=MarsEnvironment)
async def get_environment(sol: int):
    """Get environmental data for a specific sol"""
    try:
        data = await fetch_environment_data(sol)
        return MarsEnvironment(**data)
    except Exception as e:
        logger.error(f"Error fetching environment data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/terrain", response_model=TerrainData)
async def get_terrain(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
):
    """Get terrain data for coordinates"""
    try:
        data = await fetch_terrain_data(lat, lon)
        return TerrainData(**data)
    except Exception as e:
        logger.error(f"Error fetching terrain data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/science-targets")
async def get_science_targets(
    lat: float = Query(...),
    lon: float = Query(...),
    radius_km: float = Query(5.0, description="Search radius in km")
):
    """Get science targets in area"""
    try:
        targets = await fetch_science_targets(lat, lon, radius_km)
        return {"targets": targets, "count": len(targets)}
    except Exception as e:
        logger.error(f"Error fetching science targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rover/status")
async def get_rover_status():
    """Get current rover status"""
    try:
        status = await fetch_rover_status()
        return status
    except Exception as e:
        logger.error(f"Error fetching rover status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/integrated", response_model=IntegratedData)
async def get_integrated_data(
    sol: int = Query(...),
    lat: float = Query(...),
    lon: float = Query(...),
    radius_km: float = Query(5.0)
):
    """Get all integrated data for mission planning"""
    try:
        # Fetch all data concurrently
        env_task = fetch_environment_data(sol)
        terrain_task = fetch_terrain_data(lat, lon)
        targets_task = fetch_science_targets(lat, lon, radius_km)
        rover_task = fetch_rover_status()
        
        env_data, terrain_data, targets, rover_status = await asyncio.gather(
            env_task, terrain_task, targets_task, rover_task
        )
        
        return IntegratedData(
            environment=MarsEnvironment(**env_data),
            terrain=TerrainData(**terrain_data),
            science_targets=targets,
            rover_status=rover_status,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error fetching integrated data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data-sources/register")
async def register_data_source(
    name: str,
    url: str,
    api_key: Optional[str] = None
):
    """Register a new external data source"""
    # In production, store in database
    return {
        "status": "registered",
        "name": name,
        "url": url,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
