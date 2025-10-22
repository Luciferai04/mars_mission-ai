#!/usr/bin/env python3
"""
Planning Service
Orchestrates mission planning using Vision, MARL, and Data Integration services
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import httpx
import logging
from datetime import datetime
import asyncio
import os
import sys
from pathlib import Path

# Ensure project src is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.autonomous_experiments import ExperimentDesigner
from src.core.strategic_planner import StrategicPlanner
from src.integrations.jpl_tools import export_apgen, export_plexil

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Planning Service",
    description="Mars mission planning orchestration",
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

# Service URLs from environment
VISION_SERVICE_URL = os.getenv("VISION_SERVICE_URL", "http://vision-service:8002")
MARL_SERVICE_URL = os.getenv("MARL_SERVICE_URL", "http://marl-service:8003")
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://data-integration-service:8004")


# Pydantic models
class MissionRequest(BaseModel):
    sol: int
    lat: float
    lon: float
    battery_soc: float
    time_budget_min: int
    objectives: List[str]
    constraints: Optional[Dict[str, Any]] = None


class MissionPlan(BaseModel):
    plan_id: str
    status: str
    mission_context: Dict[str, Any]
    integrated_data: Dict[str, Any]
    vision_analysis: Optional[Dict[str, Any]]
    optimized_actions: List[Dict[str, Any]]
    expected_completion: int
    total_power: float
    total_time: int
    rl_confidence: float
    created_at: str


async def call_service(url: str, method: str = "GET", json: Optional[Dict] = None) -> Dict:
    """Helper to call microservices"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if method == "GET":
                response = await client.get(url)
            elif method == "POST":
                response = await client.post(url, json=json)
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Service call failed: {url} - {e}")
            raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint with dependency status"""
    services = {}
    
    # Check all dependent services
    for name, url in [
        ("vision", f"{VISION_SERVICE_URL}/health"),
        ("marl", f"{MARL_SERVICE_URL}/health"),
        ("data", f"{DATA_SERVICE_URL}/health")
    ]:
        try:
            result = await call_service(url)
            services[name] = result.get("status", "unknown")
        except:
            services[name] = "unavailable"
    
    all_healthy = all(s == "healthy" for s in services.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "service": "planning-service",
        "version": "1.0.0",
        "dependencies": services
    }


@app.post("/plan", response_model=MissionPlan)
async def create_mission_plan(request: MissionRequest):
    """Create optimized mission plan using all services"""
    plan_id = f"PLAN_{int(datetime.utcnow().timestamp())}"
    
    try:
        # Step 1: Fetch integrated data
        logger.info(f"Fetching integrated data for {request.lat}, {request.lon}")
        data_url = (
            f"{DATA_SERVICE_URL}/integrated?"
            f"sol={request.sol}&lat={request.lat}&lon={request.lon}"
        )
        integrated_data = await call_service(data_url)
        
        # Step 2: Prepare mission context for MARL
        mission_context = {
            "lat": request.lat,
            "lon": request.lon,
            "battery_soc": request.battery_soc,
            "time_budget_min": request.time_budget_min,
            "targets": integrated_data["science_targets"],
            "sol": request.sol,
            "temp": integrated_data["environment"]["temp_c"],
            "dust": integrated_data["environment"]["dust_opacity"]
        }
        
        # Step 3: Run MARL optimization
        logger.info("Running MARL optimization")
        marl_url = f"{MARL_SERVICE_URL}/optimize"
        marl_result = await call_service(marl_url, method="POST", json=mission_context)
        
        # Step 4: Build complete plan
        plan = MissionPlan(
            plan_id=plan_id,
            status="completed",
            mission_context=mission_context,
            integrated_data=integrated_data,
            vision_analysis=None,  # Can be added later with image upload
            optimized_actions=marl_result["optimized_actions"],
            expected_completion=marl_result["expected_completion"],
            total_power=marl_result["total_power"],
            total_time=marl_result["total_time"],
            rl_confidence=marl_result["rl_confidence"],
            created_at=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Plan {plan_id} created successfully")
        return plan
        
    except Exception as e:
        logger.error(f"Plan creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Planning failed: {str(e)}")


@app.post("/plan/with-vision")
async def create_plan_with_vision(
    sol: int,
    lat: float,
    lon: float,
    battery_soc: float,
    time_budget_min: int,
    image: UploadFile = File(...)
):
    """Create mission plan with terrain image analysis"""
    plan_id = f"PLAN_{int(datetime.utcnow().timestamp())}"
    
    try:
        # Step 1: Analyze terrain image
        logger.info("Analyzing terrain image")
        vision_url = f"{VISION_SERVICE_URL}/analyze"
        
        image_data = await image.read()
        files = {"file": (image.filename, image_data, image.content_type)}
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(vision_url, files=files)
            response.raise_for_status()
            vision_analysis = response.json()
        
        # Step 2: Get integrated data
        data_url = f"{DATA_SERVICE_URL}/integrated?sol={sol}&lat={lat}&lon={lon}"
        integrated_data = await call_service(data_url)
        
        # Step 3: Enrich targets with vision analysis
        targets = integrated_data["science_targets"]
        
        # Add hazards from vision as avoidance targets
        if vision_analysis.get("hazards"):
            for hazard in vision_analysis["hazards"]:
                targets.append({
                    "id": f"HAZARD_{hazard['type']}",
                    "name": f"Hazard: {hazard['type']}",
                    "lat": lat + hazard.get("offset_lat", 0.0),
                    "lon": lon + hazard.get("offset_lon", 0.0),
                    "priority": -10,  # Negative priority = avoid
                    "instruments": []
                })
        
        # Step 4: Run MARL optimization with enriched targets
        mission_context = {
            "lat": lat,
            "lon": lon,
            "battery_soc": battery_soc,
            "time_budget_min": time_budget_min,
            "targets": targets,
            "sol": sol,
            "temp": integrated_data["environment"]["temp_c"],
            "dust": integrated_data["environment"]["dust_opacity"]
        }
        
        marl_url = f"{MARL_SERVICE_URL}/optimize"
        marl_result = await call_service(marl_url, method="POST", json=mission_context)
        
        # Step 5: Build plan with vision
        plan = MissionPlan(
            plan_id=plan_id,
            status="completed",
            mission_context=mission_context,
            integrated_data=integrated_data,
            vision_analysis=vision_analysis,
            optimized_actions=marl_result["optimized_actions"],
            expected_completion=marl_result["expected_completion"],
            total_power=marl_result["total_power"],
            total_time=marl_result["total_time"],
            rl_confidence=marl_result["rl_confidence"],
            created_at=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Plan {plan_id} with vision created successfully")
        return plan
        
    except Exception as e:
        logger.error(f"Plan creation with vision failed: {e}")
        raise HTTPException(status_code=500, detail=f"Planning failed: {str(e)}")


@app.post("/experiments/propose")
async def propose_experiments(sol: int, lat: float, lon: float, k: int = 3):
    """Propose autonomous experiments based on integrated data"""
    try:
        data_url = f"{DATA_SERVICE_URL}/integrated?sol={sol}&lat={lat}&lon={lon}"
        integrated_data = await call_service(data_url)
        designer = ExperimentDesigner()
        proposals = designer.propose_experiments(integrated_data, k=k)
        return {"proposals": proposals, "count": len(proposals)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/plan/long-term")
async def plan_long_term(horizon_days: int = 90):
    """Generate a long-term strategic plan"""
    try:
        planner = StrategicPlanner()
        plan = planner.generate({}, horizon_days=horizon_days)
        return plan.__dict__
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export/jpl")
async def export_jpl(plan: Dict[str, Any], fmt: str = "plexil"):
    """Export mission plan to JPL planning tool formats (simplified)."""
    try:
        if fmt.lower() == "apgen":
            content = export_apgen(plan)
            return {"format": "apgen", "content": content}
        content = export_plexil(plan)
        return {"format": "plexil", "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/services/status")
async def get_services_status():
    """Get detailed status of all services"""
    services = {}
    
    # Vision service
    try:
        vision_health = await call_service(f"{VISION_SERVICE_URL}/health")
        services["vision"] = {
            "status": vision_health.get("status"),
            "version": vision_health.get("version"),
            "models": vision_health.get("models_loaded")
        }
    except:
        services["vision"] = {"status": "unavailable"}
    
    # MARL service
    try:
        marl_health = await call_service(f"{MARL_SERVICE_URL}/health")
        marl_stats = await call_service(f"{MARL_SERVICE_URL}/agents/stats")
        services["marl"] = {
            "status": marl_health.get("status"),
            "version": marl_health.get("version"),
            "agents": marl_health.get("agents"),
            "episodes": marl_stats.get("episodes"),
            "confidence": marl_stats.get("agent_confidence")
        }
    except:
        services["marl"] = {"status": "unavailable"}
    
    # Data integration service
    try:
        data_health = await call_service(f"{DATA_SERVICE_URL}/health")
        services["data"] = {
            "status": data_health.get("status"),
            "version": data_health.get("version"),
            "sources": data_health.get("data_sources")
        }
    except:
        services["data"] = {"status": "unavailable"}
    
    return {
        "planning_service": "operational",
        "services": services,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
