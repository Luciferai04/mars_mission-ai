#!/usr/bin/env python3
"""
FastAPI Web Interface for Mars Mission Planning Assistant

Provides REST API endpoints for mission planning, terrain analysis,
and NASA data integration as specified in project requirements.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_pipeline.dem_processor import DEMProcessor
from path_planner import PathPlanner
from resource_optimizer import ResourceOptimizer
from constraint_solver import ConstraintSolver
from report_generator import make_report

# Initialize FastAPI app
app = FastAPI(
    title="Mars Mission Planning Assistant",
    description="AI-powered mission planning for NASA Mars 2020 Perseverance operations",
    version="1.0.0",
)

# Add CORS middleware to allow dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins including file://
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
dem_processor = DEMProcessor(cache_dir=os.getenv("CACHE_DIR", "./data/cache/"))
path_planner = PathPlanner()
resource_optimizer = ResourceOptimizer()
constraint_solver = ConstraintSolver()

logger = logging.getLogger(__name__)


# Pydantic Models
class CommWindow(BaseModel):
    start_min: int
    end_min: int


class PlanConstraints(BaseModel):
    time_budget_min: int = 480
    power_budget_wh: Optional[float] = None
    data_budget_mb: Optional[float] = 250.0
    max_drive_m: Optional[float] = 200.0
    comm_windows: Optional[List[CommWindow]] = []


class MissionPlanRequest(BaseModel):
    goals: List[str]
    constraints: PlanConstraints
    include_report: bool = False


class RouteRequest(BaseModel):
    dem_path: str
    start_lat: float
    start_lon: float
    goal_lat: float
    goal_lon: float
    slope_max_deg: Optional[float] = 30.0


class ScenarioRequest(BaseModel):
    scenario: str


class SlopeComputeRequest(BaseModel):
    dem_path: str
    max_safe_slope: float = 30.0


class EnvironmentSearchRequest(BaseModel):
    params: Dict[str, Any]


# API Endpoints


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <html>
        <head><title>Mars Mission Planning Assistant</title></head>
        <body>
            <h1>Mars Mission Planning Assistant API</h1>
            <p>AI-powered mission planning for NASA Mars 2020 Perseverance operations</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><a href="/docs">API Documentation (Swagger UI)</a></li>
                <li><a href="/health">Health Check</a></li>
                <li><a href="/dashboard">Mission Dashboard</a></li>
            </ul>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "Mars Mission Planning Assistant",
        "version": "1.0.0",
    }


@app.post("/plan")
async def generate_mission_plan(request: MissionPlanRequest):
    """Generate comprehensive mission plan from goals and constraints.

    Example payload:
    {
        "goals": ["Plan a 3-sol mission to investigate delta deposits"],
        "constraints": {
            "time_budget_min": 480,
            "comm_windows": [{"start_min": 420, "end_min": 450}]
        },
        "include_report": true
    }
    """
    try:
        # Import planner modules
        from gpt4_planner import MissionPlannerLLM

        planner = MissionPlannerLLM()

        context = {
            "objectives": request.goals,
            "constraints": request.constraints.dict(),
            "rover_state": {},
        }

        # Generate initial plan
        draft_plan = planner.generate_plan(context)

        # Optimize resources
        activities = draft_plan.get("activities", [])
        optimized_activities, power_summary = resource_optimizer.optimize_power_usage(
            activities,
            mmrtg_output_w=110.0,
            time_budget_min=request.constraints.time_budget_min,
            battery_capacity_ah=43.0,
            battery_count=2,
        )

        # Validate constraints
        validation = constraint_solver.evaluate(
            optimized_activities, request.constraints.dict()
        )

        if not validation["ok"]:
            # Attempt repair
            optimized_activities = constraint_solver.repair(
                optimized_activities, request.constraints.dict()
            )

        plan = {
            "goals": request.goals,
            "activities": optimized_activities,
            "power": power_summary,
            "validation": validation,
        }

        response = {"plan": plan}

        if request.include_report:
            response["report"] = make_report(plan)

        return response

    except Exception as e:
        logger.error(f"Mission planning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Mission planning error: {str(e)}")


@app.post("/route_from_dem")
async def compute_route(request: RouteRequest):
    """Compute safest route between lat/lon over a DEM GeoTIFF.

    Example:
    {
        "dem_path": "./data/dem/jezero_demo.tif",
        "start_lat": 18.4, "start_lon": 77.5,
        "goal_lat": 18.45, "goal_lon": 77.55,
        "slope_max_deg": 30.0
    }
    """
    try:
        if not os.path.exists(request.dem_path):
            raise HTTPException(
                status_code=404, detail=f"DEM file not found: {request.dem_path}"
            )

        # Load DEM and compute slope
        elevation, metadata = dem_processor.load_dem(request.dem_path)
        slope_deg = dem_processor.compute_slope(elevation, metadata["pixel_size"])

        # Plan route
        path_planner.slope_max_deg = request.slope_max_deg
        route = path_planner.plan_route_from_slope(
            slope_deg,
            metadata["transform"],
            request.start_lat,
            request.start_lon,
            request.goal_lat,
            request.goal_lon,
        )

        # Calculate distance estimate
        path_length = len(route.get("path_rc", []))
        pixel_size_m = (metadata["pixel_size"][0] + metadata["pixel_size"][1]) / 2
        estimated_distance_m = path_length * pixel_size_m

        return {
            "success": route["success"],
            "start_rc": route["start_rc"],
            "goal_rc": route["goal_rc"],
            "path_length_steps": path_length,
            "estimated_distance_m": estimated_distance_m,
            "pixel_size_m": pixel_size_m,
            "message": "Route found" if route["success"] else "No feasible route",
        }

    except Exception as e:
        logger.error(f"Route computation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Route error: {str(e)}")


@app.get("/traverse")
async def get_traverse_data():
    """Get live JPL M20 traverse JSON feed."""
    import httpx

    traverse_url = "https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_traverse.json"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(traverse_url, timeout=10.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch traverse data: {e}")
        raise HTTPException(
            status_code=502, detail="Unable to fetch traverse data from NASA"
        )


@app.get("/waypoints")
async def get_waypoints_data():
    """Get live JPL M20 waypoints JSON feed."""
    import httpx

    waypoints_url = (
        "https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_waypoints.json"
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(waypoints_url, timeout=10.0)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch waypoints data: {e}")
        raise HTTPException(
            status_code=502, detail="Unable to fetch waypoints data from NASA"
        )


@app.post("/plan_scenario")
async def plan_scenario(request: ScenarioRequest):
    """Plan mission from scenario description.

    Example: {"scenario": "Dust storm incoming - adjust 5-day mission plan"}
    """
    try:
        # Parse scenario and generate objectives
        scenario = request.scenario.lower()

        goals = []
        constraints = {"time_budget_min": 480}

        if "dust storm" in scenario or "storm" in scenario:
            goals = [
                "Secure rover in safe location",
                "Minimize power consumption",
                "Maintain thermal control",
                "Store critical science data",
            ]
            constraints["time_budget_min"] = 240  # Shorter operations

        elif "sample" in scenario:
            goals = [
                "Navigate to sample collection site",
                "Perform sample acquisition",
                "Cache sample for MSR",
            ]

        elif "emergency" in scenario:
            goals = [
                "Assess system status",
                "Execute contingency procedures",
                "Establish communication",
            ]
            constraints["time_budget_min"] = 120

        else:
            goals = ["Nominal science operations"]

        # Generate plan
        plan_request = MissionPlanRequest(
            goals=goals, constraints=PlanConstraints(**constraints), include_report=True
        )

        return await generate_mission_plan(plan_request)

    except Exception as e:
        logger.error(f"Scenario planning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Scenario error: {str(e)}")


@app.post("/plan_daily")
async def plan_daily_mission(include_report: bool = False):
    """Plan using live traverse data to seed current location."""
    try:
        # Fetch current rover position (not used in this simplified example)
        _ = await get_traverse_data()

        # Extract latest position (simplified)
        _latest_position = {"lat": 18.4447, "lon": 77.4508}  # Jezero Crater approximate

        goals = [
            "Daily system health check",
            "Science imaging operations",
            "Data transmission to Earth",
        ]

        plan_request = MissionPlanRequest(
            goals=goals,
            constraints=PlanConstraints(time_budget_min=480),
            include_report=include_report,
        )

        return await generate_mission_plan(plan_request)

    except Exception as e:
        logger.error(f"Daily planning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Daily plan error: {str(e)}")


@app.post("/export_sequence")
async def export_command_sequence(plan: Dict[str, Any]):
    """Convert plan JSON into simple command sequence."""
    try:
        activities = plan.get("activities", [])

        sequence = {"sequence_id": f"SEQ_{len(activities):04d}", "commands": []}

        for i, activity in enumerate(activities):
            command = {
                "cmd_id": i + 1,
                "type": activity.get("type", "unknown"),
                "start_time": activity.get("start_min", 0),
                "duration": activity.get("duration_min", 0),
                "parameters": {
                    k: v
                    for k, v in activity.items()
                    if k not in ["type", "start_min", "duration_min", "id"]
                },
            }
            sequence["commands"].append(command)

        return sequence

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Export error: {str(e)}")


@app.post("/dem/upload")
async def upload_dem(file: UploadFile = File(...)):
    """Upload and manage DEM files locally."""
    dem_dir = Path(os.getenv("DEM_TIF_PATH", "./data/dem/"))
    dem_dir.mkdir(parents=True, exist_ok=True)

    file_path = dem_dir / file.filename

    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return {
            "filename": file.filename,
            "path": str(file_path),
            "size_mb": len(content) / (1024 * 1024),
            "status": "uploaded",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/dem/list")
async def list_dems():
    """List available DEM files."""
    dem_dir = Path(os.getenv("DEM_TIF_PATH", "./data/dem/"))

    if not dem_dir.exists():
        return {"dems": []}

    dems = []
    for dem_file in dem_dir.glob("*.tif"):
        stat = dem_file.stat()
        dems.append(
            {
                "filename": dem_file.name,
                "path": str(dem_file),
                "size_mb": stat.st_size / (1024 * 1024),
            }
        )

    return {"dems": dems}


@app.post("/dem/compute_slope")
async def compute_slope_grid(request: SlopeComputeRequest):
    """Compute and cache slope grid from DEM."""
    try:
        result = dem_processor.compute_and_cache_slope(
            request.dem_path, request.max_safe_slope
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Slope computation error: {str(e)}"
        )


@app.post("/environment_search")
async def search_environment_data(request: EnvironmentSearchRequest):
    """Proxy to NASA PDS search with caller-supplied params."""
    import httpx

    pds_base_url = os.getenv(
        "NASA_PDS_BASE_URL", "https://pds-imaging.jpl.nasa.gov/data/"
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                pds_base_url, params=request.params, timeout=15.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"PDS search failed: {e}")
        raise HTTPException(status_code=502, detail=f"PDS search error: {str(e)}")


@app.get("/dashboard", response_class=HTMLResponse)
async def mission_dashboard():
    """Simple web UI dashboard."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mars Mission Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: #eee; }
            h1 { color: #16f; }
            .card { background: #252540; padding: 20px; margin: 20px 0; border-radius: 8px; }
            button { background: #16f; color: white; padding: 10px 20px; border: none; 
                    border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background: #0d4; }
            pre { background: #1a1a1a; padding: 15px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1> Mars Mission Planning Dashboard</h1>
        
        <div class="card">
            <h2>Quick Actions</h2>
            <button onclick="checkHealth()">Health Check</button>
            <button onclick="getTraverse()">Fetch Traverse Data</button>
            <button onclick="planDaily()">Generate Daily Plan</button>
        </div>
        
        <div class="card">
            <h2>Status</h2>
            <pre id="output">Ready. Click a button to interact with the API.</pre>
        </div>
        
        <script>
            const output = document.getElementById('output');
            
            async function checkHealth() {
                const res = await fetch('/health');
                output.textContent = JSON.stringify(await res.json(), null, 2);
            }
            
            async function getTraverse() {
                output.textContent = 'Fetching traverse data...';
                try {
                    const res = await fetch('/traverse');
                    output.textContent = JSON.stringify(await res.json(), null, 2);
                } catch(e) {
                    output.textContent = 'Error: ' + e.message;
                }
            }
            
            async function planDaily() {
                output.textContent = 'Generating daily mission plan...';
                try {
                    const res = await fetch('/plan_daily', {method: 'POST'});
                    output.textContent = JSON.stringify(await res.json(), null, 2);
                } catch(e) {
                    output.textContent = 'Error: ' + e.message;
                }
            }
        </script>
    </body>
    </html>
    """


# Run with: uvicorn src.interfaces.web_interface:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web_interface:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
    )
