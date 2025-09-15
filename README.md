# Mars Mission Planning Assistant

Cutting-edge AI-assisted mission planning for NASA’s Mars 2020 Perseverance operations, built to reduce planning time from hours to minutes while preserving safety and operational authenticity.

Key capabilities
- Terrain analysis (vision): hazard detection, target selection, traversability
- Mission planning (LLM + constraints): ASPEN-inspired iterative repair, resource timelines
- Real NASA data: PDS/ODE, live traverse/waypoints feeds, MEDA environmental data via PDS
- Human-readable mission reports and API endpoints

Quickstart
1) Create and activate a virtual environment
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   ```
2) Install dependencies
   ```bash
   pip install -U pip && pip install -r requirements.txt
   ```
3) Configure environment
   - Copy `.env.example` to `.env` and set: NASA_API_KEY, OPENAI_API_KEY, CACHE_DIR, DEM_TIF_PATH
4) Run quality checks and tests
   ```bash
   pre-commit run --all-files
   pytest -q
   ```
5) Launch API (FastAPI + Uvicorn)
   ```bash
   uvicorn src.interfaces.web_interface:app --reload
   ```

API endpoints
- GET /health: basic health check
- POST /plan: generate a mission plan from goals and constraints
  - Example payload:
    {"goals": ["Plan a 3-sol mission to investigate delta deposits"], "constraints": {"time_budget_min": 480, "comm_windows": [{"start_min": 420, "end_min": 450}] }}
  - Optional: include_report: true to attach a human-readable report
- POST /route_from_dem: compute safest route between lat/lon over a DEM GeoTIFF
  - Body: { dem_path, start_lat, start_lon, goal_lat, goal_lon, slope_max_deg? }
  - Returns route success flag, pixel size, and estimated path distance
- GET /traverse and GET /waypoints: live JPL M20 JSON feeds
- POST /plan_scenario: { scenario: "Dust storm incoming - adjust 5-day mission plan" }
- POST /plan_daily: plan using live traverse to seed current location; optional include_report
- POST /export_sequence: convert a plan JSON into a simple command sequence
- POST /dem/upload and GET /dem/list: manage DEM files locally for routing
- POST /dem/compute_slope: compute and cache slope grid; returns stats and cache path
- POST /environment_search: proxy to NASA PDS search with caller-supplied params
- GET /dashboard: simple web UI

Utilities
- make fetch-demo-dem url=<PDS_or_ODE_DEM_URL> out=jezero_demo.tif
  - Or run: python scripts/fetch_demo_dem.py <URL> <OUTPUT_PATH>

Training (optional)
- Install training deps only when needed: pip install -r requirements-train.txt
- Build a hazard dataset from a DEM: make build-hazard-ds dem=data/dem/jezero_demo.tif out=data/cache/hazard_manifest.json lat=18.4
- Train a simple hazard model: make train-hazard manifest=data/cache/hazard_manifest.json out=models/hazard_threshold.json
- Evaluate: make eval-hazard manifest=data/cache/hazard_manifest.json model=models/hazard_threshold.json
See docs/training_guide.md for details.

Scenario packs and CLI
- Scenarios: scenarios/daily_plan.json, scenarios/emergency_replan.json, scenarios/long_term.json
- Run against API: make run-scenario file=scenarios/daily_plan.json (server must be running)
- CLI local planning: make plan-cli file=scenarios/daily_plan.json out=data/exports/daily_plan.json

Mastcam-Z metadata
- Fetch via NASA API (requires NASA_API_KEY): make fetch-mastcamz-meta sol=1000 out=data/cache/mastcamz_meta.json
- Only real NASA data is used; do not fabricate labels. Derive labels conservatively or via human annotation.

Project layout
See PROJECT_STRUCTURE.txt and the src/ tree for modules.

Notes
- Path contains a space: quote the project path as "mars mission_ai" in commands
- Geospatial deps (rasterio/shapely) may require GDAL/PROJ; use Dockerfile or see docs/deployment_instructions.md for macOS setup
