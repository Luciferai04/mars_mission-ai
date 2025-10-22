# Getting Started with Mars Mission Planning Assistant

##  Quick Start (5 Minutes)

### 1. Clone and Setup
```bash
cd mars_mission-ai
make setup
source .venv/bin/activate
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your keys:
# - OPENAI_API_KEY=sk-...
# - NASA_API_KEY=... (optional, get from https://api.nasa.gov)
```

### 3. Start the Server
```bash
make run-dev
```

### 4. Test the API
Open your browser to:
- **Dashboard**: http://localhost:8000/dashboard
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

##  Prerequisites

- **Python 3.10+**
- **GDAL/PROJ** (for DEM processing)
  - macOS: `brew install gdal`
  - Ubuntu: `sudo apt-get install gdal-bin libgdal-dev`
- **OpenAI API Key** (required for GPT-4 planning)
- **NASA API Key** (optional but recommended)

---

##  Example Usage

### Plan a Daily Mission
```bash
curl -X POST http://localhost:8000/plan_daily \
  -H "Content-Type: application/json"
```

### Plan from Scenario File
```bash
make run-scenario file=scenarios/daily_plan.json
```

### Compute Route from DEM
```bash
curl -X POST http://localhost:8000/route_from_dem \
  -H "Content-Type: application/json" \
  -d '{
    "dem_path": "./data/dem/jezero_demo.tif",
    "start_lat": 18.4, "start_lon": 77.5,
    "goal_lat": 18.45, "goal_lon": 77.55,
    "slope_max_deg": 30.0
  }'
```

---

##  Project Structure

```
mars_mission-ai/
 src/
    core/               # Core algorithms
    interfaces/         # FastAPI web interface
    data_pipeline/      # DEM & NASA data processing
    utils/              # Helper functions
 scenarios/              # Example mission scenarios
    daily_plan.json
    emergency_replan.json
    long_term.json
 data/
    dem/                # DEM GeoTIFF files
    cache/              # Processed data cache
 tests/                  # Test suite (TODO)
 docs/                   # Documentation (TODO)
 models/                 # Trained models
 scripts/                # Utility scripts
 requirements.txt        # Python dependencies
 .env.example           # Configuration template
 Makefile               # Automation commands
 README.md              # Main documentation
```

---

##  Available Commands

### Development
```bash
make setup          # Initial setup with venv
make install        # Install dependencies only
make run            # Start production server
make run-dev        # Start development server with reload
make format         # Format code with black
make lint           # Run linting checks
make test           # Run test suite
make clean          # Clean cache files
```

### Data Operations
```bash
# Download a DEM
make fetch-demo-dem url=https://example.com/dem.tif

# Compute and cache slope from DEM
curl -X POST http://localhost:8000/dem/compute_slope \
  -H "Content-Type: application/json" \
  -d '{"dem_path": "./data/dem/jezero_demo.tif"}'

# List available DEMs
curl http://localhost:8000/dem/list
```

### Mission Planning
```bash
# Plan from scenario
make run-scenario file=scenarios/emergency_replan.json

# Plan daily mission
curl -X POST http://localhost:8000/plan_daily

# Plan from custom goals
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "goals": ["Investigate delta deposits", "Collect rock sample"],
    "constraints": {"time_budget_min": 480},
    "include_report": true
  }'
```

---

##  API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/plan` | POST | Generate mission plan |
| `/route_from_dem` | POST | Compute optimal route |
| `/traverse` | GET | Live rover traverse data |
| `/waypoints` | GET | Live rover waypoints |
| `/plan_scenario` | POST | Plan from scenario description |
| `/plan_daily` | POST | Generate daily plan |
| `/export_sequence` | POST | Export command sequence |
| `/dem/upload` | POST | Upload DEM file |
| `/dem/list` | GET | List available DEMs |
| `/dem/compute_slope` | POST | Compute slope from DEM |
| `/dashboard` | GET | Interactive web dashboard |

Full API documentation: http://localhost:8000/docs

---

##  Example Scenarios

### 1. Daily Nominal Operations
```json
{
  "goals": [
    "Perform daily system health checks",
    "Conduct Mastcam-Z panoramic imaging",
    "Analyze rock sample with SuperCam",
    "Complete data transmission to Earth"
  ],
  "constraints": {
    "time_budget_min": 480,
    "power_budget_wh": 1200
  }
}
```

### 2. Emergency Dust Storm Response
```json
{
  "scenario": "Dust storm incoming - adjust mission plan",
  "goals": [
    "Secure rover in safe location",
    "Minimize power consumption",
    "Maintain thermal control"
  ],
  "constraints": {
    "time_budget_min": 120,
    "power_budget_wh": 400
  }
}
```

### 3. Long-Term Sample Collection
```json
{
  "goals": [
    "Navigate to delta region",
    "Collect 3 high-priority rock samples",
    "Cache samples for MSR"
  ],
  "constraints": {
    "time_budget_min": 2400,
    "max_drive_m": 500
  }
}
```

---

##  Troubleshooting

### GDAL Import Error
```bash
# macOS
brew install gdal
pip install --no-cache-dir GDAL==$(gdal-config --version)

# Ubuntu
sudo apt-get install gdal-bin libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install GDAL
```

### OpenAI API Key Issues
Make sure your `.env` file has:
```
OPENAI_API_KEY=sk-your-key-here
```

### NASA API Rate Limits
The NASA API has rate limits. If you get 429 errors:
1. Get a free API key from https://api.nasa.gov
2. Add to `.env`: `NASA_API_KEY=your-key-here`

---

##  Next Steps

1. **Read IMPROVEMENTS.md** - Understand current state and roadmap
2. **Explore scenarios/** - See example mission plans
3. **Check API docs** - http://localhost:8000/docs
4. **Download DEMs** - Get real Mars terrain data
5. **Contribute** - Add tests, improve NASA integration, enhance docs

---

##  Contributing

Priority areas needing work (see IMPROVEMENTS.md):
1. **Testing** - Write unit and integration tests
2. **NASA Data** - Implement PDS/ODE API clients
3. **Documentation** - Add deployment guides and examples
4. **Features** - Real GPT-4V terrain analysis, advanced scheduling

---

##  Resources

- [Mars 2020 Mission](https://mars.nasa.gov/mars2020/)
- [Perseverance Rover](https://mars.nasa.gov/mars2020/spacecraft/rover/)
- [NASA PDS](https://pds.nasa.gov/)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [FastAPI Docs](https://fastapi.tiangolo.com/)

---

##  License

See project LICENSE file for details.
