# API Documentation

## Base URL
```
http://localhost:8000  (local)
https://your-domain.com  (production)
```

## Interactive Documentation
- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`

---

## Core Endpoints

### Health & Status

#### `GET /health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "service": "Mars Mission Planning Assistant",
  "version": "1.0.0"
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### Mission Planning

#### `POST /plan`
Generate comprehensive mission plan from goals and constraints.

**Request Body:**
```json
{
  "goals": [
    "Perform daily system health checks",
    "Conduct Mastcam-Z panoramic imaging",
    "Analyze rock sample with SuperCam"
  ],
  "constraints": {
    "time_budget_min": 480,
    "power_budget_wh": 1200,
    "data_budget_mb": 250,
    "max_drive_m": 100,
    "comm_windows": [
      {"start_min": 420, "end_min": 450}
    ]
  },
  "include_report": true
}
```

**Response:**
```json
{
  "plan": {
    "goals": ["..."],
    "activities": [
      {
        "id": "pre_checks",
        "type": "system_checks",
        "duration_min": 10,
        "power_w": 50,
        "start_min": 0
      }
    ],
    "power": {
      "budget_wh": 1200,
      "consumed_wh": 850
    },
    "validation": {
      "ok": true,
      "violations": []
    }
  },
  "report": "Mars Mission Plan Report\n..."
}
```

**Python Example:**
```python
import requests

response = requests.post(
    'http://localhost:8000/plan',
    json={
        'goals': ['Investigate delta deposits'],
        'constraints': {'time_budget_min': 480},
        'include_report': False
    }
)
plan = response.json()
```

---

#### `POST /plan_daily`
Generate daily mission plan using live rover data.

**Query Parameters:**
- `include_report` (bool, optional): Include human-readable report

**Response:** Same as `/plan`

**Example:**
```bash
curl -X POST "http://localhost:8000/plan_daily?include_report=true"
```

---

#### `POST /plan_scenario`
Plan mission from scenario description.

**Request Body:**
```json
{
  "scenario": "Dust storm incoming - adjust 5-day mission plan"
}
```

**Scenarios Supported:**
- Dust storm emergency
- Sample collection campaign
- Emergency system issues
- Nominal science operations

**Example:**
```bash
curl -X POST http://localhost:8000/plan_scenario \
  -H "Content-Type: application/json" \
  -d '{"scenario": "Emergency replan for dust storm"}'
```

---

### Path Planning

#### `POST /route_from_dem`
Compute optimal route between two points using DEM.

**Request Body:**
```json
{
  "dem_path": "./data/dem/jezero_demo.tif",
  "start_lat": 18.4,
  "start_lon": 77.5,
  "goal_lat": 18.45,
  "goal_lon": 77.55,
  "slope_max_deg": 30.0
}
```

**Response:**
```json
{
  "success": true,
  "start_rc": [10, 15],
  "goal_rc": [50, 55],
  "path_length_steps": 45,
  "estimated_distance_m": 4500.0,
  "pixel_size_m": 100.0,
  "message": "Route found"
}
```

**Example:**
```python
response = requests.post(
    'http://localhost:8000/route_from_dem',
    json={
        'dem_path': './data/dem/test.tif',
        'start_lat': 18.4,
        'start_lon': 77.5,
        'goal_lat': 18.45,
        'goal_lon': 77.55
    }
)
```

---

### NASA Data Integration

#### `GET /traverse`
Get live rover traverse data from JPL.

**Response:**
```json
{
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
```

---

#### `GET /waypoints`
Get rover waypoints.

**Response:** GeoJSON FeatureCollection

---

### DEM Management

#### `POST /dem/compute_slope`
Compute and cache slope from DEM.

**Request Body:**
```json
{
  "dem_path": "./data/dem/jezero.tif",
  "max_safe_slope": 30.0
}
```

**Response:**
```json
{
  "slope_cache_path": "./data/cache/jezero_slope.npz",
  "metadata": {
    "min_slope": 0.0,
    "max_slope": 45.2,
    "mean_slope": 12.5,
    "safe_percentage": 85.3
  },
  "cache_hit": false
}
```

---

#### `GET /dem/list`
List available DEM files.

**Response:**
```json
{
  "dems": [
    {
      "filename": "jezero_demo.tif",
      "path": "/path/to/dem.tif",
      "size_mb": 125.4
    }
  ]
}
```

---

#### `POST /dem/upload`
Upload new DEM file.

**Request:** multipart/form-data with file

**Example:**
```bash
curl -X POST http://localhost:8000/dem/upload \
  -F "file=@jezero.tif"
```

---

### Command Export

#### `POST /export_sequence`
Convert mission plan to command sequence.

**Request Body:**
```json
{
  "activities": [
    {"id": "act1", "type": "imaging", "duration_min": 20}
  ]
}
```

**Response:**
```json
{
  "sequence_id": "SEQ_0001",
  "commands": [
    {
      "cmd_id": 1,
      "type": "imaging",
      "start_time": 0,
      "duration": 20,
      "parameters": {}
    }
  ]
}
```

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Invalid request parameters"
}
```

### 404 Not Found
```json
{
  "detail": "DEM file not found: /path/to/dem.tif"
}
```

### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "goals"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error
```json
{
  "detail": "Mission planning error: ..."
}
```

### 502 Bad Gateway
```json
{
  "detail": "Unable to fetch traverse data from NASA"
}
```

---

## Rate Limiting

- Default: 100 requests/minute
- Burst: Up to 200 requests
- NASA API: Subject to NASA rate limits (use API key)

---

## Authentication

Currently no authentication required for local deployment.

For production:
```python
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/plan")
async def plan(request: Request, credentials: HTTPBearer = Depends(security)):
    # Verify token
    pass
```

---

## Pagination

For endpoints returning large lists:
```json
{
  "items": [],
  "page": 1,
  "per_page": 100,
  "total": 1523
}
```

---

## Webhooks (Future)

Subscribe to mission events:
```python
@app.post("/webhooks/subscribe")
async def subscribe(url: str, events: List[str]):
    # events: ["plan_complete", "route_found", "error"]
    pass
```

---

## SDK Examples

### Python Client
```python
import requests

class MarsPlanner:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def plan_mission(self, goals, constraints):
        return requests.post(
            f"{self.base_url}/plan",
            json={"goals": goals, "constraints": constraints}
        ).json()
    
    def compute_route(self, dem_path, start, goal):
        return requests.post(
            f"{self.base_url}/route_from_dem",
            json={
                "dem_path": dem_path,
                "start_lat": start[0],
                "start_lon": start[1],
                "goal_lat": goal[0],
                "goal_lon": goal[1]
            }
        ).json()

# Usage
planner = MarsPlanner()
plan = planner.plan_mission(
    goals=["Investigate crater"],
    constraints={"time_budget_min": 480}
)
```

### JavaScript Client
```javascript
const API_URL = 'http://localhost:8000';

async function planMission(goals, constraints) {
  const response = await fetch(`${API_URL}/plan`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({goals, constraints})
  });
  return response.json();
}

// Usage
const plan = await planMission(
  ['Conduct science operations'],
  {time_budget_min: 480}
);
```

---

## Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Full Workflow Test
```bash
# 1. Check health
curl http://localhost:8000/health

# 2. List DEMs
curl http://localhost:8000/dem/list

# 3. Plan mission
curl -X POST http://localhost:8000/plan \
  -H "Content-Type: application/json" \
  -d '{"goals": ["Test"], "constraints": {"time_budget_min": 480}}'

# 4. Get traverse data
curl http://localhost:8000/traverse
```

---

## Best Practices

1. **Cache Results:** DEM computations are expensive, use caching
2. **Handle Timeouts:** Some operations take >30s, set appropriate timeouts
3. **Error Handling:** Always check response status codes
4. **Pagination:** Use pagination for large result sets
5. **Rate Limiting:** Respect rate limits, especially for NASA APIs

---

## Support

- **Swagger UI**: http://localhost:8000/docs
- **Issues**: GitHub repository
- **Docs**: See GETTING_STARTED.md and deployment_instructions.md
