# Quick Reference Card

##  Quick Start (60 seconds)
```bash
make setup && source .venv/bin/activate
cp .env.example .env  # Add OPENAI_API_KEY
make run-dev
# Open: http://localhost:8000/dashboard
```

##  Testing
```bash
pytest -v                    # Run all tests
pytest -m unit              # Unit tests only
pytest --cov=src            # With coverage
```

##  NASA Data
```python
from src.data_pipeline.nasa_api_client import NASAAPIClient
client = NASAAPIClient()
telemetry = client.get_rover_telemetry(sol=1400)
meda = client.get_meda_environment_data(sol=1400)
images = client.get_mastcamz_metadata(sol=1400)
```

##  API Endpoints
| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Health check |
| `POST /plan` | Generate mission plan |
| `POST /route_from_dem` | Compute route |
| `GET /traverse` | Rover traverse data |
| `POST /dem/compute_slope` | Process DEM |
| `GET /dashboard` | Web UI |

##  Deployment
```bash
# Docker
docker-compose up -d

# Cloud (GCP)
gcloud run deploy --image gcr.io/PROJECT/mars-planner

# Local production
sudo systemctl start mars-planner
```

##  Documentation
- **Getting Started**: `GETTING_STARTED.md`
- **API Docs**: `docs/api_documentation.md`
- **Deployment**: `docs/deployment_instructions.md`
- **Implementation**: `IMPLEMENTATION_COMPLETE.md`

##  Project Status: 85% Complete
-  Testing (50+ tests)
-  NASA Integration (6 APIs)
-  Documentation (1000+ lines)
-  Production Ready

##  Quick Links
- Interactive API: http://localhost:8000/docs
- Dashboard: http://localhost:8000/dashboard
- GitHub: https://github.com/Luciferai04/mars_mission-ai
