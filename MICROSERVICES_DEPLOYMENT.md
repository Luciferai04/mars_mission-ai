# Mars Mission AI - Microservices Deployment Guide

## Architecture Overview

The Mars Mission AI system is now decomposed into specialized microservices:

### Services

1. **API Gateway (Kong)** - Port 8000/8001
   - Central entry point for all API requests
   - Rate limiting, authentication, load balancing
   - Admin API on port 8001

2. **Vision Service** - Port 8002
   - Terrain image analysis using SegFormer
   - Hazard detection and classification
   - Rock/obstacle identification

3. **MARL Service** - Port 8003
   - Multi-Agent Reinforcement Learning optimization
   - 5 specialized agents (Route, Power, Science, Hazard, Strategy)
   - Mission plan optimization with RL confidence scores

4. **Data Integration Service** - Port 8004
   - Aggregates data from multiple Mars data sources
   - Environment, terrain, science targets, rover status
   - Concurrent data fetching with async/await

5. **Planning Service** - Port 8005
   - Orchestrates all services
   - Creates complete mission plans
   - Supports vision-enhanced planning

### Supporting Infrastructure

- **PostgreSQL** - Port 5432: Main database for mission data
- **Redis** - Port 6379: Caching layer
- **Kong Database** - Separate PostgreSQL for Kong

## Directory Structure

```
mars_mission-ai/
 services/
    vision/
       app/
          main.py
       Dockerfile
       requirements.txt
    marl/
       app/
          main.py
       Dockerfile
       requirements.txt
    data-integration/
       app/
          main.py
       Dockerfile
       requirements.txt
    planning/
        app/
           main.py
        Dockerfile
        requirements.txt
 database/
    init.sql
    schema.sql
 gateway/
    kong.yml
 docker-compose.yml
 MICROSERVICES_DEPLOYMENT.md (this file)
```

## Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum (16GB recommended)
- 20GB disk space

### Step 1: Build and Start All Services

```bash
# Build and start all microservices
docker-compose up --build -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### Step 2: Configure API Gateway

```bash
# Configure Kong routes (after Kong is up)
./scripts/configure_gateway.sh
```

Or manually:

```bash
# Add Vision Service
curl -i -X POST http://localhost:8001/services/ \
  --data name=vision-service \
  --data url='http://vision-service:8002'

curl -i -X POST http://localhost:8001/services/vision-service/routes \
  --data 'paths[]=/vision' \
  --data 'strip_path=true'

# Add MARL Service
curl -i -X POST http://localhost:8001/services/ \
  --data name=marl-service \
  --data url='http://marl-service:8003'

curl -i -X POST http://localhost:8001/services/marl-service/routes \
  --data 'paths[]=/marl' \
  --data 'strip_path=true'

# Add Data Integration Service
curl -i -X POST http://localhost:8001/services/ \
  --data name=data-service \
  --data url='http://data-integration-service:8004'

curl -i -X POST http://localhost:8001/services/data-service/routes \
  --data 'paths[]=/data' \
  --data 'strip_path=true'

# Add Planning Service
curl -i -X POST http://localhost:8001/services/ \
  --data name=planning-service \
  --data url='http://planning-service:8005'

curl -i -X POST http://localhost:8001/services/planning-service/routes \
  --data 'paths[]=/planning' \
  --data 'strip_path=true'
```

### Step 3: Verify Services

```bash
# Check all services health
curl http://localhost:8002/health  # Vision
curl http://localhost:8003/health  # MARL
curl http://localhost:8004/health  # Data Integration
curl http://localhost:8005/health  # Planning

# Check Planning Service dependencies
curl http://localhost:8005/services/status
```

## Usage Examples

### 1. Create a Mission Plan (Basic)

```bash
curl -X POST http://localhost:8005/plan \
  -H "Content-Type: application/json" \
  -d '{
    "sol": 1234,
    "lat": -4.5,
    "lon": 137.4,
    "battery_soc": 85.0,
    "time_budget_min": 180,
    "objectives": ["science", "exploration"]
  }'
```

### 2. Create a Mission Plan with Vision Analysis

```bash
curl -X POST http://localhost:8005/plan/with-vision \
  -F "sol=1234" \
  -F "lat=-4.5" \
  -F "lon=137.4" \
  -F "battery_soc=85.0" \
  -F "time_budget_min=180" \
  -F "image=@terrain.jpg"
```

### 3. Direct Service Calls

```bash
# Vision Service - Analyze terrain image
curl -X POST http://localhost:8002/analyze \
  -F "file=@terrain.jpg"

# MARL Service - Get agent stats
curl http://localhost:8003/agents/stats

# MARL Service - Optimize mission
curl -X POST http://localhost:8003/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "lat": -4.5,
    "lon": 137.4,
    "battery_soc": 85.0,
    "time_budget_min": 180,
    "targets": [...],
    "sol": 1234,
    "temp": -63.0,
    "dust": 0.4
  }'

# Data Integration - Get integrated data
curl "http://localhost:8004/integrated?sol=1234&lat=-4.5&lon=137.4"
```

### 4. Through API Gateway (Kong)

```bash
# After gateway configuration
curl http://localhost:8000/vision/health
curl http://localhost:8000/marl/agents/stats
curl http://localhost:8000/data/environment/1234
curl http://localhost:8000/planning/plan -X POST -H "Content-Type: application/json" -d '{...}'
```

## Monitoring

### Container Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f vision-service
docker-compose logs -f marl-service
docker-compose logs -f planning-service
```

### Health Checks

```bash
# Check container health status
docker ps

# Detailed health check
docker inspect mars-planning --format='{{.State.Health.Status}}'
```

### Resource Usage

```bash
# Monitor resource usage
docker stats
```

## Scaling

### Horizontal Scaling

```bash
# Scale specific services
docker-compose up -d --scale vision-service=3
docker-compose up -d --scale marl-service=2
```

### Load Balancing via Kong

Kong automatically load balances across scaled instances.

## Troubleshooting

### Service Not Starting

```bash
# Check logs
docker-compose logs service-name

# Rebuild specific service
docker-compose up -d --build service-name
```

### Database Connection Issues

```bash
# Check PostgreSQL
docker-compose exec postgres psql -U mars_admin -d mars_mission -c "\dt"

# Reinitialize database
docker-compose down -v
docker-compose up -d postgres
```

### Gateway Issues

```bash
# Check Kong logs
docker-compose logs kong

# Verify Kong configuration
curl http://localhost:8001/services
curl http://localhost:8001/routes
```

## Maintenance

### Backup

```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U mars_admin mars_mission > backup.sql

# Backup trained models
tar -czf models_backup.tar.gz models/
```

### Updates

```bash
# Update specific service
docker-compose build service-name
docker-compose up -d --no-deps service-name

# Update all services
docker-compose down
docker-compose up --build -d
```

### Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Remove unused images
docker image prune -a
```

## Production Considerations

### Security

1. Change default passwords in `docker-compose.yml`
2. Enable Kong authentication plugins
3. Use environment variables for secrets
4. Enable TLS/SSL for external access

### Performance

1. Increase container resources in `docker-compose.yml`
2. Use production-grade databases (managed PostgreSQL)
3. Add Redis clustering for caching
4. Enable Kong rate limiting

### Monitoring

1. Add Prometheus + Grafana for metrics
2. Use ELK stack for centralized logging
3. Set up alerting for service failures
4. Monitor MARL agent performance

## Integration with Existing System

The microservices can coexist with the monolithic API:

```yaml
# docker-compose.yml
services:
  # Add legacy API
  legacy-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8001:8000"
    networks:
      - mars-network
```

## Development Mode

```bash
# Start with hot-reload
docker-compose -f docker-compose.dev.yml up

# Run single service for development
docker-compose up vision-service postgres redis
```

## Service Dependencies

```
Planning Service
     Vision Service
     MARL Service
     Data Integration Service
             PostgreSQL
             Redis
```

## API Documentation

Once services are running, access interactive API docs:

- Vision: http://localhost:8002/docs
- MARL: http://localhost:8003/docs
- Data Integration: http://localhost:8004/docs
- Planning: http://localhost:8005/docs
- Kong Admin: http://localhost:8001

## Support

For issues, check:
1. Service logs: `docker-compose logs -f`
2. Health endpoints: `/health` on each service
3. Kong admin API: http://localhost:8001
4. Planning service status: http://localhost:8005/services/status
