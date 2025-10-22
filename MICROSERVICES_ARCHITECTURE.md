# Mars Mission Planning Assistant - Microservices Architecture

## Overview

A minimal, scalable microservices architecture designed for production deployment of the Mars Mission Planning Assistant with proper service separation, API gateway, and containerization.

---

##  Architecture Diagram

```
                           
                              API Gateway   
                              (nginx/Kong)  
                           
                                    
                
                                                      
            
          Planning API     Vision API        MARL API     
          (Mission Plans)   (Terrain)       (RL Agents)    
            
                                                      
                
                                    
                            
                              Data Service  
                              (NASA APIs)   
                            
                                    
                
                                                      
            
           PostgreSQL         Redis         MinIO (S3)    
           (Metadata)        (Cache)        (DEM Files)   
            
```

---

##  Core Microservices

### 1. API Gateway Service
**Purpose:** Central entry point, routing, auth, rate limiting

**Technology:** nginx or Kong  
**Port:** 8000 (external)  
**Responsibilities:**
- Route requests to appropriate services
- Authentication/Authorization (JWT)
- Rate limiting and DDoS protection
- SSL/TLS termination
- API versioning
- Request/response logging

**Endpoints:**
- `/api/v1/planning/*` → Planning API
- `/api/v1/vision/*` → Vision API
- `/api/v1/marl/*` → MARL API
- `/api/v1/data/*` → Data Service

---

### 2. Mission Planning Service
**Purpose:** Core mission planning and optimization

**Technology:** Python (FastAPI)  
**Port:** 8001 (internal)  
**Container:** `planning-service`  
**Resources:** 2 CPU, 4GB RAM

**Responsibilities:**
- Generate mission plans (GPT-4 integration)
- Resource optimization (power, time, thermal)
- Constraint solving (ASPEN-inspired)
- Multi-sol planning
- Activity sequencing
- Report generation

**Key Endpoints:**
```
POST /plan                  # Generate mission plan
POST /plan_scenario         # Quick scenario planning
POST /plan_daily            # Daily operations
POST /optimize_resources    # Resource optimization
POST /export_sequence       # Command sequence export
GET  /health                # Service health check
```

**Dependencies:**
- Data Service (rover state, targets)
- Redis (caching plans)
- PostgreSQL (plan history)

---

### 3. Vision Processing Service
**Purpose:** Terrain analysis and hazard detection

**Technology:** Python (FastAPI + PyTorch)  
**Port:** 8002 (internal)  
**Container:** `vision-service`  
**Resources:** 4 CPU, 8GB RAM (GPU optional)

**Responsibilities:**
- Terrain hazard classification
- Image preprocessing
- Vision model inference (ConvNeXt-Tiny)
- Hazard map generation
- Batch image processing
- Model versioning

**Key Endpoints:**
```
POST /classify_hazard       # Classify single image
POST /classify_batch        # Batch classification
POST /generate_hazard_map   # Create hazard map from DEM
POST /analyze_terrain       # Full terrain analysis
GET  /model_info            # Model version and stats
GET  /health                # Service health check
```

**Dependencies:**
- MinIO/S3 (image storage)
- Redis (inference cache)
- GPU (optional, for faster inference)

**Model Storage:**
- Models loaded at startup
- Hot-reload capability for new models
- A/B testing support

---

### 4. MARL Optimization Service
**Purpose:** Reinforcement learning-based optimization

**Technology:** Python (FastAPI)  
**Port:** 8003 (internal)  
**Container:** `marl-service`  
**Resources:** 2 CPU, 4GB RAM

**Responsibilities:**
- Load trained RL agents
- Coordinate multi-agent decisions
- Generate optimized action sequences
- Training mode (optional)
- Policy updates
- Agent performance tracking

**Key Endpoints:**
```
POST /optimize              # RL-based optimization
POST /train                 # Train agents (optional)
GET  /agents/stats          # Agent statistics
GET  /agents/confidence     # RL confidence metrics
POST /agents/reload         # Hot-reload trained models
GET  /health                # Service health check
```

**Dependencies:**
- Data Service (mission context)
- Redis (Q-table cache)
- PostgreSQL (training history)

**Model Storage:**
- Agent policies loaded at startup
- Checkpoint management
- Incremental learning support

---

### 5. Data Integration Service
**Purpose:** NASA data fetching and caching

**Technology:** Python (FastAPI)  
**Port:** 8004 (internal)  
**Container:** `data-service`  
**Resources:** 1 CPU, 2GB RAM

**Responsibilities:**
- NASA API client
- Rover traverse data (M20_traverse.json)
- MEDA environmental data
- Mastcam-Z metadata
- DEM file management
- Data caching strategy
- Rate limiting to NASA APIs

**Key Endpoints:**
```
GET  /traverse              # Live rover traverse
GET  /navcam/recent         # Recent Navcam images
GET  /navcam/metadata       # Mastcam-Z metadata
GET  /environment           # MEDA environmental data
POST /dem/upload            # Upload DEM file
GET  /dem/list              # List available DEMs
GET  /health                # Service health check
```

**Dependencies:**
- NASA PDS API (external)
- NASA Mars API (external)
- MinIO/S3 (DEM storage)
- Redis (data cache)
- PostgreSQL (metadata)

---

##  Shared Data Stores

### 6. PostgreSQL Database
**Purpose:** Structured data storage

**Port:** 5432 (internal)  
**Container:** `postgres`  
**Resources:** 2 CPU, 4GB RAM

**Schemas:**
```sql
-- Mission plans
mission_plans (
    id UUID PRIMARY KEY,
    created_at TIMESTAMP,
    goals JSONB,
    activities JSONB,
    constraints JSONB,
    status VARCHAR(50)
)

-- MARL training history
marl_episodes (
    id SERIAL PRIMARY KEY,
    episode_num INT,
    reward FLOAT,
    agent_stats JSONB,
    created_at TIMESTAMP
)

-- Vision model results
vision_results (
    id UUID PRIMARY KEY,
    image_path VARCHAR(255),
    classification VARCHAR(50),
    confidence FLOAT,
    model_version VARCHAR(50),
    created_at TIMESTAMP
)

-- Audit logs
audit_logs (
    id SERIAL PRIMARY KEY,
    service VARCHAR(50),
    action VARCHAR(100),
    user_id VARCHAR(100),
    details JSONB,
    created_at TIMESTAMP
)
```

---

### 7. Redis Cache
**Purpose:** High-speed caching and message queue

**Port:** 6379 (internal)  
**Container:** `redis`  
**Resources:** 1 CPU, 2GB RAM

**Use Cases:**
- Mission plan caching (TTL: 1 hour)
- Vision inference results (TTL: 24 hours)
- NASA API response cache (TTL: 15 minutes)
- MARL Q-value cache (persistent)
- Rate limiting counters
- Service health status
- Job queue (Celery backend)

**Key Patterns:**
```
plan:{plan_id} → Mission plan JSON
vision:{image_hash} → Classification result
nasa:traverse → Latest traverse data
marl:agents:{agent_name} → Agent Q-values
health:{service_name} → Health status
```

---

### 8. MinIO (S3-compatible Storage)
**Purpose:** Object storage for large files

**Port:** 9000 (internal), 9001 (console)  
**Container:** `minio`  
**Resources:** 1 CPU, 2GB RAM

**Buckets:**
```
mars-dems/           # DEM GeoTIFF files
   jezero_*.tif
mars-images/         # Navcam/Mastcam images
   navcam/
   mastcam/
mars-models/         # Trained models
   vision/
   marl/
mars-reports/        # Generated reports
   mission_*.pdf
```

**Access:**
- Internal: S3 API
- External: Pre-signed URLs (time-limited)
- Backup: S3 replication

---

##  Supporting Services

### 9. Message Queue (Optional)
**Technology:** RabbitMQ or Kafka  
**Purpose:** Async job processing

**Use Cases:**
- Long-running MARL training
- Batch vision processing
- Report generation
- NASA data sync

---

### 10. Monitoring Stack

**Prometheus** (Metrics)
- Service metrics
- API latency
- Resource usage
- Request rates

**Grafana** (Visualization)
- Real-time dashboards
- Alert visualization
- Performance graphs

**Loki** (Logs)
- Centralized logging
- Log aggregation
- Search and filter

---

##  Docker Compose Configuration

```yaml
version: '3.8'

services:
  # API Gateway
  gateway:
    image: nginx:alpine
    ports:
      - "8000:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - planning-service
      - vision-service
      - marl-service
      - data-service
    networks:
      - mars-net

  # Mission Planning Service
  planning-service:
    build: ./services/planning
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/mars
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - postgres
      - redis
    networks:
      - mars-net
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G

  # Vision Processing Service
  vision-service:
    build: ./services/vision
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/mars
      - REDIS_URL=redis://redis:6379/1
      - MINIO_URL=minio:9000
      - MODEL_PATH=/models/terrain_vision_convnext.pth
    volumes:
      - ./models:/models:ro
    depends_on:
      - postgres
      - redis
      - minio
    networks:
      - mars-net
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  # MARL Optimization Service
  marl-service:
    build: ./services/marl
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/mars
      - REDIS_URL=redis://redis:6379/2
      - MODEL_DIR=/models/marl
    volumes:
      - ./models/marl:/models/marl:ro
    depends_on:
      - postgres
      - redis
    networks:
      - mars-net
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G

  # Data Integration Service
  data-service:
    build: ./services/data
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/mars
      - REDIS_URL=redis://redis:6379/3
      - MINIO_URL=minio:9000
      - NASA_API_KEY=${NASA_API_KEY}
    depends_on:
      - postgres
      - redis
      - minio
    networks:
      - mars-net

  # PostgreSQL
  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=mars
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - mars-net

  # Redis
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - mars-net

  # MinIO
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - minio-data:/data
    ports:
      - "9001:9001"
    networks:
      - mars-net

  # Prometheus (Monitoring)
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - mars-net

  # Grafana (Visualization)
  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - mars-net

volumes:
  postgres-data:
  redis-data:
  minio-data:
  prometheus-data:
  grafana-data:

networks:
  mars-net:
    driver: bridge
```

---

##  Security & Authentication

### API Gateway Security
```nginx
# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

# JWT validation
auth_jwt "Mars Mission API";
auth_jwt_key_file /etc/nginx/jwt_key.pem;

# Headers
add_header X-Frame-Options "DENY";
add_header X-Content-Type-Options "nosniff";
add_header X-XSS-Protection "1; mode=block";
```

### Service-to-Service Auth
- mTLS (mutual TLS) for internal communication
- Service mesh (Istio) for advanced scenarios
- API keys in environment variables

### Secrets Management
```bash
# Use Docker secrets or Kubernetes secrets
docker secret create nasa_api_key nasa_key.txt
docker secret create openai_api_key openai_key.txt
```

---

##  Service Communication Patterns

### Synchronous (REST)
```
Gateway → Services: HTTP/REST
Services → PostgreSQL: SQL
Services → Redis: Redis Protocol
Services → MinIO: S3 API
```

### Asynchronous (Optional)
```
Services → RabbitMQ → Workers
Services → Kafka → Consumers
```

### Service Discovery
- Docker DNS (service names)
- Consul (advanced scenarios)
- Kubernetes DNS (for K8s deployment)

---

##  Deployment Strategies

### Development
```bash
docker-compose up -d
```

### Staging
```bash
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
```

### Production (Kubernetes)
```bash
kubectl apply -f k8s/
```

---

##  Scaling Strategy

### Horizontal Scaling
```yaml
# Planning Service: 2-10 replicas
planning-service:
  deploy:
    replicas: 2
    autoscaling:
      min: 2
      max: 10
      cpu_threshold: 70%

# Vision Service: 1-5 replicas (GPU limited)
vision-service:
  deploy:
    replicas: 1
    autoscaling:
      min: 1
      max: 5
      cpu_threshold: 80%

# MARL Service: 2-5 replicas
marl-service:
  deploy:
    replicas: 2
    autoscaling:
      min: 2
      max: 5
      cpu_threshold: 70%
```

### Database Scaling
- PostgreSQL: Read replicas for heavy queries
- Redis: Redis Cluster for high availability
- MinIO: Distributed mode for large files

---

##  Health Checks & Monitoring

### Health Check Endpoints
```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "planning-service",
        "version": "1.0.0",
        "dependencies": {
            "postgres": check_postgres(),
            "redis": check_redis()
        }
    }
```

### Metrics Collection
```python
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('api_requests_total', 'Total API requests')
REQUEST_LATENCY = Histogram('api_request_duration_seconds', 'Request latency')

@app.middleware("http")
async def add_metrics(request, call_next):
    REQUEST_COUNT.inc()
    with REQUEST_LATENCY.time():
        response = await call_next(request)
    return response
```

---

##  Directory Structure

```
mars_mission-ai/
 docker-compose.yml
 docker-compose.staging.yml
 docker-compose.prod.yml
 nginx.conf
 prometheus.yml
 init.sql

 services/
    planning/
       Dockerfile
       requirements.txt
       app/
          main.py
          routers/
          models/
          utils/
       tests/
   
    vision/
       Dockerfile
       requirements.txt
       app/
       tests/
   
    marl/
       Dockerfile
       requirements.txt
       app/
       tests/
   
    data/
        Dockerfile
        requirements.txt
        app/
        tests/

 k8s/                      # Kubernetes manifests
    deployments/
    services/
    configmaps/
    secrets/

 models/
     vision/
     marl/
```

---

##  Benefits of This Architecture

| Benefit | Description |
|---------|-------------|
| **Scalability** | Each service scales independently |
| **Fault Isolation** | One service failure doesn't crash everything |
| **Technology Flexibility** | Each service can use optimal tech stack |
| **Team Autonomy** | Teams can work on services independently |
| **Easy Deployment** | Deploy services individually |
| **Monitoring** | Granular metrics per service |
| **Cost Efficiency** | Scale only what needs scaling |

---

##  Migration Path

### Phase 1: Monolith → Modular (Current)
Current single FastAPI app with modules

### Phase 2: Service Extraction
1. Extract Vision Service (week 1)
2. Extract MARL Service (week 2)
3. Extract Data Service (week 3)
4. Deploy API Gateway (week 4)

### Phase 3: Containerization
1. Docker images per service
2. Docker Compose setup
3. Local testing
4. Staging deployment

### Phase 4: Production
1. Kubernetes manifests
2. CI/CD pipelines
3. Monitoring setup
4. Production deployment

---

##  Quick Start

```bash
# 1. Clone repository
git clone <repo>
cd mars_mission-ai

# 2. Set environment variables
cp .env.example .env
# Edit .env with API keys

# 3. Build and start services
docker-compose up -d

# 4. Check service health
curl http://localhost:8000/health

# 5. Access services
# API Gateway: http://localhost:8000
# Grafana: http://localhost:3000
# MinIO Console: http://localhost:9001
```

---

**This microservices architecture is production-ready, scalable, and follows industry best practices for modern cloud-native applications! **
