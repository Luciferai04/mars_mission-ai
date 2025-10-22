# Deployment Instructions

## Overview

This guide covers deploying the Mars Mission Planning Assistant in various environments: local development, Docker containers, cloud platforms, and production.

---

## Local Development Deployment

### Prerequisites
- Python 3.10+
- GDAL/PROJ libraries
- OpenAI API key
- (Optional) NASA API key

### Setup

```bash
# 1. Clone and navigate
cd mars_mission-ai

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 5. Create data directories
mkdir -p data/{dem,cache} models logs

# 6. Run server
uvicorn src.interfaces.web_interface:app --reload --host 0.0.0.0 --port 8000
```

### Verify Installation
```bash
# Health check
curl http://localhost:8000/health

# View dashboard
open http://localhost:8000/dashboard
```

---

## Docker Deployment

### Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directories
RUN mkdir -p data/dem data/cache models logs

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.interfaces.web_interface:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NASA_API_KEY=${NASA_API_KEY}
      - CACHE_DIR=/app/data/cache
      - DEM_TIF_PATH=/app/data/dem
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
```

### Build and Run

```bash
# Build image
docker build -t mars-mission-planner:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_key_here \
  -v $(pwd)/data:/app/data \
  --name mars-planner \
  mars-mission-planner:latest

# Or use docker-compose
docker-compose up -d

# View logs
docker logs -f mars-planner

# Stop container
docker stop mars-planner
```

---

## Cloud Deployment

### AWS Elastic Beanstalk

1. **Install EB CLI**
```bash
pip install awsebcli
```

2. **Initialize EB**
```bash
eb init -p python-3.11 mars-mission-planner
```

3. **Create environment**
```bash
eb create mars-planner-prod
```

4. **Set environment variables**
```bash
eb setenv OPENAI_API_KEY=your_key NASA_API_KEY=your_key
```

5. **Deploy**
```bash
eb deploy
```

### Google Cloud Run

1. **Build and push image**
```bash
# Configure gcloud
gcloud config set project YOUR_PROJECT_ID

# Build image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/mars-mission-planner

# Deploy
gcloud run deploy mars-mission-planner \
  --image gcr.io/YOUR_PROJECT_ID/mars-mission-planner \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=your_key,NASA_API_KEY=your_key \
  --memory 2Gi \
  --cpu 2
```

### Azure Container Instances

```bash
# Create resource group
az group create --name mars-planner-rg --location eastus

# Create container
az container create \
  --resource-group mars-planner-rg \
  --name mars-mission-planner \
  --image mars-mission-planner:latest \
  --dns-name-label mars-planner \
  --ports 8000 \
  --environment-variables \
    OPENAI_API_KEY=your_key \
    NASA_API_KEY=your_key \
  --memory 2 \
  --cpu 2
```

### Heroku

1. **Create Heroku app**
```bash
heroku create mars-mission-planner
```

2. **Add buildpacks**
```bash
heroku buildpacks:add --index 1 heroku/python
heroku buildpacks:add --index 2 https://github.com/heroku/heroku-geo-buildpack.git
```

3. **Set config vars**
```bash
heroku config:set OPENAI_API_KEY=your_key NASA_API_KEY=your_key
```

4. **Deploy**
```bash
git push heroku main
```

---

## Production Deployment

### Architecture

```

  Load Balancer  

         
    
      Nginx   (Reverse Proxy)
    
         
    
       Gunicorn     (WSGI Server)
    
            
    
     FastAPI App    
    
            
    
      Cache / Data / Logs  
    
```

### Nginx Configuration

`/etc/nginx/sites-available/mars-planner`:

```nginx
upstream mars_planner {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name mars-planner.example.com;
    
    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name mars-planner.example.com;
    
    ssl_certificate /etc/letsencrypt/live/mars-planner.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mars-planner.example.com/privkey.pem;
    
    location / {
        proxy_pass http://mars_planner;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /static {
        alias /var/www/mars-planner/static;
        expires 30d;
    }
}
```

### Systemd Service

`/etc/systemd/system/mars-planner.service`:

```ini
[Unit]
Description=Mars Mission Planning Assistant
After=network.target

[Service]
Type=notify
User=mars-planner
WorkingDirectory=/opt/mars-mission-ai
Environment="PATH=/opt/mars-mission-ai/.venv/bin"
EnvironmentFile=/opt/mars-mission-ai/.env
ExecStart=/opt/mars-mission-ai/.venv/bin/gunicorn \
    -k uvicorn.workers.UvicornWorker \
    -w 4 \
    -b 127.0.0.1:8000 \
    --timeout 120 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    src.interfaces.web_interface:app

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable mars-planner
sudo systemctl start mars-planner
sudo systemctl status mars-planner
```

---

## Security Considerations

### API Keys
- **Never commit API keys** to version control
- Use environment variables or secret managers
- Rotate keys regularly
- Use separate keys for dev/staging/production

### Network Security
- Use HTTPS in production (Let's Encrypt for free SSL)
- Implement rate limiting
- Configure CORS appropriately
- Use firewall rules to restrict access

### Application Security
```python
# Add to web_interface.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["your-domain.com", "*.your-domain.com"]
)
```

---

## Monitoring and Logging

### Application Logging
```python
# Configure in src/interfaces/web_interface.py
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("mars_planner")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    'logs/mars_planner.log',
    maxBytes=10485760,  # 10MB
    backupCount=5
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
```

### Prometheus Metrics
```bash
pip install prometheus-fastapi-instrumentator

# In web_interface.py
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### Health Checks
Configure health check endpoint for load balancers:
```python
@app.get("/healthz")
async def healthz():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
```

---

## Scaling

### Horizontal Scaling
- Use load balancer (Nginx, HAProxy, AWS ALB)
- Deploy multiple app instances
- Share cache/data via NFS or object storage (S3)

### Vertical Scaling
- Increase CPU/memory for compute-intensive operations
- Optimize DEM processing with caching
- Use async operations for I/O

### Performance Optimization
```python
# Use Redis for caching
import redis
cache = redis.Redis(host='localhost', port=6379, db=0)

# Cache DEM slope computations
@app.post("/dem/compute_slope")
async def compute_slope(request: SlopeComputeRequest):
    cache_key = f"slope:{hash(request.dem_path)}"
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)
    # ... compute slope ...
    cache.setex(cache_key, 3600, json.dumps(result))
    return result
```

---

## Backup and Recovery

### Data Backup
```bash
# Backup DEM data and cache
tar -czf backup-$(date +%Y%m%d).tar.gz data/ models/

# Upload to S3
aws s3 cp backup-$(date +%Y%m%d).tar.gz s3://mars-planner-backups/
```

### Database Backup (if using PostgreSQL)
```bash
pg_dump -U mars_planner -h localhost mars_db > backup.sql
```

### Automated Backups
```bash
# Add to crontab
0 2 * * * /opt/mars-mission-ai/scripts/backup.sh
```

---

## Troubleshooting

### Common Issues

**GDAL Import Error**
```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install GDAL

# macOS
brew install gdal
pip install GDAL==$(gdal-config --version)
```

**Port Already in Use**
```bash
# Find process using port 8000
lsof -i :8000
# Kill process
kill -9 <PID>
```

**Memory Issues with Large DEMs**
- Increase container/instance memory
- Process DEMs in chunks
- Use lower resolution DEMs for testing

### Logs Location
- Application: `logs/mars_planner.log`
- Docker: `docker logs mars-planner`
- Systemd: `journalctl -u mars-planner -f`

---

## Maintenance

### Regular Tasks
- Update dependencies: `pip list --outdated`
- Clear old cache: `find data/cache -mtime +30 -delete`
- Rotate logs: Configured via RotatingFileHandler
- Monitor disk space: `df -h`

### Updates
```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart service
sudo systemctl restart mars-planner
```

---

## Support

For deployment issues:
1. Check logs first
2. Review this documentation
3. Consult GETTING_STARTED.md
4. Open GitHub issue with deployment details
