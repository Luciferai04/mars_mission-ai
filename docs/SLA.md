# Service Level Objectives (SLOs) and Operational Playbook

## SLOs
- Availability: 99.5% monthly for Planning, MARL, Data, Vision
- Latency (P95):
  - MARL optimize < 120 ms
  - Data integrated < 450 ms
  - Planning /plan < 1800 ms
  - Vision classify_hazard < 350 ms (CPU), < 150 ms (GPU)
- Error rate: < 1% 5xx per service per day

## Monitoring
- Health endpoints: /health for each service (compose/CI smoke tests)
- Logs: stdout/stderr; optional ELK/Grafana (see README)
- Metrics (optional): add Prometheus exporters per service

## Alerts (examples)
- Service unavailable > 2 min
- P95 latency above SLO for 10 min
- 5xx rate > 1% for 10 min

## Incident Response
1. Identify impacted service via /services/status
2. Inspect service logs: `docker compose logs <service>`
3. Roll restart: `docker compose restart <service>`
4. If model/weights issue: reload agents (`/agents/reload`) or vision `/reload_model`
5. If data source issue: set USE_REAL_NASA=false or rotate keys

## Runbooks
- MARL model refresh: train with scripts/train_marl.py, then POST /agents/reload
- Vision model update: mount new .pth at /models and POST /reload_model
- Data integration switch: set USE_REAL_NASA=true and NASA_API_KEY, redeploy

## Security
- Store keys as environment secrets (never commit)
- Enable Kong rate limits and auth if exposed publicly
- Run `bandit -r src/` locally; Trivy scans in CI for images
