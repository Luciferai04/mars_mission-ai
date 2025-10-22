#!/bin/bash
# Optional Kong security hardening: enable key-auth and rate limiting per service
# Usage: ./scripts/configure_gateway_auth.sh

set -euo pipefail
KONG_ADMIN_URL=${KONG_ADMIN_URL:-http://localhost:8001}
API_KEY=${API_KEY:-demo-key}

wait_kong() {
  echo "Waiting for Kong Admin API at $KONG_ADMIN_URL ..."
  for i in {1..60}; do
    if curl -sf "$KONG_ADMIN_URL" >/dev/null; then echo "Kong ready"; return 0; fi
    sleep 2
  done
  echo "Kong not reachable" >&2; exit 1
}

wait_kong

# Enable Prometheus metrics plugin (global)
curl -sf -X POST "$KONG_ADMIN_URL/plugins" \
  --data "name=prometheus" >/dev/null || true

# Create a consumer and key
curl -sf -X POST "$KONG_ADMIN_URL/consumers" --data "username=demo" >/dev/null || true
curl -sf -X POST "$KONG_ADMIN_URL/consumers/demo/key-auth" --data "key=${API_KEY}" >/dev/null || true

echo "Configured consumer 'demo' with apikey=${API_KEY}"

# Attach key-auth to services (uncomment to enforce globally)
for svc in vision-service marl-service data-service planning-service; do
  echo "Attaching key-auth to $svc"
  curl -sf -X POST "$KONG_ADMIN_URL/services/$svc/plugins" \
    --data "name=key-auth" \
    --data "config.key_names=apikey" \
    --data "config.hide_credentials=true" >/dev/null || true
  # Per-service rate limit baseline
  curl -sf -X POST "$KONG_ADMIN_URL/services/$svc/plugins" \
    --data "name=rate-limiting" \
    --data "config.minute=600" \
    --data "config.policy=local" >/dev/null || true
  echo "  -> key-auth and rate-limiting added"
done

echo "Kong auth/rate-limit configuration complete. Use header: apikey: ${API_KEY}"
