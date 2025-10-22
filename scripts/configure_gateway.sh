#!/bin/bash
# Configure Kong API Gateway for Mars Mission Services

KONG_ADMIN_URL="http://localhost:8001"

echo "Configuring Kong API Gateway..."

# Wait for Kong to be ready
echo "Waiting for Kong to start..."
until curl -s "${KONG_ADMIN_URL}" > /dev/null; do
    echo "Kong not ready yet, waiting..."
    sleep 2
done
echo "Kong is ready!"

# Configure Vision Service
echo "Configuring Vision Service..."
curl -i -X POST ${KONG_ADMIN_URL}/services/ \
  --data name=vision-service \
  --data url='http://vision-service:8002'

curl -i -X POST ${KONG_ADMIN_URL}/services/vision-service/routes \
  --data 'paths[]=/vision' \
  --data 'strip_path=true'

# Configure MARL Service
echo "Configuring MARL Service..."
curl -i -X POST ${KONG_ADMIN_URL}/services/ \
  --data name=marl-service \
  --data url='http://marl-service:8003'

curl -i -X POST ${KONG_ADMIN_URL}/services/marl-service/routes \
  --data 'paths[]=/marl' \
  --data 'strip_path=true'

# Configure Data Integration Service
echo "Configuring Data Integration Service..."
curl -i -X POST ${KONG_ADMIN_URL}/services/ \
  --data name=data-service \
  --data url='http://data-integration-service:8004'

curl -i -X POST ${KONG_ADMIN_URL}/services/data-service/routes \
  --data 'paths[]=/data' \
  --data 'strip_path=true'

# Configure Planning Service
echo "Configuring Planning Service..."
curl -i -X POST ${KONG_ADMIN_URL}/services/ \
  --data name=planning-service \
  --data url='http://planning-service:8005'

curl -i -X POST ${KONG_ADMIN_URL}/services/planning-service/routes \
  --data 'paths[]=/planning' \
  --data 'strip_path=true'

# Enable rate limiting (optional)
echo "Enabling rate limiting..."
curl -i -X POST ${KONG_ADMIN_URL}/plugins/ \
  --data "name=rate-limiting" \
  --data "config.minute=100" \
  --data "config.policy=local"

# Enable request logging (optional)
echo "Enabling request logging..."
curl -i -X POST ${KONG_ADMIN_URL}/plugins/ \
  --data "name=file-log" \
  --data "config.path=/tmp/kong_requests.log"

echo "Kong configuration complete!"
echo "Services available at:"
echo "  Vision: http://localhost:8000/vision"
echo "  MARL: http://localhost:8000/marl"
echo "  Data: http://localhost:8000/data"
echo "  Planning: http://localhost:8000/planning"
