import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  vus: 5,
  duration: '30s',
};

const BASE = __ENV.BASE_URL || 'http://localhost:8005';

export default function () {
  const payload = JSON.stringify({
    sol: 1600,
    lat: 18.4447,
    lon: 77.4508,
    battery_soc: 0.65,
    time_budget_min: 480,
    objectives: ['traverse', 'image', 'sample'],
  });
  const res = http.post(`${BASE}/plan`, payload, { headers: { 'Content-Type': 'application/json' } });
  check(res, {
    'status is 200': (r) => r.status === 200,
    'has plan_id': (r) => r.body && r.body.includes('plan_id'),
  });
  sleep(1);
}
