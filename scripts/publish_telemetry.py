#!/usr/bin/env python3
"""
Publish synthetic (or bridged) telemetry to the Data Integration service.

Usage:
  python scripts/publish_telemetry.py --url http://localhost:8004 --rate 2

Optionally pipe real sources here and adapt the payload mapping.
"""
import time
import argparse
import requests
import random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8004", help="Base URL of data-integration service")
    p.add_argument("--rate", type=float, default=1.0, help="Messages per second")
    args = p.parse_args()

    interval = 1.0 / max(args.rate, 0.1)
    endpoint = f"{args.url.rstrip('/')}/publish/telemetry"

    try:
        while True:
            payload = {
                "battery_soc": round(random.uniform(40, 95), 1),
                "power_generation": round(random.uniform(80, 180), 1),
                "power_consumption": round(random.uniform(60, 140), 1),
                "temp_c": round(random.uniform(-80, -10), 1),
                "dust_opacity": round(random.uniform(0.2, 0.8), 2),
            }
            r = requests.post(endpoint, json=payload, timeout=5)
            r.raise_for_status()
            time.sleep(interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
