#!/usr/bin/env python3
"""
Autonomous experiment design heuristics.
"""
from __future__ import annotations

from typing import List, Dict, Any


class ExperimentDesigner:
    def propose_experiments(
        self, integrated_data: Dict[str, Any], k: int = 3
    ) -> List[Dict[str, Any]]:
        targets = integrated_data.get("science_targets", [])
        env = integrated_data.get("environment", {})
        proposed = []
        for t in sorted(targets, key=lambda x: x.get("priority", 0), reverse=True)[:k]:
            proposed.append(
                {
                    "id": f"EXP_{t.get('id', 'T')}",
                    "name": f"Investigate {t.get('name', 'Target')}",
                    "lat": t.get("lat"),
                    "lon": t.get("lon"),
                    "steps": ["approach", "image", "analyze", "sample"],
                    "estimated_time_min": 90,
                    "estimated_power_wh": 120.0,
                    "risk": "LOW" if env.get("dust_opacity", 0.3) < 0.5 else "MEDIUM",
                }
            )
        return proposed
