#!/usr/bin/env python3
"""
Long-term strategic planner producing multi-sol to multi-month objectives.
"""
from __future__ import annotations

from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StrategicPlan:
    horizon_days: int
    objectives: list[dict]
    generated_at: str


class StrategicPlanner:
    def generate(
        self, mission_context: Dict[str, Any], horizon_days: int = 90
    ) -> StrategicPlan:
        # Very simple heuristic plan over horizon
        objectives = []
        start = datetime.utcnow()
        for week in range(max(1, horizon_days // 7)):
            objectives.append(
                {
                    "week": week + 1,
                    "focus": (
                        "systematic_survey"
                        if week % 2 == 0
                        else "targeted_investigation"
                    ),
                    "milestones": ["map_terrain", "collect_samples", "analyze_imagery"],
                }
            )
        return StrategicPlan(
            horizon_days=horizon_days,
            objectives=objectives,
            generated_at=start.isoformat(),
        )
