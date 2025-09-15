from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConstraintSolver:
    """Resource and constraint solver using an iterative-repair style API.

    This class is intentionally conservative and fast; you can swap in OR-Tools
    modeling here for tighter guarantees.
    """

    def evaluate(self, activities: List[Dict[str, Any]], constraints: Dict[str, Any]) -> Dict[str, Any]:
        violations: List[str] = []

        # Time budget check
        time_budget = float(constraints.get("time_budget_min", 480))
        total_time = 0.0
        for a in activities:
            total_time += float(a.get("duration_min", 0))
        if total_time > time_budget * 1.05:  # 5% tolerance
            violations.append(f"time_budget_exceeded: {total_time:.1f} > {time_budget:.1f} min")

        # Energy and peak power checks (approximate)
        power = constraints.get("power", {})
        mmrtg_output_w = float(power.get("mmrtg_output_w", 110))
        peak_demand_w = float(power.get("peak_demand_w", 900))
        # Peak instantaneous power must not exceed peak demand
        for a in activities:
            p = float(a.get("power_w", a.get("peak_power_w", 0)))
            if p > peak_demand_w:
                violations.append(f"peak_power_exceeded:{p}W>{peak_demand_w}W in {a.get('id')}")

        # Communication constraint placeholder: ensure at least one post_tx activity exists
        comm_windows = int(constraints.get("comm_windows_per_sol", constraints.get("communication_windows", 1)))
        if comm_windows >= 1:
            if not any(a.get("type") == "data_tx" for a in activities):
                violations.append("missing_data_transmission_activity")

        ok = len(violations) == 0
        return {"ok": ok, "violations": violations}

    def repair(self, activities: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Naive repair: ensure a data_tx exists if comm windows available
        comm_windows = int(constraints.get("comm_windows_per_sol", constraints.get("communication_windows", 1)))
        if comm_windows >= 1 and not any(a.get("type") == "data_tx" for a in activities):
            activities = activities + [{"id": "post_tx", "type": "data_tx", "duration_min": 10, "power_w": 180}]
        return activities
