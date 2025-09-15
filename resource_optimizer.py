from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ResourceOptimizer:
    """Optimize power usage and manage thermal constraints.

    This mirrors NASA-style reasoning without simulating mission data.
    Methods accept abstracted activity dictionaries and constraint dicts.
    """

    safety_margin_pct: float = 0.2  # 20% margin on power/time

    def estimate_mmrtg_energy_wh(self, mmrtg_output_w: float, time_budget_min: float) -> float:
        """Energy available from MMRTG over a given time budget."""
        return mmrtg_output_w * (time_budget_min / 60.0)

    def estimate_total_budget_wh(
        self,
        mmrtg_output_w: float,
        time_budget_min: float,
        battery_capacity_ah: float | None = None,
        battery_count: int = 0,
        battery_bus_v: float = 28.0,
        use_battery: bool = True,
    ) -> float:
        base = self.estimate_mmrtg_energy_wh(mmrtg_output_w, time_budget_min)
        if use_battery and battery_capacity_ah and battery_count > 0:
            # Very conservative: allow 20% of nominal battery capacity per sol
            usable_ah = 0.2 * battery_capacity_ah * battery_count
            base += usable_ah * battery_bus_v
        # Apply safety margin reserve
        return base * (1.0 - self.safety_margin_pct)

    def estimate_activity_energy_wh(self, activity: Dict[str, Any]) -> float:
        """Estimate energy for an activity; uses fields if provided.

        Recognized fields: power_w, duration_min, energy_wh, peak_power_w.
        """
        if "energy_wh" in activity and isinstance(activity["energy_wh"], (int, float)):
            return float(activity["energy_wh"])
        power = float(activity.get("power_w") or activity.get("peak_power_w") or 100.0)
        duration = float(activity.get("duration_min", 15))
        return power * (duration / 60.0)

    def optimize_power_usage(
        self,
        activities: List[Dict[str, Any]],
        mmrtg_output_w: float,
        time_budget_min: float,
        battery_capacity_ah: float | None = None,
        battery_count: int = 0,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Conservatively prune/scale activities to fit within energy budget."""
        budget_wh = self.estimate_total_budget_wh(
            mmrtg_output_w, time_budget_min, battery_capacity_ah, battery_count
        )
        kept: List[Dict[str, Any]] = []
        total = 0.0
        for a in activities:
            e = self.estimate_activity_energy_wh(a)
            if total + e <= budget_wh:
                kept.append(a)
                total += e
            else:
                # Defer or downscale non-essential activities
                if a.get("type") in {"imaging", "sampling", "analysis"}:
                    scaled = dict(a)
                    scaled_duration = max(5.0, float(a.get("duration_min", 15)) * 0.5)
                    scaled["duration_min"] = scaled_duration
                    e_scaled = self.estimate_activity_energy_wh(scaled)
                    if total + e_scaled <= budget_wh:
                        kept.append(scaled)
                        total += e_scaled
                    else:
                        kept.append({**a, "status": "deferred"})
                else:
                    kept.append({**a, "status": "deferred"})
        return kept, {"budget_wh": budget_wh, "consumed_wh": total}

    def thermal_management(
        self,
        planned_activities: List[Dict[str, Any]],
        environmental: Dict[str, Any],
        operating_temp_min_c: float,
        operating_temp_max_c: float,
    ) -> List[Dict[str, Any]]:
        """Remove or flag activities outside thermal operating envelope."""
        temp = environmental.get("temperature_c")
        if temp is None:
            return planned_activities
        adjusted: List[Dict[str, Any]] = []
        for a in planned_activities:
            if operating_temp_min_c <= temp <= operating_temp_max_c:
                adjusted.append(a)
            else:
                adjusted.append({**a, "status": "await_thermal_ok"})
        return adjusted