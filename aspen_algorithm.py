from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .constraint_solver import ConstraintSolver
from .timeline_manager import TimelineManager
from .resource_optimizer import ResourceOptimizer
from .path_planner import PathPlanner
from .ortools_scheduler import OrToolsScheduler


@dataclass
class MissionPlanner:
    """ASPEN-inspired mission planner using iterative repair.

    - Generates activities from goals (via LLM or rule base)
    - Evaluates and repairs plans against constraints
    - Produces a scheduled, resource-feasible plan
    """

    iterative_repair_algorithm: bool = True
    constraint_satisfaction: ConstraintSolver = ConstraintSolver()
    resource_timeline: TimelineManager = TimelineManager()
    resource_optimizer: ResourceOptimizer = ResourceOptimizer()
    path_planner: PathPlanner = PathPlanner()
    ortools_scheduler: OrToolsScheduler = OrToolsScheduler()

    def generate_mission_plan(self, goals: List[str], constraints: Dict[str, Any], rover_state: Dict[str, Any]) -> Dict[str, Any]:
        # Seed activities from goals (high-level)
        activities: List[Dict[str, Any]] = [
            {"id": "pre_checks", "type": "system_checks", "duration_min": 10, "power_w": 150},
        ]

        # Optional: route planning if DEM and route provided
        route_info = None
        route = constraints.get("route", {})
        dem_ctx = constraints.get("dem", {})
        slope_deg = dem_ctx.get("slope_deg")  # expected np.ndarray when provided by caller
        raster_transform = dem_ctx.get("transform")
        if route and slope_deg is not None and raster_transform is not None:
            route_info = self.path_planner.plan_route_from_slope(
                slope_deg,
                raster_transform,
                route.get("start_lat"),
                route.get("start_lon"),
                route.get("goal_lat"),
                route.get("goal_lon"),
            )
            if route_info.get("success"):
                path_len = len(route_info.get("path_rc", []))
                est_distance_m = float(path_len) * float(constraints.get("dem", {}).get("pixel_size_m", 20.0))
                activities.append(
                    {
                        "id": "drive_segment",
                        "type": "drive",
                        "distance_m": est_distance_m,
                        "duration_min": max(10, int(est_distance_m / max(constraints.get("drive_speed_m_per_min", 8), 1))),
                        "power_w": 200,
                    }
                )

        # Science operations suggested by goals
        if any("sample" in g.lower() or "geolog" in g.lower() for g in goals):
            activities.append({"id": "science_op", "type": "imaging", "instrument": "Mastcam-Z", "duration_min": 20, "power_w": 250})

        activities.append({"id": "post_tx", "type": "data_tx", "duration_min": 10, "power_w": 180})

        # Evaluate and repair
        eval_res = self.constraint_satisfaction.evaluate(activities, constraints)
        if not eval_res.get("ok"):
            activities = self.constraint_satisfaction.repair(activities, constraints)

        # Power optimization
        power = constraints.get("power", {})
        mmrtg_output_w = float(power.get("mmrtg_output_w", 110))
        time_budget_min = float(constraints.get("time_budget_min", 480))
        battery_capacity_ah = power.get("battery_capacity_ah", 43)
        battery_count = power.get("battery_count", 2)
        activities_opt, power_stats = self.resource_optimizer.optimize_power_usage(
            activities, mmrtg_output_w, time_budget_min, battery_capacity_ah, battery_count
        )

        # Thermal management
        environmental = constraints.get("environmental", {})
        therm = constraints.get("thermal", {})
        activities_therm = self.resource_optimizer.thermal_management(
            activities_opt,
            environmental,
            float(therm.get("operating_temp_min_c", -40)),
            float(therm.get("operating_temp_max_c", 40)),
        )

        # Schedule using OR-Tools if available; fallback to naive timeline
        scheduled = []
        if self.ortools_scheduler.is_available():
            scheduled = self.ortools_scheduler.schedule(activities_therm, constraints)
        if not scheduled:
            scheduled = self.resource_timeline.schedule(activities_therm, constraints)
        return {
            "goals": goals,
            "constraints": constraints,
            "rover_state": rover_state,
            "route": route_info,
            "activities": scheduled,
            "power": power_stats,
            "violations": eval_res.get("violations", []),
        }
