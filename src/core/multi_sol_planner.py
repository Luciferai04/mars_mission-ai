#!/usr/bin/env python3
"""
Multi-Sol Mission Planner

Plans rover operations across multiple Mars sols (days) with resource
continuity, route optimization, and science scheduling.
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import numpy as np


@dataclass
class SolPlan:
    """Plan for a single sol."""

    sol_number: int
    start_location: Tuple[float, float]  # (lat, lon)
    end_location: Tuple[float, float]
    waypoints: List[Tuple[float, float]]
    activities: List[Dict[str, Any]]

    # Resource projections
    power_budget_wh: float
    power_consumed_wh: float
    power_margin_wh: float

    thermal_min_c: float
    thermal_max_c: float

    drive_distance_m: float
    drive_time_hours: float

    # Science
    sample_targets: List[Dict[str, Any]]
    imaging_sessions: List[Dict[str, Any]]

    # Safety
    hazards_encountered: List[Dict[str, Any]]
    contingency_triggers: List[str]

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiSolMissionPlan:
    """Mission plan spanning multiple sols."""

    mission_name: str
    start_sol: int
    end_sol: int
    num_sols: int

    sol_plans: List[SolPlan]

    # Cumulative metrics
    total_distance_m: float
    total_power_consumed_wh: float
    total_samples_collected: int
    total_images_captured: int

    # Route optimization
    route_continuity_score: float  # 0-1
    backtracking_distance_m: float

    # Science value
    science_value_score: float  # 0-100
    high_priority_targets_visited: int

    # Safety
    overall_risk_level: str  # LOW, MEDIUM, HIGH
    contingency_plans: List[Dict[str, Any]]

    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiSolPlanner:
    """Plans rover operations across multiple Mars sols."""

    def __init__(
        self,
        dem_data: Optional[np.ndarray] = None,
        dem_bounds: Optional[Tuple[float, float, float, float]] = None,
    ):
        """
        Args:
            dem_data: Digital elevation model array
            dem_bounds: (min_lat, max_lat, min_lon, max_lon)
        """
        self.dem_data = dem_data
        self.dem_bounds = dem_bounds
        self.logger = logging.getLogger(__name__)

        # NASA operational constraints
        self.max_drive_per_sol_m = 150.0  # Conservative limit
        self.max_power_per_sol_wh = 900.0  # Battery capacity
        self.min_power_reserve_wh = 200.0  # Safety margin

        self.max_slope_deg = 30.0
        self.caution_slope_deg = 15.0

        # Activity costs
        self.power_costs_wh = {
            "drive_per_m": 5.0,
            "sample": 50.0,
            "imaging": 10.0,
            "arm_deployment": 30.0,
            "overnight_survival": 150.0,
            "communication": 20.0,
        }

        self.time_costs_hours = {
            "drive_per_m": 0.01,  # ~100m/hour
            "sample": 2.0,
            "imaging": 0.25,
            "arm_deployment": 1.0,
        }

    def plan_multi_sol_mission(
        self,
        start_location: Tuple[float, float],
        target_locations: List[Dict[str, Any]],
        num_sols: int,
        mission_objectives: Dict[str, Any],
        environmental_forecast: Dict[int, Dict[str, Any]],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> MultiSolMissionPlan:
        """Plan mission across multiple sols.

        Args:
            start_location: Initial rover position (lat, lon)
            target_locations: List of target dicts with location, priority, type
            num_sols: Number of sols to plan
            mission_objectives: Science goals, sampling targets, exploration areas
            environmental_forecast: Per-sol weather predictions {sol: {temp, dust, wind}}
            constraints: Additional mission constraints

        Returns:
            Complete multi-sol mission plan
        """
        self.logger.info(f"Planning {num_sols}-sol mission from {start_location}")

        constraints = constraints or {}

        # Optimize route across all sols
        route_plan = self._optimize_multi_sol_route(
            start_location, target_locations, num_sols, constraints
        )

        # Allocate activities to sols
        sol_plans = []
        current_location = start_location

        for sol_idx in range(num_sols):
            sol_number = constraints.get("start_sol", 0) + sol_idx

            # Get route segment for this sol
            sol_route = route_plan["sol_routes"][sol_idx]

            # Get environmental conditions
            env = environmental_forecast.get(sol_number, {})

            # Plan activities for this sol
            sol_plan = self._plan_single_sol(
                sol_number=sol_number,
                start_location=current_location,
                route_segment=sol_route,
                targets_in_range=sol_route["targets"],
                environment=env,
                objectives=mission_objectives,
                cumulative_state=self._get_cumulative_state(sol_plans),
            )

            sol_plans.append(sol_plan)
            current_location = sol_plan.end_location

        # Build complete mission plan
        mission_plan = self._build_mission_plan(
            sol_plans, mission_objectives, constraints
        )

        # Generate contingency plans
        mission_plan.contingency_plans = self._generate_contingency_plans(
            mission_plan, environmental_forecast
        )

        return mission_plan

    def _optimize_multi_sol_route(
        self,
        start: Tuple[float, float],
        targets: List[Dict[str, Any]],
        num_sols: int,
        constraints: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Optimize route across multiple sols minimizing backtracking."""

        # Sort targets by priority and spatial clustering
        sorted_targets = self._prioritize_and_cluster_targets(start, targets)

        # Divide targets across sols
        sol_routes = []
        current_pos = start
        remaining_targets = sorted_targets.copy()

        for sol_idx in range(num_sols):
            # Determine how many targets can be reached this sol
            sol_route = {
                "sol": sol_idx,
                "start": current_pos,
                "waypoints": [],
                "targets": [],
                "end": current_pos,
                "distance_m": 0.0,
            }

            distance_budget = self.max_drive_per_sol_m

            while remaining_targets and distance_budget > 10.0:
                # Find nearest high-priority target
                next_target = self._find_next_optimal_target(
                    current_pos, remaining_targets, distance_budget
                )

                if not next_target:
                    break

                # Add to route
                target_loc = (next_target["lat"], next_target["lon"])
                distance = self._calculate_distance(current_pos, target_loc)

                sol_route["waypoints"].append(target_loc)
                sol_route["targets"].append(next_target)
                sol_route["distance_m"] += distance

                current_pos = target_loc
                distance_budget -= distance
                remaining_targets.remove(next_target)

            sol_route["end"] = current_pos
            sol_routes.append(sol_route)

        return {
            "sol_routes": sol_routes,
            "unvisited_targets": remaining_targets,
            "total_distance_m": sum(r["distance_m"] for r in sol_routes),
        }

    def _prioritize_and_cluster_targets(
        self, start: Tuple[float, float], targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Sort targets by priority and spatial proximity."""

        # Calculate distance from start
        for target in targets:
            target["_distance_from_start"] = self._calculate_distance(
                start, (target["lat"], target["lon"])
            )

        # Multi-criteria sort: priority first, then proximity
        priority_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

        sorted_targets = sorted(
            targets,
            key=lambda t: (
                -priority_map.get(t.get("priority", "MEDIUM"), 2),
                t["_distance_from_start"],
            ),
        )

        return sorted_targets

    def _find_next_optimal_target(
        self,
        current: Tuple[float, float],
        candidates: List[Dict[str, Any]],
        distance_budget: float,
    ) -> Optional[Dict[str, Any]]:
        """Find next target optimizing priority and distance."""

        reachable = []

        for target in candidates:
            target_loc = (target["lat"], target["lon"])
            dist = self._calculate_distance(current, target_loc)

            if dist <= distance_budget:
                priority_score = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(
                    target.get("priority", "MEDIUM"), 2
                )

                # Score combines priority and proximity
                score = priority_score * 100 - dist

                reachable.append((score, target))

        if not reachable:
            return None

        # Return highest score
        reachable.sort(key=lambda x: x[0], reverse=True)
        return reachable[0][1]

    def _plan_single_sol(
        self,
        sol_number: int,
        start_location: Tuple[float, float],
        route_segment: Dict[str, Any],
        targets_in_range: List[Dict[str, Any]],
        environment: Dict[str, Any],
        objectives: Dict[str, Any],
        cumulative_state: Dict[str, Any],
    ) -> SolPlan:
        """Plan activities for a single sol."""

        # Initialize power budget
        power_available = self.max_power_per_sol_wh
        power_used = 0.0

        # Account for overnight survival
        power_used += self.power_costs_wh["overnight_survival"]

        # Account for communication
        power_used += self.power_costs_wh["communication"]

        # Drive activities
        drive_distance = route_segment["distance_m"]
        drive_power = drive_distance * self.power_costs_wh["drive_per_m"]
        drive_time = drive_distance * self.time_costs_hours["drive_per_m"]

        power_used += drive_power

        # Science activities at targets
        activities = []
        sample_targets = []
        imaging_sessions = []

        for target in targets_in_range:
            # Determine activities based on target type
            target_activities = self._plan_target_activities(
                target, objectives, power_available - power_used
            )

            for activity in target_activities:
                power_used += activity["power_wh"]
                activities.append(activity)

                if activity["type"] == "sample":
                    sample_targets.append(target)
                elif activity["type"] == "imaging":
                    imaging_sessions.append(activity)

        # Safety checks
        power_margin = power_available - power_used

        contingency_triggers = []
        if power_margin < self.min_power_reserve_wh:
            contingency_triggers.append("LOW_POWER_MARGIN")

        if environment.get("temperature_c", -60) < -90:
            contingency_triggers.append("EXTREME_COLD")

        if environment.get("dust_opacity", 0.5) > 2.0:
            contingency_triggers.append("DUST_STORM")

        return SolPlan(
            sol_number=sol_number,
            start_location=start_location,
            end_location=route_segment["end"],
            waypoints=route_segment["waypoints"],
            activities=activities,
            power_budget_wh=power_available,
            power_consumed_wh=power_used,
            power_margin_wh=power_margin,
            thermal_min_c=environment.get("temperature_c", -60) - 10,
            thermal_max_c=environment.get("temperature_c", -60) + 20,
            drive_distance_m=drive_distance,
            drive_time_hours=drive_time,
            sample_targets=sample_targets,
            imaging_sessions=imaging_sessions,
            hazards_encountered=[],
            contingency_triggers=contingency_triggers,
            metadata={
                "environment": environment,
                "planned_at": datetime.utcnow().isoformat(),
            },
        )

    def _plan_target_activities(
        self, target: Dict[str, Any], objectives: Dict[str, Any], power_budget: float
    ) -> List[Dict[str, Any]]:
        """Plan activities for a specific target."""

        activities = []

        target_type = target.get("type", "observation")
        priority = target.get("priority", "MEDIUM")

        # Always image
        if power_budget >= self.power_costs_wh["imaging"]:
            activities.append(
                {
                    "type": "imaging",
                    "target": target,
                    "power_wh": self.power_costs_wh["imaging"],
                    "duration_hours": self.time_costs_hours["imaging"],
                }
            )
            power_budget -= self.power_costs_wh["imaging"]

        # Sample high-priority targets
        if priority == "HIGH" and target_type in ["rock", "sample_site", "geological"]:
            sample_power = (
                self.power_costs_wh["arm_deployment"] + self.power_costs_wh["sample"]
            )

            if power_budget >= sample_power:
                activities.append(
                    {
                        "type": "sample",
                        "target": target,
                        "power_wh": sample_power,
                        "duration_hours": (
                            self.time_costs_hours["arm_deployment"]
                            + self.time_costs_hours["sample"]
                        ),
                    }
                )

        return activities

    def _get_cumulative_state(self, sol_plans: List[SolPlan]) -> Dict[str, Any]:
        """Get cumulative state from completed sol plans."""

        if not sol_plans:
            return {
                "total_distance_m": 0.0,
                "total_power_used_wh": 0.0,
                "samples_collected": 0,
                "images_captured": 0,
            }

        return {
            "total_distance_m": sum(p.drive_distance_m for p in sol_plans),
            "total_power_used_wh": sum(p.power_consumed_wh for p in sol_plans),
            "samples_collected": sum(len(p.sample_targets) for p in sol_plans),
            "images_captured": sum(len(p.imaging_sessions) for p in sol_plans),
        }

    def _build_mission_plan(
        self,
        sol_plans: List[SolPlan],
        objectives: Dict[str, Any],
        constraints: Dict[str, Any],
    ) -> MultiSolMissionPlan:
        """Build complete mission plan from sol plans."""

        cumulative = self._get_cumulative_state(sol_plans)

        # Calculate route metrics
        backtracking = self._calculate_backtracking(sol_plans)
        continuity_score = 1.0 - (
            backtracking / max(cumulative["total_distance_m"], 1.0)
        )

        # Science value
        science_score = self._calculate_science_value(sol_plans, objectives)
        high_pri_visited = sum(
            len([t for t in p.sample_targets if t.get("priority") == "HIGH"])
            for p in sol_plans
        )

        # Risk assessment
        risk_level = self._assess_overall_risk(sol_plans)

        return MultiSolMissionPlan(
            mission_name=constraints.get(
                "mission_name", f'Mission_{datetime.utcnow().strftime("%Y%m%d")}'
            ),
            start_sol=sol_plans[0].sol_number if sol_plans else 0,
            end_sol=sol_plans[-1].sol_number if sol_plans else 0,
            num_sols=len(sol_plans),
            sol_plans=sol_plans,
            total_distance_m=cumulative["total_distance_m"],
            total_power_consumed_wh=cumulative["total_power_used_wh"],
            total_samples_collected=cumulative["samples_collected"],
            total_images_captured=cumulative["images_captured"],
            route_continuity_score=continuity_score,
            backtracking_distance_m=backtracking,
            science_value_score=science_score,
            high_priority_targets_visited=high_pri_visited,
            overall_risk_level=risk_level,
            contingency_plans=[],
            metadata={
                "planned_at": datetime.utcnow().isoformat(),
                "objectives": objectives,
                "constraints": constraints,
            },
        )

    def _calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Calculate distance between two positions (simplified)."""
        # Haversine or simple Euclidean for small distances
        lat_diff = abs(pos1[0] - pos2[0])
        lon_diff = abs(pos1[1] - pos2[1])

        # Rough approximation: 1 degree ≈ 59 km on Mars at equator
        return ((lat_diff**2 + lon_diff**2) ** 0.5) * 59000.0

    def _calculate_backtracking(self, sol_plans: List[SolPlan]) -> float:
        """Calculate total backtracking distance."""
        # Simplified: assume minimal backtracking with good planning
        return 0.0

    def _calculate_science_value(
        self, sol_plans: List[SolPlan], objectives: Dict[str, Any]
    ) -> float:
        """Calculate overall science value score."""

        total_samples = sum(len(p.sample_targets) for p in sol_plans)
        total_images = sum(len(p.imaging_sessions) for p in sol_plans)

        # Simple scoring
        sample_value = total_samples * 20.0
        image_value = total_images * 5.0

        return min(sample_value + image_value, 100.0)

    def _assess_overall_risk(self, sol_plans: List[SolPlan]) -> str:
        """Assess overall mission risk level."""

        contingencies = sum(len(p.contingency_triggers) for p in sol_plans)

        if contingencies >= len(sol_plans):
            return "HIGH"
        elif contingencies > 0:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_contingency_plans(
        self, mission_plan: MultiSolMissionPlan, env_forecast: Dict[int, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate contingency plans for expected hazards."""

        contingencies = []

        # Dust storm contingency
        contingencies.append(
            {
                "trigger": "DUST_STORM",
                "condition": "Dust opacity > 2.0",
                "action": "Enter safe mode, minimize power usage, wait for clearing",
                "fallback_sols": 1,
                "power_reserve_required_wh": 300.0,
            }
        )

        # Low power contingency
        contingencies.append(
            {
                "trigger": "LOW_POWER",
                "condition": "Available power < 200 Wh",
                "action": "Cancel science activities, optimize solar array positioning",
                "fallback_sols": 0,
                "power_reserve_required_wh": self.min_power_reserve_wh,
            }
        )

        # Equipment failure
        contingencies.append(
            {
                "trigger": "EQUIPMENT_FAILURE",
                "condition": "Critical subsystem fault",
                "action": "Safe rover, activate redundant systems, contact mission control",
                "fallback_sols": 2,
                "power_reserve_required_wh": 400.0,
            }
        )

        return contingencies
