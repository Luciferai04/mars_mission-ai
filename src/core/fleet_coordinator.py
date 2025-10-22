#!/usr/bin/env python3
"""
Multi-rover coordination utilities.

Provides a simple coordinator to generate joint actions for a fleet of rovers
using the existing MARL system, with conflict resolution for hazards and power.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .multi_agent_rl import MultiAgentRLSystem, AgentState, Action


@dataclass
class RoverContext:
    rover_id: str
    state: AgentState
    priority: float = 1.0  # higher = more important


class MultiRoverCoordinator:
    def __init__(self, marl_system: Optional[MultiAgentRLSystem] = None) -> None:
        self.marl = marl_system or MultiAgentRLSystem()

    def coordinate_fleet(self, rovers: List[RoverContext]) -> Dict[str, Action]:
        # Sort by priority (desc)
        rovers_sorted = sorted(rovers, key=lambda r: r.priority, reverse=True)
        assigned: Dict[str, Action] = {}

        # Naive conflict resolution: avoid multiple rovers targeting same cell
        occupied_targets = set()

        for rc in rovers_sorted:
            action = self.marl.coordinate_action(rc.state, training=False)
            # Resolve target conflicts by adjusting to 'wait' if conflict
            if action.target and action.target in occupied_targets:
                action = Action(action_type="wait", duration=10, power_required=0.0)
            if action.target:
                occupied_targets.add(action.target)
            assigned[rc.rover_id] = action

        return assigned
