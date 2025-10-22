#!/usr/bin/env python3
"""
Multi-Agent Reinforcement Learning for Mars Mission Planning

Implements cooperative multi-agent RL with specialized agents for:
- Route planning optimization
- Power management
- Science target selection
- Hazard avoidance
- Long-term strategy

Each agent learns from mission outcomes and shares knowledge to improve
collective decision-making over time.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import random
from pathlib import Path
import logging
import os
import math

# Optional DQN support
try:  # pragma: no cover - optional import
    from .dqn_agent import DQNAgent
except Exception:  # pragma: no cover
    DQNAgent = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """State representation for RL agents"""

    position: Tuple[float, float]
    battery_soc: float  # State of charge (0-1)
    time_remaining: int  # Minutes
    hazard_map: np.ndarray
    science_targets: List[Dict]
    current_sol: int
    temperature: float
    dust_opacity: float
    completed_objectives: List[str] = field(default_factory=list)

    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector for neural network"""
        return np.array(
            [
                self.position[0],
                self.position[1],
                self.battery_soc,
                self.time_remaining / 1440.0,  # Normalize to sols
                self.current_sol / 100.0,  # Normalize
                self.temperature / 100.0,  # Normalize
                self.dust_opacity,
                len(self.science_targets),
                len(self.completed_objectives),
            ]
        )


@dataclass
class Action:
    """Action representation for agents"""

    action_type: str  # 'drive', 'sample', 'image', 'wait', 'recharge'
    target: Optional[Tuple[float, float]] = None
    duration: int = 0
    power_required: float = 0.0
    expected_reward: float = 0.0


@dataclass
class Experience:
    """Experience tuple for replay buffer"""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for training"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class BaseAgent:
    """Base RL agent with Q-learning"""

    def __init__(
        self,
        agent_id: str,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table (simplified - in production use neural network)
        self.q_table = {}
        self.replay_buffer = ReplayBuffer()

        # Training metrics
        self.episode_rewards = []
        self.episode_count = 0

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions in given state"""
        state_key = self._state_to_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        return self.q_table[state_key]

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            q_values = self.get_q_values(state)
            return int(np.argmax(q_values))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Update Q-values using TD learning"""
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        # Initialize Q-tables if needed
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)

        # Q-learning update
        current_q = self.q_table[state_key][action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state_key])

        # Update Q-value
        self.q_table[state_key][action] += self.lr * (target_q - current_q)

        # Store experience
        experience = Experience(state, action, reward, next_state, done)
        self.replay_buffer.push(experience)

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state vector to hashable key"""
        return tuple(np.round(state, 2))

    def save(self, path: str):
        """Save agent's learned policy"""
        data = {
            "agent_id": self.agent_id,
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load agent's learned policy"""
        with open(path, "r") as f:
            data = json.load(f)
        self.q_table = {eval(k): np.array(v) for k, v in data["q_table"].items()}
        self.epsilon = data["epsilon"]
        self.episode_rewards = data["episode_rewards"]


class RouteAgent(BaseAgent):
    """Specialized agent for route planning"""

    def __init__(self):
        super().__init__(
            agent_id="route_planner",
            state_dim=9,
            action_dim=8,  # 8 directional moves
            learning_rate=0.001,
            gamma=0.95,
        )
        self.action_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    def compute_reward(
        self, state: AgentState, action: Action, next_state: AgentState
    ) -> float:
        """Reward function for route planning"""
        reward = 0.0

        # Positive reward for reaching science targets
        if len(next_state.completed_objectives) > len(state.completed_objectives):
            reward += 50.0

        # Penalty for hazardous terrain
        if action.action_type == "drive":
            # Assume hazard_map values: 0=safe, 1=caution, 2=hazard
            pos_x = int(next_state.position[0]) % state.hazard_map.shape[0]
            pos_y = int(next_state.position[1]) % state.hazard_map.shape[1]
            hazard_level = state.hazard_map[pos_x, pos_y]
            reward -= hazard_level * 10.0

        # Penalty for power consumption
        reward -= action.power_required * 0.1

        # Bonus for efficient movement toward targets
        if len(state.science_targets) > 0:
            target = state.science_targets[0]
            old_dist = np.sqrt(
                (state.position[0] - target["lat"]) ** 2
                + (state.position[1] - target["lon"]) ** 2
            )
            new_dist = np.sqrt(
                (next_state.position[0] - target["lat"]) ** 2
                + (next_state.position[1] - target["lon"]) ** 2
            )
            if new_dist < old_dist:
                reward += 5.0

        return reward


class PowerAgent(BaseAgent):
    """Specialized agent for power management"""

    def __init__(self):
        super().__init__(
            agent_id="power_manager",
            state_dim=9,
            action_dim=5,  # power modes
            learning_rate=0.001,
            gamma=0.98,
        )
        self.action_names = [
            "full_power",
            "nominal",
            "conservative",
            "minimal",
            "recharge",
        ]

    def compute_reward(
        self, state: AgentState, action: Action, next_state: AgentState
    ) -> float:
        """Reward function for power management"""
        reward = 0.0

        # Large penalty for running out of power
        if next_state.battery_soc < 0.1:
            reward -= 100.0

        # Reward for maintaining healthy battery level
        if 0.3 <= next_state.battery_soc <= 0.8:
            reward += 10.0

        # Penalty for over-conservative power use (missed opportunities)
        if next_state.battery_soc > 0.9 and len(state.science_targets) > 0:
            reward -= 5.0

        # Bonus for completing objectives with good power management
        if (
            len(next_state.completed_objectives) > len(state.completed_objectives)
            and next_state.battery_soc > 0.2
        ):
            reward += 30.0

        return reward


class ScienceAgent(BaseAgent):
    """Specialized agent for science target selection"""

    def __init__(self):
        super().__init__(
            agent_id="science_planner",
            state_dim=9,
            action_dim=10,  # select from top 10 targets
            learning_rate=0.001,
            gamma=0.95,
        )

    def compute_reward(
        self, state: AgentState, action: Action, next_state: AgentState
    ) -> float:
        """Reward function for science planning"""
        reward = 0.0

        # Major reward for completing high-priority science
        if len(next_state.completed_objectives) > len(state.completed_objectives):
            reward += 100.0

        # Bonus for sample collection
        if action.action_type == "sample":
            reward += 50.0

        # Consider diversity of targets
        target_types = set(
            [obj.split("_")[0] for obj in next_state.completed_objectives]
        )
        reward += len(target_types) * 10.0

        # Time efficiency bonus
        if action.duration < 30:  # Quick operations
            reward += 5.0

        return reward


class HazardAgent(BaseAgent):
    """Specialized agent for hazard avoidance"""

    def __init__(self):
        super().__init__(
            agent_id="hazard_avoider",
            state_dim=9,
            action_dim=4,  # risk levels
            learning_rate=0.001,
            gamma=0.99,
        )
        self.action_names = ["avoid_all", "conservative", "balanced", "aggressive"]

    def compute_reward(
        self, state: AgentState, action: Action, next_state: AgentState
    ) -> float:
        """Reward function for hazard avoidance"""
        reward = 0.0

        # Severe penalty for entering hazardous terrain
        if hasattr(state, "hazard_map"):
            pos_x = int(next_state.position[0]) % state.hazard_map.shape[0]
            pos_y = int(next_state.position[1]) % state.hazard_map.shape[1]
            hazard = state.hazard_map[pos_x, pos_y]
            if hazard >= 2:  # HAZARD level
                reward -= 200.0
            elif hazard >= 1:  # CAUTION level
                reward -= 50.0

        # Reward for safe operation
        if next_state.battery_soc > 0.2 and len(next_state.completed_objectives) > 0:
            reward += 20.0

        return reward


class StrategyAgent(BaseAgent):
    """Specialized agent for long-term strategic planning"""

    def __init__(self):
        super().__init__(
            agent_id="strategist",
            state_dim=9,
            action_dim=6,  # strategy types
            learning_rate=0.0005,
            gamma=0.99,
        )
        self.action_names = [
            "aggressive_exploration",
            "systematic_survey",
            "targeted_investigation",
            "sample_collection",
            "conservative_ops",
            "opportunistic",
        ]

    def compute_reward(
        self, state: AgentState, action: Action, next_state: AgentState
    ) -> float:
        """Reward function for strategic planning"""
        reward = 0.0

        # Long-term progress reward
        completion_rate = len(next_state.completed_objectives) / max(
            len(state.science_targets), 1
        )
        reward += completion_rate * 100.0

        # Bonus for efficient multi-sol planning
        if next_state.current_sol > state.current_sol:
            objectives_per_sol = (
                len(next_state.completed_objectives) / next_state.current_sol
            )
            reward += objectives_per_sol * 50.0

        # Penalty for inefficient operations
        if next_state.time_remaining < 60 and len(next_state.science_targets) > 5:
            reward -= 30.0

        return reward


# ---------------------------
# DQN-backed agent wrappers
# ---------------------------
if DQNAgent is not None:

    class DeepRouteAgent(DQNAgent):
        def __init__(self):
            super().__init__(
                agent_id="route_planner_dqn",
                state_dim=9,
                action_dim=8,
                reward_fn=self.compute_reward,
            )
            self.action_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

        def compute_reward(
            self, state: AgentState, action: Action, next_state: AgentState
        ) -> float:
            reward = 0.0
            if len(next_state.completed_objectives) > len(state.completed_objectives):
                reward += 50.0
            if action.action_type == "drive":
                pos_x = int(next_state.position[0]) % state.hazard_map.shape[0]
                pos_y = int(next_state.position[1]) % state.hazard_map.shape[1]
                hazard_level = state.hazard_map[pos_x, pos_y]
                reward -= hazard_level * 10.0
            reward -= action.power_required * 0.1
            if len(state.science_targets) > 0:
                target = state.science_targets[0]
                old_dist = np.sqrt(
                    (state.position[0] - target["lat"]) ** 2
                    + (state.position[1] - target["lon"]) ** 2
                )
                new_dist = np.sqrt(
                    (next_state.position[0] - target["lat"]) ** 2
                    + (next_state.position[1] - target["lon"]) ** 2
                )
                if new_dist < old_dist:
                    reward += 5.0
            return reward

    class DeepPowerAgent(DQNAgent):
        def __init__(self):
            super().__init__(
                agent_id="power_manager_dqn",
                state_dim=9,
                action_dim=5,
                reward_fn=self.compute_reward,
            )
            self.action_names = [
                "full_power",
                "nominal",
                "conservative",
                "minimal",
                "recharge",
            ]

        def compute_reward(
            self, state: AgentState, action: Action, next_state: AgentState
        ) -> float:
            reward = 0.0
            if next_state.battery_soc < 0.1:
                reward -= 100.0
            if 0.3 <= next_state.battery_soc <= 0.8:
                reward += 10.0
            if next_state.battery_soc > 0.9 and len(state.science_targets) > 0:
                reward -= 5.0
            if (
                len(next_state.completed_objectives) > len(state.completed_objectives)
                and next_state.battery_soc > 0.2
            ):
                reward += 30.0
            return reward

    class DeepScienceAgent(DQNAgent):
        def __init__(self):
            super().__init__(
                agent_id="science_planner_dqn",
                state_dim=9,
                action_dim=10,
                reward_fn=self.compute_reward,
            )

        def compute_reward(
            self, state: AgentState, action: Action, next_state: AgentState
        ) -> float:
            reward = 0.0
            if len(next_state.completed_objectives) > len(state.completed_objectives):
                reward += 100.0
            if action.action_type == "sample":
                reward += 50.0
            target_types = set(
                [obj.split("_")[0] for obj in next_state.completed_objectives]
            )
            reward += len(target_types) * 10.0
            if action.duration < 30:
                reward += 5.0
            return reward

    class DeepHazardAgent(DQNAgent):
        def __init__(self):
            super().__init__(
                agent_id="hazard_avoider_dqn",
                state_dim=9,
                action_dim=4,
                reward_fn=self.compute_reward,
            )
            self.action_names = ["avoid_all", "conservative", "balanced", "aggressive"]

        def compute_reward(
            self, state: AgentState, action: Action, next_state: AgentState
        ) -> float:
            reward = 0.0
            if hasattr(state, "hazard_map"):
                pos_x = int(next_state.position[0]) % state.hazard_map.shape[0]
                pos_y = int(next_state.position[1]) % state.hazard_map.shape[1]
                hazard = state.hazard_map[pos_x, pos_y]
                if hazard >= 2:
                    reward -= 200.0
                elif hazard >= 1:
                    reward -= 50.0
            if (
                next_state.battery_soc > 0.2
                and len(next_state.completed_objectives) > 0
            ):
                reward += 20.0
            return reward

    class DeepStrategyAgent(DQNAgent):
        def __init__(self):
            super().__init__(
                agent_id="strategist_dqn",
                state_dim=9,
                action_dim=6,
                reward_fn=self.compute_reward,
            )
            self.action_names = [
                "aggressive_exploration",
                "systematic_survey",
                "targeted_investigation",
                "sample_collection",
                "conservative_ops",
                "opportunistic",
            ]

        def compute_reward(
            self, state: AgentState, action: Action, next_state: AgentState
        ) -> float:
            reward = 0.0
            completion_rate = len(next_state.completed_objectives) / max(
                len(state.science_targets), 1
            )
            reward += completion_rate * 100.0
            if next_state.current_sol > state.current_sol:
                objectives_per_sol = (
                    len(next_state.completed_objectives) / next_state.current_sol
                )
                reward += objectives_per_sol * 50.0
            if next_state.time_remaining < 60 and len(next_state.science_targets) > 5:
                reward -= 30.0
            return reward


class MultiAgentRLSystem:
    """Coordinated multi-agent reinforcement learning system"""

    def __init__(self, models_dir: str = "models/marl", algorithm: str = "tabular"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Decide backend algorithm
        algo_env = os.getenv("MARL_ALGO", "").lower()
        self.algorithm = (algo_env or algorithm).lower()

        # Initialize specialized agents
        if self.algorithm == "dqn" and DQNAgent is not None:
            self.agents = {
                "route": DeepRouteAgent(),
                "power": DeepPowerAgent(),
                "science": DeepScienceAgent(),
                "hazard": DeepHazardAgent(),
                "strategy": DeepStrategyAgent(),
            }
            logger.info("Initialized MARL system with DQN agents")
        else:
            self.agents = {
                "route": RouteAgent(),
                "power": PowerAgent(),
                "science": ScienceAgent(),
                "hazard": HazardAgent(),
                "strategy": StrategyAgent(),
            }
            if self.algorithm == "dqn" and DQNAgent is None:
                logger.warning(
                    "Requested DQN but torch is not available. Falling back to tabular agents."
                )

        # Coordination mechanism
        self.coordination_weights = {
            "route": 0.25,
            "power": 0.25,
            "science": 0.25,
            "hazard": 0.15,
            "strategy": 0.10,
        }

        # Training history
        self.training_episodes = 0
        self.collective_rewards = []

        logger.info("Multi-agent RL system initialized with 5 specialized agents")

    def coordinate_action(self, state: AgentState, training: bool = False) -> Action:
        """Coordinate actions from all agents to select best action"""

        state_vector = state.to_vector()
        agent_actions = {}
        agent_q_values = {}

        # Get action preferences from each agent
        for name, agent in self.agents.items():
            action_idx = agent.select_action(state_vector, training=training)
            q_values = agent.get_q_values(state_vector)
            agent_actions[name] = action_idx
            agent_q_values[name] = q_values

        # Weighted voting for action selection
        combined_action = self._weighted_action_selection(agent_actions, agent_q_values)

        # Heuristic fallback if Q-values are uninformative
        route_q = agent_q_values.get("route")
        if route_q is not None and (
            np.var(route_q) < 1e-8 or np.allclose(route_q, route_q[0])
        ):
            heuristic = self._heuristic_route_action(state)
            if heuristic is not None:
                combined_action = heuristic

        # Convert to Action object
        return self._action_idx_to_action(combined_action, state)

    def _weighted_action_selection(
        self, agent_actions: Dict, agent_q_values: Dict
    ) -> int:
        """Combine agent actions using weighted voting"""

        # Simple approach: use route agent's action with input from others
        route_action = agent_actions["route"]

        # Modify based on other agents' concerns
        hazard_q = agent_q_values["hazard"][agent_actions["hazard"]]
        if hazard_q < -50:  # High hazard concern
            # Choose safer action from route options
            route_action = agent_actions["hazard"] % self.agents["route"].action_dim

        power_q = agent_q_values["power"][agent_actions["power"]]
        if power_q < -20:  # Power concern
            # Suggest conservative action
            route_action = min(route_action, 2)  # Limit to conservative moves

        return route_action

    def _action_idx_to_action(self, action_idx: int, state: AgentState) -> Action:
        """Convert action index to Action object"""

        # Map route agent's action to actual action
        directions = [
            (0, 10),
            (7, 7),
            (10, 0),
            (7, -7),
            (0, -10),
            (-7, -7),
            (-10, 0),
            (-7, 7),
        ]

        if action_idx < len(directions):
            dx, dy = directions[action_idx]
            new_pos = (state.position[0] + dx, state.position[1] + dy)

            return Action(
                action_type="drive",
                target=new_pos,
                duration=20,
                power_required=110.0 * 0.33,  # 110W for 20 min
                expected_reward=0.0,
            )
        else:
            return Action(action_type="wait", duration=10, power_required=50.0 * 0.17)

    def _heuristic_route_action(self, state: AgentState) -> int | None:
        """Choose a direction toward nearest target while avoiding hazards."""
        if not state.science_targets:
            return None
        target = state.science_targets[0]
        dx = target["lat"] - state.position[0]
        dy = target["lon"] - state.position[1]
        angle = math.atan2(dy, dx)  # radians
        # 8-direction quantization (0:E, then CCW)
        dirs = [
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ]
        angles = [math.atan2(d[1], d[0]) for d in dirs]
        idx = int(
            np.argmin(
                [abs((a - angle + math.pi) % (2 * math.pi) - math.pi) for a in angles]
            )
        )

        # Hazard-aware tweak: if target cell ahead is hazardous, pick next safest
        def hazard_for(idx: int) -> float:
            step = [
                (0, 10),
                (7, 7),
                (10, 0),
                (7, -7),
                (0, -10),
                (-7, -7),
                (-10, 0),
                (-7, 7),
            ][idx]
            nx = int(state.position[0] + step[0]) % state.hazard_map.shape[0]
            ny = int(state.position[1] + step[1]) % state.hazard_map.shape[1]
            return float(state.hazard_map[nx, ny])

        base_h = hazard_for(idx)
        if base_h >= 2:
            # pick among all directions with minimal hazard
            idx = int(np.argmin([hazard_for(i) for i in range(8)]))
        return idx

    def train_episode(self, initial_state: AgentState, max_steps: int = 100) -> float:
        """Train all agents for one episode"""

        state = initial_state
        total_reward = 0.0

        for step in range(max_steps):
            # Select coordinated action
            action = self.coordinate_action(state, training=True)

            # Simulate environment step (simplified)
            next_state, reward, done = self._simulate_step(state, action)

            # Update all agents
            state_vector = state.to_vector()
            next_state_vector = next_state.to_vector()

            for name, agent in self.agents.items():
                agent_reward = agent.compute_reward(state, action, next_state)
                action_idx = agent.select_action(state_vector, training=False)
                agent.update(
                    state_vector, action_idx, agent_reward, next_state_vector, done
                )

            total_reward += reward
            state = next_state

            if done:
                break

        # Decay exploration
        for agent in self.agents.values():
            agent.decay_epsilon()

        self.training_episodes += 1
        self.collective_rewards.append(total_reward)

        return total_reward

    def _simulate_step(
        self, state: AgentState, action: Action
    ) -> Tuple[AgentState, float, bool]:
        """Simulate environment step (simplified simulation)"""

        # Create next state
        next_state = AgentState(
            position=action.target if action.target else state.position,
            battery_soc=max(
                0, state.battery_soc - action.power_required / 2064.0
            ),  # 2064Wh capacity
            time_remaining=state.time_remaining - action.duration,
            hazard_map=state.hazard_map,
            science_targets=state.science_targets.copy(),
            current_sol=state.current_sol,
            temperature=state.temperature,
            dust_opacity=state.dust_opacity,
            completed_objectives=state.completed_objectives.copy(),
        )

        # Check if target reached
        if action.action_type == "drive" and len(state.science_targets) > 0:
            target = state.science_targets[0]
            dist = np.sqrt(
                (next_state.position[0] - target["lat"]) ** 2
                + (next_state.position[1] - target["lon"]) ** 2
            )
            if dist < 5:  # Within 5 units
                next_state.completed_objectives.append(target["id"])
                next_state.science_targets.pop(0)

        # Calculate reward
        reward = sum(
            agent.compute_reward(state, action, next_state)
            for agent in self.agents.values()
        ) / len(self.agents)

        # Check if done
        done = (
            next_state.battery_soc < 0.05
            or next_state.time_remaining <= 0
            or len(next_state.science_targets) == 0
        )

        return next_state, reward, done

    def save_all_agents(self):
        """Save all agent policies"""
        for name, agent in self.agents.items():
            ext = (
                ".pt"
                if self.algorithm == "dqn"
                and DQNAgent is not None
                and hasattr(agent, "policy_net")
                else ".json"
            )
            path = self.models_dir / f"{name}_agent{ext}"
            agent.save(str(path))
            logger.info(f"Saved {name} agent to {path}")

    def load_all_agents(self):
        """Load all agent policies"""
        for name, agent in self.agents.items():
            # Prefer DQN checkpoint if present
            pt_path = self.models_dir / f"{name}_agent.pt"
            json_path = self.models_dir / f"{name}_agent.json"
            load_path = pt_path if pt_path.exists() else json_path
            if load_path.exists():
                agent.load(str(load_path))
                logger.info(f"Loaded {name} agent from {load_path}")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            "episodes": self.training_episodes,
            "avg_reward": (
                np.mean(self.collective_rewards[-100:])
                if self.collective_rewards
                else 0
            ),
            "total_experiences": sum(
                len(agent.replay_buffer) for agent in self.agents.values()
            ),
            "agent_epsilons": {
                name: agent.epsilon for name, agent in self.agents.items()
            },
            "recent_rewards": (
                self.collective_rewards[-10:] if self.collective_rewards else []
            ),
        }

    def optimize_mission_plan(self, mission_context: Dict) -> Dict:
        """Use trained agents to optimize mission plan"""

        # Convert mission context to state
        state = AgentState(
            position=(
                mission_context.get("lat", 18.4),
                mission_context.get("lon", 77.4),
            ),
            battery_soc=mission_context.get("battery_soc", 0.7),
            time_remaining=mission_context.get("time_budget_min", 480),
            hazard_map=np.random.rand(100, 100),  # Placeholder
            science_targets=mission_context.get("targets", []),
            current_sol=mission_context.get("sol", 1597),
            temperature=mission_context.get("temp", -60),
            dust_opacity=mission_context.get("dust", 0.4),
        )

        # Generate optimized plan
        actions = []
        for _ in range(10):  # Generate 10 actions
            action = self.coordinate_action(state, training=False)
            actions.append(
                {
                    "type": action.action_type,
                    "target": action.target,
                    "duration": action.duration,
                    "power": action.power_required,
                }
            )

            # Update state (simplified)
            state, _, done = self._simulate_step(state, action)
            if done:
                break

        return {
            "optimized_actions": actions,
            "expected_completion": len([a for a in actions if a["type"] == "sample"]),
            "total_power": sum(a["power"] for a in actions),
            "total_time": sum(a["duration"] for a in actions),
            "rl_confidence": np.mean(
                [1 - agent.epsilon for agent in self.agents.values()]
            ),
        }


# Training script
def train_marl_system(episodes: int = 1000, save_interval: int = 100):
    """Train the multi-agent system"""

    system = MultiAgentRLSystem()

    logger.info(f"Starting MARL training for {episodes} episodes...")

    for episode in range(episodes):
        # Create random initial state
        initial_state = AgentState(
            position=(18.4 + np.random.randn() * 0.01, 77.4 + np.random.randn() * 0.01),
            battery_soc=0.7 + np.random.rand() * 0.2,
            time_remaining=480,
            hazard_map=np.random.rand(100, 100),
            science_targets=[
                {
                    "id": f"target_{i}",
                    "lat": 18.4 + np.random.randn() * 0.02,
                    "lon": 77.4 + np.random.randn() * 0.02,
                }
                for i in range(5)
            ],
            current_sol=1597,
            temperature=-60 + np.random.randn() * 10,
            dust_opacity=0.4 + np.random.rand() * 0.2,
        )

        reward = system.train_episode(initial_state)

        if (episode + 1) % 10 == 0:
            stats = system.get_training_stats()
            logger.info(
                f"Episode {episode + 1}/{episodes}: "
                f"Reward={reward:.2f}, Avg={stats['avg_reward']:.2f}"
            )

        if (episode + 1) % save_interval == 0:
            system.save_all_agents()
            logger.info(f"Checkpoint saved at episode {episode + 1}")

    system.save_all_agents()
    logger.info("Training complete!")

    return system


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    system = train_marl_system(episodes=500)

    # Test the system
    print("\n=== Testing Trained MARL System ===")
    stats = system.get_training_stats()
    print(f"Total episodes: {stats['episodes']}")
    print(f"Average reward: {stats['avg_reward']:.2f}")
    print(f"Agent exploration rates: {stats['agent_epsilons']}")
