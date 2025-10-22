#!/usr/bin/env python3
"""
Generic Deep Q-Network (DQN) agent usable by MARL specialized agents.

Design goals:
- Keep torch import optional to avoid import-time failures when unused
- Expose the same interface methods as tabular BaseAgent: select_action, update, decay_epsilon, save, load
- Accept an external reward function: reward_fn(state: Any, action: Action, next_state: Any) -> float
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple
import numpy as np
import random
from pathlib import Path

# Optional torch import
try:  # pragma: no cover - optional import
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover - keep runtime safe if torch not present
    torch = None
    nn = None
    optim = None


class _MLP(nn.Module):  # type: ignore[misc]
    def __init__(
        self, input_dim: int, output_dim: int, hidden: Tuple[int, int] = (128, 128)
    ):
        super().__init__()
        self.net = nn.Sequential(  # type: ignore[attr-defined]
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], output_dim),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


class DQNAgent:
    """
    Minimal DQN agent with target network and replay buffer.
    Expects numeric state vectors and discrete action space.
    """

    def __init__(
        self,
        agent_id: str,
        state_dim: int,
        action_dim: int,
        reward_fn: Callable[[Any, Any, Any], float],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        batch_size: int = 64,
        target_update: int = 200,
        hidden: Tuple[int, int] = (128, 128),
        device: Optional[str] = None,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for DQNAgent but is not installed.")

        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_fn = reward_fn

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy_net = _MLP(state_dim, action_dim, hidden).to(self.device)
        self.target_net = _MLP(state_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory: list[tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.max_mem = 100_000
        self.train_steps = 0
        self.episode_rewards: list[float] = []

    # Interface compatibility methods
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        self.policy_net.eval()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
                0
            )
            q = self.policy_net(s).squeeze(0).cpu().numpy()
        return q

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        q = self.get_q_values(state)
        return int(np.argmax(q))

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if len(self.memory) >= self.max_mem:
            self.memory.pop(0)
        self.memory.append(
            (
                state.astype(np.float32),
                action,
                float(reward),
                next_state.astype(np.float32),
                bool(done),
            )
        )

    def _train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(
            np.stack(states), dtype=torch.float32, device=self.device
        )
        actions_t = torch.tensor(
            actions, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        rewards_t = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        next_states_t = torch.tensor(
            np.stack(next_states), dtype=torch.float32, device=self.device
        )
        dones_t = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        q_values = self.policy_net(states_t).gather(1, actions_t)
        # Double DQN: action selection by policy_net, evaluation by target_net
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            targets = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    # Maintain interface compatibility with BaseAgent.update(...)
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        self.push(state, action, reward, next_state, done)
        return self._train_step()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # Persistence
    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "agent_id": self.agent_id,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "epsilon": self.epsilon,
            "train_steps": self.train_steps,
            "episode_rewards": self.episode_rewards,
            "state_dict": self.policy_net.state_dict(),
        }
        torch.save(data, str(p))

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["state_dict"])
        self.target_net.load_state_dict(ckpt["state_dict"])
        self.epsilon = float(ckpt.get("epsilon", self.epsilon))
        self.train_steps = int(ckpt.get("train_steps", 0))
        self.episode_rewards = list(ckpt.get("episode_rewards", []))
