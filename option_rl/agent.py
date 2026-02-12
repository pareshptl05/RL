"""Tabular Q-learning agent with state discretization."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass
class AgentConfig:
    learning_rate: float = 0.1
    discount: float = 0.98
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995


class StateDiscretizer:
    def __init__(self, spot_bins: int = 12, time_bins: int = 10):
        self.spot_bins = spot_bins
        self.time_bins = time_bins

    def transform(self, state: tuple[float, float, float, int]) -> tuple[int, int, int]:
        _, spot, time_left, position = state

        spot_bucket = min(max(int((spot - 60) / 80 * self.spot_bins), 0), self.spot_bins - 1)
        time_bucket = min(max(int(time_left / (30 / 365) * self.time_bins), 0), self.time_bins - 1)

        return spot_bucket, time_bucket, int(position)


class QLearningAgent:
    def __init__(self, config: AgentConfig, discretizer: StateDiscretizer, n_actions: int = 3):
        self.config = config
        self.discretizer = discretizer
        self.n_actions = n_actions
        self.q_table: dict[tuple[int, int, int], list[float]] = {}
        self.epsilon = config.epsilon_start

    def _ensure_state(self, state_key: tuple[int, int, int]) -> None:
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0 for _ in range(self.n_actions)]

    def act(self, state: tuple[float, float, float, int], greedy: bool = False) -> int:
        key = self.discretizer.transform(state)
        self._ensure_state(key)

        if (not greedy) and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return max(range(self.n_actions), key=lambda a: self.q_table[key][a])

    def update(
        self,
        state: tuple[float, float, float, int],
        action: int,
        reward: float,
        next_state: tuple[float, float, float, int],
        done: bool,
    ) -> None:
        s_key = self.discretizer.transform(state)
        n_key = self.discretizer.transform(next_state)
        self._ensure_state(s_key)
        self._ensure_state(n_key)

        q_sa = self.q_table[s_key][action]
        next_best = 0.0 if done else max(self.q_table[n_key])
        target = reward + self.config.discount * next_best

        self.q_table[s_key][action] = q_sa + self.config.learning_rate * (target - q_sa)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
