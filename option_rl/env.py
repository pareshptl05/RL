"""Simple options trading environment for reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class EnvConfig:
    s0: float = 100.0
    strike: float = 100.0
    rate: float = 0.01
    volatility: float = 0.2
    maturity_years: float = 30 / 365
    steps: int = 30
    transaction_cost: float = 0.05
    max_position: int = 1


class OptionTradingEnv:
    """A minimal episodic environment.

    State variables are:
      - step index
      - spot price
      - time to maturity
      - current position in the option {0, 1}

    Actions are:
      - 0: hold
      - 1: buy one option if flat
      - 2: sell/close one option if long

    Reward is marked-to-market PnL delta minus transaction cost.
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self.dt = config.maturity_years / config.steps
        self._price_path: list[float] = []
        self._step = 0
        self.position = 0
        self.prev_option_price = 0.0

    def reset(self) -> tuple[float, float, float, int]:
        self._step = 0
        self.position = 0
        self._price_path = self._simulate_price_path()
        time_left = self.config.maturity_years
        self.prev_option_price = self._option_price(self._price_path[0], time_left)
        return self._get_state()

    def _simulate_price_path(self) -> list[float]:
        prices = [self.config.s0]
        drift = (self.config.rate - 0.5 * self.config.volatility**2) * self.dt
        diffusion = self.config.volatility * math.sqrt(self.dt)
        for _ in range(self.config.steps):
            z = random.gauss(0.0, 1.0)
            next_price = prices[-1] * math.exp(drift + diffusion * z)
            prices.append(next_price)
        return prices

    def _option_price(self, spot: float, tau: float) -> float:
        if tau <= 1e-10:
            return max(spot - self.config.strike, 0.0)
        vol_sqrt_tau = self.config.volatility * math.sqrt(tau)
        d1 = (
            math.log(spot / self.config.strike)
            + (self.config.rate + 0.5 * self.config.volatility**2) * tau
        ) / vol_sqrt_tau
        d2 = d1 - vol_sqrt_tau
        nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2)))
        nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2)))
        discounted_strike = self.config.strike * math.exp(-self.config.rate * tau)
        return spot * nd1 - discounted_strike * nd2

    def _get_state(self) -> tuple[float, float, float, int]:
        spot = self._price_path[self._step]
        time_left = max(self.config.maturity_years - self._step * self.dt, 0.0)
        return (float(self._step), spot, time_left, self.position)

    def step(self, action: int) -> tuple[tuple[float, float, float, int], float, bool]:
        current_spot = self._price_path[self._step]
        time_left = max(self.config.maturity_years - self._step * self.dt, 0.0)
        current_option_price = self._option_price(current_spot, time_left)

        reward = self.position * (current_option_price - self.prev_option_price)

        if action == 1 and self.position < self.config.max_position:
            self.position += 1
            reward -= self.config.transaction_cost
        elif action == 2 and self.position > 0:
            self.position -= 1
            reward -= self.config.transaction_cost

        self.prev_option_price = current_option_price
        self._step += 1

        done = self._step >= self.config.steps
        if done and self.position > 0:
            terminal_spot = self._price_path[self._step]
            payoff = max(terminal_spot - self.config.strike, 0.0)
            reward += payoff - self.prev_option_price
            self.position = 0

        return self._get_state(), reward, done
