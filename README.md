# OptionRL: Reinforcement Learning for Option Trading

This repository contains a starter implementation of a reinforcement learning (RL) agent for trading **European call options** in a simulated market.

## What is included

- A synthetic geometric Brownian motion (GBM) price simulator.
- A small options trading environment with transaction costs.
- A tabular Q-learning agent with state discretization.
- A simple training/evaluation script.

## Project structure

- `option_rl/env.py` – environment dynamics and reward function.
- `option_rl/agent.py` – discretizer and Q-learning implementation.
- `train.py` – training loop and evaluation output.

## Quick start

```bash
python3 train.py --episodes 400
```

Example output:

- Mean training reward over the last N episodes.
- Evaluation reward with exploration disabled.

## Notes and next steps

This is intentionally lightweight so you can iterate quickly. To make this “nifty” for production-like workflows, consider:

1. Replace GBM with historical options chain + underlying data.
2. Add richer action space (delta hedging, strikes, expiries).
3. Add risk-aware rewards (Sharpe, CVaR, drawdown penalties).
4. Upgrade to deep RL (DQN/PPO/SAC) with function approximation.
5. Add walk-forward validation and realistic slippage models.
