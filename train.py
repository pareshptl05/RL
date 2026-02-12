"""Train a simple Q-learning option trading agent."""

from __future__ import annotations

import argparse
import statistics

from option_rl.agent import AgentConfig, QLearningAgent, StateDiscretizer
from option_rl.env import EnvConfig, OptionTradingEnv


def run_episode(env: OptionTradingEnv, agent: QLearningAgent, train: bool = True) -> float:
    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.act(state, greedy=not train)
        next_state, reward, done = env.step(action)

        if train:
            agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

    if train:
        agent.decay_epsilon()

    return total_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=50)
    args = parser.parse_args()

    env = OptionTradingEnv(EnvConfig())
    agent = QLearningAgent(AgentConfig(), StateDiscretizer())

    rewards = []
    for _ in range(args.episodes):
        rewards.append(run_episode(env, agent, train=True))

    eval_rewards = [run_episode(env, agent, train=False) for _ in range(args.eval_episodes)]

    print(f"Training episodes: {args.episodes}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Mean training reward (last 50): {statistics.fmean(rewards[-50:]):.4f}")
    print(f"Mean eval reward: {statistics.fmean(eval_rewards):.4f}")


if __name__ == "__main__":
    main()
