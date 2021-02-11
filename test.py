import numpy as np
import gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import argparse
import gym_turbine


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        help='Path to data .csv file.',
    )
    parser.add_argument(
        '--agent',
        help='Path to agent .pkl file or model .zip file.',
    )
    parser.add_argument(
        '--time',
        type=int,
        default=50,
        help='Max simulation time (seconds).',
    )
    args = parser.parse_args()

    agent_path = args.agent
    env = gym.make("TurbineStab-v0")
    agent = PPO.load(agent_path)
    done = False
    env.reset()
    while not done:
        action, _states = agent.predict(env.observation, deterministic=True)
        _, _, done, _ = env.step(action)
        env.render()
