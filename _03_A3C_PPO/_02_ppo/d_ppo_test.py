# https://gymnasium.farama.org/environments/box2d/bipedal_walker/
import os

import gymnasium as gym
import torch
from b_actor_and_critic import Actor
from e_get_configs import get_environment_config

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


def test(env: gym.Env, actor: Actor, num_episodes: int) -> None:
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = env.reset()

        episode_steps = 0

        done = False

        while not done:
            episode_steps += 1
            action = actor.get_action(observation, exploration=False)

            next_observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(i, episode_steps, episode_reward))


def main_play(num_episodes: int, config) -> None:
    env = gym.make(config["env_name"], render_mode="human")

    actor = Actor(n_features=config["n_features"], n_actions=config["n_actions"])
    model_params = torch.load(os.path.join(MODEL_DIR, "appo_{0}_latest.pth".format(config["env_name"])), weights_only=True)
    actor.load_state_dict(model_params)
    actor.eval()

    test(env=env, actor=actor, num_episodes=num_episodes)

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 3
    # ENV_NAME = "BipedalWalker-v3"

    env_name = "lunar_lander"
    config = get_environment_config(env_name)

    main_play(num_episodes=NUM_EPISODES, config=config)
