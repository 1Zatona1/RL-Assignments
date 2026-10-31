import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import gymnasium_env
from game import run_episodes


def record(episodes=5, gamma=0.9):
    """Record one video per episode into the videos/ folder."""
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder="videos",
        episode_trigger=lambda ep: True,
        name_prefix="policy-iteration",
    )
    run_episodes(env, episodes, gamma)
    env.close()  # needed so the last video is written to disk
    print("Videos saved in the videos/ folder.")


if __name__ == "__main__":
    record()