import gymnasium as gym
import gymnasium_env 

env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")
obs, info = env.reset(seed=42)

for _ in range(20):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()

env.close()
