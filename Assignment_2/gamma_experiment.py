import numpy as np
import gymnasium as gym
import gymnasium_env
from policy_iteration import build_model, policy_iteration

SIZE = 5

env = gym.make("gymnasium_env/GridWorld-v0")   # no rendering, so it runs fast

for g in (0.5, 0.9, 0.99):
    iters = []
    for seed in range(200):
        obs, _ = env.reset(seed=seed)
        ns, rw, term = build_model(obs, SIZE, env.unwrapped._action_to_direction)
        _, _, n = policy_iteration(ns, rw, term, gamma=g)
        iters.append(n)
    print(f"gamma={g}: mean={np.mean(iters):.2f}, max={max(iters)}")

env.close()