import numpy as np
import gymnasium as gym
import gymnasium_env
from policy_iteration import state_index, build_model, policy_iteration

SIZE = 5
GAMMA = 0.9

env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")

for episode in range(5):
    obs, info = env.reset()

    # plan for this maze
    next_state, reward, terminal = build_model(obs, SIZE, env.unwrapped._action_to_direction)
    pi, V, iters = policy_iteration(next_state, reward, terminal, gamma=GAMMA)
    print(f"episode {episode}: policy converged in {iters} iterations")

    # follow the policy
    done, steps = False, 0
    while not done and steps < 50:
        action = int(np.argmax(pi[state_index(obs["agent"], SIZE)]))
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    print(f"  finished in {steps} steps, last reward = {r}")

env.close()