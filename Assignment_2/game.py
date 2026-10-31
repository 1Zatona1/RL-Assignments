import numpy as np
import gymnasium as gym
import gymnasium_env
from policy_iteration import state_index, build_model, policy_iteration

SIZE = 5


def run_episodes(env, episodes=5, gamma=0.9):
    """For each new maze: plan with policy iteration, then follow the policy."""
    for episode in range(1, episodes + 1):
        obs, info = env.reset()

        # plan for this maze
        next_state, reward, terminal = build_model(obs, SIZE, env.unwrapped._action_to_direction)
        pi, V, iters = policy_iteration(next_state, reward, terminal, gamma=gamma)

        # follow the policy
        done, steps, r = False, 0, 0
        while not done and steps < 50:
            action = int(np.argmax(pi[state_index(obs["agent"], SIZE)]))
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        if r == 1:
            outcome = "reached the goal"
        elif r == -1:
            outcome = "hit a bad cell"
        else:
            outcome = "stopped (no safe path to the goal)"
        print(f"Episode {episode}: converged in {iters} iterations, {steps} steps, {outcome}")


def watch(episodes=5, gamma=0.9):
    """Watch the agent in a pygame window."""
    env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")
    run_episodes(env, episodes, gamma)
    env.close()


if __name__ == "__main__":
    watch()