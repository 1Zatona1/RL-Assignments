import numpy as np
import pytest
import gymnasium as gym
import gymnasium_env
from policy_iteration import (
    state_index,
    build_model,
    initial_policy,
    policy_evaluation,
    policy_improvement,
    policy_iteration,
)

SIZE = 5
GAMMA = 0.9


@pytest.fixture
def env():
    """A fresh environment for each test, closed afterwards."""
    env = gym.make("gymnasium_env/GridWorld-v0")
    yield env
    env.close()


@pytest.fixture
def maze(env):
    """One fixed maze (seed 0) with its model tables."""
    obs, _ = env.reset(seed=0)
    next_state, reward, terminal = build_model(obs, SIZE, env.unwrapped._action_to_direction)
    return obs, next_state, reward, terminal


def test_layouts(env):
    """Environment: observation keys exist and the 4 special cells never overlap."""
    for seed in range(1000):
        obs, _ = env.reset(seed=seed)
        assert set(obs.keys()) == {"agent", "target", "bad_1", "bad_2"}
        cells = {tuple(obs[k]) for k in obs}
        assert len(cells) == 4, f"overlap with seed {seed}"


def test_build_model(env):
    """Model: next_state / reward / terminal must match what env.step really does."""
    u = env.unwrapped
    for seed in range(20):
        obs, _ = env.reset(seed=seed)
        next_state, reward, terminal = build_model(obs, SIZE, u._action_to_direction)

        assert next_state.shape == (SIZE * SIZE, 4)
        assert reward.shape == (SIZE * SIZE, 4)
        assert terminal.sum() == 3  # goal + 2 bad cells

        for x in range(SIZE):
            for y in range(SIZE):
                s = state_index((x, y), SIZE)

                if terminal[s]:
                    assert (next_state[s] == s).all()
                    assert (reward[s] == 0).all()
                    continue

                for a in range(4):
                    # put the real env's agent on (x, y), take action a, compare
                    u._agent_location = np.array([x, y])
                    o, r, term, _, _ = u.step(a)
                    assert state_index(o["agent"], SIZE) == next_state[s, a]
                    assert r == reward[s, a]
                    assert term == terminal[next_state[s, a]]


def test_initial_policy(maze):
    """Policy: each non-terminal row sums to 1, terminal rows are 0, no wall bumps."""
    obs, next_state, reward, terminal = maze
    pi = initial_policy(next_state, terminal)

    assert pi.shape == (SIZE * SIZE, 4)
    for s in range(SIZE * SIZE):
        if terminal[s]:
            assert pi[s].sum() == 0
        else:
            assert abs(pi[s].sum() - 1) < 1e-9
            wall_bump = next_state[s] == s
            assert (pi[s][wall_bump] == 0).all()


def test_policy_evaluation(maze):
    """Evaluation: terminal V = 0, Bellman equation holds, and a hand-checked case."""
    obs, next_state, reward, terminal = maze

    # 1) random policy: values satisfy the equation we agreed on
    pi = initial_policy(next_state, terminal)
    V = policy_evaluation(pi, next_state, reward, terminal, gamma=GAMMA)
    assert (V[terminal] == 0).all()
    for s in range(SIZE * SIZE):
        if terminal[s]:
            continue
        expected = sum(
            pi[s, a] * (reward[s, a] + GAMMA * V[next_state[s, a]]) for a in range(4)
        )
        assert abs(V[s] - expected) < 1e-4

    # 2) hand-checked: a policy that steps straight into the goal from one cell
    #    must give that cell V = 1 (reward +1, then the goal is worth 0)
    s, a = np.argwhere(reward == 1)[0]
    pi_one = np.zeros_like(pi)
    pi_one[s, a] = 1.0
    V_one = policy_evaluation(pi_one, next_state, reward, terminal, gamma=GAMMA)
    assert abs(V_one[s] - 1.0) < 1e-9


def test_policy_improvement(maze):
    """Improvement: one action per row, no wall bumps, and it heads for the goal."""
    obs, next_state, reward, terminal = maze

    pi = initial_policy(next_state, terminal)
    V = policy_evaluation(pi, next_state, reward, terminal, gamma=GAMMA)
    new_pi = policy_improvement(V, next_state, reward, terminal, gamma=GAMMA)

    for s in range(SIZE * SIZE):
        if terminal[s]:
            assert new_pi[s].sum() == 0
        else:
            assert new_pi[s].sum() == 1 and (new_pi[s] == 1).sum() == 1
            a = new_pi[s].argmax()
            assert next_state[s, a] != s  # not a wall bump

    # a cell next to the goal must step straight into it
    s, a = np.argwhere(reward == 1)[0]
    assert new_pi[s, a] == 1


def test_policy_iteration(env):
    """Final policy is optimal and reaches the goal whenever that is possible."""
    all_iters, solvable = [], 0

    for seed in range(200):
        obs, _ = env.reset(seed=seed)
        next_state, reward, terminal = build_model(obs, SIZE, env.unwrapped._action_to_direction)
        pi, V, iters = policy_iteration(next_state, reward, terminal, gamma=GAMMA)
        all_iters.append(iters)

        # 1) V satisfies the Bellman optimality equation: best legal action matches V
        for s in range(SIZE * SIZE):
            if terminal[s]:
                continue
            q = reward[s] + GAMMA * V[next_state[s]]
            q[next_state[s] == s] = -np.inf
            assert abs(V[s] - q.max()) < 1e-4

        # 2) if a safe path to the goal exists (V > 0), following pi must reach the goal
        if V[state_index(obs["agent"], SIZE)] > 0:
            solvable += 1
            terminated, r, steps = False, 0, 0
            while not terminated and steps < 50:
                a = int(np.argmax(pi[state_index(obs["agent"], SIZE)]))
                obs, r, terminated, truncated, _ = env.step(a)
                steps += 1
            assert terminated and r == 1, f"seed {seed}: did not reach the goal"

    # shown with: uv run pytest -s
    print(f"\nlayouts with a safe path: {solvable}/200")
    print(f"iterations: min={min(all_iters)}, max={max(all_iters)}, mean={np.mean(all_iters):.2f}")