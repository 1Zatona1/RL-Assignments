"""Policy iteration for the Grid Maze (pure NumPy, no gym or pygame)."""

import numpy as np


def state_index(pos, size):
    """Map an (x, y) cell to a single state number in 0 .. size*size - 1."""
    return int(pos[0]) * size + int(pos[1])


def build_model(obs, size, action_to_direction):
    """Build the known, deterministic model of one maze layout.

    Args:
        obs: observation dict from env.reset() with keys
             "agent", "target", "bad_1", "bad_2" (each an [x, y] array).
        size: grid side length (5 for a 5x5 maze).
        action_to_direction: dict {action: np.array([dx, dy])}. Pass the
             env's own mapping (env.unwrapped._action_to_direction) so the
             model and the environment can never disagree.

    Returns:
        next_state: int array (n_states, n_actions), the state reached by
                    taking action a in state s.
        reward:     float array (n_states, n_actions), the reward for that move.
        terminal:   bool array (n_states,), True for the goal and bad cells.
    """
    n_states = size * size
    n_actions = len(action_to_direction)

    goal = state_index(obs["target"], size)
    bad_cells = {
        state_index(obs["bad_1"], size),
        state_index(obs["bad_2"], size),
    }

    terminal = np.zeros(n_states, dtype=bool)
    terminal[goal] = True
    for b in bad_cells:
        terminal[b] = True

    next_state = np.zeros((n_states, n_actions), dtype=int)
    reward = np.zeros((n_states, n_actions), dtype=float)

    for x in range(size):
        for y in range(size):
            s = x * size + y
            for a in range(n_actions):
                if terminal[s]:
                    # Absorbing: stay put, no further reward.
                    next_state[s, a] = s
                    continue

                # Same rule as env.step: move, then clip at the walls.
                nx, ny = np.clip(
                    np.array([x, y]) + action_to_direction[a], 0, size - 1
                )
                s_next = int(nx) * size + int(ny)
                next_state[s, a] = s_next

                if s_next == goal:
                    reward[s, a] = 1.0
                elif s_next in bad_cells:
                    reward[s, a] = -1.0

    return next_state, reward, terminal