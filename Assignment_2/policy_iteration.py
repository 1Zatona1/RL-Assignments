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

def initial_policy(next_state, terminal):
    """Uniform random policy over legal moves only (no wall bumps)."""
    n_states, n_actions = next_state.shape
    pi = np.zeros((n_states, n_actions))

    for s in range(n_states):
        if terminal[s]:
            continue  # no actions in the goal or bad cells
        legal = next_state[s] != s          # True where the move changes the cell
        pi[s, legal] = 1 / legal.sum()

    return pi

def policy_evaluation(pi, next_state, reward, terminal, gamma=0.9, theta=1e-6):
    """Compute V for a given policy by repeating the update until it stops changing.
 
    V_new[s] = sum_a pi[s, a] * (reward[s, a] + gamma * V[next_state[s, a]])
 
    Args:
        pi:         (n_states, n_actions) array, pi[s, a] = probability of action a in state s.
        next_state: from build_model.
        reward:     from build_model.
        terminal:   from build_model. Terminal states keep V = 0.
        gamma:      discount factor.
        theta:      stop when the largest change in V is below this.
 
    Returns:
        V: (n_states,) array of state values under pi.
    """
    n_states, n_actions = pi.shape
    V = np.zeros(n_states)  # initial guess V0 = 0
 
    while True:
        V_new = np.zeros(n_states)
 
        for s in range(n_states):
            if terminal[s]:
                continue  # episode is over here, V stays 0
 
            for a in range(n_actions):
                V_new[s] += pi[s, a] * (reward[s, a] + gamma * V[next_state[s, a]])
 
        delta = np.max(np.abs(V_new - V))
        V = V_new
 
        if delta < theta:
            return V

def policy_improvement(V, next_state, reward, terminal, gamma=0.9):
    """Make the policy greedy with respect to V.
 
    For each state, score every action with
        q[a] = reward[s, a] + gamma * V[next_state[s, a]]
    and give probability 1 to the best legal action (wall bumps are skipped).
 
    Returns:
        new_pi: (n_states, n_actions) array with a single 1 in each non-terminal row.
    """
    n_states, n_actions = next_state.shape
    new_pi = np.zeros((n_states, n_actions))
 
    for s in range(n_states):
        if terminal[s]:
            continue  # no action in the goal or bad cells
 
        q = reward[s] + gamma * V[next_state[s]]  # score of the 4 actions
        q[next_state[s] == s] = -np.inf           # never pick a wall bump
        new_pi[s, np.argmax(q)] = 1.0
 
    return new_pi

def policy_iteration(next_state, reward, terminal, gamma=0.9, theta=1e-6):
    """Alternate evaluation and improvement until the policy stops changing.
 
    Returns:
        pi:         the final policy, one 1 per non-terminal row.
        V:          the values of the final policy.
        iterations: number of evaluate + improve rounds, including the last
                    round that confirmed nothing changed.
    """
    pi = initial_policy(next_state, terminal)  # start from the random policy
    iterations = 0
 
    while True:
        V = policy_evaluation(pi, next_state, reward, terminal, gamma, theta)
        new_pi = policy_improvement(V, next_state, reward, terminal, gamma)
        iterations += 1
 
        if np.array_equal(new_pi, pi):  # nothing changed, so the policy is stable
            return new_pi, V, iterations
 
        pi = new_pi
 