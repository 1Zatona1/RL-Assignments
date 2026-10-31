# Assignment 2: Grid Maze with Policy Iteration

CMPS458 Reinforcement Learning, Fall 2026.

A custom Gymnasium environment (a 5x5 "Grid Maze") solved with Policy Iteration, a dynamic programming method.

## 1. Overview

- **Environment:** the agent starts on a random cell and must reach the goal cell while avoiding 2 bad cells. The goal and bad cells are placed randomly on every reset.
- **Observation:** 8 integers, the (x, y) coordinates of the agent, the goal and the 2 bad cells.
- **Actions:** 4 discrete moves (right, up, left, down), deterministic. Moving into a wall leaves the agent in place.
- **Rewards:** +1 on reaching the goal, -1 on reaching a bad cell, 0 otherwise. The episode ends on the goal or a bad cell.
- **Algorithm:** for each new maze, the known model is built from the observation and policy iteration (policy evaluation + policy improvement) is run until the policy stops changing. The agent then follows the resulting policy.

## 2. Project Structure

```
Assignment_2/
├── gymnasium_env/
│   ├── __init__.py          # registers gymnasium_env/GridWorld-v0
│   └── envs/
│       ├── __init__.py
│       └── grid_world.py    # the environment and pygame rendering
├── policy_iteration.py      # build_model, evaluation, improvement, policy iteration
├── main.py                  # tests for the environment and each function
├── game.py                  # watch the agent in a pygame window
├── record.py                # record videos with RecordVideo
├── gamma_experiment.py      # iteration counts for different gamma values
├── REPORT.md                # outcome summary and answers to the questions
├── videos/                  # recorded videos
└── pyproject.toml
```

## 3. How to Run

```
cd Assignment_2
uv sync
uv run test_policy_iteration.py    # tests
uv run game.py                     # watch the agent in a pygame window
uv run record.py                   # record videos into videos/
```

Dependencies are listed in `pyproject.toml` and installed by `uv sync`.

## 4. Report

The outcome summary and the answers to the assignment questions are in [REPORT.md](REPORT.md).