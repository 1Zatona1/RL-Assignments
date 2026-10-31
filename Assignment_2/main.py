import subprocess
import sys

from game import watch
from record import record

BANNER = """
==============================================
  Welcome to the Grid Maze!
  An agent plans its way to the goal (green)
  and avoids the bad cells (red X) using
  Policy Iteration.
==============================================
"""

MENU = """
  1) Watch the agent (pygame window)
  2) Record videos
  3) Run the tests
  4) Run the gamma experiment
  q) Quit
"""


def ask_number(prompt, default, cast):
    """Ask for a number; pressing Enter keeps the default."""
    answer = input(f"{prompt} (default {default}): ").strip()
    if not answer:
        return default
    try:
        return cast(answer)
    except ValueError:
        print(f"Not a valid number, using {default}.")
        return default

def ask_gamma(default=0.9):
    """Ask for gamma until it is a number with 0 < gamma < 1."""
    while True:
        answer = input(f"Discount factor gamma, above 0 and below 1 (default {default}): ").strip()
        if not answer:
            return default
        try:
            gamma = float(answer)
        except ValueError:
            print("Please enter a number.")
            continue
        if 0 < gamma < 1:
            return gamma
        print("gamma must be above 0 and below 1. At 1, equally good policies can swap forever and the algorithm never stops.")

def run_script(*args):
    """Run another Python script or module with the same interpreter."""
    subprocess.run([sys.executable, *args])


def main():
    print(BANNER)

    while True:
        print(MENU)
        choice = input("Choose: ").strip().lower()

        if choice == "1":
            episodes = ask_number("How many episodes?", 5, int)
            gamma = ask_gamma()
            watch(episodes, gamma)
        elif choice == "2":
            episodes = ask_number("How many episodes?", 5, int)
            gamma = ask_gamma()
            record(episodes, gamma)
        elif choice == "3":
            run_script("-m", "pytest", "-v")
        elif choice == "4":
            run_script("gamma_experiment.py")
        elif choice in ("q", "quit", "exit"):
            print("Goodbye!")
            break
        else:
            print("Please choose 1, 2, 3, 4 or q.")


if __name__ == "__main__":
    main()