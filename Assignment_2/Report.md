# Assignment 2: Policy Iteration


**Students:**    
Ahmed Mohamed Farouk - 1220223  
Omar Ashraf Moursi - 1220132    

**Repository:** https://github.com/1Zatona1/RL-Assignments.git 

---

## 1. Outcome Summary

**Environment**     
- 5x5 grid, with 8-integer observation: (x, y) of the agent, the goal and the 2 bad cells
- 4 deterministic actions (right, up, left, down); moving into a wall leaves the agent in place
- 1 terminating goal cell with reward +1
- 2 terminating bad cells, each with reward -1
- all other cells give reward 0
- the agent, goal and bad cell positions are randomized on every reset, and never overlap  


**Algorithm**       
- The model (next state, reward, terminal) is built from each new maze, and policy iteration is run separately for it
- The initial policy is random, with a uniform distribution over LEGAL actions
- Evaluation and improvement are repeated until the policy stops changing
- gamma = 0.9 and theta = 1e-6



**Results**
- 3.85 iterations on average with a maximum of 7 over 200 mazes

**Videos:** `videos/`

---

## 2. Questions

### Q1. What is the state-space size of the 5x5 Grid Maze problem?

**Answer:** 25 possible postions for goal * 24 possible positions for the first bad cell * 23 possible postions for the second bad cell * 22 possible postions for the agent start.


### Q2. How to optimize the policy iteration for the Grid Maze problem?

**Answer:**     
1- begin each evaluation from the previous V instead of zeros, since the new policy is close to the old one.    
2- use the already-updated values during the same sweep instead of keeping a separate copy, so information spreads faster and fewer sweeps are needed.

### Q3. How many iterations did it take to converge on a stable policy for the 5x5 maze?

**Answer:** With gamma = 0.9, policy iteration converged in about 4 iterations on average (3.85 over 200 random 5x5 mazes), with a maximum of 7. The count alsp includes the last iteration that confirms the policy no longer changes.

| gamma | mean iterations | max iterations |
|-------|-----------------|----------------|
| 0.5   |     4.06        |     7          |
| 0.9   |     3.85        |     7          |
| 0.99  |     3.65        |     6          |



### Q4. Explain, with an example, how policy iteration behaves with multiple goal cells.

**Answer:** With multiple goal cells, policy iteration needs no change to the algorithm. Every goal is a terminal state with +1 reward, so the values reflect the discounted distance to the nearest goal, and the final policy sends each cell toward its closest goal.

**Example:** a row of 5 cells with goals at both ends, gamma = 0.9:

G c1 c2 c3 G

Cells next to a goal have value 1 (c1 and c3), and c2 has value 0.9 (two steps from either goal). The policy is c1 → left, c3 → right, and c2 can go either way, since both goals are equally close.

we used this rule gamma^(steps-1) 

### Q5. Can policy iteration work on a 100x100 maze? Explain why.

**Answer:** Yes, but it gets slow. A 100x100 maze has 10,000 states per layout, which fits in memory, but each sweep of policy evaluation is way bigger than on 5x5, and the longer paths need many more sweeps. Planning one policy for all layouts (goal and bad cells included) would be infeasible, since there would be about 10^16 states.


### Q6. Can policy iteration work on a continuous-space maze? Explain why.

**Answer:** No. Policy iteration stores a value and an action for each state in a table and sweeps over every state, so it needs a finite (discrete) set of states. A continuous maze has infinitely many positions, so no table can hold them.


### Q7. Can policy iteration work with moving bad cells (like Pac-Man ghosts)? Explain why.

**Answer:** No. Policy iteration needs a fixed model, but with moving bad cells the rewards keep changing, so the values computed for one set of positions become wrong as soon as they move, and they never settle on one stable policy. It could only work if the ghosts’ positions were included in the state and their movement were known in advance, which makes the state space much larger.

---