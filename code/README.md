# Assignment 2: Reinforcement Learning

## Quick start guide

1. Execute the command `python main.py` in the shell to start the training.

## Run with specified Environment and Hyperparameters

Environment and Hyperparameters can be configured within the file `python main.py`.

1. Open script `main.py` with an editor.
2. Find the section `Definition of the environment`.
    - You should see content similar to
```python
# ---- Definition of the environment ----
# define playground
M = 10                              # length of the gridworld ---> number of columns
N = 10                              # height of the gridworld ---> number of rows
N_states =  M * N                   # total number of states
states_matrix = np.eye(N_states)
N_actions = 4                                           # number of possible actions in each state: 1->N 2->E 3->S 4->W
action_col_change = np.array([-1,0,+1,0])               # number of cell shifted in vertical as a function of the action
action_row_change = np.array([0,+1,0,-1])               # number of cell shifted in horizontal as a function of the action

# define walls
walls = [
    (5, 0),(5, 1),(5, 2),(5, 3),
    (3, 9),(3, 8),(3, 7),(3, 6),(3, 5),
    (9, 6),(8, 6),(7, 6)
]

# define goal
RAND_GOAL = True   # whether to use a random goal for each repetition
DEF_GOAL = [2, 9]   # goal position if not random

```
3. Edit environment
4. Find the section `Hyperparameters`.
    - You should see content similar to
```python
# ---- Hyperparameters ----
# training
nrepetitions = 100  # number of runs for the algorithm
nTrials = 2000      # should be integer >0
nSteps = 50         # maximum number of allowed steps
learningRate = 0.9  # should be real, Greater than 0
epsilon = 0.2       # should be real, Greater or Equal to 0; epsion=0 Greedy, otherwise epsilon-Greedy (exploration factor)
gamma = 0.9         # should be real, positive, smaller than 1 (discount factor)
lam = 0.3           # decay controlling for eligibility trace in SARSA(lambda). should be real, positive. Switch to SARSA when set to -1

# Rewards and Penalizes
REWARD = 1              # only when the robot reaches the charger, sited in End state
PENALIZE_EDGE = -0.1    # penalize when hit edge
PENALIZE_WALL = -1      # penalize when hit wall
PENALIZE_EXCEEDED = -1  # penalize when steps exceeded

```
5. Edit hyperparameters
6. Execute the command `python main.py` in the shell to start the training.

## Output

 - the learning curve will be logged into directory `./learning_curve/`
 - the graph of preferred direction will be saved into directory `./preferred_direc/`, numbered by runs.
 - the mean and std of extra steps will be outputs in the terminal after the training is finished.
