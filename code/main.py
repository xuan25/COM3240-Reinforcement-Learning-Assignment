import os
from matplotlib import patches, pyplot as plt
import numpy as np
from numpy.lib.function_base import average

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



def get_Q_nn(input_vector, weights):
    """Get Q-value from a neural network.

    Args:
        input_vector: A vector representing the current state 
        weights: Weight matrix for neural networks

    Returns:
        Q-value vector
    """
    # compute Qvalues. Qvalue=logsig(weights*input).
    Q = 1 / ( 1 + np.exp( - weights.dot(input_vector)))
    return Q

def homing_nn(n_trials, n_steps, learning_rate, eps, gamma, lam, goal):
    """Run homing learning for one time.

    Args:
        n_trials: number of trials
        n_steps: max number of steps
        learning_rate: learning rate for neural network
        eps: eps value for eps-greedy policy
        gamma: gamma value for SARSA weight update
        lam: lambda value for eligibility trace
        goal: position of the goal

    Returns:
        optimal step curve in terms of minimum steps to the goal, 
        learning curve in terms of number of steps it takes to the goal, 
        weights of the neural network
    """
    End = goal
    s_end = np.ravel_multi_index(End,dims=(M,N),order='F')  #terminal state. Conversion in single index

    ## variables
    weights = np.random.rand(N_actions, N_states)
    learning_curve = np.zeros((1, n_trials))
    optimal_step_curve = np.zeros((1, n_trials))

    ## SARSA

    # start trials
    for trial in range(n_trials):

        # Initialization
        Start = np.array([np.random.randint(M),np.random.randint(N)])   #random start
        s_start = np.ravel_multi_index(Start,dims=(M,N),order='F')      #conversion in single index

        state = Start                                                   #set current state
        s_index = s_start                                               #conversion in single index
        step = 0

        optimal_step = np.abs(End[0] - Start[0]) + np.abs(End[1] - Start[1])
        optimal_step_curve[0,trial] = optimal_step

        if lam > 0:
            e = np.zeros([N_actions, N_states])

        # start steps
        while s_index != s_end and step < n_steps:

            step += 1
            learning_curve[0,trial] = step

            input_vector = states_matrix[:,s_index].reshape(N_states,1)         #convert the state into an input vector

            # compute Qvalues
            q_s = get_Q_nn(input_vector, weights)

            # eps-greedy policy
            greedy = (np.random.rand() > eps)               # 1--->greedy action 0--->non-greedy action
            if greedy:
                action = np.argmax(q_s)                     # pick best action
            else:
                action = np.random.randint(N_actions)       # pick random action

            state_new = np.array([0,0])
            # move into a new state
            state_new[0] = state[0] + action_col_change[action]
            state_new[1] = state[1] + action_row_change[action]

            r = 0
            # put the robot back in grid if it goes out. Consider also the option to give a negative reward
            if state_new[0] < 0:
                state_new[0] = 0
                r = PENALIZE_EDGE
            if state_new[0] >= M:
                state_new[0] = M-1
                r = PENALIZE_EDGE
            if state_new[1] < 0:
                state_new[1] = 0
                r = PENALIZE_EDGE
            if state_new[1] >= N:
                state_new[1] = N-1
                r = PENALIZE_EDGE

            # walls hittest
            if (state_new[0], state_new[1]) in walls:
                state_new = state
                r = PENALIZE_WALL

            s_index_new = np.ravel_multi_index(state_new,dims=(M,N),order='F')  #conversion in a single index

            # update Qvalues. Only if is not the first step
            if step > 1:
                delta = r_old - q_sa_old + gamma * q_s[action]
                if lam > 0:
                    dw = learning_rate * delta * e
                else:
                    dw = learning_rate * delta * output_old.dot(input_old.T)
                    
                weights += dw

            if lam > 0:
                e = gamma * lam * e
                e[action, s_index] = e[action, s_index] + 1

            # store variables for sarsa computation in the next step
            output = np.zeros((N_actions,1))
            output[action] = 1

            # update variables
            input_old = input_vector
            output_old = output
            q_sa_old = q_s[action]
            r_old = r

            state[0] = state_new[0]
            state[1] = state_new[1]
            s_index = s_index_new

            # trial ends
            if s_index == s_end or step == n_steps:
                if s_index == s_end:
                    # reward reached
                    r_old = REWARD
                elif step == n_steps:
                    # steps exceeded
                    r_old = PENALIZE_EXCEEDED
                # update weights
                delta = r_old - q_sa_old
                if lam > 0:
                    dw = learning_rate * delta * e
                else:
                    dw = learning_rate * delta * output_old.dot(input_old.T)
                weights += dw

    return optimal_step_curve, learning_curve, weights


def draw_learning_curves(learning_curves, save_path=None):
    """draw learning curves

    Args:
        learning_curves: learning curves
        save_path: path to save the image, None for show in window
    """
    n_trials = learning_curves.shape[1]

    plt.figure()
    means = np.mean(learning_curves, axis = 0)
    errors_std = np.std(learning_curves, axis = 0)

    plt.fill_between(np.arange(n_trials), means-errors_std, means+errors_std, alpha=1, edgecolor='silver', facecolor='silver', linewidth=0)
    plt.plot(np.arange(n_trials), means, color='black', marker='o', linestyle='solid', linewidth=0.5, markersize=0)
    plt.xlabel('Trials')
    plt.ylabel('Average Steps')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def draw_preferred_direc(weights, goal, walls, save_path=None):
    """draw preferred directions for each position in the playground.

    Args:
        weights: weight of the neural network
        goal: goal position
        walls: walls position
        save_path: path to save the image, None for show in window
    """
    plt.figure()
    
    plt.xlim((1, 1+M))
    plt.ylim((1, 1+N))
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().invert_yaxis()
    plt.xticks(np.arange(1, 1+M, step=1), ha='left')
    plt.yticks(np.arange(1, 1+N, step=1), va='top')
    plt.grid()

    for x in range(M):
        for y in range(N):
            if x == goal[0] and y == goal[1]:
                continue
            if (x, y) in walls:
                continue

            s_test = np.ravel_multi_index(np.array([x, y]),dims=(M,N),order='F')
            input_vector = states_matrix[:,s_test].reshape(N_states,1)
            Q = 1 / ( 1 + np.exp( - weights.dot(input_vector)))
            action = np.argmax(Q)

            r_c = action_row_change[action]
            c_c = action_col_change[action]
            
            plt.arrow(x+1.5, y+1.5, c_c * 0.3, r_c * 0.3, head_width=0.06, head_length=0.1)

    goal_rect = patches.Rectangle([goal[0]+1, goal[1]+1], 1, 1, linewidth=1, edgecolor='lime', facecolor='lime')
    plt.gca().add_patch(goal_rect)

    for wall in walls:
        wall_rect = patches.Rectangle([wall[0]+1, wall[1]+1], 1, 1, linewidth=1, edgecolor='silver', facecolor='silver')
        plt.gca().add_patch(wall_rect)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def save_learning_curves(learning_curves, save_path='learning_curves.csv'):
    """save learning curves to a csv file

    Args:
        learning_curves: learning curves
        save_path: path to save the image, None for show in window
    """
    n_trials = learning_curves.shape[1]
    means = np.mean(learning_curves, axis = 0)
    errors_std = np.std(learning_curves, axis = 0)

    import csv
    with open(save_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['trail', 'mean', 'std'])
        for i in range(n_trials):
            csvwriter.writerow([i, means[i], errors_std[i]])



if __name__ == '__main__':
    # mkdir for outputs
    for d in [ './learning_curve/', './preferred_direc/' ]:
        if not os.path.exists(d):
            os.makedirs(d)

    # trace var
    learning_curves = np.zeros((nrepetitions, nTrials))
    extra_steps_curves = np.zeros((nrepetitions, nTrials))

    # init goal
    if not RAND_GOAL:
        goal = np.array(DEF_GOAL)

    # start repetitions
    for i in range(nrepetitions):
        print('{}/{}'.format(i, nrepetitions))

        # init goal
        if RAND_GOAL:
            goal = np.array([np.random.randint(0, high=M), np.random.randint(0, high=N)])

        # homing
        optimal_step_curve, learning_curve, weights = homing_nn(nTrials,nSteps,learningRate,epsilon,gamma,lam, goal)

        # trace
        learning_curves[i,:] = learning_curve
        extra_steps_curves[i,:] = learning_curve - optimal_step_curve
        draw_preferred_direc(weights, goal, walls, save_path='./preferred_direc/{}.png'.format(i))

    draw_learning_curves(learning_curves, save_path='./learning_curve/learning_curve.png')
    save_learning_curves(learning_curves, save_path='./learning_curve/learning_curve.csv')
    # draw_learning_curves(extra_steps_curves, save_path='./learning_curve/extra_steps_curves.png')
    # save_learning_curves(extra_steps_curves, save_path='./learning_curve/extra_steps_curves.csv')

    # calc mean extra steps
    extra_steps_over_runs = np.average(extra_steps_curves, axis=1)
    average_extra_steps = np.average(extra_steps_over_runs)
    std_extra_steps = np.std(extra_steps_over_runs)
    print('[extra_steps] mean: {}, std:{}'.format(average_extra_steps, std_extra_steps))
