from __future__ import print_function
import numpy as np
import pandas as pd
import time

np.random.seed(2) # reproducible

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9
ALPHA = 0.1 # lr
LAMDA = 0.9 # discount fator
MAX_EPISODES = 13 # maximum epoch
FRESH_TIME = 0.01 # control vectity

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns = actions, # action's name
    )
    # print(table)
    return table

# build_q_table(N_STATES, ACTIONS)

def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0): # act non-greedy or state-action
        action_name = np.random.choice(ACTIONS)
    else: # act greedy
        action_name = state_actions.argmax()
    return action_name

def get_env_feedback(S, A):
    # This is how agent will interact with the environment
    if A == 'right': # move right
        if S == N_STATES - 2:# terminate
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else: # move left
        R = 0
        if S == 0:
            S_ = S # reach the wall
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction))
        print('\r                              ')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction))
        time.sleep(FRESH_TIME)

def q_learning():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.ix[S, A]
            if S_ != 'terminal':
                q_target = R + LAMDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True

            q_table.ix[S, A] += ALPHA*(q_target - q_predict)
            S = S_

            update_env(S, episode, step_counter+1)
            step_counter += 1
    return q_table

if __name__ == '__main__':
    q_table = q_learning()
    print('\r\nQ-table:\n')
    print(q_table)
