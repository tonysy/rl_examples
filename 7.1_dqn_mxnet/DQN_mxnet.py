"""
DQN with MXNet
"""
import numpy as np 
import gym

# Hyper Params
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
DISCOUNT_FACTOR = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000

env = gym.make('CartPole-v0')
env = env.unwrapped

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]

class Net