"""
DQN with Pytorch
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np 
import gym
import pdb

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

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_space = self.out(x)
        return action_space

class DQN(object):

    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.loss_list = []
    def choose_actions(self, observation):
        observation = Variable(torch.unsqueeze(torch.FloatTensor(observation), 0))
        if np.random.uniform() < EPSILON:
            action_value = self.eval_net.forward(observation)
            print(action_value)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
            print(action)
           
        else:
            action = np.random.randint(0, N_ACTIONS)
        
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a,r], s_))

        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)

        # pdb.set_trace()
        q_next = self.target_net(b_s_).detach() # detach from graph, don't backpropagate
    
        q_target = b_r + GAMMA * q_next.max(1)[0]
        
        loss = self.loss_func(q_eval, q_target)
        print loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.loss_list)), self.loss_list)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
 
dqn = DQN()
for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_actions(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
        if done:
            break
        
        s = s_


