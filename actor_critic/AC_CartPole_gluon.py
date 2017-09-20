"""Actor-Critic using TD-error as the Advantage, Reinforcement learning.

The cartpole example, Policy is oscillated.
"""
import numpy as np 
import mxnet as mx
import gym


from mxnet import nd, autograd
from mxnet import gluon

import pdb

np.random.seed(2)

# HyperParameters
MAX_EPISODE = 5000
DISPLAY_REWARD_THRESHOLD = 200
MAX_EP_STEPS = 1000 # maximum time step in one episode
RENDER = False
GAMMA = 0.9
LR_A = 0.0005
LR_C = 0.005

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

ctx = mx.cpu()

class Actor(object):
    def __init__(self, n_features, n_actions, lr=0.001):
        self.loss = gluon.loss.L2Loss()
        self.optimizer = 'adam'
        self.lr = lr
        self.batch_size = 1
    
        self.actor_net = gluon.nn.Sequential()
        with self.actor_net.name_scope():
            self.actor_net.add(gluon.nn.Dense(20, in_units=n_features, activation='relu'))
            self.actor_net.add(gluon.nn.Dense(n_actions, in_units=20))
        self.actor_net.initialize()

        # self.actor_net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
        self.trainer = gluon.Trainer(self.actor_net.collect_params(), 'adam', {'learning_rate': self.lr})

    def learn(self, s, a, td_error):
        s = s[np.newaxis, :]
        
        # out_log_softmax = mx.nd.log_softmax(self.actor_net(s))
        # exp_v = mx.nd.mean(out_log_softmax * td_error)

        s_data = nd.array(s).as_in_context(ctx)
        with autograd.record():
            out_softmax = mx.nd.softmax(self.actor_net(s_data))
            log_prob = mx.nd.log(out_softmax)[0, a]
            exp_v = mx.nd.mean(log_prob * td_error)
            loss = -exp_v
        loss.backward()
        self.trainer.step(self.batch_size)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = mx.nd.softmax(self.actor_net(mx.nd.array(s))).asnumpy()
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())

class Critic(object):
    def __init__(self,n_features, lr=0.01):
        self.loss = gluon.loss.L2Loss()
        self.optimizer = 'adam'
        self.lr = lr
        self.batch_size = 1

    
        self.critic_net = gluon.nn.Sequential()
        with self.critic_net.name_scope():
            self.critic_net.add(gluon.nn.Dense(20, in_units=n_features, activation='relu'))
            self.critic_net.add(gluon.nn.Dense(1, in_units=20))
        self.critic_net.initialize()
        # self.critic_net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
        self.trainer = gluon.Trainer(self.critic_net.collect_params(), 'adam', {'learning_rate': self.lr})

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        s_data = nd.array(s).as_in_context(ctx)
        r_data = nd.array([r]).as_in_context(ctx)
        s__data = nd.array(s_).as_in_context(ctx)

        with autograd.record():
            v_ = self.critic_net(s__data)
            td_error = r_data + GAMMA * v_ - self.critic_net(s_data)
            
            loss = mx.nd.square(td_error)
        loss.backward()
        self.trainer.step(self.batch_size)
        return td_error

actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(n_features=N_F, lr=LR_C)

        
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
            
            



        
    
