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

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001
LR_C = 0.002

GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'

ctx = mx.cpu()

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound

        # self.actor_net, self.actor_trainer = self._build_actor(lr_a)
        # self.critic_net, self.critic_trainer = self._build_critic()

        self.actor_net_eval, self.actor_trainer_eval = self._build_actor(LR_A)
        self.actor_net_target, self.actor_trainer_target = self._build_actor(LR_A)

        self.critic_net_eval, self.critic_trainer_eval = self._build_critic(LR_C)
        self.critic_net_target, self.critic_trainer_target = self._build_critic(LR_C)

    def _build_actor(self, lr):
        actor_net = gluon.nn.Sequential()
        with actor_net.name_scope():
            actor_net.add(gluon.nn.Dense(30, in_units=self.s_dim, activation='relu'))
            actor_net.add(gluon.nn.Dense(self.a_dim, in_units=30, activation='tanh'))
        actor_net.initialize()
        actor_trainer = gluon.Trainer(actor_net.collect_params(), 'adam', {'learning_rate': lr})
        return actor_net, actor_trainer

    def _build_critic(self, lr):
        critic_net = gluon.nn.Sequential()
        with critic_net.name_scope():
            critic_net.add(gluon.nn.Dense(1, in_units=30))
        critic_net.initialize()
        critic_trainer = gluon.Trainer(critic_net.collect_params(), 'adam', {'learning_rate': lr})
        return critic_net, critic_trainer
    
    def _get_critic_output(self, net, s, a):
        w1_s = nd.random_normal(shape=(self.s_dim, 30))
        w1_a = nd.random_normal(shape=(self.a_dim, 30))
        b1 = nd.zeros((1, 30))
        params = [w1_s, w1_a, b1]
        for param in params:
            param.attach_grad()
        critic_input = nd.relu(nd.dot(s, w1_s) + nd.dot(a, w1_a) + b1)

        output = net(critic_input)
        return output

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch_transitions = self.memory[indices, :]
        batch_s = batch_transitions[:, :self.s_dim]
        batch_a = batch_transitions[:, self.s_dim: self.s_dim + self.a_dim]
        batch_r = batch_transitions[:, -self.s_dim - 1: -self.s_dim]
        batch_s_ = batch_transitions[:, -self.s_dim:]

        # actor_net_eval, actor_trainer_eval = self._build_actor(LR_A)
        # actor_net_target, actor_trainer_target = self._build_actor(LR_A)

        # critic_net_eval, critic_trainer_eval = self._build_critic(LR_C)
        # critic_net_target, critic_trainer_target = self._build_critic(LR_C)

        soft_replace(self.actor_net_eval, self.actor_net_target)
        soft_replace(self.critic_trainer_eval, self.critic_trainer_target)


        batch_s.as_in_context(ctx)
        with autograd.record():
            a = self.actor_net_eval(batch_s)
            a_ = self.actor_net_target(batch_s_)
            
            q = self._get_critic_output(self.critic_net_eval, batch_s, batch_a)
            q_ = self._get_critic_output(batch_s_, a_)
            
            q_target = batch_r + GAMMA * q_

            td_error = gluon.loss.L2Loss(q_target, q)
            a_loss = - nd.mean(q)
            
        td_errors.backward()
        a_loss.backward()
        
        self.actor_trainer_eval.step(BATCH_SIZE)
        self.critic_net_eval.step(BATCH_SIZE)


    
    def soft_replace(self, net_eval, net_target):
        for i in range(len(net_eval)):
            target_param_keys = net_target[i].collect_params().keys()
            eval_param_keys = net_eval[i].collect_params().keys()

            for j in range(len(target_param_keys)):
                eval_data = net_eval[i].collect_params()[eval_param_keys[j]].data()
                target_data = net_target[i].collect_params()[target_param_keys[j]].data()
                replace_data = (1 - TAU) * target_data + TAU * eval_data

                net_target[i].collect_params()[target_param_keys[j]].set_data(replace_data)      

    
    def choose_action(self, s):
        action = self.actor_net_eval(s)[0]

        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1


env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r/10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995
            ddpg.learn()
        
        s = s_
        ep_reward += r 
        if j == MAX_EP_STEPS - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var,)
            if ep_reward > -300: RENDER = True
            break
