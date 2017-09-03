
# coding: utf-8

# In[1]:

import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
import gym


# In[2]:

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
DISCOUNTOR_FACTOR = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000


# In[4]:

# env

env = gym.make('CartPole-v0')
env = env.unwrapped

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ctx = mx.cpu()


# In[17]:

# Net
def Net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(10,activation="relu"))
        net.add(gluon.nn.Dense(N_ACTIONS))
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    return net


# In[24]:

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = 'sgd'
        self.loss = gluon.loss.L2Loss()
        
    def choose_action(self, observation):
        observation = nd.array(observation).reshape((1,4))

        if np.random.uniform() < EPSILON:
            action_value = self.eval_net(observation)
           
            action = nd.argmax(action_value, axis=1)[0].astype(int).asnumpy()[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        
        return action
    
    def store_transition(self, s, a, r, s_):
        # print(s,a,r,s_)
        # s = nd.array(s).reshape((1,))
        # a = nd.array([a])#.reshape((1,))
        # r = nd.array([r])#.reshape((1,))
        # s_ = nd.array(s_).reshape((1,))
        # print(s,a,r,s_)
        transition = np.hstack((s, [a, r], s_))
    
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.initialize(self.eval_net.collect_params())
            # pass
        
        self.learn_step_counter += 1
        
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = nd.array(b_memory[:, :N_STATES])
        b_a = nd.array(b_memory[:, N_STATES:N_STATES+1])
        b_r = nd.array(b_memory[:, N_STATES+1])
        b_s_ = nd.array(b_memory[:, -N_STATES:])
        
        q_next = self.target_net(b_s_)
        q_target = b_r + DISCOUNTOR_FACTOR * nd.max_axis(q_next,axis=1)
        eval_trainer = gluon.Trainer(self.eval_net.collect_params(), 'sgd', {'learning_rate': 0.01})
        
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(b_s, q_target), batch_size=32, shuffle=True)
        for data, label in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            
            with autograd.record():
                q_eval = self.eval_net(data)
                q_eval = nd.pick(q_eval, b_a, 1)

                loss = self.loss(q_eval, label)
            
            loss.backward()
            eval_trainer.step(32)


# In[25]:

dqn = DQN()


# In[26]:

for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
	env.render()
        a = dqn.choose_action(s)

        s_, r, done, info = env.step(a)

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
                      '| Ep_reward: ', round(ep_r, 2))

        if done:
            break

        s = s_


# In[ ]:



