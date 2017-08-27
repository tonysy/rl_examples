import gym
from rl_brain import DeepQNetwork

env = gym.make("MountainCar-v0")
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.001,
                  e_greedy=0.9,
                  replace_target_iter=300,
                  memory_size=300,
                  e_greedy_increment=0.0002,)

total_step = 0

print(env.action_space)
print(env.action_space.n)