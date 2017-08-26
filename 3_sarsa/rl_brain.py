import numpy as np 
import pandas as pd 

class  RL(object):
	"""docstring for  RL"""
    def __init__(self, actions, lr=0.01, reward_decay=0.9, e_greedy=0.9):
        uper( RL, self).__init__()
        self.actions = actions
        self.lr = lr
        self.discount_factor = reward_decay
        self.epsilon = e_greedy
        
        self.q_table = pd.DataFrame(columns=self.actions)
    
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.discount_factor * self.q_table.ix[s_,:].max()
        else:
            q_target = r

        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    


# off-policy
class QLearningTable(RL):
	"""docstring for QLearningTable"""
	def __init__(self, arg):
		super(QLearningTable, self).__init__()
		self.arg = arg

# on-policy
class SarsaTable(object):
	"""docstring for SarsaTable"""
	def __init__(self, arg):
		super(SarsaTable, self).__init__()
		self.arg = arg
