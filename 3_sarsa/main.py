from maze_env import Maze
from rl_brain import QLearningTabel

def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        # RL choose action base on observation
        action = RL.choose_action(str(observation))

        while True:
            # refresh env
            env.render()

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL choose action based on next observation
            action_ = RL.choose_action(str(observation_))
            
            # RL learn from this transition(s, a, r, s_, a_) ===> Sarsa
            RL.learn(str(observation), action, reward, str(observation_), action_)

            observation = observation_
            action = action_

            if done:
                break

    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTabel(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
