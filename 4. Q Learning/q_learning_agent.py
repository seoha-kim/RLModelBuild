import numpy as np
import random
from collections import defaultdict
from environment import Env

class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.9
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def learn(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        q_2 = reward + self.discount_factor * max(self.q_table[next_state])  # Ballman optimizataion equation
        self.q_table[state][action] += self.learning_rate * (q_2 - q_1)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)  # random
        else:
            state_action = self.q_table[state]
            action = self.arg_max(state_action)  # q-function
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)


if __name__ == "__main__":
    env = Env()
    agent = QLearningAgent(actions=list(range(env.n_actions)))
    for episode in range(1000):
        state = env.reset()
        while True:
            env.render()

            action = agent.get_action(str(state))
            next_state, reward, done = env.step(action)
            agent.learn(str(state), action, reward, str(next))
            state = next_state
            env.print_value_all(agent.q_table)
            if done:
                break

# https://github.com/rlcode/reinforcement-learning-kr-v2