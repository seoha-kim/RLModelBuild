import numpy as np
import random
from collections import defaultdict
from environment import Env


class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # q-function update
    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_state_q = self.q_table[next_state][next_action]
        new_q = (current_q + self.learning_rate * (reward + self.discount_factor * next_state_q - current_q))
        self.q_table[state][action] = new_q

    # epsilon greedy policy
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
    agent = SARSAgent(actions=list(range(env.n_actions)))
    for episode in range(1000):
        state = env.reset()  # initialize game environment / state
        action = agent.get_action(str(state))  # choose action for now states
        while True:
            env.render()
            next_state, reward, done = env.step(action)  # next state, reward, episode end
            next_action = agent.get_action(str(next_state))  # choose action for next states
            agent.learn(str(state), action, reward, str(next_state), next_action)  # q-function update

            state = next_state
            action = next_action
            env.print_value_all(agent.q_table)
            if done:
                break

# https://github.com/rlcode/reinforcement-learning-kr-v2