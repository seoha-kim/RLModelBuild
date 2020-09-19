import numpy as np
from environment import GraphicDisplay, Env

class PolicyIteration:
    def __init__(self, env):
        self.env = env # initialize environment
        self.value_table = [[0.0]*env.width for _ in range(env.height)] # initialize value function (2D list)
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]]*env.width for _ in range(env.height)] # initialize policy
        self.policy_table[2][2] = [] # final state
        self.discount_factor = 0.9 # discount factor

    # policy evaluation to calculate next value function through Bellman expectation equation
    def policy_evaluation(self):
        # initialize next value function
        next_value_table = [[0.0]*self.env.width for _ in range(self.env.height)]
        # calculate Bellman expectation equation for all states
        for state in self.env.get_all_states():
            value = 0.0
            # value function of final state (=0)
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = value
                continue

            # Bellman expectation equation
            for action in self.env.possible_actions:
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value += (self.get_policy(state)[action] * (reward + self.discount_factor * next_value))
            next_value_table[state[0]][state[1]] = value
        self.value_table = next_value_table

    # Greedy policy improvement for value function now
    def policy_improvement(self):
        next_policy = self.policy_table
        for state in self.env.get_all_states():
            if state == [2, 2]:
                continue

            value_list = []
            # initialize policy to return
            result = [0.0, 0.0, 0.0, 0.0]

            # calculate [reward + (discount factor * value function for next state]
            for index, action in enumerate(self.env.possible_actions):
                next_state = self.env.state_after_action(state, action)
                reward = self.env.get_reward(state, action)
                next_value = self.get_value(next_state)
                value = reward + self.discount_factor * next_value
                value_list.append(value)

            # Greedy policy improvement for actions to maximize rewards
            max_idx_list = np.argwhere(value_list == np.amax(value_list))
            max_idx_list = max_idx_list.flatten().tolist()
            prob = 1 / len(max_idx_list)
            for idx in max_idx_list:
                result[idx] = prob
            next_policy[state[0]][state[1]] = result
        self.policy_table = next_policy

    # return random actions for specific states and policies
    def get_action(self, state):
        policy = self.get_policy(state)
        policy = np.array(policy)
        return np.random.choice(4, 1, p=policy)[0]

    # return policy for states
    def get_policy(self, state):
        return self.policy_table[state[0]][state[1]]

    # return value of value functions
    def get_value(self, state):
        return self.value_table[state[0]][state[1]]


if __name__ == "__main__":
    env = Env()
    policy_iteration = PolicyIteration(env)
    grid_world = GraphicDisplay(policy_iteration)
    grid_world.mainloop()

# https://github.com/rlcode/reinforcement-learning-kr-v2 참고