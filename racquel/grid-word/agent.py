import random
import numpy as np


class Agent:
    def __init__(self, grid_size=(5, 5)):
        self.actions = ["up", "down", "left", "right"]
        self.q_table = np.zeros(shape=(grid_size[0], grid_size[1], len(self.actions)), dtype=np.float32)
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.alpha = 0.1
        self.gamma = 0.99

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return self.actions[np.argmax(self.q_table[state])]

    def update_policy(self, state, action, reward, next_state, done):
        action_idx = self.actions.index(action)
        current_q = self.q_table[state][action_idx]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action_idx] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
