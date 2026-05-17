import random
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class QLearningAgent:
    def __init__(
        self,
        n_actions:       int,
        alpha:           float = 0.1,
        gamma:           float = 0.95,
        epsilon_start:   float = 1.0,
        epsilon_end:     float = 0.05,
        epsilon_decay:   float = 0.999,
        optimistic_init: float = 0.0,
    ):
        self.n_actions       = n_actions
        self.alpha           = alpha
        self.gamma           = gamma
        self.epsilon         = epsilon_start
        self.epsilon_end     = epsilon_end
        self.epsilon_decay   = epsilon_decay
        self.optimistic_init = optimistic_init
        # setting the initial Q value to zero
        self.Q: Dict[Tuple, float] = defaultdict(lambda: self.optimistic_init)
        self.episode_count = 0


# the valid actions will be provided by the environment 

    def select_action(self, state: Tuple, valid_actions: List[int]) -> int:
        if not valid_actions:
            raise ValueError("No valid actions available.")

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        return self._greedy_action(state, valid_actions)

    def _greedy_action(self, state: Tuple, valid_actions: List[int]) -> int:
        best_idx    = valid_actions[0]
        best_value  = self.Q[(state, valid_actions[0])]
        for idx in valid_actions[1:]:
            val = self.Q[(state, idx)]
            if val > best_value:
                best_value = val
                best_idx   = idx

        return best_idx

    def update(
        self,
        state:        Tuple,
        action_idx:   int,
        reward:       float,
        next_state:   Tuple,
        next_valid:   List[int],
        done:         bool,
    ):

        current_q = self.Q[(state, action_idx)]
        if done or not next_valid:
            target = reward
        else:
            best_next_q = max(self.Q[(next_state, a)] for a in next_valid)
            target      = reward + self.gamma * best_next_q

        self.Q[(state, action_idx)] += self.alpha * (target - current_q)


    def decay_epsilon(self):
        """Call once per episode after the episode ends."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episode_count += 1


    def q_table_size(self) -> int:
        return len(self.Q)

    def best_q_for_state(self, state: Tuple, valid_actions: List[int]) -> float:
        if not valid_actions:
            return 0.0
        return max(self.Q[(state, a)] for a in valid_actions)