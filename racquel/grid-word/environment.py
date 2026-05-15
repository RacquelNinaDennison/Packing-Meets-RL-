class Environment:
    def __init__(self, grid_config={}):
      self.env_name = "GridWorld"
      self.grid = grid_config.get("grid", None)
      self.agent_position = grid_config.get("agent_position", None)
      self.reward_config = grid_config.get("reward_config", None)
      self.obstacle_positions = grid_config.get("obstacle_positions", None)
      self.goal_position = grid_config.get("goal_position", None)
      self.initial_position = self.agent_position

    def get_state(self):
        return self.agent_position
        
    def get_reward(self):
        if(self.is_terminal()):
            return self.reward_config.get("terminal_reward", 0)
        elif(self.agent_position in self.obstacle_positions):
            return self.reward_config.get("obstacle_reward", 0)
        else:
            return self.reward_config.get("step_reward", 0)

    def is_terminal(self):
        return self.agent_position == self.goal_position

    def reset(self):
        self.agent_position = self.initial_position

    def step(self, action):
        if(action == "up"):
            new_position = (self.agent_position[0] - 1, self.agent_position[1])
        elif(action == "down"):
            new_position = (self.agent_position[0] + 1, self.agent_position[1])
        elif(action == "left"):
            new_position = (self.agent_position[0], self.agent_position[1] - 1)
        elif(action == "right"):
            new_position = (self.agent_position[0], self.agent_position[1] + 1)
        if(new_position[0] < 0 or new_position[0] >= len(self.grid) or new_position[1] < 0 or new_position[1] >= len(self.grid[0])):
            return self.agent_position, self.get_reward(), False
        if new_position in self.obstacle_positions:
            return self.agent_position, self.get_reward(), False
        else:
            self.agent_position = new_position
            if(self.is_terminal()):
                return self.agent_position, self.get_reward(), True
            else:
                return self.agent_position, self.get_reward(), False