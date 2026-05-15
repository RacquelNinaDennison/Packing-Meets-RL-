from environment import Environment

def main():
    environment = Environment({
        "grid": [[0, 0, 0, 0, 2],
                 [0, 0, -1, 0, 0],
                 [0, -1, 0, 0, 0],
                 [0, 0, 0, -1, 0],
                 [0, 0, 0, 0, 0]],
        "agent_position": (4, 0),
        "goal_position": (0, 4),
        "obstacle_positions": [(1, 3), (2, 1), (3, 3)],
        "reward_config": {
            "step_reward": -0.01,
            "terminal_reward": 100,
            "obstacle_reward": -10,

        }
    })
    print(environment.get_state())
    print(environment.get_reward())
    print(environment.is_terminal())
    print(environment.reset())
    print(environment.step("up"))


if __name__ == "__main__":
    main()