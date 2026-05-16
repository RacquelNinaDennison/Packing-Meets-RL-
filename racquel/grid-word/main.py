from environment import Environment
from agent import Agent
import matplotlib.pyplot as plt

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
            "step_reward": -0.02,
            "terminal_reward": 100,
            "obstacle_reward": -10,
            "boundary_reward": -5,
        }
    })
    agent = Agent(grid_size=(5, 5))
    episode_rewards = []
    episode_steps = []

    for episode in range(1000):
        environment.reset()
        state = environment.get_state()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < 200:
            action = agent.get_action(state)
            next_state, reward, done = environment.step(action)
            agent.update_policy(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        if episode % 100 == 0:
            print(f"Episode {episode}: Steps = {steps}, Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")

    print("\nQ-table best actions per cell:")
    actions = ["up", "down", "left", "right"]
    arrows = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
    for r in range(5):
        row = ""
        for c in range(5):
            best = actions[agent.q_table[r, c].argmax()]
            row += f" {arrows[best]} "
        print(row)

    plot_results(episode_rewards, episode_steps)

def plot_results(episode_rewards, episode_steps):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(episode_rewards)
    ax1.set_title("Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")

    ax2.plot(episode_steps)
    ax2.set_title("Steps per Episode (Learning Curve)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")

    plt.tight_layout()
    plt.savefig("learning_curves.png")
    plt.show()

if __name__ == "__main__":
    main()
