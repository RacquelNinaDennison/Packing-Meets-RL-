import random
from collections import deque
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from modules.environment import MultiBatchingEnv
from modules.agent import QLearningAgent


def run_episode(env: MultiBatchingEnv, agent: QLearningAgent) -> Dict:
    state   = env.reset()
    total_r = 0.0
    steps   = 0

    while not env.done:
        valid  = env.valid_action_indices()
        action = agent.select_action(state, valid)
        next_s, reward, done = env.step(action)
        next_valid = env.valid_action_indices() if not done else []
        agent.update(state, action, reward, next_s, next_valid, done)
        state    = next_s
        total_r += reward
        steps   += 1

    summary = env.summary()
    summary["total_reward"] = total_r
    summary["steps"]        = steps
    return summary


def train(
    n_episodes:        int   = 10000,
    alpha:             float = 0.3,
    gamma:             float = 0.99,
    epsilon_start:     float = 1.0,
    epsilon_end:       float = 0.05,
    epsilon_decay:     float = 0.9995,
    lam:               float = 200.0,
    mu:                float = 100.0,
    feasibility_bonus: float = 500.0,
    log_every:         int   = 500,
    seed:              int   = 42,
):
    random.seed(seed)
    env   = MultiBatchingEnv(lam=lam, mu=mu, feasibility_bonus=feasibility_bonus)
    agent = QLearningAgent(
        n_actions      = env.n_actions,
        alpha          = alpha,
        gamma          = gamma,
        epsilon_start  = epsilon_start,
        epsilon_end    = epsilon_end,
        epsilon_decay  = epsilon_decay,
        optimistic_init= 0.0,
    )

    best_cost = float("inf")
    best_sol  = None
    logs      = []
    window_f  = deque(maxlen=log_every)
    window_c  = deque(maxlen=log_every)

    print(f"Training: {n_episodes} episodes | "
          f"α={alpha} γ={gamma} | lam={lam} mu={mu} bonus={feasibility_bonus}")
    print(f"Actions: {env.n_actions}\n")
    print(f"{'Episode':>8}  {'Feasible%':>10}  {'Avg Cost':>10}  "
          f"{'Best Cost':>10}  {'Epsilon':>8}  {'Q-entries':>10}")

    for ep in range(1, n_episodes + 1):
        log = run_episode(env, agent)
        agent.decay_epsilon()
        logs.append(log)

        window_f.append(int(log["feasible"]))
        if log["feasible"]:
            window_c.append(log["total_cost"])
            if log["total_cost"] < best_cost:
                best_cost = log["total_cost"]
                best_sol  = log

        if ep % log_every == 0:
            feas_pct = 100 * sum(window_f) / len(window_f)
            avg_cost = sum(window_c) / len(window_c) if window_c else float("inf")
            print(f"{ep:>8}  {feas_pct:>9.1f}%  {avg_cost:>10.1f}  "
                  f"{best_cost:>10.1f}  {agent.epsilon:>8.4f}  "
                  f"{agent.q_table_size():>10}")

    _print_solution(best_sol, best_cost)
    return agent, logs, best_sol


def _print_solution(sol: Optional[Dict], best_cost: float):
    print(f"Best cost found : {best_cost:.1f}")
    if sol is None:
        print("No feasible solution found.")
        return

    print("\nLoad assignments:")
    for (frm, to, tr, part), qty in sorted(sol["load"].items()):
        if qty > 0:
            print(f"  {frm} -> {to} via {tr}  [{part}] : {qty} units")

    print("\nFrequencies (dispatches):")
    for (frm, to, tr), freq in sorted(sol["frequencies"].items()):
        if freq > 0:
            print(f"  {frm} -> {to} via {tr} : {freq}")

    print(f"\nFeasible        : {sol['feasible']}")
    print(f"Flow violations : {sol['flow_violation']}")
    print(f"Unmet demand    : {sol['unmet_demand']}")


def plot_curves(logs: List[Dict], window: int = 100):
    """Plot learning curve (reward) and step curve."""
    rewards = [l["total_reward"] for l in logs]
    steps = [l["steps"] for l in logs]
    episodes = range(1, len(logs) + 1)


    def smooth(values, w):
        if len(values) < w:
            w = len(values)
        return np.convolve(values, np.ones(w) / w, mode="valid")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)


    ax1.plot(episodes, rewards, alpha=0.3, color="blue", label="Raw")
    smoothed_r = smooth(rewards, window)
    ax1.plot(range(window, window + len(smoothed_r)), smoothed_r,
             color="blue", linewidth=2, label=f"Rolling avg ({window})")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Learning Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)


    ax2.plot(episodes, steps, alpha=0.3, color="green", label="Raw")
    smoothed_s = smooth(steps, window)
    ax2.plot(range(window, window + len(smoothed_s)), smoothed_s,
             color="green", linewidth=2, label=f"Rolling avg ({window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps per Episode")
    ax2.set_title("Step Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    feasible = [int(l["feasible"]) for l in logs]
    smoothed_f = smooth(feasible, window)
    ax3.plot(range(window, window + len(smoothed_f)), smoothed_f,
             color="red", linewidth=2)
    ax3.set_ylabel("Feasible %")
    ax3.set_title("Feasibility Rate")
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()

    


if __name__ == "__main__":
    agent, logs, best_sol = train()
    plot_curves(logs)