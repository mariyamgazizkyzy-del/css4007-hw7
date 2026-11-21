"""Part D2: train epsilon-greedy Q-learning agents on Blackjack-v1.

Run:  python d2_qlearn.py            (50,000 games per run)
      python d2_qlearn.py 2000       (quick test with fewer games)

Runs: seeds 11, 22, 33  x  epsilon 0.2 and 0.0  (6 runs).
Every run gets a brand-new QLearner(epsilon, seed) and a brand-new environment
(made with make_env()); the environment is seeded only at the very first reset,
later resets continue the same random sequence.

Output (all in outputs/qlearning/):
  learning_curve.csv                   mean training reward per 1,000-game block
  q_seed<seed>_eps<eps>.csv            final Q-values + visit counts, one row per state
                                       (also used as the frozen policy by d3_eval.py)
  greedy_seed<seed>_eps<eps>_noace.csv greedy hit/stick table, no usable ace
  greedy_seed<seed>_eps<eps>_ace.csv   greedy hit/stick table, usable ace
Training reward = total reward of each game (+1 win, 0 draw, -1 loss) INCLUDING
the exploration moves, so the epsilon=0.2 curve is not the greedy performance.
"""
import csv, os, sys
from common import make_env, obs_key, END, SAFETY_LIMIT
from learners import QLearner

SEEDS = [11, 22, 33]
EPSILONS = [0.2, 0.0]
N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
BLOCK = 1_000
OUT_DIR = os.path.join("outputs", "qlearning")

DEALER_CARDS = list(range(1, 11))          # 1 = ace, ... , 10
SUMS_NOACE = list(range(4, 22))            # 4..21
SUMS_ACE = list(range(12, 22))             # 12..21


def greedy_label(q_stick, q_hit):
    """Greedy action from Q-values. Equal Q-values (incl. never-updated) = 'tie'."""
    if q_stick == q_hit:
        return "tie"
    return "stick" if q_stick > q_hit else "hit"


def train(epsilon, seed, n_games):
    """One run: fresh learner, fresh env. Returns (agent, block_means, cutoffs)."""
    env = make_env()
    agent = QLearner(epsilon=epsilon, seed=seed)
    block_means, block_sum, cutoffs = [], 0.0, 0
    for game in range(1, n_games + 1):
        # seed only at the very first reset; later resets continue the random sequence
        obs, _ = env.reset(seed=seed) if game == 1 else env.reset()
        s = obs_key(obs)
        total = 0.0
        for _ in range(SAFETY_LIMIT):
            a = agent.choose_action(s)
            next_obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s_next = END if done else obs_key(next_obs)   # terminal successor -> END
            agent.update(s, a, r, s_next)
            total += r
            s = s_next
            if done:
                break
        else:
            cutoffs += 1                                  # 100 actions without ending
        block_sum += total
        if game % BLOCK == 0:
            block_means.append(block_sum / BLOCK)
            block_sum = 0.0
    return agent, block_means, cutoffs


def save_q(agent, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["player_sum", "dealer", "usable_ace", "q_stick", "q_hit",
                    "visits_stick", "visits_hit", "greedy"])
        for s in sorted(agent.Q):
            q0, q1 = agent.Q[s][0], agent.Q[s][1]
            w.writerow([s[0], s[1], s[2], repr(q0), repr(q1),
                        agent.visits.get((s, 0), 0), agent.visits.get((s, 1), 0),
                        greedy_label(q0, q1)])


def save_greedy_tables(agent, tag):
    """H = hit, S = stick, '-' = tie / state never seen. Rows: player sum, cols: dealer."""
    for ace, sums, name in ((0, SUMS_NOACE, "noace"), (1, SUMS_ACE, "ace")):
        path = os.path.join(OUT_DIR, f"greedy_{tag}_{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["player_sum"] + [f"dealer_{d}" for d in DEALER_CARDS])
            for ps in sums:
                row = []
                for d in DEALER_CARDS:
                    q = agent.Q.get((ps, d, ace))
                    lab = greedy_label(q[0], q[1]) if q is not None else "tie"
                    row.append({"hit": "H", "stick": "S"}.get(lab, "-"))
                w.writerow([ps] + row)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    curve_rows = []
    for seed in SEEDS:
        for eps in EPSILONS:
            agent, means, cutoffs = train(eps, seed, N_GAMES)
            tag = f"seed{seed}_eps{eps}"
            for b, m in enumerate(means, start=1):
                curve_rows.append([seed, eps, b, b * BLOCK, repr(m)])
            save_q(agent, os.path.join(OUT_DIR, f"q_{tag}.csv"))
            save_greedy_tables(agent, tag)
            last = means[-1] if means else float("nan")
            print(f"seed {seed} eps {eps}: {N_GAMES} games, states={len(agent.Q)}, "
                  f"last-block mean reward={last:.4f}, cut-offs={cutoffs} -> {tag}")
    with open(os.path.join(OUT_DIR, "learning_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "epsilon", "block", "games_end", "mean_reward"])
        w.writerows(curve_rows)


if __name__ == "__main__":
    main()
