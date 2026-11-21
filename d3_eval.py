"""Part D3: evaluate the frozen greedy Q-tables and the fixed baseline on NEW games.

Run:  python d3_eval.py              (10,000 games per policy per eval seed)
      python d3_eval.py 1000         (quick test)
Needs: outputs/qlearning/q_seed<seed>_eps<eps>.csv   (from d2_qlearn.py)
Output: outputs/evaluation/eval_results.csv   one row per (policy, eval seed)
        outputs/evaluation/eval_summary.csv   per policy, summed over the 3 eval seeds

Evaluation rules: greedy action only, no exploration, no learning/update (a policy
is just a lookup table here). Every (policy, eval seed) pair gets a brand-new
environment seeded at its first reset, so each policy starts from the same seed.
Frozen-table rule: if Q(stick) == Q(hit) (a tie), or the state is not in the table
(never seen in training), the policy sticks. Such decisions are counted.
"""
import csv, os, sys
from common import make_env, fixed_policy, obs_key, SAFETY_LIMIT

TRAIN_SEEDS = [11, 22, 33]
EPSILONS = [0.2, 0.0]
EVAL_SEEDS = [1011, 1022, 1033]
N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
Q_DIR = os.path.join("outputs", "qlearning")
OUT_DIR = os.path.join("outputs", "evaluation")


class FrozenGreedyPolicy:
    """Lookup table state -> action built from saved Q-values. Never changes."""

    def __init__(self, path):
        self.table = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                s = (int(row["player_sum"]), int(row["dealer"]), int(row["usable_ace"]))
                self.table[s] = row["greedy"]          # 'hit' / 'stick' / 'tie'
        self.unseen = 0                                 # state not in table
        self.ties = 0                                   # Q(stick) == Q(hit)

    def __call__(self, s):
        lab = self.table.get(s)
        if lab is None:
            self.unseen += 1
            return 0
        if lab == "tie":
            self.ties += 1
            return 0
        return 1 if lab == "hit" else 0


def evaluate(policy, eval_seed, n_games):
    env = make_env()
    wins = losses = draws = cutoffs = 0
    total = 0.0
    for game in range(1, n_games + 1):
        obs, _ = env.reset(seed=eval_seed) if game == 1 else env.reset()
        for _ in range(SAFETY_LIMIT):
            obs, reward, terminated, truncated, _ = env.step(policy(obs_key(obs)))
            if terminated or truncated:
                break
        else:
            cutoffs += 1                                # excluded from the tallies
            continue
        total += reward
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    played = wins + losses + draws
    return wins, losses, draws, cutoffs, (total / played if played else float("nan"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    policies = []                                       # (label, train_seed, eps, policy)
    for eps in EPSILONS:
        for ts in TRAIN_SEEDS:
            path = os.path.join(Q_DIR, f"q_seed{ts}_eps{eps}.csv")
            policies.append((f"q_seed{ts}_eps{eps}", ts, eps, FrozenGreedyPolicy(path)))
    policies.append(("baseline_hit_below_20", "", "", fixed_policy))

    rows = []
    for label, ts, eps, pol in policies:
        for es in EVAL_SEEDS:
            u0, t0 = getattr(pol, "unseen", 0), getattr(pol, "ties", 0)
            w, l, d, c, mean_r = evaluate(pol, es, N_GAMES)
            unseen = getattr(pol, "unseen", 0) - u0
            ties = getattr(pol, "ties", 0) - t0
            rows.append([label, ts, eps, es, N_GAMES, w, l, d, c, repr(mean_r), unseen, ties])
            print(f"{label:24s} eval_seed {es}: W={w} L={l} D={d} cut={c} "
                  f"mean reward={mean_r:+.4f} (unseen={unseen}, ties={ties})")

    header = ["policy", "train_seed", "train_epsilon", "eval_seed", "games", "wins",
              "losses", "draws", "cutoffs", "mean_reward", "unseen_decisions", "tie_decisions"]
    with open(os.path.join(OUT_DIR, "eval_results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

    # summary per policy over the three eval seeds
    summary = {}
    for r in rows:
        s = summary.setdefault(r[0], [r[1], r[2], 0, 0, 0, 0])
        s[2] += r[5]; s[3] += r[6]; s[4] += r[7]; s[5] += r[4] - r[8]
    with open(os.path.join(OUT_DIR, "eval_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "train_seed", "train_epsilon", "games", "wins", "losses",
                    "draws", "win_rate", "mean_reward"])
        print("\nSummary (3 eval seeds pooled):")
        for label, (ts, eps, wi, lo, dr, n) in summary.items():
            mean_r = (wi - lo) / n
            w.writerow([label, ts, eps, n, wi, lo, dr, wi / n, mean_r])
            print(f"  {label:24s} W={wi} L={lo} D={dr}  win rate={wi/n:.4f}  mean reward={mean_r:+.4f}")


if __name__ == "__main__":
    main()
