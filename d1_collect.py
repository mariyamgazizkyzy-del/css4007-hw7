"""Part D1: collect games with the fixed hit-below-20 policy and log every step.

Run:  python d1_collect.py            (10,000 games for seeds 11, 22, 33)
Output: outputs/fixed_policy_logs/games_seed<seed>.csv
"""
import csv, os, sys
from common import make_env, fixed_policy, obs_key, END, SAFETY_LIMIT

SEEDS = [11, 22, 33]
N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
OUT_DIR = os.path.join("outputs", "fixed_policy_logs")
HEADER = ["seed", "game", "step", "obs", "action", "reward", "next_obs", "terminal", "cutoff"]

def collect(seed, n_games):
    env = make_env()
    rows, cutoffs = [], 0
    for game in range(1, n_games + 1):
        # seed only at the very first reset; later resets continue the random sequence
        obs, _ = env.reset(seed=seed) if game == 1 else env.reset()
        game_rows, cut = [], False
        for step in range(1, SAFETY_LIMIT + 1):
            action = fixed_policy(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            nxt = END if done else obs_key(next_obs)   # terminal successor -> END
            game_rows.append([seed, game, step, obs_key(obs), action, reward, nxt, int(done), 0])
            obs = next_obs
            if done:
                break
        else:
            cut = True                                 # 100 actions without ending
        if cut:
            cutoffs += 1
            for r in game_rows:
                r[-1] = 1                              # mark so learners can exclude it
        rows.extend(game_rows)
    return rows, cutoffs

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for seed in SEEDS:
        rows, cutoffs = collect(seed, N_GAMES)
        path = os.path.join(OUT_DIR, f"games_seed{seed}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)
        print(f"seed {seed}: {N_GAMES} games, {len(rows)} transitions, cut-offs={cutoffs} -> {path}")
