"""Part D1 driver: feed the SAME saved games, in the SAME order, to the three
fixed-policy evaluators and save estimates after 100, 1,000 and 10,000 games.

Run:  python d1_estimate.py
Needs: outputs/fixed_policy_logs/games_seed<seed>.csv  (from d1_collect.py)
Output: outputs/fixed_policy_estimates/seed<seed>_games<N>.csv   (long format)
        outputs/fixed_policy_estimates/model_based_convergence.csv

This file only does input/output plumbing; the learning is in learners.py.
Missing estimate = empty value cell (never a filled-in zero).
"""
import ast, csv, os
from collections import defaultdict
from common import END, fixed_policy
from learners import ModelBasedEvaluator, MonteCarloEvaluator, TD0Evaluator

SEEDS = [11, 22, 33]
CHECKPOINTS = [100, 1_000, 10_000]
LOG_DIR = os.path.join("outputs", "fixed_policy_logs")
OUT_DIR = os.path.join("outputs", "fixed_policy_estimates")

def load_games(seed):
    """Return (list of games, number of excluded cut-off games).
    Each game is a list of (s, a, r, s_next) with s_next == END at the end."""
    games, order, cutoff_ids = {}, [], set()
    with open(os.path.join(LOG_DIR, f"games_seed{seed}.csv")) as f:
        for row in csv.DictReader(f):
            g = int(row["game"])
            if g not in games:
                games[g] = []; order.append(g)
            nxt = END if row["next_obs"] == END else ast.literal_eval(row["next_obs"])
            games[g].append((ast.literal_eval(row["obs"]), int(row["action"]),
                             float(row["reward"]), nxt))
            if int(row["cutoff"]):
                cutoff_ids.add(g)
    # cut-off games are not complete returns: exclude them from every learner
    return [games[g] for g in order if g not in cutoff_ids], len(cutoff_ids)

def run_model(mb):
    """Works with evaluate() returning V or (V, converged, sweeps), or raising."""
    try:
        res = mb.evaluate(fixed_policy)
    except RuntimeError as e:
        return {}, False, None, str(e)
    if isinstance(res, tuple):
        V, converged, sweeps = res
    else:
        V, converged, sweeps = res, True, None
    return V, converged, sweeps, ""

def save_checkpoint(seed, n, mb, mc, td, visits, conv_rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    V_mb, converged, sweeps, err = run_model(mb)
    conv_rows.append([seed, n, converged, sweeps, err])
    path = os.path.join(OUT_DIR, f"seed{seed}_games{n}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "player_sum", "dealer", "usable_ace", "value", "visits"])
        for s in sorted(visits):
            for name, V in (("model_based", V_mb), ("monte_carlo", mc.V), ("td0", td.V)):
                v = V.get(s)                      # None -> empty cell = missing estimate
                w.writerow([name, s[0], s[1], s[2], "" if v is None else repr(v), visits[s]])
    print(f"seed {seed}: saved checkpoint after {n} games -> {path} "
          f"(model converged={converged}, sweeps={sweeps})")

def main():
    conv_rows = []
    for seed in SEEDS:
        games, excluded = load_games(seed)
        if excluded:
            print(f"seed {seed}: excluded {excluded} cut-off games")
        mb, mc, td = ModelBasedEvaluator(), MonteCarloEvaluator(), TD0Evaluator()
        visits = defaultdict(int)                 # nonterminal visits per state
        for n, game in enumerate(games, start=1):
            for s, a, r, s_next in game:
                mb.update(s, a, r, s_next)
                td.update(s, r, s_next)
                visits[s] += 1
            mc.add_game(game)
            if n in CHECKPOINTS:
                save_checkpoint(seed, n, mb, mc, td, visits, conv_rows)
    with open(os.path.join(OUT_DIR, "model_based_convergence.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "games", "converged", "sweeps", "error"])
        w.writerows(conv_rows)

if __name__ == "__main__":
    main()
