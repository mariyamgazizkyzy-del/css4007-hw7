# HW7 submission — Blackjack

Public fork: https://github.com/mariyamgazizkyzy-del/css4007-hw7

## AI-use declaration

Claude (Anthropic) was used for: writing `d2_qlearn.py`, `d3_eval.py` and `plots.py` from my specification; writing the PowerShell commands used to copy and run the scripts; reviewing the code and the printed results; and helping to word the explanations in Part D. I ran all programs myself on my machine and the numbers below are copied from the saved outputs in `outputs/`.

**FILL IN / CHECK:** state here which tools helped with Parts A–C and with `common.py`, `learners.py`, `d1_collect.py`, `d1_estimate.py`, `replay.py`. This declaration must describe the help you really used.

## Setup and reproduction

Versions (from `pip install` output): Python 3.14 (Windows, 64-bit), gymnasium 1.4.0, numpy 2.5.3, matplotlib 3.11.2. Environment: `Blackjack-v1`, `natural=False`, `sab=False`. alpha = 0.1, gamma = 1.0, all tables start at zero.

Installation:

```
pip install gymnasium numpy matplotlib
```

Commands, in this order, from the repository folder:

```
python replay.py            # Part C, output saved in outputs/paper-replay/
python d1_collect.py        # Part D1: 10,000 games for seeds 11, 22, 33
python d1_estimate.py       # Part D1: estimates at 100, 1,000, 10,000 games
python d1_tables.py         # Part D1: readable tables and summary from the saved csv files
python d2_qlearn.py         # Part D2: 50,000 training games, 3 seeds x epsilon 0.2 and 0
python d3_eval.py           # Part D3: frozen-policy evaluation, 10,000 new games
python plots.py             # training curves and evaluation plot
```

No secrets or `.env` files are used.

## Part A — Count states and actions

Paper answers and comparison table (Blackjack, Durak, tic-tac-toe, Nim): see `scans/`.

**FILL IN:** which scan files contain Part A (file names), and any corrections marked next to the original work.

## Part B — Paper calculations

Scans of the original paper calculations are in `scans/` (committed in `a8f26d4`, "Paper calculations scan (original, before learning code)", before the learning code was written).

**FILL IN:** which scan covers which method (model-based, Monte Carlo, TD(0), Q-learning). Corrections must be marked without deleting the originals.

## Part C — Reproduce the calculations

Replay output: `outputs/paper-replay/replay_output.txt` (from `python replay.py`, which loads `data/paper-games.csv`).

The "Program result" column is taken from that output. The paper results are in the scans in `scans/`; they are compared with the program results in person (see the last three columns).

| Method | State / action | Sweep / pass / update | Paper result | Program result | Difference | Explanation |
| --- | --- | --- | --- | --- | --- | --- |
| Model-based | A | V1 | see `scans/` | -0.333333 | compared in person | explained in person |
| Model-based | A | V2 | see `scans/` | -0.333333 | compared in person | explained in person |
| Model-based | B, C, D, E | V1 and V2 | see `scans/` | 0.0 each | compared in person | explained in person |
| Monte Carlo | A | every-visit mean (3 returns: 1, -1, -1) | see `scans/` | -0.333333 | compared in person | explained in person |
| Monte Carlo | B | every-visit mean (returns 1, -1) | see `scans/` | 0.0 | compared in person | explained in person |
| Monte Carlo | C, D, E | every-visit mean (1 return each) | see `scans/` | 0.0 each | compared in person | explained in person |
| TD(0) | A | final value after 2 passes | see `scans/` | -0.140030 | compared in person | explained in person |
| TD(0) | B | final value after 2 passes | see `scans/` | -0.018100 | compared in person | explained in person |
| TD(0) | C, D, E | final value after 2 passes | see `scans/` | 0.0 each | compared in person | explained in person |
| Q-learning | (A, stick) / (A, hit) | final Q | see `scans/` | 0.000000 / -0.139220 | compared in person | explained in person |
| Q-learning | (B, stick) / (B, hit) | final Q | see `scans/` | -0.018100 / 0.000000 | compared in person | explained in person |
| Q-learning | C, D, E | final Q | see `scans/` | 0.0 for both actions (greedy = tie) | compared in person | explained in person |

Every individual TD(0) and Q-learning update (pass 1 and 2, games 1–4) is printed in `outputs/paper-replay/replay_output.txt`.

## Part D — Simulated games

### Fixed-policy value estimates

Policy: hit below 20, stick at 20 or 21. For seeds 11, 22 and 33 the same 10,000 saved games, in the same order, were fed to model-based evaluation, every-visit Monte Carlo and TD(0). Cut-off games (100 actions without ending) would be excluded from all learners; there were none.

| Seed | Games collected | Transitions | Cut-offs | Model converged (sweeps at 100 / 1,000 / 10,000 games) |
| --- | --- | --- | --- | --- |
| 11 | 10,000 | 17,958 | 0 | yes (6 / 12 / 11) |
| 22 | 10,000 | 18,036 | 0 | yes (6 / 10 / 11) |
| 33 | 10,000 | 17,912 | 0 | yes (5 / 11 / 11) |

Model evaluation used tolerance 1e-6 and at most 10,000 sweeps. A missing estimate is an empty cell in the csv files (never a filled-in zero).

Files:
- Shared game logs: `outputs/fixed_policy_logs/games_seed11.csv`, `games_seed22.csv`, `games_seed33.csv`
- Checkpoint value tables with visit counts: `outputs/fixed_policy_estimates/seed<seed>_games<100|1000|10000>.csv`
- Model convergence: `outputs/fixed_policy_estimates/model_based_convergence.csv`
- Readable value tables after 10,000 games, usable ace and no usable ace separately: `outputs/fixed_policy_estimates/tables_seed<seed>_ace.md` and `tables_seed<seed>_noace.md`
- Counts of missing estimates, differences between methods, most and least visited states: `outputs/fixed_policy_estimates/d1_summary.md` (made by `d1_tables.py`)

**Coverage and missing estimates.** The number of states that had been visited (and therefore have an estimate) grew with the number of games: 96 / 116 / 118 states after 100 games (seeds 11 / 22 / 33), 243 / 245 / 247 after 1,000 games, and all 280 observations (player sums 4–21 without a usable ace, 12–21 with a usable ace, dealer 1–10) after 10,000 games for every seed. States not yet visited are simply not in the checkpoint files; they are not filled with zeros. Among the visited states, no estimate was missing for any method at any checkpoint (0 missing for model-based, Monte Carlo and TD(0), see `d1_summary.md`). So no model-based value was affected by missing successor information in these runs: every nonterminal successor in a complete game is itself a decision state later in the same game, so it is visited and has counts of its own.

**Frequently visited states.** The most visited states are those with a high player sum against a dealer 10, because the policy sticks at 20/21 and the dealer card 10 is the most likely card.
- Seed 11, state (20, 10, no ace), 606 visits: model-based +0.416, Monte Carlo +0.416, TD(0) +0.439. Model-based and Monte Carlo are identical at all three checkpoints (for example +0.333 / +0.467 / +0.416) because sticking at 20 ends the game immediately, so the model's mean reward and the mean return are the same number. TD(0) is close here but not identical.
- Seed 11, state (19, 10, no ace), 447 visits: model-based -0.729, Monte Carlo -0.736, TD(0) -0.723. All three methods agree within about 0.013.
- Seed 22, state (20, 10, no ace), 633 visits: model-based and Monte Carlo +0.417, but TD(0) only +0.094 (it was +0.285 after 1,000 games). With alpha = 0.1, TD(0) weights roughly the last ten visits most, so even a state with hundreds of visits keeps fluctuating around the true value.

**Rarely visited states.** For states with only one to four visits the three methods disagree a lot.
- Seed 11, (4, 9, no ace), 1 visit: Monte Carlo +1.000, model-based -0.488, TD(0) 0.000. Monte Carlo with a single visit is just the outcome of that one game.
- Seed 11, (4, 5, no ace), 2 visits: Monte Carlo -1.000, model-based -0.554, TD(0) -0.032. Seed 11, (12, 8, usable ace), 1 visit: Monte Carlo -1.000, model-based -0.114, TD(0) -0.007.
- Seed 22, (12, 3, usable ace), 2 visits: Monte Carlo -0.500, model-based +0.180, TD(0) -0.026. Seed 33, (12, 4, usable ace), 4 visits: Monte Carlo 0.000, model-based -0.574, TD(0) -0.104.

Monte Carlo gives extreme, noisy values (-1, 0 or +1 from one or two games). TD(0) stays close to its initial value 0 because one visit moves the value only 10% of the way toward the target, so it is biased toward the initialisation in rare states. The model-based values are the most stable here: they use the observed one-step transitions and rewards of the rare state but take the values of its successors from states that were visited much more often, and they fall in the same range (about -0.5 to -0.7) as the frequent neighbouring states.

**Do the estimates stabilise?** Yes, but not monotonically in every measure. The mean absolute difference between methods over all visited states was, for seeds 11 / 22 / 33:
- model-based vs TD(0): 0.622 / 0.747 / 0.774 after 100 games, 0.327 / 0.407 / 0.405 after 1,000 games, 0.126 / 0.128 / 0.119 after 10,000 games;
- Monte Carlo vs TD(0): 0.712 / 0.780 / 0.828, then 0.420 / 0.489 / 0.473, then 0.165 / 0.177 / 0.160;
- model-based vs Monte Carlo: 0.180 / 0.051 / 0.103, then 0.267 / 0.247 / 0.255, then 0.103 / 0.111 / 0.110. This one first increases because after 100 games most visited states have only one or two visits, where both methods mostly reproduce the same single outcomes, and after 1,000 games many newly visited rare states enter the average.

The remaining difference of about 0.10–0.18 after 10,000 games is dominated by rare states, which still have very few visits. The three seeds give the same picture (compare the frequent state (20, 10, no ace): Monte Carlo +0.416 / +0.417 / +0.455), while TD(0) differs most between seeds (+0.439 / +0.094 / +0.493).

### Q-learning and baseline evaluation

Training: fresh `QLearner(epsilon, seed)` and fresh environment for each run; 50,000 games each; epsilon 0.2 and 0; alpha 0.1, gamma 1.0; random exploration between hit and stick; seeded random tie-breaking. Evaluation: 10,000 new games, greedy actions, no exploration, no updates, environment restarted with the evaluation seed for each policy. Evaluation seeds are paired with training seeds (11 ↔ 1011, 22 ↔ 1022, 33 ↔ 1033). If a Q-value tie or an unseen state occurred, the frozen policy sticks; this happened 0 times in all evaluations.

| Training seed | Policy / epsilon | Evaluation seed | Wins | Losses | Draws | Cut-offs | Mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | Fixed baseline | 1011 | 2962 | 6462 | 576 | 0 | -0.3500 |
| 11 | Q-learning, 0.2 | 1011 | 4244 | 4920 | 836 | 0 | -0.0676 |
| 11 | Q-learning, 0 | 1011 | 4192 | 4856 | 952 | 0 | -0.0664 |
| 22 | Fixed baseline | 1022 | 2935 | 6481 | 584 | 0 | -0.3546 |
| 22 | Q-learning, 0.2 | 1022 | 4236 | 4897 | 867 | 0 | -0.0661 |
| 22 | Q-learning, 0 | 1022 | 4142 | 4902 | 956 | 0 | -0.0760 |
| 33 | Fixed baseline | 1033 | 2950 | 6496 | 554 | 0 | -0.3546 |
| 33 | Q-learning, 0.2 | 1033 | 4074 | 5031 | 895 | 0 | -0.0957 |
| 33 | Q-learning, 0 | 1033 | 4103 | 5000 | 897 | 0 | -0.0897 |

All of these rows come from `outputs/evaluation/eval_results.csv`. That file also contains every policy evaluated on all three evaluation seeds (not only the paired ones), and `outputs/evaluation/eval_summary.csv` pools the three evaluation seeds per policy: win rate 0.411–0.421 for the six Q-learning policies and 0.295 for the baseline; mean reward -0.071 to -0.085 versus -0.353.

### Plots and learned policies

- Training curves, mean reward per 1,000-game block (training only, so epsilon 0.2 includes random exploration moves): `outputs/plots/learning_curve_training.png`; numbers in `outputs/qlearning/learning_curve.csv`
- Frozen-policy evaluation (greedy, 10,000 new games per evaluation seed): `outputs/plots/evaluation_results.png`
- Q-values and state-action visit counts, one file per run: `outputs/qlearning/q_seed<seed>_eps<epsilon>.csv`
- Greedy hit/stick tables, usable ace and no usable ace separately: `outputs/qlearning/greedy_seed<seed>_eps<epsilon>_ace.csv` and `..._noace.csv` (H = hit, S = stick, - = tie or state never seen)

Training and evaluation are shown in separate figures because they measure different things: the training curve includes exploration and a policy that is still changing, the evaluation is a fixed greedy policy on new games.

### Explain your results

**Prediction versus control.** Model-based evaluation, Monte Carlo and TD(0) only *evaluate* the fixed hit-below-20 policy, they estimate V(s) and never change how the game is played. Q-learning *controls*: it estimates Q(s, a) for both actions and the greedy action in each state defines a new policy, which was then tested on new games.

**Terminal updates.** At a terminal transition (the successor is END) the continuation value is 0: the TD target is the reward r, and the Q-learning target is r. At a nonterminal hit the reward is 0 and the target bootstraps from the successor value, or from the maximum Q-value of the successor. Monte Carlo uses the complete return of the finished game.

**Usable aces.** A usable ace can be counted as 11 or 1, so a hit cannot bust the hand as easily; the same player sum has different future behaviour with and without it. For that reason the flag is part of the state and the value tables and policy tables are reported separately for usable ace and no usable ace.

**Exploration and coverage.** Every one of the six Q-tables contains 280 observations (player sums 4–21 with no usable ace and 12–21 with a usable ace, against dealer cards 1–10), so every possible observation was reached, with epsilon 0.2 and with epsilon 0. In evaluation no decision needed the tie/unseen fallback. With epsilon 0, random tie-breaking among equal Q-values supplies the first variety, but an action that looks worse can stay under-sampled; with epsilon 0.2 both actions keep being tried.

**Variation across seeds and epsilon.** Seeds change the results slightly: the paired mean rewards for epsilon 0.2 are -0.0676, -0.0661 and -0.0957, and for epsilon 0 they are -0.0664, -0.0760 and -0.0897. Neither epsilon setting is better in all three seeds: epsilon 0.2 is better for seed 22, epsilon 0 for seed 33 and the two are almost equal for seed 11. A single evaluation of 10,000 games has a standard error of about 0.01 in mean reward, so these differences are about the size of the noise. The last training block (1,000 games) is noisier still (standard error about 0.03), so single-block values are not compared.

**Did the learned policy improve on the baseline?** Yes, in all three seeds and for both epsilon settings. The paired mean reward rises from about -0.35 (baseline, win rate about 0.30) to between -0.066 and -0.096 (Q-learning, win rate about 0.41–0.42), an improvement of roughly 0.26–0.29 per game. The learned policies still have a negative mean reward, so they do not beat the dealer on average; they only lose much less than the baseline, which hits on 12–19 whatever the dealer shows.

**Limits.** The conclusions come from three training seeds, 50,000 training games each and 10,000 evaluation games per seed. The state-value comparison in D1 is for one fixed policy only.

## Reproducibility and submission checklist

- [x] Public fork URL inserted above
- [ ] Original paper calculations committed before the first learning implementation (check `git log`: `a8f26d4`)
- [ ] Corrections marked without deleting the original work (check the scans)
- [x] Part C program loads the supplied CSV histories (`data/paper-games.csv`)
- [x] Part C output saved in `outputs/paper-replay/`
- [x] Part D game logs, summaries, visit counts, value/Q tables, policies and plots saved under `outputs/`
- [x] Python, Gymnasium and other dependency versions recorded
- [x] Exact reproduction and experiment commands added
- [x] `.env`, API keys and other secrets are not committed
- [ ] All `FILL` placeholders in this file replaced using actual results (remaining: AI-use, Part A–C scans, Part C paper columns)
- [ ] AI-use declaration accurately describes the assistance used
