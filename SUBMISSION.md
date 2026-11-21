# HW6 submission

## AI-use declaration

I used **Claude (Anthropic)** in a chat session for: explaining the assignment, writing the Python code
(`floors.py`, `solver.py`, `part_a_check.py`, `part_b.py`, `part_c.py`, `checks.py`), running the experiments,
and drafting the explanations in this file. The paper calculations in Part A were checked against
`outputs/part_a_check.txt` (produced by code). I reviewed the results and am responsible for the submission.

## Setup and reproduction

Python 3.12.3, `gymnasium==1.4.0`, `numpy==2.5.4` (see `requirements.txt`, `outputs/part_b_versions.txt`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python part_a_check.py   # Part A numbers for checking the paper work -> outputs/part_a_check.txt
python part_b.py         # Part B -> outputs/part_b_*.csv/md/txt
python checks.py         # rule checks -> outputs/rule_checks.txt
python part_c.py         # Part C -> outputs/part_c_*.csv/md/txt
```

Floor rules are in `data/maps.json` and are read by `floors.py`; `solver.py` only calls
`GridWorld.outcomes()`, so a new floor type needs only a new entry in `maps.json` (this is how `D` and `W` were added).

## Part A — Paper calculations

**Scan/photo: `part_a/paper_calculations.jpg` — **Scan/photo:** [page 1](part_a/paper_calculations.jpg), [page 2](part_a/paper_calculations_2.jpg) (committed after the code; see git history)**
The numbers below are the results the paper work must match (they are verified in `outputs/part_a_check.txt`).

**1. Model.** States: the nonterminal, non-wall tiles — (1,1) (1,2) (1,3) (2,1) (2,3) (3,1) (3,2) (3,3) (3,4);
G and H are terminal (value 0), the wall is not a state. Actions: up, right, down, left. Transitions: deterministic on
`S`, `.`, `M`; on `I` the chosen direction has probability 0.8 and each perpendicular direction 0.1; moves into a wall
or outside the map leave the agent in place. Reward: the reward of the tile where the move ends
(-1 floor, -3 mud, -1 ice, +10 goal, -10 hole). The agent maximizes the expected discounted sum of rewards with gamma = 0.9.

**2. V1** (V0 = 0, so V1(s) = best expected immediate reward):

```text
 -1    -1    10     G
 -1     #    -1     H
 -1    -1    -1    -1
```

(1,3) = 10 because "right" enters G for +10. At (1,2) mud: staying costs -3 but moving right/left costs -1, so V1 = -1.
At ice (2,3): "left" gives 0.8*(-1) [wall, stay] + 0.1*(-1) + 0.1*(-1) = -1, which beats "up" (-1.9, because of the 10% hole slip).

**3. V2 using only V1** (each Q = sum of p*(r + 0.9*V1(s'))):

(1,2) mud: up = -3 + 0.9(-1) = -3.9 (blocked, stays on M); right = -1 + 0.9(10) = **8.0**; down = -3.9 (wall); left = -1 + 0.9(-1) = -1.9. V2 = **8.0**.

(1,3): up = -1 + 0.9(10) = 8.0 (blocked, stays); right = 10 + 0.9(0) = **10.0**; down = -1 + 0.9(-1) = -1.9; left = -3 + 0.9(-1) = -3.9 (enters mud). V2 = **10.0**.

(2,3) ice:
- up = 0.8(-1 + 0.9·10) + 0.1(-1 + 0.9·(-1)) + 0.1(-10 + 0) = 6.4 - 0.19 - 1.0 = **5.21**
- right = 0.8(-10) + 0.1(-1 + 0.9·10) + 0.1(-1 + 0.9·(-1)) = -8 + 0.8 - 0.19 = -7.39
- down = 0.8(-1 + 0.9·(-1)) + 0.1(-1 + 0.9·(-1)) + 0.1(-10) = -1.52 - 0.19 - 1.0 = -2.71
- left = 0.8(-1 + 0.9·(-1)) + 0.1(-1 + 0.9·10) + 0.1(-1 + 0.9·(-1)) = -1.52 + 0.8 - 0.19 = -0.91

V2 = **5.21**.

**4. Arrows after two rounds:** (1,2) → right; (1,3) → right; (2,3) → up. No ties at these three positions.
Two rounds do not prove the final optimal policy because V2 is only a 2-step lookahead estimate: values keep changing
until convergence (the solver needed 12 iterations), and an action that looks best with V1 can change when later values
propagate (for example, values at (2,3) change from 5.21 to 5.82). Only convergence of V (and a greedy policy from it) gives the optimal policy.

## Part B — FrozenLake

### Fixed policy, written before the experiments

State numbering is 4·row + col. Gymnasium actions: 0 left, 1 down, 2 right, 3 up.

| State | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 10 | 13 | 14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Action | right | right | down | left | left | down | right | right | down | right | right |

Reasoning: the safe route on the map is 0 → 1 → 2 → 6 → 10 → 14 → 15 (right, right, down, down, down, right), which
avoids all holes (5, 7, 11, 12) when moves are deterministic. The other states (3, 4, 8, 9, 13) are fallback moves that
lead back toward this route or toward the goal. The policy was not changed after seeing results.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no | random | 11 | 0 | 50 | 0 | 0 | 7.98 |
| no | random | 22 | 0 | 50 | 0 | 0 | 6.42 |
| no | random | 33 | 1 | 49 | 0 | 2 | 6.66 |
| no | fixed | 11 | 50 | 0 | 0 | 100 | 6.00 |
| no | fixed | 22 | 50 | 0 | 0 | 100 | 6.00 |
| no | fixed | 33 | 50 | 0 | 0 | 100 | 6.00 |
| yes | random | 11 | 0 | 50 | 0 | 0 | 8.30 |
| yes | random | 22 | 2 | 48 | 0 | 4 | 7.76 |
| yes | random | 33 | 0 | 50 | 0 | 0 | 6.74 |
| yes | fixed | 11 | 0 | 50 | 0 | 0 | 6.68 |
| yes | fixed | 22 | 3 | 47 | 0 | 6 | 6.94 |
| yes | fixed | 33 | 4 | 46 | 0 | 8 | 7.60 |

Episode rows: `outputs/part_b_episodes.csv`; example traces: `outputs/part_b_traces.txt` (one trace for each (condition, policy, outcome) that occurred).
Time limits never occurred: every episode ended in a goal or a hole before 100 steps.

**Slipping.** Without slipping the fixed policy reaches the goal in all 150 episodes in exactly 6 steps, while random
actions reach it in only 1 of 150. With slipping, the intended move happens only with probability 1/3 (the other two
directions are perpendicular), so the same fixed policy drops to 0-8% success (0/50, 3/50, 4/50). Slipping pushes the agent
off the route and often into a hole, and my route passes next to holes, so it is fragile. Random actions are also near 0%.

**Variation across batches.** With the deterministic fixed policy the batches are identical (50/50/50). With random
actions and slipping, the counts differ between batches (0, 2, 0 goals; 0, 3, 4 for the fixed policy), which shows the
sampling noise of 50 episodes. Because these successes are rare, the differences between batches (and between random
and fixed under slipping) are small and not strongly reliable.

**One success is not enough evidence**: a single success can be luck (for instance random actions reached the goal once without slipping), whereas many episodes with several batches give an estimate of the success rate and its variability.

**Not training.** The fixed policy never changes and nothing is learned from the rewards; the episodes only measure how good
a fixed (or random) rule is. Training would update the policy or values from experience.

## Part C — Custom solver

### Compare code with paper

Code (`outputs/part_c_value_policy_maps.txt`): V1 equals the paper map above; V2(1,2) = 8.0, V2(1,3) = 10.0, V2(2,3) = 5.21 with the same
best actions (right, right, up). No discrepancies were found between the paper calculations and the code.

### Values, policies and convergence

Maps: `outputs/part_c_value_policy_maps.txt`. Tolerance 1e-6, safeguard 10000.

| Map | Iterations | Final max change | Converged |
| --- | --- | --- | --- |
| Paper | 12 | 2.165e-07 | yes |
| Extended | 11 | 2.406e-07 | yes |

Final values — paper map (extended map identical except at (3,1) = 1.502 instead of 1.536):

```text
 4.20   8.00  10.00    G
 2.78    #     5.82    H
 1.54   2.82   4.24   2.82
```

Policy, paper map `>` right `^` up `<` left (extended map differs only at (3,1): `^` instead of `>`):

```text
 >   >   >   G
 ^   #   ^   H
 >   >   ^   <
```

Tie-breaking: actions whose Q-values are within 1e-9 of the maximum are tied; the first in the order up, right, down, left is chosen.
No ties occurred at the final values.

### Simulation results

Seeds: episode seed 1000 × batch seed + i; 100-step limit; discounted return r0 + 0.9 r1 + 0.81 r2 + ...

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| paper | 11 | 47 | 3 | 0 | 5.04 | 2.230 |
| paper | 22 | 44 | 6 | 0 | 4.96 | 1.442 |
| paper | 33 | 42 | 8 | 0 | 4.88 | 0.970 |
| extended | 11 | 50 | 0 | 0 | 5.00 | 1.502 |
| extended | 22 | 50 | 0 | 0 | 5.00 | 1.502 |
| extended | 33 | 50 | 0 | 0 | 5.00 | 1.502 |

(Files: `outputs/part_c_episodes.csv`, `outputs/part_c_sim_summary.md`.)

### Explain your results

**What changed with sand and water.** Almost nothing in values: only the start tile changed, V(3,1) from 1.536 to 1.502, and its
action changed from right to up. Sand at (3,2) costs -2 instead of -1, which lowers the lower route. Water at (3,4) is
entered only from (3,3) via a move to the right, which the policy never uses, so it costs nothing; values at (3,2) and
(3,4) stay the same because the reward belongs to the tile being entered, not the tile being left.

**Cost and risk.** On the paper map the best route from the start goes along the bottom and up through the ice (-1 per
step, V = 1.54), which is cheap but risky: 10% slip into the hole per attempt at the ice. On the extended map sand makes the lower route more expensive, so the
policy switches to the safe route through the mud (up, up, right, right, right: one -3 step, no randomness), with V = 1.50 slightly below
the ice route's 1.54 on the paper map but with no risk. The policy near the ice at (2,3) always goes up (towards G), never right (towards H).

**Simulation vs values.** On the paper map the policy reached the goal in 47, 44, 42 of 50 episodes; the mean discounted returns (2.23, 1.44, 0.97; mean about 1.55) vary between batches but average close to V(start) = 1.54. On the extended map the route is deterministic, so every episode gives 1.502, exactly the computed
value. Finite simulations only estimate the infinite-horizon value: they have a 100-step cut-off and random sampling error, whereas V is an exact expectation over infinitely many steps.

### Rule checks

`outputs/rule_checks.txt`: all checks passed — boundary and wall moves (stay in place, reward of the ending tile), ice probabilities sum to 1 for all four
actions, rewards for mud (-3), sand (-2), water (-5), goal (+10), hole (-10), moving into ice being deterministic, and no transitions or rewards after G/H.
