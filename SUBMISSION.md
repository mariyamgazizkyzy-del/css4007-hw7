# HW7 submission — Blackjack

## AI-use declaration

Which tools did you use, and for what?

## Setup and reproduction

Record dependency versions, installation steps and exact commands for the
paper replay, training and frozen-policy evaluation.

## Part A — Count states and actions

Link your paper answers and comparison table for Blackjack, Durak, tic-tac-toe
and Nim. Show formulas, assumptions, example legal actions and hidden information.

## Part B — Paper calculations

Link scans/photos of all four methods. Commit the original paper work before
implementing learners. Keep originals alongside any corrections.

## Part C — Reproduce the calculations

Link the fixed-history replay output in outputs/paper-replay/.

| Method | State / action | Sweep / pass / update | Paper result | Program result | Difference | Explanation |
| --- | --- | --- | --- | --- | --- | --- |

## Part D — Simulated games

### Fixed-policy value estimates

Link the shared logs, checkpoint value tables and visit counts. Explain
missing states and differences among model-based, Monte Carlo and TD estimates.

### Q-learning and baseline evaluation

| Training seed | Policy / epsilon | Evaluation seed | Wins | Losses | Draws | Cut-offs | Mean reward |
| --- | --- | --- | --- | --- | --- | --- | --- |

### Plots and learned policies

Link training curves, Q-tables, hit/stick policy tables and state-action counts.
Separate usable-ace and no-usable-ace results.

### Explain your results

Address prediction versus control, terminal updates, usable aces, exploration,
coverage, variation across seeds and whether the learned policy improved.
# HW7 --- Learn Blackjack Four Ways

**CSS-4007 · Artificial Intelligence · Fall 2026**

**Student:** Mariam Gazizkyzy\
**Repository:** `https://github.com/mariyamgazizkyzy-del/css4007-hw7` 
replace with your public fork URL)\
**Environment:** Gymnasium `Blackjack-v1`, `natural=False`, `sab=False`

> **Draft status:** Parts A and B below contain the hand calculations
> based on the assignment's supplied histories. Parts C and D require
> running the code in the student's own fork. Their results are
> deliberately left as fields to complete; no simulated results,
> dependency versions, or repository commits are invented.

## AI-use declaration

I used ChatGPT to help explain the assignment, check mathematical
calculations, structure this report, and draft explanations. I am
responsible for reviewing the calculations, implementing and running the
programs, checking the outputs, and submitting my own work. The final
declaration should be updated to match the actual tools and assistance
used.

## Blackjack as an MDP

A decision state is the player's observation
`(player_sum, dealer_showing, usable_ace)`. The action is `0 = stick` or
`1 = hit`. A nonterminal hit has reward 0. At the end of a game the
reward is +1 for a win, -1 for a loss, or 0 for a draw. A game
terminates when the player sticks and the dealer completes its turn, or
when the player busts. The usable-ace flag is needed because an ace
currently counted as 11 may later be changed to 1; therefore the same
total can have different future transitions depending on this flag. The
dealer's hidden card is not part of the player's observation and must
not be exposed to the learner.

Parameters: discount `gamma = 1.0`; TD/Q-learning rate `alpha = 0.1`;
initial V and Q values are zero. The fixed policy for model-based
evaluation, Monte Carlo and TD(0) is hit below 20 and stick at 20 or 21.

# Part A --- States and actions

## A1. Blackjack

The restricted decision-observation table has 10 player sums (12--21),
10 dealer up-card values (1--10), and 2 usable-ace values:

`10 × 10 × 2 = 200` observation entries.

There are two action labels (hit and stick), so the restricted Q-table
has `200 × 2 = 400` state-action entries. These are counts for the
stated restricted representation, not the entire environment:
observations with player sums below 12 and the terminal marker are
excluded. The full simulator also has hidden card information and
turn/game status that are not all included in the player's observation.

Two different card situations can produce `(18, 6, no usable ace)`: a
player holding 10+8, or a player holding 5+6+7, while the dealer shows
6. The dealer's hidden card must not be provided because the player
cannot observe it; including it would leak hidden information and change
the learning problem.

## A2. Durak

Before any cards are exposed, the number of unordered six-card hands is:

`C(36,6) = 36! / (6! × 30!) = 1,947,792`.

Order is ignored because a hand is a set of cards, not a sequence. A
hand alone is not a complete game state.

Given our six cards and one exposed trump card outside our hand, 29
cards remain available for the opponent's six-card hand:

`C(29,6) = 29! / (6! × 23!) = 475,020`

This counts candidate opponent hands only. It does not count the
remaining stock order or all other hidden state components.

On an empty table, an attacker with six distinct cards has six legal
one-card opening actions.

For the defense example, hearts are trump and the attack is 10♣. From
6♣, 9♣, A♣, 7♥, Q♥, 8♠, the legal defense cards are A♣, 7♥ and Q♥. The
defender may also pick up. Thus there are `3 + 1 = 4` legal actions.

A full simulator state needs both hands, attack/defense pairs on the
table, trump card and suit, ordered stock, discarded cards, phase, and
player to move. The player observes their own hand and public
table/trump information, but not the opponent's hand or the order of the
stock. The same visible table can therefore hide different states and
produce different future outcomes. No single total for all reachable
Durak states is asserted here.

## A3. Tic-tac-toe

Each of nine squares has three encodings (empty, X, O), so all board
encodings number `3^9 = 19,683`. This is an upper bound on encodings,
not the exact number of legally reachable positions: some encodings have
impossible X/O counts, and some continue after a win even though play
must stop at the first win.

There are nine action labels (place a mark in one of nine squares). In
the supplied board:

    X O X
    . O .
    . . .

There are two X marks and two O marks, so it is X's turn. The five legal
placements are row 2 column 1, row 2 column 3, row 3 column 1, row 3
column 2, and row 3 column 3. The board contents determine the player to
move in a legal position because X starts and turns alternate.
Multiplying every board encoding by two would not count legal states
exactly: many boards are unreachable, and the turn is determined by the
mark counts for legal boards.

## A4. Nim

The three piles are labeled and bounded by 2, 3 and 4 stones. The number
of pile configurations is:

`(2+1)(3+1)(4+1) = 3 × 4 × 5 = 60`.

Including one of two players to move gives `60 × 2 = 120` encodings.
This is an encoding count, not a claim that every encoding is reachable
from the specified initial position and starting player.

At `(2,3,4)`, actions are `(pile, stones removed)`: pile 1 remove 1 or
2; pile 2 remove 1, 2 or 3; pile 3 remove 1, 2, 3 or 4. Total:
`2+3+4=9`.

At `(0,1,2)`, actions are `(2,1)`, `(3,1)`, `(3,2)`, for 3 legal
actions. At `(0,0,0)`, the position is terminal and has 0 legal actions.
"Choose a pile" is incomplete because the number of stones removed must
also be specified. If piles were unlabeled, permutations of the same
pile sizes would represent the same configuration, reducing the count.

## A5. Comparison

  ----------------------------------------------------------------------------------------------------------------
  Game          Representation           Count formula Count type    Action                  Example Hidden
                                                                     representation            legal information
                                                                                             actions 
  ------------- ---------------- --------------------- ------------- ------------------ ------------ -------------
  Blackjack     Player sum,              `10×10×2=200` Restricted    Hit / stick            2 action Dealer hole
                dealer up-card,                        observation                           labels; card
                usable ace                             count                            legal choice 
                                                                                          depends on 
                                                                                               state 

  Durak         Hands, table,      `C(36,6)=1,947,792` Partial       Play a card / pick   4 in given Opponent hand
                trump, stock,             for one hand component     up / end attack         defense and stock
                phase, turn                            count                                         order

  Tic-tac-toe   Nine squares              `3^9=19,683` Upper bound   Place mark in a      5 in given None under
                                                       on legal      square                    board perfect
                                                       positions                                     observation

  Nim           Three labeled            `3×4×5×2=120` Encoding      `(pile, amount)`           9 at None under
                pile sizes and                         count                               `(2,3,4)` perfect
                player to move                                                                       observation
  ----------------------------------------------------------------------------------------------------------------

Durak can require an especially large Q-table because the state contains
combinations of cards, hidden hands, stock order, table layout and
phase. Blackjack also grows when the observation range is expanded. The
number of action labels is not necessarily the number of legal actions
in a particular state: for example, a Nim pile cannot lose more stones
than it currently contains.

# Part B --- Paper calculations for the supplied histories

States: A=(18,6,no usable ace), B=(20,6,no usable ace), C=(17,6,usable
ace), D=(17,6,no usable ace), E=(21,6,no usable ace). END is terminal
and has continuation value 0.

## B1. Model-based learning

For observed transitions:

`P(s'|s,a) = count(s,a,s') / count(s,a)`\
`R(s,a,s') = mean reward observed for that triple`

  --------------------------------------------------------------------------
  State-action   Next state            Count       Estimated     Mean reward
                                                 probability 
  -------------- ----------- --------------- --------------- ---------------
  A, hit         B                         2             2/3               0

  A, hit         END                       1             1/3              -1

  B, stick       END                       2               1               0

  C, hit         D                         1               1               0

  D, hit         E                         1               1               0

  E, stick       END                       1               1               0
  --------------------------------------------------------------------------

For B, stick, both observed terminal rewards are included:
`(+1 + -1)/2 = 0`.

Synchronous policy evaluation uses only the previous sweep:

`V_(k+1)(s) = Σ_s' P(s'|s,π(s)) [R(s,π(s),s') + gamma V_k(s')]`.

Starting from all zeros: - `V1(A) = (2/3)(0) + (1/3)(-1) = -1/3`. -
`V1(B) = (1/2)(+1) + (1/2)(-1) = 0`. - `V1(C)=V1(D)=V1(E)=0`. -
`V2(A) = (2/3)(0 + V1(B)) + (1/3)(-1 + 0) = -1/3`. -
`V2(B)=V2(C)=V2(D)=V2(E)=0`.

  State     V0              V1              V2
  ------- ---- --------------- ---------------
  A          0   -0.3333333333   -0.3333333333
  B          0               0               0
  C          0               0               0
  D          0               0               0
  E          0               0               0

All other state-action pairs are unobserved. Empirical probabilities
from these few constructed histories need not equal the true
probabilities of Blackjack.

## B2. Direct evaluation (every-visit Monte Carlo)

Returns from each nonterminal visit: - Game 1: A gets +1; B gets +1. -
Game 2: A gets -1. - Game 3: C, D and E each get 0. - Game 4: A gets -1;
B gets -1.

`V(s) = sum of returns / number of visits`.

  State   Returns        Visits            Mean return
  ------- ------------ -------- ----------------------
  A       +1, -1, -1          3   -1/3 = -0.3333333333
  B       +1, -1              2                      0
  C       0                   1                      0
  D       0                   1                      0
  E       0                   1                      0

## B3. TD(0)

Update rule: `V(s) ← V(s) + 0.1 × [r + V(s') - V(s)]`, with `V(END)=0`.
Updates are immediate and values carry between games.

First pass, games 1--4:

  Game/step   State     Old V    r   Successor V   Target   New V
  ----------- ------- ------- ---- ------------- -------- -------
  1/1         A             0    0             0        0       0
  1/2         B             0   +1             0        1     0.1
  2/1         A             0   -1             0       -1    -0.1
  3/1         C             0    0             0        0       0
  3/2         D             0    0             0        0       0
  3/3         E             0    0             0        0       0
  4/1         A          -0.1    0           0.1      0.1   -0.08
  4/2         B           0.1   -1             0       -1   -0.01

Second pass, without resetting:

  Game/step   State       Old V    r   Successor V   Target      New V
  ----------- ------- --------- ---- ------------- -------- ----------
  1/1         A           -0.08    0         -0.01    -0.01     -0.073
  1/2         B           -0.01   +1             0        1      0.091
  2/1         A          -0.073   -1             0       -1    -0.1657
  3/1         C               0    0             0        0          0
  3/2         D               0    0             0        0          0
  3/3         E               0    0             0        0          0
  4/1         A         -0.1657    0         0.091    0.091   -0.14003
  4/2         B           0.091   -1             0       -1    -0.0181

Final TD values: A=-0.14003, B=-0.0181, C=D=E=0.

## B4. Q-learning

Update rule: `Q(s,a) ← Q(s,a) + 0.1 × [r + max_a' Q(s',a') - Q(s,a)]`.
Both actions are included in the maximum, including unobserved actions
initialized to zero. Terminal continuation is zero.

First pass: \| Game/step \| State-action \| Old Q \| r \| Max next Q \|
Target \| New Q \| \|---\|---\|---:\|---:\|---:\|---:\|---:\| \| 1/1 \|
A, hit \| 0 \| 0 \| 0 \| 0 \| 0 \| \| 1/2 \| B, stick \| 0 \| +1 \| 0 \|
1 \| 0.1 \| \| 2/1 \| A, hit \| 0 \| -1 \| 0 \| -1 \| -0.1 \| \| 3/1 \|
C, hit \| 0 \| 0 \| 0 \| 0 \| 0 \| \| 3/2 \| D, hit \| 0 \| 0 \| 0 \| 0
\| 0 \| \| 3/3 \| E, stick \| 0 \| 0 \| 0 \| 0 \| 0 \| \| 4/1 \| A, hit
\| -0.1 \| 0 \| 0.1 \| 0.1 \| -0.08 \| \| 4/2 \| B, stick \| 0.1 \| -1
\| 0 \| -1 \| -0.01 \|

Second pass: \| Game/step \| State-action \| Old Q \| r \| Max next Q \|
Target \| New Q \| \|---\|---\|---:\|---:\|---:\|---:\|---:\| \| 1/1 \|
A, hit \| -0.08 \| 0 \| 0 \| 0 \| -0.072 \| \| 1/2 \| B, stick \| -0.01
\| +1 \| 0 \| 1 \| 0.091 \| \| 2/1 \| A, hit \| -0.072 \| -1 \| 0 \| -1
\| -0.1648 \| \| 3/1 \| C, hit \| 0 \| 0 \| 0 \| 0 \| 0 \| \| 3/2 \| D,
hit \| 0 \| 0 \| 0 \| 0 \| 0 \| \| 3/3 \| E, stick \| 0 \| 0 \| 0 \| 0
\| 0 \| \| 4/1 \| A, hit \| -0.1648 \| 0 \| 0.091 \| 0.091 \| -0.13922
\| \| 4/2 \| B, stick \| 0.091 \| -1 \| 0 \| -1 \| -0.0181 \|

Final Q-table (unobserved actions retain their initialized zero): \|
State \| Q(hit) \| Q(stick) \| Greedy action \| \|---\|---:\|---:\|---\|
\| A \| -0.13922 \| 0 \| Stick \| \| B \| 0 \| -0.0181 \| Hit \| \| C \|
0 \| 0 \| Tie \| \| D \| 0 \| 0 \| Tie \| \| E \| 0 \| 0 \| Tie \|

An untried action's zero estimate is not evidence that it is good; it
only indicates that no update has been made for that action. These four
methods need not agree because they estimate different quantities, use
different update rules, and process the limited histories differently.
Replaying fixed histories checks update calculations, not exploration or
policy quality.

# Part C --- Exact reproduction in code

**Status: complete only after the fixed-history program has been run and
its outputs checked. Do not fill this section with guessed results.**

-   Reproduction command: `[ADD THE ACTUAL COMMAND]`
-   Input files: `data/paper-games.csv`, `data/states.csv`
-   Program file(s): `[ADD FILE NAMES]`
-   Output directory: `outputs/paper-replay/`
-   Python version: `[RECORD python --version]`
-   Dependency versions:
    `[RECORD installed versions, including Gymnasium]`

Attach or summarize the program output for: 1. Transition counts,
probabilities and mean rewards. 2. Both synchronous model-based sweeps.
3. Every-visit Monte Carlo returns, visit counts and averages. 4. Every
TD(0) update in both passes. 5. Every Q-learning update in both passes.
6. Paper-versus-program comparison, including full-precision
differences.

  -------------------------------------------------------------------------------------------
  Method       State /        Sweep/pass/update      Paper result       Program    Difference
               state-action                                              result 
  ------------ -------------- ------------------- --------------- ------------- -------------
  Model        A              V1                    -0.3333333333       \[RUN\]       \[RUN\]

  Model        A              V2                    -0.3333333333       \[RUN\]       \[RUN\]

  Monte Carlo  A              One pass              -0.3333333333       \[RUN\]       \[RUN\]

  TD(0)        A              Pass 2 final               -0.14003       \[RUN\]       \[RUN\]

  TD(0)        B              Pass 2 final                -0.0181       \[RUN\]       \[RUN\]

  Q-learning   A, hit         Pass 2 final               -0.13922       \[RUN\]       \[RUN\]

  Q-learning   B, stick       Pass 2 final                -0.0181       \[RUN\]       \[RUN\]
  -------------------------------------------------------------------------------------------

Explain any discrepancies here:
`[COMPARE ACTUAL OUTPUT WITH PAPER WORK; KEEP ORIGINAL CALCULATIONS VISIBLE AND MARK CORRECTIONS.]`

# Part D --- Learning from simulated Blackjack games

**Status: pending actual Gymnasium experiments. All fields below must be
populated from saved outputs.**

## D1. Fixed-policy evaluation

Environment: `Blackjack-v1`, `natural=False`, `sab=False`. Policy: hit
below 20, stick at 20 or 21. For each seed 11, 22 and 33, collect 10,000
games. The first reset is seeded; later resets continue the same
environment random sequence. Log each transition and map terminal
successors to END for the learners.

  -------------------------------------------------------------------------
             Seed Games collected        Cut-offs Model         Notes
                                                  convergence / 
                                                  sweeps        
  --------------- --------------- --------------- ------------- -----------
               11          10,000         \[RUN\] \[RUN\]       \[RUN\]

               22          10,000         \[RUN\] \[RUN\]       \[RUN\]

               33          10,000         \[RUN\] \[RUN\]       \[RUN\]
  -------------------------------------------------------------------------

For each seed, feed the same saved game sequence to model-based
evaluation, every-visit Monte Carlo and TD(0). Save estimates at 100,
1,000 and 10,000 games. Model evaluation uses tolerance `1e-6` and at
most 10,000 synchronous sweeps. Unobserved state-action pairs must
remain marked unobserved; do not substitute hidden game rules or present
initialization zeros as empirical estimates.

Insert the generated state-value tables here, separately for usable ace
and no usable ace. Include visit counts and mark missing/unobserved
estimates explicitly: `[LINK OR TABLE FROM outputs/]`.

Discuss several frequently visited and rarely visited states using
actual counts and estimates: `[ADD ACTUAL COMPARISONS]`. Explain whether
estimates stabilize as the number of games increases and identify
model-based values affected by missing successor information:
`[ADD FROM OUTPUTS]`.

## D2. Q-learning

Train a fresh Q-table for each seed 11, 22 and 33 for 50,000 games. Use
`alpha=0.1`, `gamma=1.0`, epsilon 0.2, uniform random exploration
between hit and stick, and seeded random tie-breaking. Repeat with
epsilon 0. Save Q-values, state-action visit counts and greedy policies
separately for usable-ace and no-usable-ace observations.

    Training seed   Epsilon    Games Output file(s)
  --------------- --------- -------- ----------------
               11       0.2   50,000 \[PATH\]
               22       0.2   50,000 \[PATH\]
               33       0.2   50,000 \[PATH\]
               11         0   50,000 \[PATH\]
               22         0   50,000 \[PATH\]
               33         0   50,000 \[PATH\]

## D3. Frozen-policy evaluation

Evaluate each frozen Q-learning policy and the fixed-policy baseline for
10,000 new games. Use evaluation seeds 1011, 1022 and 1033, paired with
training seeds. Restart the evaluation environment with the same seed
for each policy. Use greedy actions, no exploration and no learning
updates.

  -------------------------------------------------------------------------------------------------
    Training   Epsilon   Evaluation Policy            Wins    Losses     Draws   Cut-offs      Mean
        seed                   seed                                                          reward
  ---------- --------- ------------ ------------ --------- --------- --------- ---------- ---------
          11       ---         1011 Fixed          \[RUN\]   \[RUN\]   \[RUN\]    \[RUN\]   \[RUN\]
                                    baseline                                              

          11       0.2         1011 Q-learning     \[RUN\]   \[RUN\]   \[RUN\]    \[RUN\]   \[RUN\]

          11         0         1011 Q-learning     \[RUN\]   \[RUN\]   \[RUN\]    \[RUN\]   \[RUN\]

          22       ---         1022 Fixed          \[RUN\]   \[RUN\]   \[RUN\]    \[RUN\]   \[RUN\]
                                    baseline                                              

          22       0.2         1022 Q-learning     \[RUN\]   \[RUN\]   \[RUN\]    \[RUN\]   \[RUN\]

          22         0         1022 Q-learning     \[RUN\]   \[RUN\]   \[RUN\]    \[RUN\]   \[RUN\]

          33       ---         1033 Fixed          \[RUN\]   \[RUN\]   \[RUN\]    \[RUN\]   \[RUN\]
                                    baseline                                              

          33       0.2         1033 Q-learning     \[RUN\]   \[RUN\]   \[RUN\]    \[RUN\]   \[RUN\]

          33         0         1033 Q-learning     \[RUN\]   \[RUN\]   \[RUN\]    \[RUN\]   \[RUN\]
  -------------------------------------------------------------------------------------------------

Plot training mean reward in blocks of 1,000 games. Clearly distinguish
training results from frozen-policy evaluation results. Report cut-offs
separately; they are not wins, losses, draws, or complete Monte Carlo
returns. Safety limit: 100 player actions per game.

**Did Q-learning improve on the baseline?**
`[ANSWER ONLY AFTER COMPARING ALL THREE SEEDS AND BOTH EPSILON SETTINGS; INCLUDE NEGATIVE RESULTS.]`

## D4. Discussion

-   Model-based learning estimates a transition/reward model and
    evaluates a supplied policy.
-   Monte Carlo estimates state values from complete game returns.
-   TD(0) estimates state values using one-step bootstrapped updates.
-   Q-learning estimates action values and can derive a greedy action
    policy.
-   Terminal transitions use zero continuation value. Nonterminal
    transitions bootstrap from the successor value or maximum successor
    Q-value.
-   The usable-ace flag distinguishes observations with different future
    card-handling behavior.
-   Exploration can expose the learner to both actions; with epsilon 0,
    unvisited actions may remain poorly estimated. Limited visits make
    estimates noisy, especially for rare observations.

Add conclusions based on the actual experiment:
`[DISCUSS OBSERVED VISIT FREQUENCIES, CONVERGENCE, SEED VARIATION, EPSILON COMPARISON, AND BASELINE COMPARISON.]`

# Reproducibility and submission checklist

-   [ ] Public fork URL inserted above; no pull request to starter.
-   [ ] Original paper calculations photographed/scanned and committed
    before the first learning implementation.
-   [ ] Corrections are marked without deleting the original work.
-   [ ] Part C program loads the supplied CSV histories exactly; no
    replacement histories.
-   [ ] Part C output is saved in `outputs/paper-replay/`.
-   [ ] Part D game logs, summaries, visit counts, value/Q tables,
    policies and plots are saved under `outputs/`.
-   [ ] Python, Gymnasium and other dependency versions recorded.
-   [ ] Exact reproduction and experiment commands added.
-   [ ] `.env`, API keys and other secrets are not committed.
-   [ ] All `[RUN]`, `[PATH]` and `[ADD ...]` placeholders are replaced
    using actual results.
-   [ ] AI-use declaration accurately describes the assistance used.
