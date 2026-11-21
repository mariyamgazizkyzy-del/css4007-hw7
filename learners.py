
from collections import defaultdict
import numpy as np

from common import END

ALPHA = 0.1
GAMMA = 1.0
ACTIONS = (0, 1)


class ModelBasedEvaluator:
    """Estimate transitions and evaluate a fixed policy."""

    def __init__(self):
        self.counts = defaultdict(int)
        self.reward_sums = defaultdict(float)
        self.transitions = defaultdict(int)
        self.states = set()

    def update(self, s, a, r, s_next):
        key = (s, a)
        triple = (s, a, s_next)

        self.counts[key] += 1
        self.reward_sums[triple] += r
        self.transitions[triple] += 1

        self.states.add(s)
        if s_next != END:
            self.states.add(s_next)

    def evaluate(self, policy, tol=1e-6, max_sweeps=10_000):
        V = {s: 0.0 for s in self.states}
        V[END] = 0.0

        converged = False
        sweeps = 0

        for sweep in range(max_sweeps):
            new_V = V.copy()

            for s in self.states:
                a = policy(s)
                key = (s, a)
                total = self.counts.get(key, 0)

                if total == 0:
                    new_V[s] = None
                    continue

                value = 0.0
                observed = True

                for (state, action, s_next), count in self.transitions.items():
                    if (state, action) != key:
                        continue

                    probability = count / total
                    mean_reward = (
                        self.reward_sums[(state, action, s_next)] / count
                    )

                    if s_next == END:
                        next_value = 0.0
                    else:
                        next_value = V.get(s_next)

                    if next_value is None:
                        observed = False
                        break

                    value += probability * (
                        mean_reward + GAMMA * next_value
                    )

                new_V[s] = value if observed else None

            deltas = [
                abs(new_V[s] - V[s])
                for s in self.states
                if new_V[s] is not None and V[s] is not None
            ]

            delta = max(deltas, default=0.0)
            V = new_V
            sweeps = sweep + 1

            # tol=0 disables early stopping, allowing exact sweep counts.
            if tol > 0 and delta < tol:
                converged = True
                break

        return V, converged, sweeps


class MonteCarloEvaluator:
    """Every-visit Monte Carlo evaluator."""

    def __init__(self):
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        self.visits = defaultdict(int)
        self.V = {}

    def add_game(self, transitions):
        G = 0.0

        for s, a, r, s_next in reversed(transitions):
            G = r + GAMMA * G

            self.returns_sum[s] += G
            self.returns_count[s] += 1
            self.visits[s] += 1

            self.V[s] = (
                self.returns_sum[s] / self.returns_count[s]
            )


class TD0Evaluator:
    """TD(0) state-value evaluator."""

    def __init__(self):
        self.V = {END: 0.0}
        self.visits = defaultdict(int)

    def update(self, s, r, s_next):
        old_value = self.V.get(s, 0.0)

        if s_next == END:
            next_value = 0.0
        else:
            next_value = self.V.get(s_next, 0.0)

        self.V[s] = old_value + ALPHA * (
            r + GAMMA * next_value - old_value
        )

        self.visits[s] += 1


class QLearner:
    """Epsilon-greedy Q-learning agent."""

    def __init__(self, epsilon=0.2, seed=11):
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: {0: 0.0, 1: 0.0})
        self.visits = defaultdict(int)

        self.explore_rng = np.random.default_rng(seed)
        self.tie_rng = np.random.default_rng(seed + 1)

    def choose_action(self, s):
        if self.explore_rng.random() < self.epsilon:
            return int(self.explore_rng.choice(ACTIONS))

        values = self.Q[s]
        best_value = max(values.values())

        best_actions = [
            action
            for action, value in values.items()
            if value == best_value
        ]

        return int(self.tie_rng.choice(best_actions))

    def update(self, s, a, r, s_next):
        if s_next == END:
            target = r
        else:
            next_values = self.Q.get(
                s_next, {0: 0.0, 1: 0.0}
            )
            target = r + GAMMA * max(next_values.values())

        self.Q[s][a] += ALPHA * (
            target - self.Q[s][a]
        )

        self.visits[(s, a)] += 1
