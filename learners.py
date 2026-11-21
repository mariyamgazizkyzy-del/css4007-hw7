from collections import defaultdict
from common import END

ALPHA = 0.1
GAMMA = 1.0
ACTIONS = (0, 1)

class ModelBasedEvaluator:
    def __init__(self):
        self.counts = defaultdict(int)
        self.reward_sums = defaultdict(float)
        self.transitions = defaultdict(int)
        self.states = set()

    def update(self, s, a, r, s_next):
        key = (s, a)
        self.counts[key] += 1
        self.reward_sums[key] += r
        self.transitions[(s, a, s_next)] += 1
        self.states.add(s)

    def evaluate(self, policy, tol=1e-6, max_sweeps=10_000):
        V = {s: 0.0 for s in self.states}
        V[END] = 0.0

        for sweep in range(max_sweeps):
            new_V = V.copy()
            for s in self.states:
                a = policy(s)
                key = (s, a)

                if key not in self.counts:
                    continue

                total = self.counts[key]
                value = 0.0

                for (state, action, s_next), count in self.transitions.items():
                    if (state, action) != key:
                        continue

                    probability = count / total
                    mean_reward = (
                        self.reward_sums[key] / total
                    )
                    value += probability * (
                        mean_reward + GAMMA * V.get(s_next, 0.0)
                    )

                new_V[s] = value

            delta = max(
                (abs(new_V[s] - V[s]) for s in self.states),
                default=0.0
            )
            V = new_V

            if delta < tol:
                return V

        raise RuntimeError("Model-based evaluation did not converge")

class MonteCarloEvaluator:
    def __init__(self):
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        self.V = {}

    def add_game(self, transitions):
        G = 0.0

        for s, a, r, s_next in reversed(transitions):
            G = r + GAMMA * G
            self.returns_sum[s] += G
            self.returns_count[s] += 1
            self.V[s] = (
                self.returns_sum[s] / self.returns_count[s]
            )

class TD0Evaluator:
    def __init__(self):
        self.V = {END: 0.0}

    def update(self, s, r, s_next):
        old_value = self.V.get(s, 0.0)
        next_value = 0.0 if s_next == END else self.V.get(s_next, 0.0)

        self.V[s] = old_value + ALPHA * (
            r + GAMMA * next_value - old_value
        )

class QLearner:
    def __init__(self, epsilon=0.1, seed=42):
        import numpy as np

        self.epsilon = epsilon
        self.Q = defaultdict(lambda: {0: 0.0, 1: 0.0})

        self.explore_rng = np.random.default_rng(seed)
        self.tie_rng = np.random.default_rng(seed + 1)

    def choose_action(self, s):
        import numpy as np

        if self.explore_rng.random() < self.epsilon:
            return int(self.explore_rng.choice(ACTIONS))

        values = self.Q[s]
        best_value = max(values.values())

        best_actions = [
            action for action, value in values.items()
            if value == best_value
        ]

        return int(self.tie_rng.choice(best_actions))

    def update(self, s, a, r, s_next):
        if s_next == END:
            target = r
        else:
            target = r + GAMMA * max(self.Q[s_next].values())

        self.Q[s][a] += ALPHA * (
            target - self.Q[s][a]
        )