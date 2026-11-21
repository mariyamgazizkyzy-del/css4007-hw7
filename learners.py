"""Part D learners: YOU write these (the assignment requires it).

Test every update first on the supplied paper histories (data/paper-games.csv):
  TD(0)  final: A=-0.14003, B=-0.0181
  Q      final: Q(A,hit)=-0.13922, Q(B,stick)=-0.0181
Keep alpha=0.1, gamma=1.0, V(END)=0, Q(END,.)=0. Actions: 0=stick, 1=hit.
"""
from common import END

class ModelBasedEvaluator:
    """Count (s,a,s') -> P and mean R; synchronous sweeps with the fixed policy.
    Tolerance 1e-6, max 10,000 sweeps; report nonconvergence. Keep unobserved
    pairs marked as unobserved (do not fill with zeros)."""
    def update(self, s, a, r, s_next): raise NotImplementedError
    def evaluate(self, policy, tol=1e-6, max_sweeps=10_000): raise NotImplementedError

class MonteCarloEvaluator:
    """Every-visit Monte Carlo: after each game, walk backwards G = r + gamma*G."""
    def add_game(self, transitions): raise NotImplementedError

class TD0Evaluator:
    """V(s) += alpha * (r + gamma*V(s') - V(s)); V(END) = 0."""
    def update(self, s, r, s_next): raise NotImplementedError

class QLearner:
    """Epsilon-greedy with two seeded generators (explore branch, tie-breaking).
    Target = r + gamma * max(Q(s',hit), Q(s',stick)); terminal target = r."""
    def choose_action(self, s): raise NotImplementedError
    def update(self, s, a, r, s_next): raise NotImplementedError
