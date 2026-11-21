"""Shared helpers for Part D. Environment setup only; no learning code here."""
import gymnasium as gym

ENV_ID = "Blackjack-v1"
SAFETY_LIMIT = 100          # max player actions per game
END = "END"                 # terminal marker given to learners

def make_env():
    return gym.make(ENV_ID, natural=False, sab=False)

def fixed_policy(obs):
    """Hit below 20, stick at 20 or 21. 0 = stick, 1 = hit."""
    return 1 if obs[0] < 20 else 0

def obs_key(obs):
    """Gymnasium observation -> hashable (player_sum, dealer_showing, usable_ace)."""
    return (int(obs[0]), int(obs[1]), int(obs[2]))
