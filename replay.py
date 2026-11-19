"""Part C: replay the supplied paper histories. Run: python replay.py"""
import csv, sys
from collections import defaultdict

ALPHA, GAMMA = 0.1, 1.0
STATES = ["A", "B", "C", "D", "E"]
ACTIONS = [0, 1]  # 0 = stick, 1 = hit
POLICY = {"A": 1, "B": 0, "C": 1, "D": 1, "E": 0}  # hit below 20, stick at 20/21

def load(path="data/paper-games.csv"):
    games = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            games[int(r["game"])].append(
                (r["state"], int(r["action"]), r["next_state"], float(r["reward"])))
    return [games[k] for k in sorted(games)]

def model_based(games, sweeps=2):
    cnt = defaultdict(int); tot = defaultdict(lambda: defaultdict(int)); rsum = defaultdict(float)
    for g in games:
        for s, a, s2, r in g:
            cnt[(s, a)] += 1; tot[(s, a)][s2] += 1; rsum[(s, a, s2)] += r
    print("== Model-based: counts, T, R ==")
    for (s, a), n in sorted(cnt.items()):
        for s2, k in sorted(tot[(s, a)].items()):
            print(f"  ({s},{a})->{s2}: n={k}/{n}  T={k/n:.6f}  R={rsum[(s,a,s2)]/k:.6f}")
    V = {s: 0.0 for s in STATES}; out = []
    for i in range(1, sweeps + 1):
        new = {}
        for s in STATES:
            a = POLICY[s]; n = cnt.get((s, a), 0)
            if n == 0: new[s] = 0.0; continue
            new[s] = sum(k / n * (rsum[(s, a, s2)] / k + GAMMA * (V[s2] if s2 != "END" else 0))
                         for s2, k in tot[(s, a)].items())
        V = new; print(f"  V{i}:", {s: round(v, 6) for s, v in V.items()}); out.append(("MB", i, dict(V)))
    return out

def monte_carlo(games):
    rets = defaultdict(list)
    for g in games:
        G = 0.0; seq = []
        for s, a, s2, r in reversed(g):
            G = r + GAMMA * G; seq.append((s, G))
        for s, G in reversed(seq): rets[s].append(G)
    print("== Monte Carlo ==")
    res = {}
    for s in STATES:
        v = rets.get(s, []); res[s] = sum(v) / len(v) if v else None
        print(f"  {s}: returns={v} n={len(v)} mean={res[s]}")
    return res

def td(games, passes=2):
    V = {s: 0.0 for s in STATES}; V["END"] = 0.0
    print("== TD(0) ==")
    for p in range(1, passes + 1):
        for gi, g in enumerate(games, 1):
            for s, a, s2, r in g:
                old = V[s]; tgt = r + GAMMA * V[s2]; V[s] = old + ALPHA * (tgt - old)
                print(f"  pass{p} game{gi} {s}->{s2}: old={old:.6f} r={r} V'={V[s2]:.6f} target={tgt:.6f} new={V[s]:.6f}")
    return {s: V[s] for s in STATES}

def qlearn(games, passes=2):
    Q = {(s, a): 0.0 for s in STATES for a in ACTIONS}
    print("== Q-learning ==")
    for p in range(1, passes + 1):
        for gi, g in enumerate(games, 1):
            for s, a, s2, r in g:
                old = Q[(s, a)]
                nxt = 0.0 if s2 == "END" else max(Q[(s2, 0)], Q[(s2, 1)])
                tgt = r + GAMMA * nxt; Q[(s, a)] = old + ALPHA * (tgt - old)
                print(f"  pass{p} game{gi} ({s},{a}): old={old:.6f} r={r} maxQ'={nxt:.6f} target={tgt:.6f} new={Q[(s,a)]:.6f}")
    print("  final Q and greedy:")
    for s in STATES:
        q0, q1 = Q[(s, 0)], Q[(s, 1)]
        g = "tie" if q0 == q1 else ("stick" if q0 > q1 else "hit")
        print(f"  {s}: stick={q0:.6f} hit={q1:.6f} greedy={g}")
    return Q

if __name__ == "__main__":
    games = load()
    model_based(games); monte_carlo(games)
    v = td(games); print("  final TD V:", {k: round(x, 6) for k, x in v.items()})
    qlearn(games)
