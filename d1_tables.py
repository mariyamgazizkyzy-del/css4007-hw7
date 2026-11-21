"""Part D1 helper: turn outputs/fixed_policy_estimates/*.csv into readable tables.

Run:  python d1_tables.py
Needs: outputs/fixed_policy_estimates/seed<seed>_games<N>.csv   (from d1_estimate.py)
Output: outputs/fixed_policy_estimates/tables_seed<seed>_ace.md / _noace.md
            value tables after 10,000 games (visits + model-based / MC / TD(0)),
            empty estimate written as 'missing'
        outputs/fixed_policy_estimates/d1_summary.md
            per seed: states present and missing estimates at 100/1,000/10,000 games,
            mean |difference| between methods, most and least visited states.
Only reads saved results; nothing is re-simulated.
"""
import csv, os
SEEDS = [11, 22, 33]
CPS = [100, 1_000, 10_000]
METHODS = ["model_based", "monte_carlo", "td0"]
DIR = os.path.join("outputs", "fixed_policy_estimates")


def load(seed, n):
    data = {}
    with open(os.path.join(DIR, f"seed{seed}_games{n}.csv")) as f:
        for r in csv.DictReader(f):
            s = (int(r["player_sum"]), int(r["dealer"]), int(r["usable_ace"]))
            d = data.setdefault(s, {"visits": int(r["visits"])})
            d[r["method"]] = float(r["value"]) if r["value"] != "" else None
    return data


def fmt(v):
    return "missing" if v is None else f"{v:+.3f}"


def write_tables(seed, data):
    for ace, name in ((0, "noace"), (1, "ace")):
        path = os.path.join(DIR, f"tables_seed{seed}_{name}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Seed {seed}, after 10,000 games, usable ace = {ace}. "
                    "visits = nonterminal visits; 'missing' = no estimate; "
                    "states not listed were never visited.\n\n")
            f.write("| player sum | dealer | visits | model-based | Monte Carlo | TD(0) |\n")
            f.write("| --- | --- | --- | --- | --- | --- |\n")
            for s in sorted(k for k in data if k[2] == ace):
                d = data[s]
                f.write(f"| {s[0]} | {s[1]} | {d['visits']} | "
                        + " | ".join(fmt(d.get(m)) for m in METHODS) + " |\n")


def mean_abs_diff(data, a, b):
    xs = [abs(d[a] - d[b]) for d in data.values()
          if d.get(a) is not None and d.get(b) is not None]
    return (sum(xs) / len(xs)) if xs else float("nan")


def main():
    out = []
    for seed in SEEDS:
        by_cp = {n: load(seed, n) for n in CPS}
        write_tables(seed, by_cp[10_000])
        out.append(f"## Seed {seed}\n")
        out.append("| games | states with estimates | missing model-based | missing MC | missing TD(0) "
                   "| mean abs diff MB-MC | MB-TD | MC-TD |")
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for n in CPS:
            d = by_cp[n]
            miss = [sum(1 for v in d.values() if v.get(m) is None) for m in METHODS]
            out.append(f"| {n:,} | {len(d)} | {miss[0]} | {miss[1]} | {miss[2]} | "
                       f"{mean_abs_diff(d,'model_based','monte_carlo'):.4f} | "
                       f"{mean_abs_diff(d,'model_based','td0'):.4f} | "
                       f"{mean_abs_diff(d,'monte_carlo','td0'):.4f} |")
        final = by_cp[10_000]
        ranked = sorted(final, key=lambda s: final[s]["visits"], reverse=True)
        for title, group in (("Most visited states", ranked[:5]), ("Least visited states", ranked[-5:])):
            out.append(f"\n{title} (values after 100 / 1,000 / 10,000 games; '-' = not yet visited):\n")
            out.append("| state (sum, dealer, ace) | visits@10,000 | model-based | Monte Carlo | TD(0) |")
            out.append("| --- | --- | --- | --- | --- |")
            for s in group:
                cells = []
                for m in METHODS:
                    vals = []
                    for n in CPS:
                        d = by_cp[n].get(s)
                        vals.append("-" if d is None else fmt(d.get(m)))
                    cells.append(" / ".join(vals))
                out.append(f"| {s} | {final[s]['visits']} | " + " | ".join(cells) + " |")
        out.append("")
    path = os.path.join(DIR, "d1_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n".join(out))
    print("\nsaved", path, "and tables_seed*_ace/noace.md")


if __name__ == "__main__":
    main()
