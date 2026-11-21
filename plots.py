"""Part D plots: training learning curves and evaluation results, shown separately.

Run:  python plots.py
Needs: outputs/qlearning/learning_curve.csv   (from d2_qlearn.py)
       outputs/evaluation/eval_results.csv    (from d3_eval.py)
Output: outputs/plots/learning_curve_training.png
        outputs/plots/evaluation_results.png

Training curve: mean reward per 1,000-game block WHILE learning (epsilon=0.2 includes
exploration moves). Evaluation: greedy, no exploration, 10,000 new games per eval seed.
The two are different quantities, so they are drawn in separate figures.
"""
import csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVE = os.path.join("outputs", "qlearning", "learning_curve.csv")
EVAL = os.path.join("outputs", "evaluation", "eval_results.csv")
PLOT_DIR = os.path.join("outputs", "plots")


def plot_training():
    curves = defaultdict(list)                      # (eps, seed) -> [(games_end, mean)]
    with open(CURVE) as f:
        for r in csv.DictReader(f):
            curves[(float(r["epsilon"]), int(r["seed"]))].append(
                (int(r["games_end"]), float(r["mean_reward"])))
    epsilons = sorted({k[0] for k in curves}, reverse=True)
    fig, axes = plt.subplots(1, len(epsilons), figsize=(6 * len(epsilons), 4.5), sharey=True)
    if len(epsilons) == 1:
        axes = [axes]
    for ax, eps in zip(axes, epsilons):
        for (e, seed), pts in sorted(curves.items()):
            if e == eps:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], label=f"seed {seed}")
        ax.set_title(f"Training: epsilon = {eps}")
        ax.set_xlabel("Games played")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Mean reward per game (1,000-game blocks)")
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "learning_curve_training.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def plot_evaluation():
    by_policy = defaultdict(list)                   # policy -> [mean_reward per eval seed]
    order = []
    with open(EVAL) as f:
        for r in csv.DictReader(f):
            if r["policy"] not in by_policy:
                order.append(r["policy"])
            by_policy[r["policy"]].append(float(r["mean_reward"]))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    xs = range(len(order))
    means = [sum(by_policy[p]) / len(by_policy[p]) for p in order]
    ax.bar(xs, means, color=["tab:gray" if p.startswith("baseline") else "tab:blue"
                             for p in order], alpha=0.7)
    for x, p in zip(xs, order):                     # individual eval seeds as dots
        ax.scatter([x] * len(by_policy[p]), by_policy[p], color="black", s=14, zorder=3)
    base = [m for p, m in zip(order, means) if p.startswith("baseline")]
    if base:
        ax.axhline(base[0], color="tab:gray", linestyle="--", label="baseline mean")
        ax.legend()
    ax.set_xticks(list(xs))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Mean reward per game (greedy, no exploration)")
    ax.set_title("Evaluation on 10,000 new games per eval seed (bars = mean, dots = eval seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "evaluation_results.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_training()
    plot_evaluation()
