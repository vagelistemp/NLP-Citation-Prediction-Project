"""Draw the documentation figures from results/real_data_results.json (aggregates only).

    python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
_RES = ROOT / "results" / "real_data_results.json"
R = json.loads(_RES.read_text(encoding="utf-8")) if _RES.exists() else {}
BLUE, RED, GREEN, ORANGE, GREY, INK = "#2b6cb0", "#d1495b", "#2f855a", "#e09f1f", "#718096", "#1f2933"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#dde3ea",
                     "grid.linewidth": 0.7, "font.size": 9, "figure.dpi": 130})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def protocol_comparison():
    comp = R["protocol_comparison_original9"]
    names = ["leaky", "leak-free"]
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    for k, (label, color, key) in enumerate((("cross-validation estimate", BLUE, "cv"), ("held-out test pairs", ORANGE, "test"))):
        vals = [comp[n][key]["logloss"] for n in names]
        bars = ax.bar([i + (k - 0.5) * 0.36 for i in range(2)], vals, 0.36, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["leaky protocol\n(full-graph features)", "leak-free protocol\n(edge-out-of-fold)"])
    ax.set_ylabel("log loss (lower is better)")
    ax.set_title("Same nine features, two protocols")
    ax.legend(frameon=False)
    save(fig, "protocol_comparison.png")


def ablation():
    cum = R["cumulative_ablation"]
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    ax.plot(range(len(cum)), [c["test"]["logloss"] for c in cum], "o-", color=ORANGE, label="held-out test pairs")
    ax.plot(range(len(cum)), [c["cv"]["logloss"] for c in cum], "s--", color=BLUE, alpha=0.7, label="cross-validation")
    ax.set_xticks(range(len(cum)))
    ax.set_xticklabels([c["step"] for c in cum], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("log loss")
    ax.set_title("Adding feature groups (leak-free protocol)")
    ax.legend(frameon=False)
    save(fig, "ablation_cumulative.png")
    loo = sorted(R["leave_one_group_out"], key=lambda r: r["test"]["logloss"])
    full = R["cumulative_ablation"][-1]["test"]["logloss"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh([r["removed_group"] for r in loo], [r["test"]["logloss"] - full for r in loo], color=RED)
    ax.set_xlabel("increase in test log loss when the group is removed")
    ax.set_title("Leave-one-group-out")
    save(fig, "ablation_leave_one_out.png")


def importance():
    imp = R["importance"][:15][::-1]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.barh([r["feature"] for r in imp], [r["share"] for r in imp], color=BLUE)
    ax.set_xlabel("share of total gain")
    ax.set_title("Top 15 features")
    save(fig, "importance.png")


def calibration():
    cal = R["calibration"]
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot([0, 1], [0, 1], ":", color=GREY)
    ax.plot([c["mean_predicted"] for c in cal], [c["observed_rate"] for c in cal], "o-", color=BLUE)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed citation rate")
    ax.set_title("Calibration on held-out pairs")
    save(fig, "calibration.png")


def schematic():
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=INK, lw=1.3))

    ax.text(0.1, 6.45, "Leaky (first notebook)", fontsize=9.5, fontweight="bold", color=RED)
    box(0.1, 4.5, 3.6, 1.4, "all known edges\n(graph G)", "#edf0f3", GREY)
    box(4.5, 4.5, 5.4, 1.4, "pair (u, v) = a known edge;\nfeatures computed on G, so the\npair's own edge is inside G", "#fbe0e4", RED)
    box(10.7, 4.5, 4.4, 1.4, "classifier learns the shortcut\n\"has a reference = citation\"", "#fbe0e4", RED)
    arrow(3.7, 5.2, 4.5, 5.2)
    arrow(9.9, 5.2, 10.7, 5.2)
    ax.text(15.6, 5.2, "CV looks great,\nheld-out test is poor", fontsize=8.5, color=RED, va="center")
    ax.text(0.1, 3.5, "Leak-free (edge-out-of-fold)", fontsize=9.5, fontweight="bold", color=GREEN)
    box(0.1, 1.0, 3.6, 1.8, "fold f: hide the fold's\npositive edges\n(about 4.7% of all edges)", "#edf0f3", GREY)
    box(4.5, 1.0, 5.4, 1.8, "graph G_f = all edges except\nthe positives of fold f;\nfeatures of fold-f pairs on G_f", "#dff0e6", GREEN)
    box(10.7, 1.0, 4.4, 1.8, "train on the other folds,\nscore fold f\n(early stopping on fold f+1)", "#dff0e6", GREEN)
    arrow(3.7, 1.9, 4.5, 1.9)
    arrow(9.9, 1.9, 10.7, 1.9)
    ax.text(15.6, 1.9, "the CV estimate predicts the\nheld-out score; test pairs use\nthe full known graph", fontsize=8.5, color=GREEN, va="center")
    save(fig, "protocol_schematic.png")


if __name__ == "__main__":
    schematic()
    if (ROOT / "results" / "real_data_results.json").exists():
        protocol_comparison()
        ablation()
        importance()
        calibration()
    print("figures written to", OUT)
