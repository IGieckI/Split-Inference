"""The three Phase-1 figures (section 11.8.8), each a query over the section 11.7 tables"""

import argparse
import pathlib
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "out"

# Validated reference palette (dataviz skill, light mode).
POLICY_COLORS = {"b0": "#2a78d6", "b2": "#1baf7a", "b3": "#eda100", "learned": "#008300"}
POLICY_LABELS = {"b0": "B0 full offload", "b2": "B2 always-deepest",
                 "b3": "B3-lite best static", "learned": "epsilon-greedy (learned)"}
ARM_COLORS = {"k0": "#2a78d6", "k_shallow": "#1baf7a", "k_deep": "#eda100"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11,
})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)


def df(db, sql, params=()):
    conn = sqlite3.connect(db)
    out = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return out


def newest(pattern):
    files = sorted((ROOT / "runs").glob(pattern), key=lambda p: p.stat().st_mtime)
    assert files, f"no runs match {pattern} - run the experiments first"
    return files[-1]


# figure 1: reward vs round (learning curve)
def learning_curve(learn_db, window=25):
    d = df(learn_db, "SELECT node_id, tier, reward FROM requests ORDER BY req_id")
    fig, ax = plt.subplots(figsize=(7, 4))
    tiers = sorted(d["tier"].unique())
    colors = dict(zip(tiers, ["#2a78d6", "#1baf7a", "#eda100"]))
    for tier in tiers:
        r = d[d["tier"] == tier]["reward"].reset_index(drop=True)
        roll = r.rolling(window, min_periods=5).mean()
        ax.plot(roll.index, roll, lw=2, color=colors[tier], label=f"tier {tier}")
        ax.annotate(f"tier {tier}", (len(roll) - 1, roll.iloc[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=colors[tier], fontsize=9, va="center")
    style(ax)
    ax.set_xlabel("per-node round")
    ax.set_ylabel(f"reward, rolling mean ({window})")
    ax.set_title("epsilon-greedy learning curve - reward = -T/T_ref, failures -3", color=INK)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=150)
    plt.close(fig)


# figure 2: per-node latency CDFs
def latency_cdfs(dbs: dict):
    nodes = df(dbs["learned"], "SELECT DISTINCT node_id, tier FROM requests "
                               "ORDER BY node_id").values
    fig, axes = plt.subplots(1, len(nodes), figsize=(11, 3.6), sharey=True)
    for ax, (node, tier) in zip(np.atleast_1d(axes), nodes):
        for key, db in dbs.items():
            lat = df(db, "SELECT t_total_ms FROM requests WHERE status='OK' "
                         "AND node_id=?", (int(node),))["t_total_ms"].sort_values()
            if lat.empty:
                continue
            y = np.arange(1, len(lat) + 1) / len(lat)
            ax.plot(lat, y, lw=2, color=POLICY_COLORS[key], label=POLICY_LABELS[key])
        style(ax)
        ax.set_title(f"node {node} (tier {tier})", color=INK2)
        ax.set_xlabel("end-to-end latency (ms)")
    np.atleast_1d(axes)[0].set_ylabel("CDF")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Per-node latency CDFs - learned vs. baselines", y=1.12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "latency_cdfs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# figure 3: arm frequency per tier (the money figure)
def arm_histogram(learn_db):
    d = df(learn_db,
           "SELECT tier, action, COUNT(*) n FROM requests WHERE status='OK' "
           "AND req_id > (SELECT MAX(req_id)/2 FROM requests) "
           "GROUP BY tier, action")
    tiers = sorted(d["tier"].unique())
    arms = [a for a in ARM_COLORS if a in set(d["action"])]
    share = {t: {a: 0.0 for a in arms} for t in tiers}
    for t in tiers:
        sub = d[d["tier"] == t]
        total = sub["n"].sum()
        for _, row in sub.iterrows():
            share[t][row["action"]] = row["n"] / total * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tiers))
    w = 0.26
    for i, arm in enumerate(arms):
        vals = [share[t][arm] for t in tiers]
        bars = ax.bar(x + (i - (len(arms) - 1) / 2) * (w + 0.02), vals, w,
                      color=ARM_COLORS[arm], label=arm, zorder=3,
                      edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            if v > 0.5:
                ax.annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=8.5, color=INK2)
    style(ax)
    ax.set_xticks(x, [f"tier {t}" for t in tiers])
    ax.set_ylabel("share of picks, exploitation half (%)")
    ax.set_title("The orchestrator learns a different split per tier", color=INK)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    fig.savefig(OUT / "arm_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary_table(dbs: dict):
    print(f"{'node':>5} {'policy':>22} {'n_ok':>6} {'mean ms':>9} {'p95 ms':>9}")
    for key, db in dbs.items():
        d = df(db, "SELECT node_id, t_total_ms FROM requests WHERE status='OK'")
        for node, grp in d.groupby("node_id"):
            lat = grp["t_total_ms"]
            print(f"{node:>5} {POLICY_LABELS[key]:>22} {len(lat):>6} "
                  f"{lat.mean():>9.1f} {lat.quantile(0.95):>9.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learn-db", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    dbs = {
        "b0": newest("*_baseline_b0_*.db"),
        "b2": newest("*_baseline_b2_*.db"),
        "b3": newest("*_baseline_b3_*.db"),
        "learned": pathlib.Path(args.learn_db) if args.learn_db else newest("*_learn_*.db"),
    }
    for k, v in dbs.items():
        print(f"{k}: {v.name}")
    learning_curve(dbs["learned"])
    latency_cdfs(dbs)
    arm_histogram(dbs["learned"])
    summary_table(dbs)
    print(f"\nfigures -> {OUT}")


if __name__ == "__main__":
    main()
