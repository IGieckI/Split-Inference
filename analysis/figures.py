"""Every figure and table in the report"""

import argparse
import json
import pathlib
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "out"

# Categorical slots 1-4 of the validated reference palette
CUT_COLORS = {"k0": "#2a78d6", "k_shallow": "#eb6834", "k_deep": "#1baf7a"}
POLICY_COLORS = dict(CUT_COLORS, best="#eda100")
POLICY_LABELS = {"k0": "k0 (no split)", "k_shallow": "k_shallow",
                 "k_deep": "k_deep", "best": "best static"}

# Ordinal one-hue ramp for the stage stack
STAGES = ["device", "uplink", "queue", "server", "residual"]
STAGE_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
STAGE_HELP = {
    "device": "SPIFFS read + head inference on the MCU (device-reported)",
    "uplink": "dispatch to last fragment, minus device time (ASSIGN RTT, airtime, retransmits)",
    "queue": "waiting for the single-worker server tail",
    "server": "tail inference on the Pi (includes JPEG decode for k0)",
    "residual": "everything unaccounted for (RESULT dispatch, scheduler overhead)",
}

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


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis=axis, zorder=0)
    ax.set_axisbelow(True)


def df(db, sql, params=()):
    conn = sqlite3.connect(db)
    out = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return out


def newest(runs_dir, pattern):
    files = sorted(runs_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def ok_requests(db):
    d = df(db, "SELECT * FROM requests WHERE status='OK'")
    d["device"] = (d["t_capture_us"] + d["t_edge_us"]) / 1000
    d["uplink"] = ((d["t_last_frag"] - d["t_assign"]) * 1000 - d["device"]).clip(lower=0)
    d["queue"] = (d["t_queue_out"] - d["t_queue_in"]) * 1000
    d["server"] = d["t_tail_us"] / 1000
    d["residual"] = (d["t_total_ms"] - d[["device", "uplink", "queue", "server"]].sum(axis=1)).clip(lower=0)
    return d


def node_labels(d):
    return {int(n): f"node {int(n)} (tier {t})"
            for n, t in d[["node_id", "tier"]].drop_duplicates().values}


# figure 1: mean + p95 latency per node
def latency_by_cut(sweep):
    d = ok_requests(sweep)
    nodes = sorted(d["node_id"].unique())
    cuts = [c for c in CUT_COLORS if c in set(d["action"])]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(nodes))
    w = 0.8 / len(cuts)
    for i, cut in enumerate(cuts):
        off = (i - (len(cuts) - 1) / 2) * w
        means, p95s = [], []
        for n in nodes:
            lat = d[(d["node_id"] == n) & (d["action"] == cut)]["t_total_ms"]
            means.append(lat.mean() if len(lat) else np.nan)
            p95s.append(lat.quantile(0.95) if len(lat) else np.nan)
        bars = ax.bar(x + off, means, w * 0.92, color=CUT_COLORS[cut], label=cut,
                      zorder=3, edgecolor=SURFACE, linewidth=2)
        # p95 as a tick above the mean bar
        for xi, m, p in zip(x + off, means, p95s):
            if np.isnan(m):
                continue
            ax.plot([xi - w * 0.3, xi + w * 0.3], [p, p], color=INK2, lw=1.6, zorder=4)
        # label above the p95 rule
        for b, m, p in zip(bars, means, p95s):
            if not np.isnan(m):
                ax.annotate(f"{m:.0f}", (b.get_x() + b.get_width() / 2, max(m, p)),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=8.5, color=INK2)
    style(ax)
    labels = node_labels(d)
    ax.set_xticks(x, [labels[n] for n in nodes])
    ax.set_ylabel("end-to-end latency (ms)")
    ax.set_title("Latency per split point - bar = mean, rule = p95", color=INK)
    ax.legend(frameon=False, ncol=len(cuts), loc="upper center", bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    fig.savefig(OUT / "latency_by_cut.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# figure 2: where the time goes
def stage_breakdown(sweep):
    d = ok_requests(sweep)
    rows = (d.groupby(["node_id", "tier", "action"])[STAGES].mean()
             .reset_index().sort_values(["node_id", "action"]))
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 2.0))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for stage, color in zip(STAGES, STAGE_COLORS):
        vals = rows[stage].values
        ax.barh(y, vals, 0.68, left=left, color=color, label=stage,
                zorder=3, edgecolor=SURFACE, linewidth=2)
        left += vals
    for yi, total in zip(y, left):
        ax.annotate(f"{total:.0f} ms", (total, yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    ax.set_xlim(0, left.max() * 1.12)   # headroom so the total labels are not clipped
    style(ax, axis="x")
    ax.set_yticks(y, [f"node {int(r.node_id)} ({r.tier}) - {r.action}" for r in rows.itertuples()])
    ax.invert_yaxis()
    ax.set_xlabel("mean time per request (ms)")
    ax.set_title("Where end-to-end latency is spent, per node and split point", color=INK)
    ax.legend(frameon=False, ncol=len(STAGES), loc="upper center",
              bbox_to_anchor=(0.5, -0.9 / len(rows) - 0.04))
    fig.tight_layout()
    fig.savefig(OUT / "stage_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# figure 3: policy comparison
def policy_cdfs(policy_dbs: dict):
    any_db = next(iter(policy_dbs.values()))
    nodes = df(any_db, "SELECT DISTINCT node_id, tier FROM requests ORDER BY node_id").values
    fig, axes = plt.subplots(1, len(nodes), figsize=(3.7 * len(nodes), 3.6), sharey=True)
    for ax, (node, tier) in zip(np.atleast_1d(axes), nodes):
        for key, db in policy_dbs.items():
            lat = df(db, "SELECT t_total_ms FROM requests WHERE status='OK' AND node_id=?",
                     (int(node),))["t_total_ms"].sort_values()
            if lat.empty:
                continue
            ax.plot(lat, np.arange(1, len(lat) + 1) / len(lat), lw=2,
                    color=POLICY_COLORS[key], label=POLICY_LABELS[key])
        style(ax)
        ax.set_title(f"node {int(node)} (tier {tier})", color=INK2)
        ax.set_xlabel("end-to-end latency (ms)")
    np.atleast_1d(axes)[0].set_ylabel("CDF")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc="upper center",
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Latency distribution per splitting policy", y=1.15, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "policy_cdfs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# tables
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def tables(sweep, policy_dbs, cuts_json):
    d = ok_requests(sweep)
    allrows = df(sweep, "SELECT node_id, action, status FROM requests")
    md = ["<!-- generated by analysis/figures.py - do not edit by hand -->\n",
          "# Measured results\n"]

    # T1 - what each split point puts on the air
    md.append("\n## T1 - Bytes on the air per split point\n\n")
    t1 = []
    for cut in [c for c in CUT_COLORS if c in set(d["action"])]:
        sub = d[d["action"] == cut]
        declared = next((c.get("tensor_bytes") for c in cuts_json.get("cuts", [])
                         if c["name"] == cut), None)
        t1.append([cut, declared if declared else "JPEG (varies)",
                   f"{sub['bytes_on_air'].mean():.0f}",
                   f"{sub['n_frags'].mean():.2f}",
                   f"{sub['n_retx'].sum() / max(len(sub), 1):.3f}"])
    md.append(md_table(["cut", "activation bytes", "mean bytes on air",
                        "mean fragments", "retransmits / request"], t1))

    # T2 - the headline: latency per node per cut
    md.append("\n## T2 - End-to-end latency per node and split point (sweep)\n\n")
    t2 = []
    for (n, tier, cut), sub in d.groupby(["node_id", "tier", "action"]):
        att = allrows[(allrows["node_id"] == n) & (allrows["action"] == cut)]
        fail = (att["status"] != "OK").sum()
        t2.append([f"{int(n)} ({tier})", cut, len(sub),
                   f"{sub['t_total_ms'].mean():.1f}",
                   f"{sub['t_total_ms'].median():.1f}",
                   f"{sub['t_total_ms'].quantile(0.95):.1f}",
                   f"{fail / max(len(att), 1):.1%}"])
    md.append(md_table(["node", "cut", "n OK", "mean ms", "p50 ms", "p95 ms", "failed"], t2))

    # T3 - stage breakdown (the table view the contrast-relief rule requires)
    md.append("\n## T3 - Mean stage breakdown (ms)\n\n")
    t3 = [[f"{int(r.node_id)} ({r.tier})", r.action] + [f"{getattr(r, s):.1f}" for s in STAGES]
          for r in d.groupby(["node_id", "tier", "action"])[STAGES].mean()
                    .reset_index().itertuples()]
    md.append(md_table(["node", "cut"] + STAGES, t3))
    md.append("\n" + "\n".join(f"- **{k}** - {v}" for k, v in STAGE_HELP.items()) + "\n")

    # T4 - policy comparison (fleet-concurrent)
    if policy_dbs:
        md.append("\n## T4 - Policy comparison (whole fleet active)\n\n")
        t4 = []
        for key, db in policy_dbs.items():
            p = ok_requests(db)
            for (n, tier), grp in p.groupby(["node_id", "tier"]):
                t4.append([POLICY_LABELS[key], f"{int(n)} ({tier})",
                           "/".join(sorted(set(grp["action"]))), len(grp),
                           f"{grp['t_total_ms'].mean():.1f}",
                           f"{grp['t_total_ms'].quantile(0.95):.1f}"])
        md.append(md_table(["policy", "node", "cut actually run", "n OK", "mean ms", "p95 ms"], t4))

    # T5 - the cost of sharing the server
    if policy_dbs:
        md.append("\n## T5 - Cost of running the fleet concurrently\n\n")
        iso = d.groupby(["node_id", "action"])["t_total_ms"].mean()
        t5 = []
        for key, db in policy_dbs.items():
            p = ok_requests(db)
            for (n, tier), grp in p.groupby(["node_id", "tier"]):
                cuts_run = set(grp["action"])
                if len(cuts_run) != 1 or (n, (cut := cuts_run.pop())) not in iso.index:
                    continue
                a, b = iso.loc[(n, cut)], grp["t_total_ms"].mean()
                t5.append([POLICY_LABELS[key], f"{int(n)} ({tier})", cut,
                           f"{a:.1f}", f"{b:.1f}", f"{(b - a) / a * 100:+.1f}%"])
        md.append(md_table(["policy", "node", "cut", "isolated (T2)", "concurrent (T4)",
                            "difference"], t5))
        md.append("\nThe nodes share one single-worker server tail. This table is how much "
                  "that costs: the same node running the same cut, measured alone in the "
                  "sweep versus measured with the whole fleet active under each policy. "
                  "A policy whose tail work is expensive - k0 runs the entire model plus a "
                  "JPEG decode on the server - pays here and makes the other nodes pay too.\n")

    (OUT / "results.md").write_text("".join(md))
    print((OUT / "results.md").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = ROOT / args.runs_dir

    sweep = newest(runs, "*_sweep.db")
    assert sweep, f"no sweep run in {runs} - run `--policy sweep` first"
    policy_dbs = {}
    for key in POLICY_COLORS:
        db = newest(runs, f"*_{key}.db")
        if db:
            policy_dbs[key] = db
    print(f"sweep: {sweep.name}")
    for k, v in policy_dbs.items():
        print(f"{k}: {v.name}")

    cuts_path = ROOT / "model" / "artifacts" / "cuts.json"
    cuts_json = json.loads(cuts_path.read_text()) if cuts_path.exists() else {}

    latency_by_cut(sweep)
    stage_breakdown(sweep)
    if policy_dbs:
        policy_cdfs(policy_dbs)
    tables(sweep, policy_dbs, cuts_json)
    print(f"figures + results.md -> {OUT}")


if __name__ == "__main__":
    main()
