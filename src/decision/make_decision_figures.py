"""
Figures for the decision-layer companion paper (figures/decision/).

  fig_d1_geofence_gap.png       optimality gap vs solve time, by CO method
  fig_d2_geofence_map.png       geofences on the held-out city's service region
  fig_d3_regret.png             regret by agent x belief kernel, both DGPs
  fig_d4_kernel_value.png       paired regret difference learned - independent
  fig_d5_adaptive_design.png    LOCO kernel discrimination vs cities entered (Q2)
  fig_d6_training.png           GNN and PPO training curves

Usage:  python src/decision/make_decision_figures.py [--quick]
Missing result files are skipped with a message.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision.config import RESULTS, FIGURES  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 170, "savefig.bbox": "tight",
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.frameon": False,
})
# Categorical slots in fixed order (first three validate all-pairs).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
KERNEL_COLOR = {"learned": SERIES[0], "euclidean": SERIES[1], "independent": SERIES[2],
                "learned_nugget": SERIES[3]}
KERNEL_LABEL = {"learned": "learned", "learned_nugget": "learned + nugget",
                "euclidean": "Euclidean", "independent": "independent"}
INK, MUTED, FILL_OFF = "#1f1f1e", "#6b6a63", "#e4e2da"
DGP_TITLE = {"structured": "Structured DGP (alpha ~ N(0, sigma^2 S_learned))",
             "identity_null": "Null DGP (alpha ~ N(0, sigma^2 I))"}
AGENT_ORDER = ["oracle", "ppo", "kg", "bayes", "thompson", "ucb", "plugin",
               "round_robin", "random"]
METHOD_LABEL = {"greedy": "Greedy", "greedy_ls": "Greedy + LS", "gnn_greedy": "GNN (argmax)",
                "gnn_sample": "GNN (64 samples) + LS", "gnn_milp": "GNN-restricted MILP",
                "milp": "Full MILP"}


def _read(name, tag):
    p = RESULTS / f"{name}{tag}.csv"
    if not p.exists():
        print(f"skip: {p.name} not found")
        return None
    return pd.read_csv(p)


def fig_geofence_gap(tag):
    df = _read("geofence_benchmark", tag)
    if df is None:
        return
    s = df.groupby("method").agg(gap=("gap_bound_pct", "mean"), t=("time_s", "mean"))
    s = s.reindex([m for m in METHOD_LABEL if m in s.index])
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.scatter(s.t, s.gap, s=60, color=SERIES[0], edgecolor="white", linewidth=2, zorder=3)
    for m, r in s.iterrows():
        ax.annotate(METHOD_LABEL[m], (r.t, r.gap), xytext=(7, 4), textcoords="offset points",
                    fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Mean solve time per instance (s, log scale)")
    ax.set_ylabel("Gap to MILP dual bound\n(% of full-region premium)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Geofence CO on held-out city ({df.instance.nunique()} instances)", loc="left")
    fig.savefig(FIGURES / f"fig_d1_geofence_gap{tag}.png"); plt.close(fig)


def fig_geofence_map(tag):
    df = _read("geofence_maps", tag)
    if df is None:
        return
    polys = [np.array([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in df.cell_id]
    cols = [c for c in ["greedy_ls", "gnn_sample", "milp"] if c in df]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.3 * len(cols), 3.4))
    for ax, c in zip(np.atleast_1d(axes), cols):
        face = [SERIES[0] if v else FILL_OFF for v in df[c]]
        ax.add_collection(PolyCollection(polys, facecolors=face, edgecolors="white", linewidths=0.4))
        ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{METHOD_LABEL[c]}\n{int(df[c].sum())} cells, "
                     f"{100 * df.m[df[c] > 0].sum():.0f}% of exposure", fontsize=9.5)
    fig.suptitle("Geofences on the held-out city's candidate service area (selected cells in blue)",
                 fontsize=10, x=0.02, ha="left")
    fig.savefig(FIGURES / f"fig_d2_geofence_map{tag}.png"); plt.close(fig)


def fig_regret(tag):
    s = _read("expansion_summary", tag)
    if s is None:
        return
    agents = [a for a in AGENT_ORDER if a in set(s.agent)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)
    for ax, dgp in zip(axes, ["structured", "identity_null"]):
        sub = s[s.dgp == dgp]
        kernels = [k for k in KERNEL_COLOR if k in set(s.kernel)]
        for j, k in enumerate(kernels):
            col = KERNEL_COLOR[k]
            r = sub[sub.kernel == k].set_index("agent").reindex(agents)
            y = np.arange(len(agents)) + (j - (len(kernels) - 1) / 2) * 0.19
            ok = r.regret.notna().to_numpy()
            ax.errorbar(r.regret[ok], y[ok], xerr=[(r.regret - r.regret_lo)[ok], (r.regret_hi - r.regret)[ok]],
                        fmt="o", ms=5, color=col, ecolor=col, elinewidth=1.5, capsize=0,
                        label=KERNEL_LABEL[k] if dgp == "structured" else None)
        ax.set_yticks(np.arange(len(agents))); ax.set_yticklabels(agents)
        ax.set_xlabel("Regret vs clairvoyant (expected-claims profit)")
        ax.set_title(DGP_TITLE[dgp], loc="left", fontsize=10)
        ax.grid(axis="y", alpha=0)
    axes[0].invert_yaxis()                       # shared y: invert once
    fig.legend(*axes[0].get_legend_handles_labels(), title="Belief kernel",
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.07))
    fig.savefig(FIGURES / f"fig_d3_regret{tag}.png"); plt.close(fig)


def fig_kernel_value(tag):
    kv = _read("kernel_value", tag)
    if kv is None:
        return
    comps = [("learned_minus_independent", "learned", "learned"),
             ("learned_nugget_minus_independent", "learned_nugget", "learned + nugget")]
    agents = [a for a in AGENT_ORDER if a in set(kv.agent)]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True, sharex=True)
    for ax, dgp in zip(axes, ["structured", "identity_null"]):
        ax.axvline(0, color=MUTED, lw=1)
        for j, (comp, kern, lab) in enumerate(comps):
            r = kv[(kv.dgp == dgp) & (kv.comparison == comp)].set_index("agent").reindex(agents)
            y = np.arange(len(agents)) + (j - 0.5) * 0.25
            ok = r.mean_regret_diff.notna().to_numpy()
            ax.errorbar(r.mean_regret_diff[ok], y[ok],
                        xerr=[(r.mean_regret_diff - r.lo)[ok], (r.hi - r.mean_regret_diff)[ok]],
                        fmt="o", ms=5.5, color=KERNEL_COLOR[kern], elinewidth=1.5, capsize=0,
                        label=f"{lab} minus independent" if dgp == "structured" else None)
        ax.set_yticks(np.arange(len(agents))); ax.set_yticklabels(agents)
        ax.set_title(DGP_TITLE[dgp], loc="left", fontsize=10); ax.grid(axis="y", alpha=0)
        ax.set_xlabel("Paired regret difference (95% CI)\n(< 0: kernel decides better than independent)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    axes[0].invert_yaxis()                       # shared y: invert once
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06))
    fig.savefig(FIGURES / f"fig_d4_kernel_value{tag}.png"); plt.close(fig)


def fig_adaptive(tag):
    s = _read("adaptive_design_summary", tag)
    if s is None:
        return
    panels = [("p_win_learned_vs_independent", "Learned vs independent RE"),
              ("p_win_learned_nugget_vs_independent", "Learned + nugget vs independent RE"),
              ("p_win_learned_nugget_vs_euclidean_nugget", "Learned + nugget vs Euclidean + nugget")]
    panels = [p for p in panels if p[0] in s]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 3.8), sharey=True)
    agents = [a for a in ["round_robin", "kg", "bayes", "thompson"] if a in set(s.agent)]
    for ax, (col, title) in zip(np.atleast_1d(axes), panels):
        for i, a in enumerate(agents):
            r = s[s.agent == a].sort_values("k")
            ax.plot(r.k, r[col], "-o", lw=2, ms=5, color=SERIES[i], label=a.replace("_", "-"),
                    markeredgecolor="white", markeredgewidth=1.5)
        ax.axhline(0.8, color=MUTED, lw=1, ls="--"); ax.axhline(0.5, color=MUTED, lw=0.8, ls=":")
        ax.set_xlabel("Cities entered (incl. 4 deployed)"); ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylim(0, 1)
    np.atleast_1d(axes)[0].set_ylabel("P(first kernel wins LOCO log-lik.)")
    fig.legend(*np.atleast_1d(axes)[0].get_legend_handles_labels(), title="Expansion policy",
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1))
    fig.savefig(FIGURES / f"fig_d5_adaptive_design{tag}.png"); plt.close(fig)


def fig_training(tag):
    g = _read("geofence_gnn_history", tag)
    p = {k: _read(f"ppo_{k}_history", tag) for k in ["learned", "independent"]}
    if g is None and all(v is None for v in p.values()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    if g is not None and "gap_vs_ref" in g:
        w = max(1, len(g) // 40)
        axes[0].plot(g.step, g.gap_vs_ref.rolling(w).mean(), color=SERIES[0], lw=2)
        axes[0].axhline(0, color=MUTED, lw=1)
        axes[0].set_xlabel("Training instance")
        axes[0].set_ylabel("Greedy+LS J minus best-of-16 GNN J\n(> 0: GNN worse)")
        axes[0].set_title("GNN geofence policy (REINFORCE)", loc="left")
    for k, h in p.items():
        if h is not None:
            w = max(1, len(h) // 30)
            axes[1].plot(h.iter, h.mean_exp_profit.rolling(w).mean(), color=KERNEL_COLOR[k], lw=2, label=k)
    axes[1].set_xlabel("PPO iteration (32 episodes)"); axes[1].set_ylabel("Mean episode expected profit")
    axes[1].set_title("PPO expansion policy (randomized worlds)", loc="left")
    axes[1].legend(title="Belief kernel")
    fig.savefig(FIGURES / f"fig_d6_training{tag}.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    tag = "_quick" if ap.parse_args().quick else ""
    FIGURES.mkdir(parents=True, exist_ok=True)
    for f in [fig_geofence_gap, fig_geofence_map, fig_regret, fig_kernel_value, fig_adaptive, fig_training]:
        f(tag)
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
