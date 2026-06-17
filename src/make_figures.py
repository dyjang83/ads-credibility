"""
Figure generation for the ADS credibility paper.

Generates:
  fig_01_embedding_pca.png                2-D PCA of cell embeddings
  fig_02_exposure_distribution.png        Exposure share by city  
  fig_03_credibility_weights.png          Z by exposure under BS vs hierarchical
  fig_04_similarity_heatmap.png           Learned ODD-similarity matrix
  fig_05_posterior_lambda.png             Posterior lambda by (city, version)
  fig_06_prospective_estimates.png        Posterior predictive for new cities
  fig_07_update_first_million.png         Prior vs posterior after 1M miles
  fig_08_power_analysis.png               Forward-simulation power analysis
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_generator import (  # noqa: E402
    DEPLOYED_CITIES,
    HYPOTHETICAL_CITIES,
    SOFTWARE_VERSIONS,
)

# Consistent styling
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 170,
    "savefig.bbox": "tight",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.frameon": False,
})

CITY_COLOR = {
    "San Francisco": "#1f77b4",
    "Phoenix":       "#d62728",
    "Los Angeles":   "#9467bd",
    "Austin":        "#2ca02c",
    "Miami":         "#ff7f0e",
    "Boston":        "#17becf",
    "Denver":        "#8c564b",
}


def fig_exposure_distribution(ads: pd.DataFrame, out: Path) -> None:
    by_city = (ads.groupby("city")["exposure_million_miles"]
                  .sum().reindex(DEPLOYED_CITIES))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = [CITY_COLOR[c] for c in by_city.index]
    bars = ax.bar(by_city.index, by_city.values, color=colors)
    for bar, val in zip(bars, by_city.values):
        ax.text(bar.get_x() + bar.get_width()/2, val,
                f"{val:.1f}M", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Exposure (million autonomous miles)")
    ax.set_title("ADS exposure by city (SGO window: Jun 2025 - Apr 2026, window-matched denominators)")
    ax.set_ylim(0, by_city.max() * 1.15)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_benchmark_reduction(bench: dict, out: Path) -> None:
    rows = pd.DataFrame(bench["by_city"])
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(rows))
    w = 0.38
    ax.bar(x - w/2, rows["hdv_freq_per_million_miles"], w, label="HDV", color="#bbbbbb")
    ax.bar(x + w/2, rows["ads_freq_per_million_miles"], w, label="ADS", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(rows["city"])
    ax.set_ylabel("Claims per million miles")
    title_red = bench["overall_reduction_vs_hdv"] * 100
    # Positive title_red = ADS below HDV; negative = ADS above HDV (SGO threshold effect)
    direction = "reduction" if title_red > 0 else "increase"
    ax.set_title(f"ADS SGO crash frequency vs. HDV liability frequency by city\n"
                 f"(ADS/HDV ratio: {abs(title_red):.0f}% {direction} -- see Section 6.2 for threshold analysis)")
    ax.legend()
    for xi, (_, row) in enumerate(rows.iterrows()):
        if row["reduction_vs_hdv"] is not None:
            val = row["reduction_vs_hdv"] * 100
            # positive = ADS below HDV (good); negative = ADS above HDV (SGO threshold effect)
            label = f"{val:+.0f}%" if val < 0 else f"-{val:.0f}%"
            color = "#d62728" if val < 0 else "#2ca02c"
            ax.annotate(label,
                        (xi + w/2, row["ads_freq_per_million_miles"]),
                        xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=8, color=color)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_similarity_heatmap(S: pd.DataFrame, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    order = DEPLOYED_CITIES + HYPOTHETICAL_CITIES
    M = S.loc[order, order].to_numpy()
    im = ax.imshow(M, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(order))); ax.set_yticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right")
    ax.set_yticklabels(order)
    for i in range(len(order)):
        for j in range(len(order)):
            txt_color = "white" if M[i, j] > 0.55 else "black"
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color=txt_color)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="similarity")
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_similarity_compare(S_emb: pd.DataFrame, S_eu: pd.DataFrame, out: Path) -> None:
    order = DEPLOYED_CITIES + HYPOTHETICAL_CITIES
    abbr = {"San Francisco": "SF", "Phoenix": "PHX", "Los Angeles": "LAX",
            "Austin": "AUS", "Miami": "MIA", "Boston": "BOS", "Denver": "DEN"}
    pairs = [(i, j) for i in range(len(order)) for j in range(i+1, len(order))]
    e = [S_emb.loc[order[i], order[j]] for i, j in pairs]
    u = [S_eu.loc[order[i], order[j]] for i, j in pairs]
    labels = [f"{abbr[order[i]]}-{abbr[order[j]]}" for i, j in pairs]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(u, e, s=22, color="#1f77b4")
    for x, y, lab in zip(u, e, labels):
        ax.annotate(lab, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle="--")
    ax.set_xlabel("Euclidean-kernel similarity")
    ax.set_ylabel("Learned-embedding similarity")
    ax.set_title("Cross-city similarity: learned vs. hand-crafted")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def _heatmap_panel(ax, S: pd.DataFrame, title: str, order: list,
                   abbr: dict, tick_labels: bool = True) -> object:
    """Draw a single similarity heatmap on the given axes; return the image."""
    M = S.loc[order, order].to_numpy()
    im = ax.imshow(M, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    short = [abbr[c] for c in order]
    ax.set_xticks(np.arange(len(order)))
    ax.set_yticks(np.arange(len(order)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short if tick_labels else [""] * len(order), fontsize=7)
    n_dep = len(DEPLOYED_CITIES)
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(False)
    # orange tick marks separating deployed from hypothetical cities
    ax.axhline(n_dep - 0.5, color="darkorange", lw=1.2, linestyle="--", alpha=0.7)
    ax.axvline(n_dep - 0.5, color="darkorange", lw=1.2, linestyle="--", alpha=0.7)
    for i in range(len(order)):
        for j in range(len(order)):
            txt_col = "white" if M[i, j] > 0.55 else "black"
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.5, color=txt_col)
    ax.set_title(title, fontsize=8, pad=4)
    return im


def fig_similarity_panel(S_emb: pd.DataFrame, S_eu: pd.DataFrame,
                          S_hdv: pd.DataFrame, out: Path) -> None:
    """Figure 4: three-kernel heatmaps (Panels A–C) + learned-vs-baseline
    scatter (Panel D). Panel A must match Figure 3 exactly."""
    order = DEPLOYED_CITIES + HYPOTHETICAL_CITIES
    abbr = {"San Francisco": "SF", "Phoenix": "PHX", "Los Angeles": "LAX",
            "Austin": "AUS", "Miami": "MIA", "Boston": "BOS", "Denver": "DEN"}
    pairs = [(i, j) for i in range(len(order)) for j in range(i+1, len(order))]

    fig = plt.figure(figsize=(16.5, 9))
    # Reserve right margin explicitly so the colorbar never overlaps Panel C.
    # subplots_adjust gives precise control; the colorbar is placed via
    # fig.add_axes() at absolute figure coordinates.
    fig.subplots_adjust(left=0.05, right=0.86, top=0.92, bottom=0.52,
                        wspace=0.30)

    ax_a = fig.add_axes([0.05, 0.52, 0.25, 0.40])
    ax_b = fig.add_axes([0.33, 0.52, 0.25, 0.40])
    ax_c = fig.add_axes([0.61, 0.52, 0.25, 0.40])
    ax_cb = fig.add_axes([0.88, 0.55, 0.015, 0.34])  # colorbar — clear of Panel C
    ax_d = fig.add_axes([0.05, 0.05, 0.92, 0.38])    # scatter

    _heatmap_panel(ax_a, S_emb, "A. Learned ODD-similarity", order, abbr)
    im = _heatmap_panel(ax_b, S_eu, "B. Euclidean-kernel similarity", order, abbr,
                        tick_labels=False)
    _heatmap_panel(ax_c, S_hdv, "C. HDV-only-kernel similarity", order, abbr,
                   tick_labels=False)
    fig.colorbar(im, cax=ax_cb, label="similarity")

    # Panel D: scatter, two series (learned vs Euclidean, learned vs HDV-only)
    e_emb = [S_emb.loc[order[i], order[j]] for i, j in pairs]
    e_eu  = [S_eu.loc[order[i], order[j]] for i, j in pairs]
    e_hdv = [S_hdv.loc[order[i], order[j]] for i, j in pairs]
    labs  = [f"{abbr[order[i]]}-{abbr[order[j]]}" for i, j in pairs]
    r_eu  = float(np.corrcoef(e_emb, e_eu)[0, 1])
    r_hdv = float(np.corrcoef(e_emb, e_hdv)[0, 1])

    ax_d.scatter(e_eu, e_emb, s=28, color="#1f77b4", label="Learned vs. Euclidean",
                 marker="o", alpha=0.85)
    ax_d.scatter(e_hdv, e_emb, s=28, color="#d62728", label="Learned vs. HDV-only",
                 marker="^", alpha=0.85)
    for x, y, lab in zip(e_eu, e_emb, labs):
        ax_d.annotate(lab, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
    ax_d.plot([0, 1], [0, 1], color="gray", lw=0.8, linestyle="--")
    ax_d.axhline(0.5, color="gray", lw=0.6, linestyle=":")
    ax_d.axvline(np.median(e_eu), color="gray", lw=0.6, linestyle=":")
    ax_d.set_xlabel("Baseline kernel similarity (Euclidean ● / HDV-only ▲)", fontsize=9)
    ax_d.set_ylabel("Learned ODD-similarity", fontsize=9)
    ax_d.set_title(
        f"D. Learned vs. baseline kernels "
        f"(Pearson r: learned–Euclidean = {r_eu:.2f}; learned–HDV-only = {r_hdv:.2f})",
        fontsize=9)
    ax_d.set_xlim(-0.02, 1.02); ax_d.set_ylim(-0.02, 1.02)
    ax_d.legend(loc="lower right", fontsize=8)

    fig.savefig(out, facecolor="white", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_posterior_lambda(lam_df: pd.DataFrame, title: str, out: Path) -> None:
    # Aggregate to (city, version) by averaging over periods
    agg = (lam_df.groupby(["city", "version"])
                 .agg(lam=("lam_mean", "mean"),
                      lo=("lam_q025", "mean"),
                      hi=("lam_q975", "mean"),
                      exposure=("exposure_million_miles", "sum"),
                      claims=("claims", "sum"))
                 .reset_index())

    fig, ax = plt.subplots(figsize=(8, 4))
    cities = DEPLOYED_CITIES
    versions = sorted(lam_df["version"].unique().tolist())
    x = np.arange(len(versions))
    width = 0.18
    for i, c in enumerate(cities):
        sub = agg[agg["city"] == c].set_index("version").reindex(versions)
        pos = x + (i - (len(cities)-1)/2) * width
        ax.errorbar(pos, sub["lam"],
                    yerr=[sub["lam"] - sub["lo"], sub["hi"] - sub["lam"]],
                    fmt="o", color=CITY_COLOR[c], label=c, markersize=4, capsize=2, lw=1)
    # Abbreviate long SGO version labels
    short_ver = [v.replace("5th Generation ADS, ", "5G-").replace(
                     "6th Generation ADS, ", "6G-").replace("Version ", "v")
                 for v in versions]
    ax.set_xticks(x); ax.set_xticklabels(short_ver, rotation=15, ha="right", fontsize=9)
    ax.set_xlabel("Software version")
    ax.set_ylabel("Posterior $\\lambda$ (claims per million miles)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_credibility_weights(bs_df: pd.DataFrame, cred_df: pd.DataFrame, out: Path) -> None:
    """Section 4.3 sanity check: BS closed-form Z next to per-cell empirical Z."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Left: closed-form BS Z by city aggregate
    ax = axes[0]
    colors = [CITY_COLOR[c] for c in bs_df["city"]]
    ax.bar(bs_df["city"], bs_df["Z_buhlmann_straub"], color=colors)
    ax.set_ylabel("Buhlmann-Straub credibility Z")
    ax.set_title("Closed-form Z by city (Section 4.3 limit)")
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=20)
    for c, z in zip(bs_df["city"], bs_df["Z_buhlmann_straub"]):
        ax.text(c, z + 0.02, f"{z:.4f}", ha="center", fontsize=7)

    # Right: hierarchical Z by (city, version) cell
    # Use the versions actually present in the data (not the synthetic list)
    ax = axes[1]
    versions_present = sorted(cred_df["version"].unique().tolist())
    pivot = cred_df.pivot(index="city", columns="version", values="credibility_Z")
    pivot = pivot.reindex(DEPLOYED_CITIES)[versions_present]
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(versions_present)))
    # Abbreviate long SGO version labels for readability
    short_ver = [v.replace("5th Generation ADS, ", "5G-").replace(
                     "6th Generation ADS, ", "6G-").replace("Version ", "v")
                 for v in versions_present]
    ax.set_xticklabels(short_ver, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(DEPLOYED_CITIES))); ax.set_yticklabels(DEPLOYED_CITIES)
    ax.set_title("Z by (city, version) under hierarchical model")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if not (val != val):  # skip NaN
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=("white" if val < 0.5 else "black"))
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_prospective(prosp_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cities = prosp_df["city"].tolist()
    x = np.arange(len(cities))
    median = prosp_df["lambda_q500"].to_numpy()
    lo = prosp_df["lambda_q025"].to_numpy()
    hi = prosp_df["lambda_q975"].to_numpy()
    colors = [CITY_COLOR[c] for c in cities]
    ax.errorbar(x, median,
                yerr=[median - lo, hi - median],
                fmt="o", color="black", markersize=6, capsize=3)
    for xi, c, m in zip(x, cities, median):
        ax.scatter(xi, m, color=CITY_COLOR[c], s=80, zorder=3)
        ax.text(xi, m, f"  {m:.2f}", va="center", ha="left", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([
        f"{c}\n(nearest: {prosp_df['primary_neighbor'].iloc[i]})"
        for i, c in enumerate(cities)
    ])
    ax.set_yscale("log")
    ax.set_ylabel("$\\lambda$ (claims per million miles)")
    ax.set_title("Prospective posterior $\\lambda$ for hypothetical new cities (median, 95% CI)")
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_update_after_first_million(pros_after: pd.DataFrame, out: Path) -> None:
    """
    Scenario panel: for each prospective city, the posterior median after a
    hypothetical first million miles under three experience levels (0, 1, 3
    crashes), against the prior median. Shows the updating mechanism rather
    than depending on one arbitrary count. Priors are summarized by the median
    (the heavy-tailed draws make the mean useless, e.g. Miami's prior mean is
    far above its median).
    """
    cities = ["Miami", "Boston", "Denver"]
    counts = sorted(pros_after["first_million_observed_claims"].unique())
    fig, axes = plt.subplots(1, len(cities), figsize=(9.6, 3.8), sharey=True)
    bar_colors = {0: "#2ca02c", 1: "#1f77b4", 3: "#d62728"}
    for ax, city in zip(axes, cities):
        sub = pros_after[pros_after["city"] == city].sort_values(
            "first_million_observed_claims")
        prior_med = float(sub["prior_lambda_q500"].iloc[0])
        x = np.arange(len(sub))
        med = sub["posterior_q500"].to_numpy()
        lo = sub["posterior_q025"].to_numpy()
        hi = sub["posterior_q975"].to_numpy()
        cnts = sub["first_million_observed_claims"].to_numpy()
        ax.bar(x, med, 0.6, color=[bar_colors[int(c)] for c in cnts],
               edgecolor="#333333", zorder=2)
        ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none",
                    ecolor="#333333", capsize=3, lw=1, zorder=3)
        ax.axhline(prior_med, color="#666666", ls="--", lw=1.2, zorder=1,
                   label=f"prior median {prior_med:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(c)}" for c in cnts])
        ax.set_xlabel("crashes observed / 1M mi")
        ax.set_title(city)
        ax.set_yscale("log")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3, zorder=0)
    axes[0].set_ylabel(r"posterior median $\lambda$" "\n(claims per million miles)")
    fig.suptitle("Posterior update after first 1M miles: below-, at-, and "
                 "above-prior experience\n(dashed line = prior median; bars = "
                 "posterior median; whiskers = 95% CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_loco_comparison(seeds_df: pd.DataFrame, out: Path) -> None:
    """Per-seed total leave-one-city-out log-likelihood by model (Figure 9).

    Shows the across-seed spread that makes the four random-effects models
    indistinguishable; the single-pool baseline separates cleanly.
    """
    models = [("pool", "Single\npool", "#7f7f7f"), ("indep", "Indep\nRE", "#1f77b4"),
              ("eu", "GP\nEuclid", "#d62728"), ("hdv_only", "GP\nHDV-only", "#9467bd"),
              ("learned", "GP\nlearned", "#2ca02c")]
    fig, ax = plt.subplots(figsize=(6.2, 3.85))
    rng = np.random.default_rng(0)
    for i, (k, lab, clr) in enumerate(models):
        y = seeds_df[k].to_numpy(); x = i + rng.uniform(-0.13, 0.13, len(y))
        ax.scatter(x, y, s=26, color=clr, alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [y.mean()] * 2, color=clr, lw=2.4, zorder=4)
    ax.set_xticks(range(len(models))); ax.set_xticklabels([m[1] for m in models])
    ax.set_ylabel("Total LOCO log-likelihood")
    ax.set_title("Leave-one-city-out log-likelihood across MCMC seeds\n(bars: mean; points: individual seeds)")
    ax.axhline(seeds_df["pool"].mean(), color="#7f7f7f", ls=":", lw=1, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_power_analysis(power_df: pd.DataFrame, out: Path) -> None:
    """Forward-simulation power analysis (Figure 11).

    Summaries use the MEDIAN advantage with an inter-quartile band. At high
    claim volume the Laplace + Gauss-Hermite predictive approximation produces
    occasional large-magnitude scores; their mean is non-monotone in volume even
    though the underlying advantage is monotone, so the median is the stable
    statistic. Panel A gives the two comparisons separate y-axes because the
    learned-vs-Euclidean advantage grows much more slowly than learned-vs-
    independent-RE and would otherwise be visually flattened.
    """
    s = power_df[power_df.dgp == "structured"].sort_values("mean_total_claims")
    n = power_df[power_df.dgp == "independent_null"].sort_values("mean_total_claims")

    def _col(df, base, stat):
        # tolerate older CSVs that only carry the mean
        c = f"{base}_{stat}"
        return df[c].to_numpy() if c in df.columns else df[f"{base}_mean"].to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # ---- Panel A: structured DGP, twin y-axes ----
    xa = s.mean_total_claims.to_numpy()
    axA_l = ax[0]                      # left axis: vs independent RE
    axA_r = ax[0].twinx()             # right axis: vs Euclidean
    # vs independent RE (blue, left)
    med_i = _col(s, "learned_minus_indep", "median")
    q25_i = _col(s, "learned_minus_indep", "q25"); q75_i = _col(s, "learned_minus_indep", "q75")
    axA_l.fill_between(xa, q25_i, q75_i, color="#1f77b4", alpha=0.15)
    l1, = axA_l.plot(xa, med_i, "-o", color="#1f77b4", ms=4, label="vs. independent RE (left)")
    # vs Euclidean (red, right)
    med_e = _col(s, "learned_minus_eu", "median")
    q25_e = _col(s, "learned_minus_eu", "q25"); q75_e = _col(s, "learned_minus_eu", "q75")
    axA_r.fill_between(xa, q25_e, q75_e, color="#d62728", alpha=0.15)
    l2, = axA_r.plot(xa, med_e, "-s", color="#d62728", ms=4, label="vs. Euclidean kernel (right)")
    axA_l.axhline(0, color="#1f77b4", lw=0.6, ls=":")
    axA_l.axvline(649, color="green", ls="-.", lw=1, alpha=0.6)
    axA_l.set_xscale("log")
    axA_l.set_xlabel("expected total claims")
    axA_l.set_ylabel("median advantage vs. indep RE", color="#1f77b4")
    axA_r.set_ylabel("median advantage vs. Euclidean", color="#d62728")
    axA_l.tick_params(axis="y", labelcolor="#1f77b4")
    axA_r.tick_params(axis="y", labelcolor="#d62728")
    axA_l.set_title("A. Structured ODD process (learned correctly specified)\n"
                    "median advantage \u00b1 IQR; separate y-axes (note scales differ)")
    axA_l.legend(handles=[l1, l2], loc="upper left", fontsize=8)

    # ---- Panel B: null DGP, mean (now monotone after the quadrature fix) ----
    xb = n.mean_total_claims.to_numpy()
    for base, lab, clr, mk in [("learned_minus_indep", "vs. independent RE", "#1f77b4", "o"),
                               ("learned_minus_eu", "vs. Euclidean kernel", "#d62728", "s")]:
        y = _col(n, base, "mean")
        ax[1].plot(xb, y, "-" + mk, color=clr, ms=4, label=lab)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("expected total claims")
    ax[1].set_ylabel("mean LOCO log-likelihood advantage")
    ax[1].set_title("B. Null process (no ODD structure)\n"
                    "specificity check: learned never wins, monotone in volume")
    ax[1].legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def fig_embedding_pca(cell_emb_csv: Path, out: Path) -> None:
    """2-D projection of the 32-D cell embeddings onto the leading BETWEEN-CITY
    principal axes.

    Plain PCA on all ~5k cells finds the directions of greatest *within-city*
    cell scatter, which are not the directions that separate the city
    centroids: the city means occupy a tiny core of that cloud, so projecting
    them onto the cell-variance axes scrambles the inter-city geometry. In the
    real-data run that artifact placed Boston next to Los Angeles even though
    Boston and San Francisco are the closest pair in 32-D (cosine 0.87). We
    instead build the 2-D axes from the seven city-mean embeddings, so the
    plotted inter-city distances faithfully reflect the learned similarity
    structure; cells are projected onto the same axes for scatter context.
    """
    df = pd.read_csv(cell_emb_csv)
    embed_cols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    Z = df[embed_cols].to_numpy()

    cities = DEPLOYED_CITIES + HYPOTHETICAL_CITIES
    city_means = np.vstack([Z[df["city"].values == c].mean(axis=0) for c in cities])
    mu = city_means.mean(axis=0)

    # PCA axes from the city means (between-city structure), not the cell cloud.
    _, Sc, Vt = np.linalg.svd(city_means - mu, full_matrices=False)
    axes = Vt[:2]
    var_frac = Sc ** 2 / (Sc ** 2).sum()

    proj = (Z - mu) @ axes.T            # cells on the between-city axes
    df["pc1"] = proj[:, 0]
    df["pc2"] = proj[:, 1]
    cmean_proj = (city_means - mu) @ axes.T   # faithful centroid positions

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, c in enumerate(cities):
        sub = df[df["city"] == c]
        ax.scatter(sub["pc1"], sub["pc2"], s=8, alpha=0.4,
                   color=CITY_COLOR[c], label=c)
        cx, cy = cmean_proj[i]
        ax.scatter(cx, cy, s=120, marker="X", edgecolor="black", linewidth=1,
                   color=CITY_COLOR[c])
        ax.annotate(c, (cx, cy), xytext=(6, 6),
                    textcoords="offset points", fontsize=9,
                    fontweight="bold")
    ax.set_xlabel(f"Between-city axis 1 ({100*var_frac[0]:.1f}% of between-city var)")
    ax.set_ylabel(f"Between-city axis 2 ({100*var_frac[1]:.1f}% of between-city var)")
    ax.set_title("Learned cell embeddings: between-city 2-D projection")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.10), fontsize=8)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--embed-dir", default="results/embedding", type=Path)
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--out", default="figures", type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    ads = pd.read_csv(args.data_dir / "ads_events.csv")
    with open(args.results_dir / "benchmark_reproduction.json") as f:
        bench = json.load(f)
    S_emb = pd.read_csv(args.embed_dir / "city_similarity.csv", index_col=0)
    S_eu = pd.read_csv(args.results_dir / "city_similarity_euclidean.csv", index_col=0)
    S_hdv = pd.read_csv(args.results_dir / "city_similarity_hdv_only.csv", index_col=0)
    lam_gp = pd.read_csv(args.results_dir / "posterior_lambda_gp.csv")
    bs = pd.read_csv(args.results_dir / "buhlmann_straub_closed_form.csv")
    cred = pd.read_csv(args.results_dir / "credibility_weights_gp.csv")
    prosp = pd.read_csv(args.results_dir / "prospective_new_cities.csv")
    pros_after = pd.read_csv(args.results_dir / "prospective_after_first_million.csv")
    loco = pd.read_csv(args.results_dir / "loco_comparison.csv")

    fig_exposure_distribution(ads, args.out / "fig_02_exposure_distribution.png")
    fig_credibility_weights(bs, cred, args.out / "fig_03_credibility_weights.png")
    fig_similarity_heatmap(S_emb, "Learned ODD-similarity matrix",
                            args.out / "fig_04_similarity_heatmap.png")
    fig_posterior_lambda(lam_gp, "Posterior $\\lambda$ by (city, version): GP-prior model",
                          args.out / "fig_05_posterior_lambda.png")
    fig_prospective(prosp, args.out / "fig_06_prospective_estimates.png")
    fig_update_after_first_million(pros_after,
                                    args.out / "fig_07_update_first_million.png")
    fig_embedding_pca(args.embed_dir / "cell_embeddings.csv",
                       args.out / "fig_01_embedding_pca.png")
    power_path = args.results_dir / "power_analysis.csv"
    if power_path.exists():
        power_df = pd.read_csv(power_path, keep_default_na=False)
        for c in power_df.columns:
            if c != "dgp":
                power_df[c] = pd.to_numeric(power_df[c])
        fig_power_analysis(power_df, args.out / "fig_08_power_analysis.png")
    else:
        print("  (skipping fig_08: run src/power_analysis.py first)")
    print("Wrote figures to", args.out)
    


if __name__ == "__main__":
    main()