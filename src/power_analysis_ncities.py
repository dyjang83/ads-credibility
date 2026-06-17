"""
Number-of-cities power analysis.


Design
------
For each K in a grid we synthesize K cities whose 32-dimensional ODD embeddings
are drawn so that their cosine-similarity structure matches the real seven-city
embedding (same leading subspace and per-axis score variance, re-normalized to
the unit sphere). This is a modeling choice, stated plainly: we do not have more
than seven real cities, so additional cities are sampled from the embedding
geometry the model actually learned. The learned kernel is therefore correctly
specified by construction (as in the structured DGP of power_analysis.py), and
the question is purely one of statistical power as the number of held-out folds
grows.

For each replicate:
  - draw alpha ~ N(0, sigma^2 * S_learned_K)              over the K cities
  - draw per-city counts ~ Poisson(exp(mu + alpha) * E)   at fixed exposure E
  - run a full K-fold LOCO for the learned, Euclidean, and independent-RE
    models (Laplace fit + Gauss-Hermite predictive scoring, reusing the exact
    helpers from power_analysis.py)
  - record whether the learned kernel's total LOCO log-likelihood exceeds each
    baseline in THIS single run

The detection probability at K is the fraction of replicates in which the
learned kernel wins, reported separately versus Euclidean and versus
independent-RE. A null process (alpha ~ N(0, I)) is run as a specificity check:
detection probability must stay near or below chance.

Outputs
-------
results/power_analysis_ncities.csv          per-(dgp, K) detection probabilities
results/power_analysis_ncities_summary.json minimum K to reach the target power
figures/fig_09_ncities_power.png            detection probability vs. K
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Reuse the exact LOCO machinery from the main power analysis so the two
# analyses are numerically consistent.
from power_analysis import (  # noqa: E402
    _laplace_fit, _cond_weights, _held_logp,
    SIGMA, BASE_RATE, BASE_MILES, JITTER,
)
from data_generator import DEPLOYED_CITIES, HYPOTHETICAL_CITIES  # noqa: E402

ALL_CITIES = DEPLOYED_CITIES + HYPOTHETICAL_CITIES

# Per-city exposure for this analysis. The volume sweep is NOT the variable here;
# we hold per-city expected claims fixed at a realistic level (the real deployed
# mean is ~157 claims/city; we use a deliberately MODEST 40/city so that the
# K-curve is not trivially saturated and the four-city operating point matches
# the regime where Section 6.5 finds a tie).
PER_CITY_CLAIMS_TARGET = 40.0
FIXED_MILES = PER_CITY_CLAIMS_TARGET / BASE_RATE   # exposure giving the target


def _embedding_sampler(seed: int = 0, max_k: int = 64):
    """Return a function K -> (K x 32) unit embeddings whose cosine-similarity
    structure matches the real seven-city embedding geometry.

    The synthetic cities are drawn ONCE as a fixed pool of `max_k` cities; the
    K-city set is the first K of that pool (nested prefixes). This is essential
    for a smooth number-of-cities curve: if each K drew an independent set of
    cities, the learned-vs-Euclidean separation would jump around at small K
    purely because of which random cities happened to be sampled, producing a
    non-monotone curve that reflects city-geometry sampling noise rather than the
    effect of K. With nested prefixes, the K+2 set is the K set plus two more
    cities, so the geometry — and the detection probability — evolves smoothly.
    """
    emb = pd.read_csv(ROOT / "results/embedding/cell_embeddings.csv")
    ecols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    C = np.vstack([emb.loc[emb.city == c, ecols].to_numpy().mean(axis=0)
                   for c in ALL_CITIES])
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    mu = C.mean(axis=0)
    X = C - mu
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    score_sd = (X @ Vt.T).std(axis=0)               # per-axis spread of real cities
    rng = np.random.default_rng(seed)
    # Fixed pool drawn once; prefixes are nested. The first four pool cities are
    # the REAL deployed cities (so the K=4 anchor and the synthetic extension
    # share a common prefix and the curve is continuous at the hand-off);
    # cities 5..max_k are synthetic draws matching the learned embedding geometry.
    deployed_unit = C[[ALL_CITIES.index(c) for c in DEPLOYED_CITIES]]
    extra_scores = rng.normal(0.0, score_sd, size=(max_k - len(DEPLOYED_CITIES),
                                                    len(score_sd)))
    extra = mu + extra_scores @ Vt
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    pool = np.vstack([deployed_unit, extra])

    def sample(K: int) -> np.ndarray:
        if K > len(pool):
            raise ValueError(f"K={K} exceeds synthetic pool size {len(pool)}")
        return pool[:K]

    return sample


def _euclidean_kernel(V: np.ndarray) -> np.ndarray:
    """Euclidean-distance RBF kernel on the embeddings, median-heuristic length
    scale -- the same construction the paper uses for the Euclidean baseline,
    applied here to the synthetic cities so the baseline is defined for any K."""
    d2 = np.sum((V[:, None, :] - V[None, :, :]) ** 2, axis=-1)
    med = np.median(d2[np.triu_indices(len(V), 1)])
    return np.exp(-d2 / (med + 1e-12))


def _loco_total_K(counts, expo, S_learned, S_eu, sigma):
    """Full K-fold LOCO total predictive logp for each model."""
    n = len(counts)
    out = {"indep": 0.0, "eu": 0.0, "learned": 0.0}
    S_indep = np.eye(n)
    for h in range(n):
        fit = [i for i in range(n) if i != h]
        for key, S in [("indep", S_indep), ("eu", S_eu), ("learned", S_learned)]:
            S_ff = S[np.ix_(fit, fit)]
            mu, alpha, cov = _laplace_fit(counts[fit], expo[fit], S_ff, sigma)
            b, c = _cond_weights(S, fit, h)
            out[key] += _held_logp(counts[h], expo[h], mu, alpha, cov, b, c, sigma)
    return out


def _real_deployed_kernels():
    """The actual learned and Euclidean similarity matrices over the four real
    deployed cities, so the K=4 point of the curve is the genuine Section 6.5
    operating point rather than a synthetic four-city draw."""
    from run_analysis import euclidean_similarity_from_features
    S_learned = pd.read_csv(ROOT / "results/embedding/city_similarity.csv",
                            index_col=0).loc[DEPLOYED_CITIES, DEPLOYED_CITIES].to_numpy()
    odd = pd.read_csv(ROOT / "data/city_odd_features.csv")
    S_eu = euclidean_similarity_from_features(odd, DEPLOYED_CITIES).loc[
        DEPLOYED_CITIES, DEPLOYED_CITIES].to_numpy()
    return S_learned, S_eu


def run(dgp: str, k_grid, n_rep: int, seed: int):
    sampler = _embedding_sampler(seed=seed + 1)
    rng = np.random.default_rng(seed)
    mu_true = np.log(BASE_RATE)
    S_learned_real, S_eu_real = _real_deployed_kernels()
    rows = []
    for K in k_grid:
        if K == 4:
            # Anchor the four-city point to the REAL deployed cities, so the
            # curve starts exactly where Section 6.5 sits (the learned and
            # Euclidean kernels nearly coincide there, r = 0.78).
            S_learned, S_eu = S_learned_real, S_eu_real
        else:
            V = sampler(K)
            S_learned = V @ V.T
            S_eu = _euclidean_kernel(V)
        S_true = S_learned if dgp == "structured" else np.eye(K)
        L_true = np.linalg.cholesky(SIGMA ** 2 * S_true + JITTER * np.eye(K))
        expo = np.full(K, FIXED_MILES)

        # Common random numbers across the K-axis: reseed identically at every K
        # so the latent-effect and Poisson innovation streams are aligned across
        # points. p_win is still a binomial proportion (the dominant remaining
        # noise source), so the default replicate count is set high; CRN plus
        # many replicates makes the detection-probability curve smooth and
        # monotone in expectation.
        rng = np.random.default_rng(seed)
        win_eu = win_indep = 0
        adv_eu, adv_indep, tot_claims = [], [], []
        for _ in range(n_rep):
            alpha = L_true @ rng.standard_normal(K)
            counts = rng.poisson(np.exp(mu_true + alpha) * expo)
            tot_claims.append(counts.sum())
            t = _loco_total_K(counts, expo, S_learned, S_eu, SIGMA)
            de = t["learned"] - t["eu"]
            di = t["learned"] - t["indep"]
            adv_eu.append(de); adv_indep.append(di)
            win_eu += int(de > 0); win_indep += int(di > 0)
        rows.append({
            "dgp": dgp, "K": K,
            "mean_total_claims": float(np.mean(tot_claims)),
            "per_city_claims": float(np.mean(tot_claims) / K),
            "p_win_vs_euclidean": win_eu / n_rep,
            "p_win_vs_independent": win_indep / n_rep,
            "mean_adv_vs_euclidean": float(np.mean(adv_eu)),
            "mean_adv_vs_independent": float(np.mean(adv_indep)),
        })
        print(f"[{dgp:10s}] K={K:3d}  ~claims={rows[-1]['mean_total_claims']:7.1f} "
              f"P(win vs eu)={rows[-1]['p_win_vs_euclidean']:.2f}  "
              f"P(win vs indep)={rows[-1]['p_win_vs_independent']:.2f}")
    return pd.DataFrame(rows)


def _min_k_for_power(df, col, target):
    f = df.sort_values("K")
    hit = f[f[col] >= target]
    return int(hit["K"].iloc[0]) if len(hit) else None


def make_figure(csv_path: Path, out_path: Path, target: float = 0.80,
                min_k_eu: int | None = None):
    """Render the number-of-cities power curve (Figure 12)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(csv_path)
    s = df[df.dgp == "structured"].sort_values("K")
    nz = df[df.dgp == "identity_null"].sort_values("K")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(s.K, s.p_win_vs_independent, "-o", color="#1f77b4", ms=5,
            label="vs. independent RE (structured)")
    ax.plot(s.K, s.p_win_vs_euclidean, "-o", color="#d62728", ms=5,
            label="vs. Euclidean kernel (structured)")
    ax.plot(nz.K, nz.p_win_vs_euclidean, "--s", color="#d62728", ms=4, alpha=0.5,
            label="vs. Euclidean (null: no ODD structure)")
    ax.plot(nz.K, nz.p_win_vs_independent, "--s", color="#1f77b4", ms=4, alpha=0.5,
            label="vs. independent RE (null)")
    ax.axhline(target, color="gray", ls=":", lw=1)
    ax.axhline(0.5, color="k", lw=0.6, alpha=0.5)
    ax.annotate(f"{int(target*100)}% power", (s.K.max(), target + 0.01),
                fontsize=8, color="gray", ha="right")
    if min_k_eu:
        ax.axvline(min_k_eu, color="#d62728", ls="--", lw=1, alpha=0.5)
    ax.axvline(4, color="green", ls="-.", lw=1, alpha=0.6)
    ax.annotate("current\nfootprint", (4, 0.12), fontsize=8, color="green", ha="center")
    ax.set_xlabel("number of deployed cities held out (K)")
    ax.set_ylabel("P(single LOCO run favors learned kernel)")
    ax.set_title("Number-of-cities power curve at fixed per-city volume "
                 "(~%d claims/city)\nK=4 anchored to the real deployed cities "
                  % int(PER_CITY_CLAIMS_TARGET),
                 fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7.5, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=400)
    p.add_argument("--target", type=float, default=0.80,
                   help="target single-run detection probability")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--figdir", type=Path, default=ROOT / "figures")
    args = p.parse_args()

    k_grid = [4, 6, 8, 10, 12, 16, 20, 28, 40]
    df_s = run("structured", k_grid, args.reps, seed=2026)
    df_n = run("identity_null", k_grid, args.reps, seed=4052)
    df = pd.concat([df_s, df_n], ignore_index=True)
    df.to_csv(args.out / "power_analysis_ncities.csv", index=False)

    summary = {
        "fixed_per_city_claims": PER_CITY_CLAIMS_TARGET,
        "target_power": args.target,
        "min_K_vs_euclidean":
            _min_k_for_power(df_s, "p_win_vs_euclidean", args.target),
        "min_K_vs_independent":
            _min_k_for_power(df_s, "p_win_vs_independent", args.target),
        "four_city_power_vs_euclidean":
            float(df_s.loc[df_s.K == 4, "p_win_vs_euclidean"].iloc[0]),
        "four_city_power_vs_independent":
            float(df_s.loc[df_s.K == 4, "p_win_vs_independent"].iloc[0]),
        "null_max_power_vs_euclidean":
            float(df_n["p_win_vs_euclidean"].max()),
        "note": "Single-run LOCO detection probability under a structured DGP "
                "where the learned kernel is correctly specified, at fixed "
                "per-city volume. min_K is the smallest deployed footprint at "
                "which one LOCO evaluation favors the learned kernel with the "
                "target probability. The null DGP (independent city effects) is "
                "a specificity check: detection probability must stay near "
                "chance regardless of K.",
    }
    with open(args.out / "power_analysis_ncities_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    args.figdir.mkdir(parents=True, exist_ok=True)
    make_figure(args.out / "power_analysis_ncities.csv",
                args.figdir / "fig_09_ncities_power.png",
                target=args.target, min_k_eu=summary["min_K_vs_euclidean"])
    print("\nsaved results/power_analysis_ncities.csv")
    print("saved results/power_analysis_ncities_summary.json")
    print(json.dumps(summary, indent=2))