

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import asdict

import jax.numpy as jnp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_generator import (  # noqa: E402
    DEPLOYED_CITIES,
    HYPOTHETICAL_CITIES,
    SOFTWARE_VERSIONS,
    CITY_ODD_FEATURES,
)
from models import (  # noqa: E402
    ads_credibility_model,
    ads_credibility_gp_model,
    InferenceConfig,
    run_inference,
    diagnostics,
    buhlmann_straub_closed_form,
    save_samples,
    predict_new_city_with_S,
    leave_one_city_out_predictive_logp,
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_data(data_dir):
    data_dir = Path(data_dir)
    ads = pd.read_csv(data_dir / "ads_events.csv")
    hdv = pd.read_csv(data_dir / "hdv_claims_by_cell.csv")
    city_odd = pd.read_csv(data_dir / "city_odd_features.csv")
    with open(data_dir / "true_parameters.json") as f:
        truth = json.load(f)
    return ads, hdv, city_odd, truth


def resolve_versions(ads: pd.DataFrame) -> list:
    """Version vocabulary for the model.

    Real NHTSA SGO data (sgo_fetcher.py) carries collapsed version labels like
    "5th Generation ADS, Version 10" that are not in the synthetic
    SOFTWARE_VERSIONS list. Derive the vocabulary from the data when the column
    holds unknown labels, so the real-data path runs without code edits. Falls
    back to the synthetic constant when the data uses the synthetic labels.
    """
    observed = sorted(ads["version"].dropna().unique().tolist())
    if observed and not set(observed).issubset(set(SOFTWARE_VERSIONS)):
        return observed
    return list(SOFTWARE_VERSIONS)


def prepare_model_data(ads: pd.DataFrame, city_odd: pd.DataFrame):
    """Build the arrays passed to the NumPyro models."""
    versions = resolve_versions(ads)
    city_to_idx = {c: i for i, c in enumerate(DEPLOYED_CITIES)}
    version_to_idx = {v: i for i, v in enumerate(versions)}

    df = ads.copy()
    df = df.merge(city_odd, on="city")
    df["city_idx"] = df["city"].map(city_to_idx)
    df["version_idx"] = df["version"].map(version_to_idx)

    feat_cols = CITY_ODD_FEATURES
    X = jnp.asarray(df[feat_cols].to_numpy(dtype=float))

    return {
        "city_idx": jnp.asarray(df["city_idx"].to_numpy()),
        "version_idx": jnp.asarray(df["version_idx"].to_numpy()),
        "X": X,
        "exposure": jnp.asarray(df["exposure_million_miles"].to_numpy(dtype=float)),
        "claims": jnp.asarray(df["claims"].to_numpy(dtype=int)),
        "n_cities": len(DEPLOYED_CITIES),
        "n_versions": len(versions),
        "n_features": X.shape[1],
    }, df


# ---------------------------------------------------------------------------
# Section 6.2: benchmark reproduction
# ---------------------------------------------------------------------------

def benchmark_reproduction(ads: pd.DataFrame, hdv: pd.DataFrame) -> dict:
    """
    Approximate the Di Lillo et al. (2024) reduction by comparing observed
    ADS frequency to the matched HDV frequency in the same operating regions.
    """
    rows = []
    overall_ads_freq = ads["claims"].sum() / ads["exposure_million_miles"].sum()
    overall_hdv_freq = hdv[hdv["city"].isin(DEPLOYED_CITIES)]["claims"].sum() / \
        hdv[hdv["city"].isin(DEPLOYED_CITIES)]["exposure_million_miles"].sum()
    overall_reduction = 1.0 - overall_ads_freq / overall_hdv_freq

    for c in DEPLOYED_CITIES:
        ads_c = ads[ads["city"] == c]
        hdv_c = hdv[hdv["city"] == c]
        ads_freq = ads_c["claims"].sum() / max(ads_c["exposure_million_miles"].sum(), 1e-9)
        hdv_freq = hdv_c["claims"].sum() / max(hdv_c["exposure_million_miles"].sum(), 1e-9)
        rows.append({
            "city": c,
            "ads_claims": int(ads_c["claims"].sum()),
            "ads_exposure_million_miles": float(ads_c["exposure_million_miles"].sum()),
            "ads_freq_per_million_miles": float(ads_freq),
            "hdv_freq_per_million_miles": float(hdv_freq),
            "reduction_vs_hdv": float(1.0 - ads_freq / hdv_freq) if hdv_freq > 0 else None,
        })
    return {
        "overall_ads_freq": float(overall_ads_freq),
        "overall_hdv_freq": float(overall_hdv_freq),
        "overall_reduction_vs_hdv": float(overall_reduction),
        "by_city": rows,
        "comparison_to_paper": {
            "di_lillo_pd_reduction": 0.88,
            "di_lillo_bi_reduction": 0.92,
            "our_combined_reduction": float(overall_reduction),
        },
    }


# ---------------------------------------------------------------------------
# Section 6.3: hierarchical posterior estimates
# ---------------------------------------------------------------------------

def run_independent_re_model(data: dict, samples_dir: Path) -> dict:
    """Section 4 model with independent random effects."""
    cfg = InferenceConfig(n_warmup=1000, n_samples=1500, n_chains=2)
    samples = run_inference(ads_credibility_model, data, cfg)
    save_samples(samples, samples_dir, "indep_re")
    return samples


def run_gp_model(data: dict, S: np.ndarray, samples_dir: Path) -> dict:
    """Section 5.3 model with GP prior on city random effects."""
    # Cholesky factor (deployed cities only)
    n_dep = len(DEPLOYED_CITIES)
    S_dep = S.loc[DEPLOYED_CITIES, DEPLOYED_CITIES].to_numpy()
    # Add jitter for numerical stability
    L = np.linalg.cholesky(S_dep + 1e-6 * np.eye(n_dep))
    data_gp = dict(data)
    data_gp["L_chol"] = jnp.asarray(L)

    cfg = InferenceConfig(n_warmup=1000, n_samples=1500, n_chains=2)
    samples = run_inference(ads_credibility_gp_model, data_gp, cfg)
    save_samples(samples, samples_dir, "gp_re")
    return samples


def credibility_weights_from_posterior(samples: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Empirical credibility weight per (city, version): the share of variance
    in posterior log-lambda that aligns with the own MLE rather than the
    cross-cell pooled mean.

    We use the operational definition:
      Z_cv = corr(posterior log-lambda_cv, own MLE log-lambda_cv) ** 2 * shrink
    More simply: Z_cv = 1 - (posterior var of alpha_c shrunk to zero) ratio.

    For interpretation in the paper, we compute the practitioner-readable
    quantity:
      Z_cv = (own_exposure * own_rate * tau_c^2)
             / (own_exposure * own_rate * tau_c^2 + 1)
    using the posterior mean of tau_c. This is the Section 4.3 closed form
    evaluated at posterior-mean hyperparameters.
    """
    # Get posterior mean of tau_c (Section 4) or sigma_c (Section 5.3)
    if "tau_c" in samples:
        tau2 = float(np.mean(samples["tau_c"].reshape(-1) ** 2))
    elif "sigma_c" in samples:
        tau2 = float(np.mean(samples["sigma_c"].reshape(-1) ** 2))
    else:
        tau2 = 0.1

    rows = []
    for (city, version), grp in df.groupby(["city", "version"]):
        w = grp["exposure_million_miles"].sum()
        N = grp["claims"].sum()
        own_rate = N / w if w > 0 else 0.0
        effective_exposure = w * own_rate * tau2
        Z = effective_exposure / (effective_exposure + 1.0)
        rows.append({
            "city": city,
            "version": version,
            "exposure_million_miles": float(w),
            "claims": int(N),
            "own_freq": float(own_rate),
            "credibility_Z": float(Z),
        })
    return pd.DataFrame(rows)


def posterior_lambda_by_cell(samples: dict, df: pd.DataFrame,
                              data: dict) -> pd.DataFrame:
    """Posterior summaries of lambda for each (city, version, period) row."""
    lam = samples["lambda"]   # (chains, samples, N)
    flat = lam.reshape(-1, lam.shape[-1])
    summary = pd.DataFrame({
        "lam_mean": flat.mean(axis=0),
        "lam_q025": np.quantile(flat, 0.025, axis=0),
        "lam_q500": np.quantile(flat, 0.500, axis=0),
        "lam_q975": np.quantile(flat, 0.975, axis=0),
    })
    out = pd.concat([df[["city", "version", "period",
                          "exposure_million_miles", "claims"]].reset_index(drop=True),
                      summary], axis=1)
    return out


# ---------------------------------------------------------------------------
# Section 6.4: prospective estimation for new deployments
# ---------------------------------------------------------------------------

def prospective_new_city_estimates(
    samples_gp: dict,
    S: pd.DataFrame,
    city_odd: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each hypothetical city, derive the posterior-predictive distribution
    of lambda using the conditional Gaussian under the GP prior.

    We report the posterior median as the headline point estimate. On the
    log-scale linear predictor the posterior is approximately Gaussian, but
    after the exp() link a long right tail develops; the median is the
    coherent point summary for regulator-facing rate filings. The mean is
    also reported (with the right tail it can be inflated relative to the
    median by an order of magnitude for cells with low similarity).
    """
    S_dep = S.loc[DEPLOYED_CITIES, DEPLOYED_CITIES].to_numpy()
    odd_lookup = city_odd.set_index("city")

    rows = []
    for c in HYPOTHETICAL_CITIES:
        s_to_dep = S.loc[c, DEPLOYED_CITIES].to_numpy()
        x_new = odd_lookup.loc[c, CITY_ODD_FEATURES].to_numpy(dtype=float)
        # Posterior predictive marginalizing over version (use the latest)
        lam_draws = predict_new_city_with_S(
            samples_gp,
            S_dep_dep=S_dep,
            S_new_dep=s_to_dep,
            s_new_new=float(S.loc[c, c]),
            x_new=x_new,
            version_effect=None,
        )
        rows.append({
            "city": c,
            "primary_neighbor": DEPLOYED_CITIES[int(np.argmax(s_to_dep))],
            "max_similarity": float(s_to_dep.max()),
            "lambda_median": float(np.median(lam_draws)),
            "lambda_mean": float(np.mean(lam_draws)),
            "lambda_q025": float(np.quantile(lam_draws, 0.025)),
            "lambda_q500": float(np.quantile(lam_draws, 0.500)),
            "lambda_q975": float(np.quantile(lam_draws, 0.975)),
        })
    return pd.DataFrame(rows)


def prospective_with_first_million_miles(
    samples_gp: dict,
    S: pd.DataFrame,
    city_odd: pd.DataFrame,
    scenarios: dict,
) -> pd.DataFrame:
    """
    Approximate Bayesian update: combine the prior predictive distribution
    of lambda for each new city with a Poisson likelihood for the first
    million miles of observed claims. We discretize over the prior draws and
    reweight to form the posterior, then summarize.

    `scenarios` maps each city to a list of hypothetical first-million-mile
    claim counts to illustrate (e.g. {"Miami": [0, 1, 3], ...}). The counts
    span below-, at-, and above-prior-expectation experience so the panel
    demonstrates the updating mechanism rather than depending on one arbitrary
    integer draw.
    """
    from math import lgamma

    S_dep = S.loc[DEPLOYED_CITIES, DEPLOYED_CITIES].to_numpy()
    odd_lookup = city_odd.set_index("city")
    rows = []
    for c, n_obs_list in scenarios.items():
        s_to_dep = S.loc[c, DEPLOYED_CITIES].to_numpy()
        x_new = odd_lookup.loc[c, CITY_ODD_FEATURES].to_numpy(dtype=float)
        lam_draws = predict_new_city_with_S(
            samples_gp,
            S_dep_dep=S_dep,
            S_new_dep=s_to_dep,
            s_new_new=float(S.loc[c, c]),
            x_new=x_new,
            version_effect=None,
        )
        exposure = 1.0  # one million miles
        prior_mean = float(np.mean(lam_draws))
        prior_q025 = float(np.quantile(lam_draws, 0.025))
        prior_q500 = float(np.quantile(lam_draws, 0.500))
        prior_q975 = float(np.quantile(lam_draws, 0.975))
        for n_obs in n_obs_list:
            rate = lam_draws * exposure
            logw = n_obs * np.log(np.maximum(rate, 1e-30)) - rate - lgamma(n_obs + 1)
            w = np.exp(logw - logw.max())
            w = w / w.sum()
            rows.append({
                "city": c,
                "first_million_observed_claims": n_obs,
                "prior_lambda_mean": prior_mean,
                "prior_lambda_q025": prior_q025,
                "prior_lambda_q500": prior_q500,
                "prior_lambda_q975": prior_q975,
                "posterior_lambda_mean": float(np.sum(w * lam_draws)),
                "posterior_q025": _weighted_quantile(lam_draws, w, 0.025),
                "posterior_q500": _weighted_quantile(lam_draws, w, 0.5),
                "posterior_q975": _weighted_quantile(lam_draws, w, 0.975),
                "ess": float(1.0 / np.sum(w ** 2)),
            })
    return pd.DataFrame(rows)


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    cum = cum / cum[-1]
    return float(v[np.searchsorted(cum, q)])


# ---------------------------------------------------------------------------
# Section 6.5: baseline comparison
# ---------------------------------------------------------------------------

def euclidean_similarity_from_features(city_odd: pd.DataFrame, all_cities: list) -> pd.DataFrame:
    """Baseline kernel: Euclidean distance in raw (already-z-scored) features."""
    df = city_odd.set_index("city").loc[all_cities]
    X = df.to_numpy()
    diffs = X[:, None, :] - X[None, :, :]
    d2 = (diffs ** 2).sum(axis=-1)
    ell2 = float(np.median(d2[np.triu_indices_from(d2, k=1)])) / (2.0 * np.log(2.0))
    S = np.exp(-d2 / (2.0 * max(ell2, 1e-6)))
    return pd.DataFrame(S, index=all_cities, columns=all_cities)


def loco_comparison(
    ads: pd.DataFrame,
    city_odd: pd.DataFrame,
    S_embed: pd.DataFrame,
    S_euclid: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Leave-one-city-out predictive log-likelihood across baselines.

    For each deployed city c held out:
      - Fit each model on the remaining three cities
      - Score the held-out city's aggregate (exposure, claims)
    """
    rows = []
    for held in DEPLOYED_CITIES:
        train_cities = [c for c in DEPLOYED_CITIES if c != held]
        ads_train = ads[ads["city"].isin(train_cities)].copy()
        ads_held = ads[ads["city"] == held]
        held_exposure = float(ads_held["exposure_million_miles"].sum())
        held_claims = int(ads_held["claims"].sum())
        held_x = city_odd.set_index("city").loc[held, CITY_ODD_FEATURES].to_numpy(dtype=float)

        # Reindex training cities for the model
        city_to_idx = {c: i for i, c in enumerate(train_cities)}
        ads_train["city_idx"] = ads_train["city"].map(city_to_idx)
        versions = resolve_versions(ads)
        version_to_idx = {v: i for i, v in enumerate(versions)}
        ads_train["version_idx"] = ads_train["version"].map(version_to_idx)

        df_train = ads_train.merge(city_odd, on="city")
        X = jnp.asarray(df_train[CITY_ODD_FEATURES].to_numpy(dtype=float))
        train_data = {
            "city_idx": jnp.asarray(df_train["city_idx"].to_numpy()),
            "version_idx": jnp.asarray(df_train["version_idx"].to_numpy()),
            "X": X,
            "exposure": jnp.asarray(df_train["exposure_million_miles"].to_numpy(dtype=float)),
            "claims": jnp.asarray(df_train["claims"].to_numpy(dtype=int)),
            "n_cities": len(train_cities),
            "n_versions": len(versions),
            "n_features": X.shape[1],
        }

        # --- Baseline 1: single-pool Buhlmann-Straub on aggregated city totals
        agg = ads_train.groupby("city").agg(
            claims=("claims", "sum"), miles=("exposure_million_miles", "sum")
        )
        # Reorder to match train_cities
        agg = agg.reindex(train_cities)
        bs = buhlmann_straub_closed_form(
            agg["claims"].to_numpy(), agg["miles"].to_numpy()
        )
        # For the held city we have no own experience, so the BS predictor is
        # the grand mean.
        bs_rate = bs["X_bar"]
        bs_logp = _poisson_logp(held_claims, bs_rate * held_exposure)

        # --- Baseline 2: independent-RE hierarchical model (Section 4)
        cfg_short = InferenceConfig(n_warmup=600, n_samples=800, n_chains=1)
        samples_indep = run_inference(ads_credibility_model, train_data, cfg_short)
        indep_logp = _predict_indep_re_held(samples_indep, held_x, held_claims, held_exposure,
                                             rng_seed=DEPLOYED_CITIES.index(held))

        # --- Baseline 3: GP-prior with Euclidean kernel
        S_eu = S_euclid.loc[train_cities + [held], train_cities + [held]].to_numpy()
        S_eu_train = S_eu[:len(train_cities), :len(train_cities)]
        s_eu_to_train = S_eu[-1, :len(train_cities)]
        L_eu = np.linalg.cholesky(S_eu_train + 1e-6 * np.eye(len(train_cities)))
        train_data_eu = dict(train_data); train_data_eu["L_chol"] = jnp.asarray(L_eu)
        samples_eu = run_inference(ads_credibility_gp_model, train_data_eu, cfg_short)
        eu_logp = leave_one_city_out_predictive_logp(
            samples_eu, held_claims, held_exposure, held_x,
            similarity_to_train=s_eu_to_train,
            S_train_train=S_eu_train,
            s_self=float(S_eu[-1, -1]),
        )

        # --- Baseline 4: GP-prior with learned-embedding kernel (our method)
        S_emb_sub = S_embed.loc[train_cities + [held], train_cities + [held]].to_numpy()
        S_emb_train = S_emb_sub[:len(train_cities), :len(train_cities)]
        s_emb_to_train = S_emb_sub[-1, :len(train_cities)]
        L_emb = np.linalg.cholesky(S_emb_train + 1e-6 * np.eye(len(train_cities)))
        train_data_emb = dict(train_data); train_data_emb["L_chol"] = jnp.asarray(L_emb)
        samples_emb = run_inference(ads_credibility_gp_model, train_data_emb, cfg_short)
        emb_logp = leave_one_city_out_predictive_logp(
            samples_emb, held_claims, held_exposure, held_x,
            similarity_to_train=s_emb_to_train,
            S_train_train=S_emb_train,
            s_self=float(S_emb_sub[-1, -1]),
        )

        rows.append({
            "held_city": held,
            "held_claims": held_claims,
            "held_exposure_million_miles": held_exposure,
            "logp_pool": bs_logp,
            "logp_indep_re": indep_logp,
            "logp_gp_euclidean": eu_logp,
            "logp_gp_learned": emb_logp,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "loco_comparison.csv", index=False)
    return df


def _poisson_logp(k, rate):
    from math import lgamma, log
    rate = max(rate, 1e-30)
    return float(k * log(rate) - rate - lgamma(k + 1))


def _predict_indep_re_held(samples, held_x, held_claims, held_exposure,
                           rng_seed: int = 11):
    """
    For the independent-RE model with no own data on the held city, the
    posterior on alpha_held is just the prior N(0, tau_c^2). We marginalize
    by drawing alpha_held ~ N(0, tau_c) per posterior draw and computing the
    Poisson log-likelihood.

    rng_seed should differ across LOCO folds so that alpha_held draws are
    independent; the caller passes the held-city index as the seed offset.
    """
    from math import lgamma
    beta0 = samples["beta0"].reshape(-1)
    beta = samples["beta"].reshape(-1, samples["beta"].shape[-1])
    tau_c = samples["tau_c"].reshape(-1)
    rng = np.random.default_rng(rng_seed)
    alpha_h = rng.standard_normal(beta0.shape[0]) * tau_c
    log_lam = beta0 + beta @ held_x + alpha_h
    rate = np.exp(log_lam) * held_exposure
    rate = np.maximum(rate, 1e-30)
    logp = held_claims * np.log(rate) - rate - lgamma(held_claims + 1)
    m = logp.max()
    return float(m + np.log(np.mean(np.exp(logp - m))))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: Path, embed_dir: Path, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "posterior_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Load
    ads, hdv, city_odd, truth = load_data(data_dir)
    S_embed = pd.read_csv(embed_dir / "city_similarity.csv", index_col=0)
    data, df = prepare_model_data(ads, city_odd)
    all_cities = DEPLOYED_CITIES + HYPOTHETICAL_CITIES
    S_euclid = euclidean_similarity_from_features(city_odd, all_cities)
    S_euclid.to_csv(out_dir / "city_similarity_euclidean.csv")

    # Section 6.2
    print("[6.2] Benchmark reproduction")
    bench = benchmark_reproduction(ads, hdv)
    (out_dir / "benchmark_reproduction.json").write_text(json.dumps(bench, indent=2))
    print(f"      Overall ADS frequency:  {bench['overall_ads_freq']:.4f} per M miles")
    print(f"      Overall HDV frequency:  {bench['overall_hdv_freq']:.4f} per M miles")
    print(f"      Overall reduction:      {bench['overall_reduction_vs_hdv']*100:.1f}%")

    # Section 6.3
    print("[6.3] Fitting independent-RE hierarchical model")
    samples_indep = run_independent_re_model(data, samples_dir)
    diag_indep = diagnostics(
        samples_indep,
        ["beta0", "beta", "tau_c", "tau_v", "tau_cv", "alpha", "gamma"],
    )
    diag_indep.to_csv(out_dir / "diagnostics_indep.csv")
    worst_rhat_indep = float(pd.to_numeric(diag_indep["r_hat"], errors="coerce").max())
    print(f"      Worst R-hat: {worst_rhat_indep:.3f}")

    print("[6.3] Fitting GP-prior model")
    samples_gp = run_gp_model(data, S_embed, samples_dir)
    diag_gp = diagnostics(
        samples_gp,
        ["beta0", "beta", "sigma_c", "tau_v", "tau_cv", "alpha", "gamma"],
    )
    diag_gp.to_csv(out_dir / "diagnostics_gp.csv")
    worst_rhat_gp = float(pd.to_numeric(diag_gp["r_hat"], errors="coerce").max())
    print(f"      Worst R-hat: {worst_rhat_gp:.3f}")

    # Posterior summaries
    lam_indep = posterior_lambda_by_cell(samples_indep, df, data)
    lam_indep.to_csv(out_dir / "posterior_lambda_indep.csv", index=False)
    lam_gp = posterior_lambda_by_cell(samples_gp, df, data)
    lam_gp.to_csv(out_dir / "posterior_lambda_gp.csv", index=False)

    cred_indep = credibility_weights_from_posterior(samples_indep, df)
    cred_indep.to_csv(out_dir / "credibility_weights_indep.csv", index=False)
    cred_gp = credibility_weights_from_posterior(samples_gp, df)
    cred_gp.to_csv(out_dir / "credibility_weights_gp.csv", index=False)

    # Section 6.4
    print("[6.4] Prospective estimates for hypothetical new cities")
    pros = prospective_new_city_estimates(samples_gp, S_embed, city_odd)
    pros.to_csv(out_dir / "prospective_new_cities.csv", index=False)
    print(pros.to_string(index=False))

    # Hypothetical first million miles. Rather than one arbitrary integer count
    # per city, we show a scenario grid spanning below-, at-, and above-prior
    # experience (0, 1, and 3 crashes). On the SGO scale the prior medians are
    # ~0.76-1.28 crashes/M mi, so 1 crash is roughly the at-prior outcome, 0 is
    # a favorable surprise, and 3 an adverse one. This demonstrates the updating
    # mechanism without depending on a single Poisson draw.
    scenarios = {c: [0, 1, 3] for c in ["Miami", "Boston", "Denver"]}
    pros_post = prospective_with_first_million_miles(
        samples_gp, S_embed, city_odd, scenarios
    )
    pros_post.to_csv(out_dir / "prospective_after_first_million.csv", index=False)
    print(pros_post.to_string(index=False))

    # Section 4.3 closed-form sanity check
    print("[4.3] Buhlmann-Straub closed-form sanity check")
    agg_by_city = ads.groupby("city").agg(
        claims=("claims", "sum"), miles=("exposure_million_miles", "sum")
    ).reindex(DEPLOYED_CITIES)
    # Supply the (version, period) cells as repeated observations so the
    # within-city process variance is estimated empirically rather than via a
    # Poisson plug-in (which ignores the real version-driven over-dispersion
    # and inflates the credibility weights toward 1).
    ads_dep = ads[ads["city"].isin(DEPLOYED_CITIES)].copy()
    city_to_i = {c: i for i, c in enumerate(DEPLOYED_CITIES)}
    bs = buhlmann_straub_closed_form(
        agg_by_city["claims"].to_numpy(), agg_by_city["miles"].to_numpy(),
        cell_claims=ads_dep["claims"].to_numpy(),
        cell_exposure=ads_dep["exposure_million_miles"].to_numpy(),
        cell_city_index=ads_dep["city"].map(city_to_i).to_numpy(),
    )
    print(f"      within-city process variance v = {bs['s_var_within']:.2f} "
          f"(empirical={bs['used_empirical_within_variance']}), "
          f"between-city a = {bs['a_var_between']:.2f}")
    bs_df = pd.DataFrame({
        "city": DEPLOYED_CITIES,
        "claims": agg_by_city["claims"].to_numpy(),
        "miles": agg_by_city["miles"].to_numpy(),
        "own_rate": bs["own_estimate"],
        "Z_buhlmann_straub": bs["Z"],
        "lambda_hat": bs["lambda_hat"],
    })
    bs_df.to_csv(out_dir / "buhlmann_straub_closed_form.csv", index=False)
    print(bs_df.to_string(index=False))

    # Section 6.5
    print("[6.5] Leave-one-city-out comparison across baselines")
    loco = loco_comparison(ads, city_odd, S_embed, S_euclid, out_dir)
    print(loco.to_string(index=False))

    totals = {
        "logp_pool":        float(loco["logp_pool"].sum()),
        "logp_indep_re":    float(loco["logp_indep_re"].sum()),
        "logp_gp_euclid":   float(loco["logp_gp_euclidean"].sum()),
        "logp_gp_learned":  float(loco["logp_gp_learned"].sum()),
    }
    summary = {
        "benchmark_reduction_vs_hdv": bench["overall_reduction_vs_hdv"],
        "worst_rhat_indep": worst_rhat_indep,
        "worst_rhat_gp": worst_rhat_gp,
        "loco_total_logp": totals,
        "best_model_by_loco": max(totals, key=totals.get),
    }
    (out_dir / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--embed-dir", default="results/embedding", type=Path)
    parser.add_argument("--out", default="results", type=Path)
    args = parser.parse_args()
    main(args.data_dir, args.embed_dir, args.out)