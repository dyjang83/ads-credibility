
from __future__ import annotations
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(4)


# ---------------------------------------------------------------------------
# Parameterized version of the GP-prior model (tau scale is a script-level
# constant rather than a hard-coded literal).
# ---------------------------------------------------------------------------

def make_gp_model(tau_scale: float, tau_cv_scale: float = 0.3):
    def model(city_idx, version_idx, X, exposure, L_chol,
              claims=None, n_cities=4, n_versions=5, n_features=6):
        beta0 = numpyro.sample("beta0", dist.Normal(0.0, 2.5))
        # Regularizing prior on covariate coefficients, matching models.py
        # (see the note there): Normal(0, 0.5) on z-scored features prevents
        # the weakly-identified linear predictor from exploding out of hull.
        beta = numpyro.sample("beta",
                              dist.Normal(jnp.zeros(n_features), 0.5 * jnp.ones(n_features)))
        sigma_c = numpyro.sample("sigma_c", dist.HalfNormal(tau_scale))
        tau_v   = numpyro.sample("tau_v",   dist.HalfNormal(tau_scale))
        tau_cv  = numpyro.sample("tau_cv",  dist.HalfNormal(tau_cv_scale))

        z_alpha = numpyro.sample("z_alpha",
                                 dist.Normal(jnp.zeros(n_cities), jnp.ones(n_cities)))
        alpha = numpyro.deterministic("alpha", sigma_c * (L_chol @ z_alpha))

        gamma_raw = numpyro.sample("gamma_raw",
                                   dist.Normal(jnp.zeros(n_versions), jnp.ones(n_versions)))
        delta_raw = numpyro.sample("delta_raw",
            dist.Normal(jnp.zeros((n_cities, n_versions)),
                        jnp.ones((n_cities, n_versions))))
        gamma = numpyro.deterministic("gamma", tau_v  * gamma_raw)
        delta = numpyro.deterministic("delta", tau_cv * delta_raw)

        log_lambda = (beta0 + X @ beta + alpha[city_idx]
                      + gamma[version_idx] + delta[city_idx, version_idx])
        lam = jnp.exp(log_lambda)
        numpyro.deterministic("lambda", lam)
        rate = lam * exposure
        numpyro.sample("claims", dist.Poisson(rate), obs=claims)
    return model


def predict_new_city(samples, S_train_train, s_to_train, s_self, x_new):
    """
    Conditional Gaussian draw for alpha_new given alpha posterior on
    deployed cities, then lambda_new = exp(beta0 + x_new @ beta + alpha_new).
    """
    alpha_dep   = samples["alpha"]
    sigma_c     = samples["sigma_c"]
    beta0       = samples["beta0"]
    beta        = samples["beta"]
    n_draws     = alpha_dep.shape[0]

    S_inv = np.linalg.inv(S_train_train + 1e-6 * np.eye(S_train_train.shape[0]))
    sst   = s_to_train @ S_inv
    mu_cond_coef = sst                                # (n_train,)
    var_cond_unit = float(s_self - sst @ s_to_train)
    var_cond_unit = max(var_cond_unit, 1e-8)

    rng = np.random.default_rng(42)
    eps = rng.standard_normal(n_draws)
    mu_cond  = alpha_dep @ mu_cond_coef                              # (n_draws,)
    sigma_cond = sigma_c * np.sqrt(var_cond_unit)                    # (n_draws,)
    alpha_new = mu_cond + sigma_cond * eps                           # (n_draws,)

    log_lam = beta0 + beta @ x_new + alpha_new
    lam = np.exp(log_lam)
    return lam


def main():
    repo = Path(__file__).resolve().parents[1]  # src/ -> repo root
    out = repo / "results"
    out.mkdir(exist_ok=True, parents=True)

    ads = pd.read_csv(repo / "data" / "ads_events.csv")
    city_odd = pd.read_csv(repo / "data" / "city_odd_features.csv")
    S_embed = pd.read_csv(out / "embedding" / "city_similarity.csv", index_col=0)

    DEPLOYED = ["San Francisco", "Phoenix", "Los Angeles", "Austin"]
    NEW_CITIES = ["Boston", "Denver", "Miami"]
    CITY_ODD_FEATURES = [
        "arterial_share", "intersection_density", "signalized_fraction",
        "weather_rain_fraction", "weather_fog_fraction",
        "operating_hours_night_share",
    ]
    ads_train = ads[ads["city"].isin(DEPLOYED)].copy()
    # Derive the version vocabulary from the data so the real NHTSA SGO labels
    # ("5th Generation ADS, Version 10", ...) work without code edits, matching
    # run_analysis.resolve_versions(). The synthetic generator's labels would
    # also round-trip through sorted-unique.
    versions = sorted(ads_train["version"].dropna().unique().tolist())
    city_to_idx = {c: i for i, c in enumerate(DEPLOYED)}
    version_to_idx = {v: i for i, v in enumerate(versions)}
    ads_train["city_idx"] = ads_train["city"].map(city_to_idx)
    ads_train["version_idx"] = ads_train["version"].map(version_to_idx)
    df_train = ads_train.merge(city_odd, on="city")
    X = jnp.asarray(df_train[CITY_ODD_FEATURES].to_numpy(dtype=float))
    data = {
        "city_idx": jnp.asarray(df_train["city_idx"].to_numpy(), dtype=jnp.int32),
        "version_idx": jnp.asarray(df_train["version_idx"].to_numpy(), dtype=jnp.int32),
        "X": X,
        "exposure": jnp.asarray(df_train["exposure_million_miles"].to_numpy(dtype=float)),
        "claims": jnp.asarray(df_train["claims"].to_numpy(), dtype=jnp.int32),
        "n_cities": len(DEPLOYED),
        "n_versions": len(versions),
        "n_features": X.shape[1],
    }

    S_train_train = S_embed.loc[DEPLOYED, DEPLOYED].to_numpy()
    L = np.linalg.cholesky(S_train_train + 1e-6 * np.eye(len(DEPLOYED)))
    data["L_chol"] = jnp.asarray(L)

    sensitivity_rows = []
    for tau_scale in [0.3, 0.5, 1.0]:
        print(f"\n=== Prior scale tau_c ~ HalfNormal({tau_scale}) ===")
        model = make_gp_model(tau_scale=tau_scale, tau_cv_scale=0.3)
        mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=1500, num_chains=2,
                    progress_bar=False)
        mcmc.run(jax.random.PRNGKey(20260525), **data)
        samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        rhat = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
        worst_rhat = max(np.nanmax(s["r_hat"]) for s in rhat.values() if "r_hat" in s)
        print(f"   worst R-hat = {worst_rhat:.3f}")

        for new_city in NEW_CITIES:
            s_to_train = S_embed.loc[new_city, DEPLOYED].to_numpy()
            s_self     = float(S_embed.loc[new_city, new_city])
            x_new = city_odd.set_index("city").loc[new_city, CITY_ODD_FEATURES].to_numpy(dtype=float)
            lam_draws = predict_new_city(samples, S_train_train, s_to_train, s_self, x_new)
            q025, q500, q975 = np.percentile(lam_draws, [2.5, 50, 97.5])
            print(f"   {new_city:12s}: median {q500:.3f}, 95% CI ({q025:.3g}, {q975:.3g})")
            sensitivity_rows.append({
                "tau_c_prior_scale": tau_scale,
                "city": new_city,
                "lambda_median": float(q500),
                "lambda_q025": float(q025),
                "lambda_q975": float(q975),
                "worst_rhat": float(worst_rhat),
            })

    df = pd.DataFrame(sensitivity_rows)
    df.to_csv(out / "prior_sensitivity.csv", index=False)
    print()
    print(df.to_string(index=False))

    # ---- Compute relative changes in medians vs the baseline (0.5) ----
    pivot = df.pivot(index="city", columns="tau_c_prior_scale",
                     values="lambda_median")
    pivot["rel_03_vs_05"] = (pivot[0.3] - pivot[0.5]) / pivot[0.5]
    pivot["rel_10_vs_05"] = (pivot[1.0] - pivot[0.5]) / pivot[0.5]
    print()
    print("Relative change in posterior median vs baseline (scale=0.5):")
    print(pivot.round(3).to_string())
    pivot.to_csv(out / "prior_sensitivity_pivot.csv")


if __name__ == "__main__":
    main()
