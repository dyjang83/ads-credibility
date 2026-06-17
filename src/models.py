"""
Hierarchical Bayesian credibility model for ADS pricing.

Implements:
  - Section 4: hierarchical Poisson GLM with independent random effects on
    city, software version, and city x version interaction
  - Section 5.3: GP-prior extension where the city random effect uses a
    multivariate Gaussian prior with covariance set by the learned
    ODD-similarity matrix
  - Section 4.3: closed-form Buhlmann-Straub credibility weights for
    the limiting case (no covariates, no version effect)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS

# Allow multi-chain MCMC on a single CPU host
numpyro.set_host_device_count(4)


# ---------------------------------------------------------------------------
# Model A: Independent random effects (Section 4 / Appendix A)
# ---------------------------------------------------------------------------

def ads_credibility_model(
    city_idx,
    version_idx,
    X,
    exposure,
    claims=None,
    n_cities: int = 4,
    n_versions: int = 5,
    n_features: int = 6,
):
    """
    Hierarchical Bayesian Poisson model with non-centered random effects.

    This is the implementation that the paper's Appendix A sketches, expanded
    and tightened. Inputs:
      city_idx:    (N,) int array of city indices
      version_idx: (N,) int array of version indices
      X:           (N, n_features) covariate matrix (centered/scaled)
      exposure:    (N,) exposure in millions of miles
      claims:      (N,) observed claim counts, or None for prior predictive
    """
    # Fixed effects
    beta0 = numpyro.sample("beta0", dist.Normal(0.0, 2.5))
    # Weakly-informative prior on the covariate coefficients. The original
    # sketch used Normal(0, 2.5), but with only a handful of claims spread
    # across four cities the six ODD covariates are weakly identified
    # (Section 6.3): the posterior on beta stays close to that diffuse prior,
    # and the resulting linear predictor beta'x extrapolates explosively when
    # x is an out-of-sample city far from the training hull (the Miami
    # prospective interval blew up to ~[1e-4, 7e4] purely from this term).
    # We tighten to Normal(0, 0.5) on the z-scored features -- the same
    # regularization logic applied to tau_c below. A coefficient of 0.5
    # corresponds to a ~1.65x multiplicative change in baseline frequency per
    # 1-SD feature move; the 95% prior interval (~+/-1.0) brackets a 0.37x-2.7x
    # range per SD, which comfortably covers any plausible ODD effect while
    # preventing the unidentified tail from dominating out-of-hull prediction.
    beta = numpyro.sample(
        "beta",
        dist.Normal(jnp.zeros(n_features), 0.5 * jnp.ones(n_features)),
    )

    # Hyperpriors on random-effect scales. The original Appendix A sketch
    # used HalfNormal(1) for tau_c, but in the ADS sparse-data regime (few
    # claims per city) this prior is too diffuse: the posterior tail extends
    # to tau_c ~ 3+, which destabilizes exponential-link predictions for
    # new cities. We tighten to HalfNormal(0.5), which still admits a 95%
    # prior credible interval of roughly (0, 1.4) on the log scale -- i.e.
    # ~4x multiplicative variation in baseline frequency across cities, an
    # order of magnitude that brackets every reasonable actuarial scenario.
    tau_c = numpyro.sample("tau_c", dist.HalfNormal(0.5))
    tau_v = numpyro.sample("tau_v", dist.HalfNormal(0.5))
    tau_cv = numpyro.sample("tau_cv", dist.HalfNormal(0.3))

    # Non-centered parameterization for each random effect
    alpha_raw = numpyro.sample(
        "alpha_raw", dist.Normal(jnp.zeros(n_cities), jnp.ones(n_cities))
    )
    gamma_raw = numpyro.sample(
        "gamma_raw", dist.Normal(jnp.zeros(n_versions), jnp.ones(n_versions))
    )
    delta_raw = numpyro.sample(
        "delta_raw",
        dist.Normal(jnp.zeros((n_cities, n_versions)), jnp.ones((n_cities, n_versions))),
    )

    alpha = numpyro.deterministic("alpha", tau_c * alpha_raw)
    gamma = numpyro.deterministic("gamma", tau_v * gamma_raw)
    delta = numpyro.deterministic("delta", tau_cv * delta_raw)

    log_lambda = (
        beta0
        + X @ beta
        + alpha[city_idx]
        + gamma[version_idx]
        + delta[city_idx, version_idx]
    )
    lam = jnp.exp(log_lambda)
    numpyro.deterministic("lambda", lam)

    rate = lam * exposure
    numpyro.sample("claims", dist.Poisson(rate), obs=claims)


# ---------------------------------------------------------------------------
# Model B: GP-prior city random effects (Section 5.3)
# ---------------------------------------------------------------------------

def ads_credibility_gp_model(
    city_idx,
    version_idx,
    X,
    exposure,
    L_chol,             # Cholesky factor of the city covariance kernel
    claims=None,
    n_cities: int = 4,
    n_versions: int = 5,
    n_features: int = 6,
):
    """
    Same model as ads_credibility_model but with a multivariate-normal prior
    on the city random effect, with covariance sigma^2 * S where S is the
    learned ODD-similarity matrix and L_chol is its Cholesky factor.
    """
    beta0 = numpyro.sample("beta0", dist.Normal(0.0, 2.5))
    # Weakly-informative prior on covariate coefficients; see the matching
    # note in ads_credibility_model. Normal(0, 0.5) on z-scored features
    # regularizes the weakly-identified fixed effects so that the
    # exponential-link predictor does not explode when extrapolated to an
    # out-of-hull new city. This is the single change that fixes the Miami
    # prospective interval; it leaves the in-sample fit essentially unchanged.
    beta = numpyro.sample(
        "beta",
        dist.Normal(jnp.zeros(n_features), 0.5 * jnp.ones(n_features)),
    )

    sigma_c = numpyro.sample("sigma_c", dist.HalfNormal(0.5))
    tau_v = numpyro.sample("tau_v", dist.HalfNormal(0.5))
    tau_cv = numpyro.sample("tau_cv", dist.HalfNormal(0.3))

    # alpha ~ N(0, sigma_c^2 * S) via non-centered MVN parameterization:
    # alpha = sigma_c * L @ z,  z ~ N(0, I)
    z_alpha = numpyro.sample(
        "z_alpha", dist.Normal(jnp.zeros(n_cities), jnp.ones(n_cities))
    )
    alpha = numpyro.deterministic("alpha", sigma_c * (L_chol @ z_alpha))

    gamma_raw = numpyro.sample(
        "gamma_raw", dist.Normal(jnp.zeros(n_versions), jnp.ones(n_versions))
    )
    delta_raw = numpyro.sample(
        "delta_raw",
        dist.Normal(jnp.zeros((n_cities, n_versions)), jnp.ones((n_cities, n_versions))),
    )
    gamma = numpyro.deterministic("gamma", tau_v * gamma_raw)
    delta = numpyro.deterministic("delta", tau_cv * delta_raw)

    log_lambda = (
        beta0
        + X @ beta
        + alpha[city_idx]
        + gamma[version_idx]
        + delta[city_idx, version_idx]
    )
    lam = jnp.exp(log_lambda)
    numpyro.deterministic("lambda", lam)
    rate = lam * exposure
    numpyro.sample("claims", dist.Poisson(rate), obs=claims)


# ---------------------------------------------------------------------------
# Model C: Single-pool Buhlmann-Straub baseline (no city/version effects)
# ---------------------------------------------------------------------------

def buhlmann_straub_baseline(city_idx, version_idx, X, exposure, claims=None,
                              n_cities: int = 4, n_versions: int = 5,
                              n_features: int = 6):
    """
    Baseline: single global rate, no random effects, no covariates.
    Used to compute the trivial pooled-mean predictor.
    """
    beta0 = numpyro.sample("beta0", dist.Normal(0.0, 2.5))
    lam = jnp.exp(beta0)
    numpyro.deterministic("lambda", lam)
    rate = lam * exposure
    numpyro.sample("claims", dist.Poisson(rate), obs=claims)


# ---------------------------------------------------------------------------
# Inference driver
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    n_warmup: int = 1000
    n_samples: int = 2000
    n_chains: int = 2
    target_accept_prob: float = 0.95
    seed: int = 20260525


def run_inference(
    model,
    data: dict,
    config: InferenceConfig,
) -> dict:
    """Run NUTS MCMC and return posterior samples as a dict of arrays."""
    numpyro.set_host_device_count(config.n_chains)
    kernel = NUTS(model, target_accept_prob=config.target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=config.n_warmup,
        num_samples=config.n_samples,
        num_chains=config.n_chains,
        progress_bar=False,
    )
    rng = jax.random.PRNGKey(config.seed)
    mcmc.run(rng, **data)
    samples = mcmc.get_samples(group_by_chain=True)
    return {k: np.asarray(v) for k, v in samples.items()}


def diagnostics(samples: dict, params: list[str]) -> pd.DataFrame:
    """
    Compute R-hat and effective sample size for a list of parameters.
    Uses arviz under the hood for the standard diagnostics in Section 4.2.
    """
    import arviz as az
    # samples are (n_chains, n_samples, *param_shape)
    data_dict = {
        k: v for k, v in samples.items()
        if k in params and v.ndim >= 2
    }
    # arviz 1.x expects a nested {group: {var: array}} structure
    idata = az.from_dict({"posterior": data_dict})
    summary = az.summary(idata, var_names=list(data_dict.keys()))
    return summary


# ---------------------------------------------------------------------------
# Posterior credibility weight calculations
# ---------------------------------------------------------------------------

def buhlmann_straub_closed_form(
    claims_by_city: np.ndarray,
    exposure_by_city: np.ndarray,
    cell_claims=None,
    cell_exposure=None,
    cell_city_index=None,
) -> dict:
    """
    Section 4.3 closed-form Buhlmann-Straub credibility weights.

    Returns the per-city credibility weights Z_c, the global pooled estimate,
    and the credibility-weighted estimates lambda_hat_c. The structure
    parameters (a = Var[mu(theta)], v = E[sigma^2(theta)]) are estimated by the
    standard Buhlmann-Straub method of moments (Buhlmann & Gisler, 2005).

    Within-unit (process) variance v
    --------------------------------
    Each city is a risk observed through several (version, period) CELLS, which
    are the natural repeated observations for the Buhlmann-Straub estimator.
    When those cells are supplied (via cell_claims / cell_exposure /
    cell_city_index) we estimate v EMPIRICALLY from the weighted within-city
    scatter of the cell rates:

        v_hat = [ sum_i sum_t w_it (X_it - X_i.)^2 ] / [ sum_i (T_i - 1) ]

    This is the textbook unbiased estimator and, unlike a Poisson plug-in,
    captures the real over-dispersion of the SGO data (cell rates within a city
    vary by 8-33x more than Poisson sampling allows, driven mainly by software
    version). Using the Poisson plug-in v = X_bar instead ignores that
    dispersion and inflates every credibility weight toward 1.

    If no cell-level data are supplied, or a city has only one cell (T_i = 1, so
    it contributes no within-city degrees of freedom), we fall back to the
    Poisson plug-in v = X_bar for that estimate. The fallback is reported in the
    returned dict so callers can flag it.
    """
    K = len(claims_by_city)
    w = exposure_by_city
    X = np.divide(claims_by_city, w, out=np.zeros_like(w, dtype=float), where=w > 0)

    w_total = w.sum()
    X_bar = (w * X).sum() / w_total

    # ---- Within-unit process variance v = E[sigma^2(theta)] ----
    used_empirical_v = False
    if cell_claims is not None and cell_exposure is not None \
            and cell_city_index is not None:
        cell_claims = np.asarray(cell_claims, dtype=float)
        cell_exposure = np.asarray(cell_exposure, dtype=float)
        cell_city_index = np.asarray(cell_city_index)
        num_v, den_v = 0.0, 0.0
        for i in range(K):
            sel = cell_city_index == i
            w_it = cell_exposure[sel]
            if w_it.sum() <= 0 or sel.sum() < 2:
                continue                       # T_i < 2: no within-city d.o.f.
            X_it = np.divide(cell_claims[sel], w_it,
                             out=np.zeros_like(w_it), where=w_it > 0)
            X_i = cell_claims[sel].sum() / w_it.sum()   # exposure-weighted city rate
            num_v += (w_it * (X_it - X_i) ** 2).sum()
            den_v += (sel.sum() - 1)
        if den_v > 0 and num_v > 0:
            v_hat = num_v / den_v
            used_empirical_v = True
        else:
            v_hat = X_bar                      # fallback: Poisson plug-in
    else:
        v_hat = X_bar                          # no cells supplied: Poisson plug-in

    # ---- Between-city heterogeneity a = Var[mu(theta)] ----
    # Buhlmann-Straub unbiased method-of-moments estimator:
    #   a = ( sum_i w_i (X_i - X_bar)^2 - (K-1) v ) / ( w_tot - sum_i w_i^2 / w_tot )
    # The denominator c makes a unbiased; the (K-1) v subtraction removes the
    # expected within-unit contribution to the weighted between-city SS.
    c = w_total - (w ** 2).sum() / w_total
    numerator = (w * (X - X_bar) ** 2).sum() - (K - 1) * v_hat
    a = max(numerator / max(c, 1e-9), 1e-6)

    Kbs = v_hat / a
    Z = w / (w + Kbs)
    lam_hat = Z * X + (1.0 - Z) * X_bar

    return {
        "X_bar": float(X_bar),
        "a_var_between": float(a),
        "s_var_within": float(v_hat),
        "used_empirical_within_variance": bool(used_empirical_v),
        "K": float(Kbs),
        "Z": Z,
        "lambda_hat": lam_hat,
        "own_estimate": X,
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_samples(samples: dict, out_dir: Path, name: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{name}.npz", **samples)


def load_samples(out_dir: Path, name: str) -> dict:
    out_dir = Path(out_dir)
    data = np.load(out_dir / f"{name}.npz")
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Posterior predictive for a new city (Section 5.3)
# ---------------------------------------------------------------------------

def predict_new_city(
    samples: dict,
    similarity_to_deployed: np.ndarray,  # (n_deployed,) similarities to new city
    x_new: np.ndarray,                   # (n_features,) covariates for new city
    similarity_self: float = 1.0,
    version_idx_for_prior: Optional[int] = None,
) -> np.ndarray:
    """
    Posterior predictive for a new city's claim frequency under the GP-prior
    model. For each posterior draw:
      - draw alpha_new conditional on the deployed-city alphas and the GP
        covariance over {deployed + new}
      - optionally include a version effect for the prediction
      - return exp(beta0 + x_new'beta + alpha_new + ...)
    """
    # Shapes from samples
    beta0 = samples["beta0"]            # (chains, samples)
    beta = samples["beta"]              # (chains, samples, n_features)
    alpha_dep = samples["alpha"]        # (chains, samples, n_deployed)
    sigma_c = samples["sigma_c"]        # (chains, samples)

    # Flatten chain + sample axes
    beta0_f = beta0.reshape(-1)
    beta_f = beta.reshape(-1, beta.shape[-1])
    alpha_f = alpha_dep.reshape(-1, alpha_dep.shape[-1])
    sigma_f = sigma_c.reshape(-1)

    n_draws = beta0_f.shape[0]
    n_dep = alpha_f.shape[1]

    # Build covariance over [deployed; new]; we only know the cross-row
    # (similarity_to_deployed) and the marginal (similarity_self).
    # The full S over deployed cities is constant across draws -- we receive
    # it via S_dep_inv pre-computed externally. To keep this function
    # self-contained, we assume the caller passes a precomputed expectation.
    raise NotImplementedError(
        "Use predict_new_city_with_S below; it needs the full S over deployed cities."
    )


def predict_new_city_with_S(
    samples: dict,
    S_dep_dep: np.ndarray,               # (n_dep, n_dep) full S over deployed
    S_new_dep: np.ndarray,               # (n_dep,) similarities new -> deployed
    s_new_new: float,                    # similarity_self (S[new, new])
    x_new: np.ndarray,                   # (n_features,)
    version_effect: Optional[np.ndarray] = None,  # (n_draws,) gamma_v draws or None
    rng_seed: int = 17,
) -> np.ndarray:
    """
    For each posterior draw of (beta0, beta, alpha_deployed, sigma_c),
    sample alpha_new | alpha_deployed under the conditional Gaussian
    distribution implied by the joint MVN prior with covariance
    sigma_c^2 * S_full. Return draws of exp(beta0 + x_new'beta + alpha_new
    [+ gamma_v]).
    """
    rng = np.random.default_rng(rng_seed)
    beta0 = samples["beta0"].reshape(-1)
    beta = samples["beta"].reshape(-1, samples["beta"].shape[-1])
    alpha = samples["alpha"].reshape(-1, samples["alpha"].shape[-1])
    sigma_c = samples["sigma_c"].reshape(-1)

    n_draws = beta0.shape[0]

    # Pre-compute the constant-in-draws conditional-Normal regression weights:
    # conditional mean: mu_new|dep = (S_new_dep @ S_dep_dep^{-1}) @ alpha_dep
    # conditional variance multiplier: s_new_new - S_new_dep @ S_dep_dep^{-1} @ S_new_dep
    # The variance scales by sigma_c^2 each draw.
    S_dep_inv = np.linalg.inv(S_dep_dep + 1e-6 * np.eye(S_dep_dep.shape[0]))
    proj = S_new_dep @ S_dep_inv                # (n_dep,)
    cond_var_base = float(s_new_new - proj @ S_new_dep)
    cond_var_base = max(cond_var_base, 1e-8)

    # Conditional means per draw
    cond_means = alpha @ proj                   # (n_draws,)
    # Conditional sd per draw
    cond_sds = sigma_c * np.sqrt(cond_var_base)
    alpha_new = cond_means + cond_sds * rng.standard_normal(n_draws)

    log_lam = beta0 + beta @ x_new + alpha_new
    if version_effect is not None:
        log_lam = log_lam + version_effect
    return np.exp(log_lam)


# ---------------------------------------------------------------------------
# Out-of-sample one-city-out evaluation
# ---------------------------------------------------------------------------

def leave_one_city_out_predictive_logp(
    samples: dict,
    held_city_claims: int,
    held_city_exposure: float,
    held_city_x: np.ndarray,
    similarity_to_train: np.ndarray,
    S_train_train: np.ndarray,
    s_self: float = 1.0,
) -> float:
    """
    Predictive log-likelihood for a held-out city: integrate over the
    posterior of the model fit on the remaining cities, drawing alpha for
    the held-out city from its conditional GP and scoring the held-out
    Poisson count.
    """
    lam_draws = predict_new_city_with_S(
        samples,
        S_dep_dep=S_train_train,
        S_new_dep=similarity_to_train,
        s_new_new=s_self,
        x_new=held_city_x,
    )
    # Poisson log-pmf, averaged over draws (log-sum-exp / N)
    rate = lam_draws * held_city_exposure
    # log P(N = k | rate) = k log rate - rate - log k!
    from math import lgamma
    logp = held_city_claims * np.log(np.maximum(rate, 1e-30)) - rate - lgamma(held_city_claims + 1)
    # log mean exp over draws
    m = logp.max()
    lpd = m + np.log(np.mean(np.exp(logp - m)))
    return float(lpd)
