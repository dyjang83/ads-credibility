"""
Belief state for the decision layer: a Laplace-Gaussian posterior over
theta = (mu, alpha_1..alpha_K) under

    mu      ~ N(mu0, MU_PRIOR_SD^2)
    alpha   ~ N(0, sigma^2 S)
    n_k     ~ Poisson(exp(mu + alpha_k) * E_k)

where (n_k, E_k) are a city's cumulative claims and exposure. This is the
city-level model of power_analysis.py (same prior, same Newton iteration), fit
jointly over ALL K candidate cities: cities with zero exposure contribute only
their prior, so the joint Laplace mode/covariance reproduce `_laplace_fit` on the
observed block and the conditional-Gaussian (`_cond_weights`) on the rest. The
unit tests check both identities.
"""
from __future__ import annotations

import numpy as np

from .config import JITTER, MU_PRIOR_SD


class CityBelief:
    def __init__(self, S: np.ndarray, sigma: float, base_rate: float,
                 mu_sd: float = MU_PRIOR_SD):
        self.K = S.shape[0]
        self.S = S
        self.sigma = sigma
        self.mu0 = float(np.log(base_rate))
        self.mu_prec = 1.0 / mu_sd ** 2
        self.prior_cov_alpha = sigma ** 2 * S + JITTER * np.eye(self.K)
        self.Sig_inv = np.linalg.inv(self.prior_cov_alpha)
        self.counts = np.zeros(self.K)
        self.expo = np.zeros(self.K)
        self.mode = np.concatenate([[self.mu0], np.zeros(self.K)])
        self.cov = np.zeros((self.K + 1, self.K + 1))
        self.cov[0, 0] = mu_sd ** 2
        self.cov[1:, 1:] = self.prior_cov_alpha

    # ------------------------------------------------------------------ fit
    def update(self, counts: np.ndarray, expo: np.ndarray) -> "CityBelief":
        """Refit to cumulative (counts, exposure) per city. Warm-started."""
        self.counts = np.asarray(counts, float)
        self.expo = np.asarray(expo, float)
        mu, alpha = self.mode[0], self.mode[1:].copy()
        for _ in range(60):
            lam = np.exp(mu + alpha) * self.expo
            g_mu = np.sum(self.counts - lam) - self.mu_prec * (mu - self.mu0)
            g_al = (self.counts - lam) - self.Sig_inv @ alpha
            H = np.empty((self.K + 1, self.K + 1))
            H[0, 0] = -np.sum(lam) - self.mu_prec
            H[0, 1:] = H[1:, 0] = -lam
            H[1:, 1:] = -np.diag(lam) - self.Sig_inv
            step = np.linalg.solve(H, np.concatenate([[g_mu], g_al]))
            mu -= step[0]
            alpha -= step[1:]
            if np.max(np.abs(step)) < 1e-9:
                break
        lam = np.exp(mu + alpha) * self.expo
        H[0, 0] = -np.sum(lam) - self.mu_prec
        H[0, 1:] = H[1:, 0] = -lam
        H[1:, 1:] = -np.diag(lam) - self.Sig_inv
        self.mode = np.concatenate([[mu], alpha])
        self.cov = np.linalg.inv(-H)
        self.cov = 0.5 * (self.cov + self.cov.T)
        return self

    # ---------------------------------------------------------- predictive
    def log_rate_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and variance of eta_k = mu + alpha_k, all k."""
        m = self.mode[0] + self.mode[1:]
        v = (self.cov[0, 0] + np.diag(self.cov)[1:] + 2.0 * self.cov[0, 1:])
        return m, np.maximum(v, 1e-12)

    def mean_rate(self) -> np.ndarray:
        """Posterior mean of lambda_k = exp(eta_k) (lognormal mean)."""
        m, v = self.log_rate_moments()
        return np.exp(m + 0.5 * v)

    def eta_cov(self) -> np.ndarray:
        """Full posterior covariance of eta = mu*1 + alpha (K x K)."""
        W = np.hstack([np.ones((self.K, 1)), np.eye(self.K)])
        return W @ self.cov @ W.T

    def sample_log_rates(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """(n, K) joint draws of eta from the Gaussian approximation."""
        m, _ = self.log_rate_moments()
        L = np.linalg.cholesky(self.eta_cov() + 1e-10 * np.eye(self.K))
        return m + rng.standard_normal((n, self.K)) @ L.T

    # ------------------------------------------------------- information
    def var_reduction(self, k: int, d_expo: float) -> np.ndarray:
        """Expected reduction in Var(eta_j), all j, from d_expo more exposure in
        city k. Uses the expected Fisher information d_expo * E[lambda_k] in
        the eta_k direction (rank-one Gaussian update)."""
        C = self.eta_cov()
        f = d_expo * self.mean_rate()[k]
        ck = C[:, k]
        return f * ck ** 2 / (1.0 + f * C[k, k])

    def info_gain(self, k: int, d_expo: float) -> float:
        """Entropy reduction (nats) of the joint posterior from d_expo more
        exposure in city k (same rank-one approximation)."""
        C_kk = self.eta_cov()[k, k]
        return 0.5 * float(np.log1p(d_expo * self.mean_rate()[k] * C_kk))
