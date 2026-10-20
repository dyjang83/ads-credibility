"""
Expansion policies built directly on the posterior (baselines for PPO).

Every agent returns (city j, loading index l), or (-1, 0) to offer nothing this
quarter. All model-based agents score an offer with the same value function

  V(j, l | eta) = P_acc(eta_j, r_jl) * miles_j * [ (r_jl - e^eta_j) * min(C, H)
                                                  + RERATE * e^eta_j * max(H - C, 0) ]

where r_jl = (posterior mean rate_j) * (1 + LOADINGS[l]) is the offered rate,
H the quarters remaining, C the entry-contract length, and P_acc the operator's
acceptance curve evaluated at the agent's view eta of the true log rate. Agents
differ only in how they treat uncertainty in eta:

  plugin     eta at the posterior mean rate (ignores uncertainty)
  bayes      E_eta[V] over posterior draws (prices adverse selection)
  thompson   V at one posterior draw
  ucb        E[V] + beta * sd[V]
  kg         E[V] + one-step value of information: the expected improvement in
             the best next offer once entering j shrinks every other city's
             posterior variance (rank-one update, CityBelief.var_reduction)
  round_robin / random   model-free references
  oracle     V at the TRUE log rates (use with ExpansionEnv(oracle=True))
"""
from __future__ import annotations

import numpy as np

from .config import LOADINGS, RERATE_LOADING
from .env import CONTRACT_QUARTERS, accept_prob

LOAD = np.asarray(LOADINGS)


def offer_values(obs, eta: np.ndarray, rates: np.ndarray | None = None) -> np.ndarray:
    """V for every (draw, city, loading): eta (D, K) -> (D, K, L)."""
    H = obs.horizon - obs.t
    if rates is None:
        rates = obs.belief.mean_rate()
    r = rates[None, :, None] * (1.0 + LOAD)[None, None, :]
    lam = np.exp(eta)[:, :, None]
    p = accept_prob(eta[:, :, None], r)
    c = min(CONTRACT_QUARTERS, H)
    v = p * obs.miles[None, :, None] * ((r - lam) * c + RERATE_LOADING * lam * max(H - c, 0))
    return v


def _argmax_offer(score: np.ndarray, cand: np.ndarray) -> tuple[int, int]:
    s = np.where(cand[:, None], score, -np.inf)
    j, l = np.unravel_index(int(np.argmax(s)), s.shape)
    return (int(j), int(l)) if s[j, l] > 0 else (-1, 0)


class Agent:
    name = "agent"

    def act(self, obs, rng) -> tuple[int, int]:
        raise NotImplementedError


class RandomAgent(Agent):
    name = "random"

    def act(self, obs, rng):
        c = np.flatnonzero(obs.candidates)
        return (int(rng.choice(c)), int(rng.integers(len(LOAD)))) if len(c) else (-1, 0)


class RoundRobinAgent(Agent):
    """Fixed pool order at a fixed loading: the analogue of the nested fixed
    design in power_analysis_ncities.py."""
    name = "round_robin"

    def __init__(self, loading_idx: int = 2):
        self.l = loading_idx

    def act(self, obs, rng):
        c = np.flatnonzero(obs.candidates)
        return (int(c[0]), self.l) if len(c) else (-1, 0)


class PluginAgent(Agent):
    name = "plugin"

    def act(self, obs, rng):
        eta = np.log(obs.belief.mean_rate())[None]
        return _argmax_offer(offer_values(obs, eta)[0], obs.candidates)


class BayesAgent(Agent):
    name = "bayes"

    def __init__(self, n_draws: int = 128):
        self.n = n_draws

    def scores(self, obs, rng):
        return offer_values(obs, obs.belief.sample_log_rates(self.n, rng))

    def act(self, obs, rng):
        return _argmax_offer(self.scores(obs, rng).mean(axis=0), obs.candidates)


class ThompsonAgent(Agent):
    name = "thompson"

    def act(self, obs, rng):
        eta = obs.belief.sample_log_rates(1, rng)
        return _argmax_offer(offer_values(obs, eta)[0], obs.candidates)


class UCBAgent(BayesAgent):
    name = "ucb"

    def __init__(self, beta: float = 1.0, n_draws: int = 128):
        super().__init__(n_draws); self.beta = beta

    def act(self, obs, rng):
        V = self.scores(obs, rng)
        return _argmax_offer(V.mean(axis=0) + self.beta * V.std(axis=0), obs.candidates)


class KGAgent(BayesAgent):
    """Bayes value plus a one-step knowledge-gradient-style value of
    information. Entering j adds a year of exposure there; the rank-one update
    shrinks Var(eta_k) for every k by var_reduction. We approximate the
    preposterior by contracting each marginal's draws around its mean
    (mean-preserving) and measure the gain in the best next offer."""
    name = "kg"

    def __init__(self, n_draws: int = 128, weight: float = 1.0):
        super().__init__(n_draws); self.w = weight

    def act(self, obs, rng):
        cand = obs.candidates
        if not cand.any():
            return (-1, 0)
        eta = obs.belief.sample_log_rates(self.n, rng)
        V = offer_values(obs, eta)
        Vbar = V.mean(axis=0)
        best_now = np.max(np.where(cand[:, None], Vbar, -np.inf))
        m, v = obs.belief.log_rate_moments()
        voi = np.zeros(obs.pool.K)
        if obs.horizon - obs.t > 1:
            for j in np.flatnonzero(cand):
                dv = obs.belief.var_reduction(j, obs.miles[j] * CONTRACT_QUARTERS)
                shrink = np.sqrt(np.clip(v - dv, 1e-12, None) / v)
                eta_j = m + (eta - m) * shrink
                Vj = offer_values(obs, eta_j).mean(axis=0)
                others = cand.copy(); others[j] = False
                if others.any():
                    voi[j] = max(0.0, np.max(np.where(others[:, None], Vj, -np.inf)) - best_now)
        return _argmax_offer(Vbar + self.w * voi[:, None], cand)


class OracleAgent(Agent):
    """Clairvoyant: values offers at the true rates. Use with oracle env."""
    name = "oracle"

    def act(self, obs, rng):
        eta = obs.oracle_log_lam[None]
        return _argmax_offer(offer_values(obs, eta, np.exp(obs.oracle_log_lam))[0],
                             obs.candidates)


BASELINES = {a.name: a for a in [RandomAgent, RoundRobinAgent, PluginAgent, BayesAgent,
                                 ThompsonAgent, UCBAgent, KGAgent]}
