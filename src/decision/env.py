"""
Expansion environment: the outer, sequential decision problem.

An insurer underwrites an ADS operator's fleet. It starts with the four real
deployed cities already written; their initial exposure equals the real SGO
window exposure per city and their initial claim counts are simulated from the
world's ground truth. Each quarter t = 1..T the insurer

  1. re-rates every active city at its posterior mean times (1 + RERATE_LOADING),
     except cities still inside their 4-quarter entry contract;
  2. may offer entry terms for ONE candidate city j at loading l. The operator
     accepts with probability sigmoid(ACCEPT_A + ACCEPT_B * (log lam_true_j -
     log rate_offered)): it knows its own safety record, so underpriced offers
     are accepted more often (adverse selection);
  3. holds capital equal to VaR_99.5 - mean of next-quarter aggregate claims
     under its posterior predictive, charged at COST_OF_CAPITAL;
  4. observes Poisson claims in every active city and updates its belief.

Reward = premium - claims - capital charge. A capital SHORTFALL is a quarter in
which claims exceed premium + capital.

Each city is served on a geofence over its template region (inner CO problem,
geofence.py), solved once per template at the default instance parameters. The
geofence fixes the city's served exposure share (miles per quarter =
MILES_PER_QUARTER * share) and its true rate (exposure-weighted mean of the TRUE
cell relativities inside the geofence). The insurer's model is city-level, as
in the paper, so the geofence enters its belief only through alpha_k.

The belief kernel (learned / euclidean / independent) is a property of the
environment's insurer model, so every agent can be run under every kernel; the
data-generating kernel is a property of the World (structured / null DGP).
With `oracle=True` the insurer prices and holds capital on the TRUE rates
(clairvoyant benchmark for regret).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from .belief import CityBelief
from .config import (DATA, SIGMA, BASE_RATE, HORIZON, MILES_PER_QUARTER, LOADINGS,
                     RERATE_LOADING, COST_OF_CAPITAL, CAPITAL_Q, ACCEPT_A, ACCEPT_B)
from .geofence import make_instance, greedy_ls_solve, milp_solve
from .world import CityPool, World, service_region, DEPLOYED_CITIES

CONTRACT_QUARTERS = 4


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def accept_prob(log_lam_true, rate_offered):
    return sigmoid(ACCEPT_A + ACCEPT_B * (log_lam_true - np.log(rate_offered)))


@lru_cache(maxsize=None)
def default_geofence(template: str, solver: str = "greedy_ls", seed: int = 0) -> tuple:
    """Geofence mask for a template region at the default instance parameters
    (price 1.05, kappa 1, q 0.9, min_share 0.4). Cached: the insurer's
    cell-level belief is the prior, which does not depend on the world."""
    reg = service_region(template)
    inst = make_instance(reg, np.random.default_rng(seed))
    if solver == "milp":
        x, _ = milp_solve(inst, time_limit=120)
    else:
        x, _ = greedy_ls_solve(inst)
    if solver == "full":
        x = np.ones(reg.n, bool)
    return tuple(bool(v) for v in x)


@lru_cache(maxsize=1)
def initial_exposure() -> dict:
    ads = pd.read_csv(DATA / "ads_events.csv")
    return ads.groupby("city")["exposure_million_miles"].sum().to_dict()


@dataclass
class Obs:
    t: int
    horizon: int
    belief: CityBelief
    entered: np.ndarray
    candidates: np.ndarray
    miles: np.ndarray            # per-quarter miles if active (all cities)
    pool: CityPool
    oracle_log_lam: np.ndarray | None = None


class ExpansionEnv:
    def __init__(self, pool: CityPool, kernel: str = "learned", horizon: int = HORIZON,
                 geofence_solver: str = "greedy_ls", oracle: bool = False,
                 capital_draws: int = 256, belief_sigma: float = SIGMA,
                 belief_rate: float = BASE_RATE):
        self.pool, self.kernel, self.horizon = pool, kernel, horizon
        self.oracle = oracle
        self.capital_draws = capital_draws
        self.belief_sigma, self.belief_rate = belief_sigma, belief_rate
        self.masks = [np.array(default_geofence(t, geofence_solver)) for t in pool.template]
        self.share = np.array([service_region(t).m[mk].sum()
                               for t, mk in zip(pool.template, self.masks)])
        self.miles = MILES_PER_QUARTER * self.share

    # ------------------------------------------------------------------
    def reset(self, world: World, rng: np.random.Generator) -> Obs:
        self.world, self.rng = world, rng
        K = self.pool.K
        self.lam = np.array([world.city_rate(k, self.masks[k]) for k in range(K)])
        self.entered = np.zeros(K, bool)
        self.counts = np.zeros(K); self.expo = np.zeros(K)
        e0 = initial_exposure()
        for c in DEPLOYED_CITIES:
            k = self.pool.names.index(c)
            self.entered[k] = True
            self.expo[k] = e0[c]
            self.counts[k] = rng.poisson(self.lam[k] * e0[c])
        self.lock_rate = np.full(K, np.nan); self.lock_until = np.zeros(K, int)
        self.belief = CityBelief(self.pool.kernels[self.kernel], self.belief_sigma,
                                 self.belief_rate).update(self.counts, self.expo)
        self.t = 0
        return self._obs()

    def _obs(self) -> Obs:
        return Obs(self.t, self.horizon, self.belief, self.entered.copy(),
                   ~self.entered, self.miles, self.pool,
                   np.log(self.lam) if self.oracle else None)

    def _model_rates(self) -> np.ndarray:
        return self.lam.copy() if self.oracle else self.belief.mean_rate()

    def _capital(self, active: np.ndarray) -> float:
        mi = self.miles * active
        if self.oracle:
            mean = float(self.lam @ mi)
            tot = self.rng.poisson(mean, self.capital_draws)
        else:
            eta = self.belief.sample_log_rates(self.capital_draws, self.rng)
            tot = self.rng.poisson(np.exp(eta) @ mi)
        return float(np.quantile(tot, CAPITAL_Q) - tot.mean())

    # ------------------------------------------------------------------
    def step(self, action: tuple[int, int]):
        j, l = action
        model = self._model_rates()
        rate = model * (1.0 + RERATE_LOADING)
        locked = self.lock_until > self.t
        rate[locked] = self.lock_rate[locked]
        accepted = False
        if j >= 0 and not self.entered[j]:
            offered = model[j] * (1.0 + LOADINGS[l])
            accepted = bool(self.rng.random() < accept_prob(np.log(self.lam[j]), offered))
            if accepted:
                self.entered[j] = True
                self.lock_rate[j] = offered; self.lock_until[j] = self.t + CONTRACT_QUARTERS
                rate[j] = offered
        active = self.entered
        capital = self._capital(active)
        mi = self.miles * active
        claims = self.rng.poisson(self.lam * mi)
        premium = float(rate[active] @ mi[active])
        loss = float(claims.sum())
        charge = COST_OF_CAPITAL * capital
        reward = premium - loss - charge
        self.counts += claims; self.expo += mi
        if not self.oracle:
            self.belief.update(self.counts, self.expo)
        self.t += 1
        exp_loss = float(self.lam @ mi)
        info = {"premium": premium, "claims": loss, "capital": capital, "charge": charge,
                "exp_profit": premium - exp_loss - charge,
                "shortfall": loss > premium + capital, "accepted": accepted,
                "offered_city": j, "loading": LOADINGS[l] if j >= 0 else np.nan}
        return self._obs(), reward, self.t >= self.horizon, info


def run_episode(env: ExpansionEnv, agent, world: World, seed: int) -> dict:
    """Roll out one episode with common random numbers per (world, seed)."""
    env_rng = np.random.default_rng(seed)
    agent_rng = np.random.default_rng(seed + 7919)
    obs = env.reset(world, env_rng)
    tot, exp_tot, shortfalls, order = 0.0, 0.0, 0, []
    done = False
    while not done:
        a = agent.act(obs, agent_rng)
        obs, r, done, info = env.step(a)
        tot += r; exp_tot += info["exp_profit"]; shortfalls += info["shortfall"]
        if info["accepted"]:
            order.append(a[0])
    m, v = env.belief.log_rate_moments()
    cand = ~env.entered
    # `exp_profit` replaces realized claims by their expectation under the true
    # rates (Rao-Blackwellized; the belief still learns from realized claims).
    return {"profit": tot, "exp_profit": exp_tot, "shortfalls": shortfalls, "n_entered": len(order),
            "order": order,
            "final_sd_candidates": float(np.sqrt(v[cand]).mean()) if cand.any() else np.nan,
            "final_abs_err_all": float(np.abs(m - np.log(env.lam)).mean()),
            "counts": env.counts.copy(), "expo": env.expo.copy()}
