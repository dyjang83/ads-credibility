"""Unit tests for the decision layer (src/decision/)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import power_analysis as pa  # noqa: E402
from models import predict_new_city_with_S  # noqa: E402
from decision.belief import CityBelief  # noqa: E402
from decision.world import city_pool, sample_world, random_region, service_region  # noqa: E402
from decision.geofence import (random_instance, milp_solve, brute_force, greedy_solve,  # noqa: E402
                               greedy_ls_solve, local_search, objective, is_feasible, cvar)
from decision.env import ExpansionEnv, run_episode  # noqa: E402
from decision.agents import BASELINES, OracleAgent  # noqa: E402


@pytest.fixture(scope="module")
def pool():
    return city_pool()


# ---------------------------------------------------------------- belief
def test_belief_matches_laplace_fit_and_conditional(pool):
    """Joint fit over all cities == power_analysis._laplace_fit on the observed
    block + _cond_weights predictive for an unobserved city."""
    S = pool.kernels["learned"][:7, :7]
    obs = [0, 1, 2, 3]; h = 5
    counts = np.array([254., 135., 186., 74.]); expo = np.array([36.4, 46.6, 25.7, 7.3])
    mu, alpha, cov = pa._laplace_fit(counts, expo, S[np.ix_(obs, obs)], pa.SIGMA)
    c_all = np.zeros(7); e_all = np.zeros(7); c_all[obs] = counts; e_all[obs] = expo
    b = CityBelief(S, pa.SIGMA, pa.BASE_RATE).update(c_all, e_all)
    assert np.allclose(b.mode[0], mu, atol=1e-6)
    assert np.allclose(b.mode[1:][obs], alpha, atol=1e-6)
    sub = [0] + [i + 1 for i in obs]
    assert np.allclose(b.cov[np.ix_(sub, sub)], cov, atol=1e-6)
    bw, cw = pa._cond_weights(S, obs, h)
    w = np.concatenate([[1.0], bw])
    m_ref = w @ np.concatenate([[mu], alpha])
    v_ref = w @ cov @ w + pa.SIGMA ** 2 * cw
    m, v = b.log_rate_moments()
    # _cond_weights jitters S while CityBelief jitters sigma^2 S, so the two
    # conditionals differ at O(1e-6 / sigma^2); the fitted block above is exact.
    assert abs(m[h] - m_ref) < 1e-4 and abs(v[h] - v_ref) < 1e-4


def test_belief_conditional_matches_predict_new_city_with_S(pool):
    """With alpha effectively observed (huge exposure), the new-city log-rate
    matches the conditional Gaussian used by models.predict_new_city_with_S."""
    S = pool.kernels["learned"][:7, :7]
    dep = [0, 1, 2, 3]; h = 6
    a = np.array([0.3, -0.2, 0.1, 0.4]); mu = np.log(5.0); sig = 0.4
    E = 1e8
    c_all = np.zeros(7); e_all = np.zeros(7)
    e_all[dep] = E; c_all[dep] = np.exp(mu + a) * E
    b = CityBelief(S, sig, 5.0, mu_sd=1e-4).update(c_all, e_all)
    m, v = b.log_rate_moments()
    n = 200_000
    samples = {"beta0": np.full(n, mu), "beta": np.zeros((n, 1)),
               "alpha": np.tile(a, (n, 1)), "sigma_c": np.full(n, sig)}
    lam = predict_new_city_with_S(samples, S[np.ix_(dep, dep)], S[h, dep], S[h, h],
                                  np.zeros(1), rng_seed=0)
    assert abs(np.log(lam).mean() - m[h]) < 5e-3
    assert abs(np.log(lam).var() - v[h]) < 5e-3


def test_var_reduction_consistent_with_update(pool):
    S = pool.kernels["learned"]
    b = CityBelief(S, 0.41, 5.6)
    _, v0 = b.log_rate_moments()
    dv = b.var_reduction(5, 2.0)
    assert dv[5] > 0 and np.all(dv >= -1e-12) and np.all(dv <= v0 + 1e-12)


# ---------------------------------------------------------------- geofence
def test_cvar_matches_rockafellar_uryasev():
    rng = np.random.default_rng(3)
    L = rng.gamma(2.0, size=37)
    q = 0.9
    ru = min(t + np.maximum(L - t, 0).mean() / (1 - q) for t in L)
    assert abs(cvar(L, q) - ru) < 1e-10


@pytest.mark.parametrize("trial", range(6))
def test_milp_equals_brute_force(trial):
    rng = np.random.default_rng(100 + trial)
    reg = random_region(["Austin", "Boston", "Phoenix"][trial % 3], 12, rng)
    inst = random_instance(reg, rng, n_scen=24)
    xb, Jb = brute_force(inst)
    xm, info = milp_solve(inst, time_limit=60)
    assert is_feasible(inst, xm)
    assert abs(info["obj"] - Jb) < 1e-7
    xg, ig = greedy_ls_solve(inst)
    assert ig["obj"] <= Jb + 1e-9


def test_heuristics_feasible_and_ls_improves():
    rng = np.random.default_rng(7)
    inst = random_instance(service_region("San Francisco"), rng)
    xg, ig = greedy_solve(inst)
    xl, il = local_search(inst, xg)
    assert is_feasible(inst, xg) and is_feasible(inst, xl)
    assert il["obj"] >= ig["obj"] - 1e-12
    assert abs(objective(inst, xl) - il["obj"]) < 1e-9


# ---------------------------------------------------------------- env / agents
def test_env_deterministic_and_oracle_regret_zero(pool):
    world = sample_world(pool, "structured", np.random.default_rng(1))
    env = ExpansionEnv(pool, "learned", horizon=6)
    r1 = run_episode(env, BASELINES["bayes"](), world, seed=11)
    r2 = run_episode(env, BASELINES["bayes"](), world, seed=11)
    assert r1["exp_profit"] == r2["exp_profit"] and r1["order"] == r2["order"]
    oenv = ExpansionEnv(pool, oracle=True, horizon=6)
    o1 = run_episode(oenv, OracleAgent(), world, seed=11)
    o2 = run_episode(oenv, OracleAgent(), world, seed=11)
    assert o1["exp_profit"] - o2["exp_profit"] == 0.0


def test_null_world_ignores_kernel(pool):
    """Null DGP draws alpha independently of the embedding geometry."""
    rng = np.random.default_rng(0)
    A = np.array([sample_world(pool, "identity_null", rng).alpha for _ in range(4000)])
    C = np.corrcoef(A.T)
    off = C[~np.eye(pool.K, dtype=bool)]
    assert np.abs(off).max() < 0.08
