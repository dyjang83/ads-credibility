"""
Experiments 2-4: sequential expansion (contextual RL vs posterior baselines).

  2. Agents x belief kernels x DGPs on worlds calibrated to the fitted
     posterior (SIGMA, BASE_RATE). Metric: regret of expected-claims profit vs
     the clairvoyant oracle on the same world and seed, P(any capital
     shortfall), cities entered, final pricing error.
  3. Q1 (decision value of the learned kernel): paired per-world regret
     differences learned - independent and learned - euclidean, per agent and
     DGP. Under the null DGP the learned kernel must not help (specificity).
  4. Q2 (adaptive design): with a long horizon, record the leave-one-city-out
     comparison of the learned vs Euclidean / independent kernels on the
     entered cities each time an agent's entered count first reaches k. The
     round-robin agent is the fixed nested design of power_analysis_ncities.py.

PPO policies are trained first (one per belief kernel in --ppo-kernels) on
RANDOMIZED structured worlds and evaluated on the calibrated worlds.

Outputs (results/decision/):
  expansion_episodes.csv, expansion_summary.csv, kernel_value.csv,
  adaptive_design.csv, adaptive_design_summary.csv, ppo_<kernel>.pt,
  ppo_<kernel>_history.csv, expansion_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision.config import RESULTS, SIGMA, BASE_RATE  # noqa: E402
from decision.world import city_pool, sample_world  # noqa: E402
from decision.belief import CityBelief  # noqa: E402
from decision.env import ExpansionEnv, run_episode  # noqa: E402
from decision.agents import BASELINES, OracleAgent, RoundRobinAgent, BayesAgent, KGAgent, ThompsonAgent  # noqa: E402
from decision.ppo import ExpansionPolicy, PPOAgent, train_ppo  # noqa: E402
from power_analysis import _held_logp  # noqa: E402

KERNELS = ["learned", "learned_nugget", "euclidean", "independent"]
Q2_KERNELS = ["learned", "learned_nugget", "euclidean", "euclidean_nugget", "independent"]
Q2_PAIRS = [("learned", "independent"), ("learned", "euclidean"),
            ("learned_nugget", "independent"), ("learned_nugget", "euclidean_nugget")]
MODEL_FREE = {"random", "round_robin"}


def ci(x):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    return x.mean(), x.mean() - 1.96 * se, x.mean() + 1.96 * se


def loco_logp(S, counts, expo, entered, sigma=SIGMA, rate=BASE_RATE):
    """LOCO predictive log-likelihood over the entered cities under kernel S.
    The held-out city's predictive log-rate moments come from CityBelief and
    are scored with the quadrature of power_analysis._held_logp (passed as a
    one-parameter 'pooled' predictive with mean m_h and variance v_h)."""
    idx = np.flatnonzero(entered)
    S_e = S[np.ix_(idx, idx)]
    c0, e0 = counts[idx], expo[idx]
    tot = 0.0
    for h in range(len(idx)):
        c, e = c0.copy(), e0.copy(); c[h] = 0; e[h] = 0
        b = CityBelief(S_e, sigma, rate).update(c, e)
        m, v = b.log_rate_moments()
        tot += _held_logp(c0[h], e0[h], m[h], None, np.array([[v[h]]]), None, None,
                          sigma, pooled=True)
    return tot


def train_or_load_ppo(pool, kernel, args):
    ckpt = RESULTS / f"ppo_{kernel}{args.tag}.pt"
    pol = ExpansionPolicy()
    if ckpt.exists() and not args.retrain:
        pol.load_state_dict(torch.load(ckpt)); print(f"loaded {ckpt}")
    else:
        pol, hist = train_ppo(pool, kernel, iters=args.ppo_iters, seed=args.seed)
        torch.save(pol.state_dict(), ckpt)
        pd.DataFrame(hist).to_csv(RESULTS / f"ppo_{kernel}_history{args.tag}.csv", index=False)
    pol.eval()
    return pol


def experiment_main(pool, ppo, args):
    envs = {k: ExpansionEnv(pool, k) for k in KERNELS}
    oenv = ExpansionEnv(pool, oracle=True)
    rows = []
    t0 = time.perf_counter()
    for dgp in ["structured", "identity_null"]:
        for seed in range(args.seeds):
            for w in range(args.n_worlds):
                wid = seed * 100_000 + w
                world = sample_world(pool, dgp, np.random.default_rng(10_000 + wid))
                o = run_episode(oenv, OracleAgent(), world, seed=wid)
                for kern, env in envs.items():
                    agents = {n: A() for n, A in BASELINES.items()
                              if kern == "learned" or n not in MODEL_FREE}
                    if kern in ppo:
                        agents["ppo"] = PPOAgent(ppo[kern])
                    for name, agent in agents.items():
                        r = run_episode(env, agent, world, seed=wid)
                        rows.append({"dgp": dgp, "seed": seed, "world": w, "kernel": kern,
                                     "agent": name, "exp_profit": r["exp_profit"],
                                     "regret": o["exp_profit"] - r["exp_profit"],
                                     "any_shortfall": r["shortfalls"] > 0,
                                     "n_entered": r["n_entered"],
                                     "final_sd_candidates": r["final_sd_candidates"],
                                     "final_abs_err": r["final_abs_err_all"],
                                     "oracle_exp_profit": o["exp_profit"]})
            print(f"[{dgp}] seed {seed} done ({time.perf_counter() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    g = df.groupby(["dgp", "agent", "kernel"])
    summ = g.agg(regret=("regret", "mean"),
                 regret_lo=("regret", lambda x: ci(x)[1]), regret_hi=("regret", lambda x: ci(x)[2]),
                 p_shortfall=("any_shortfall", "mean"), n_entered=("n_entered", "mean"),
                 final_abs_err=("final_abs_err", "mean"), n=("regret", "size")).reset_index()
    p = df.pivot_table(index=["dgp", "seed", "world", "agent"], columns="kernel", values="regret")
    kv = []
    for (dgp, agent), sub in p.groupby(level=["dgp", "agent"]):
        for ref, other in [("learned", "independent"), ("learned", "euclidean"),
                           ("learned_nugget", "independent"), ("learned_nugget", "learned")]:
            if any(k not in sub or sub[k].isna().all() for k in (ref, other)):
                continue
            d = (sub[ref] - sub[other]).dropna()
            m, lo, hi = ci(d)
            kv.append({"dgp": dgp, "agent": agent, "comparison": f"{ref}_minus_{other}",
                       "mean_regret_diff": m, "lo": lo, "hi": hi, "n": len(d)})
    return summ, pd.DataFrame(kv)


def experiment_adaptive(pool, args):
    env = ExpansionEnv(pool, "learned", horizon=args.q2_horizon)
    agents = {"round_robin": RoundRobinAgent(), "bayes": BayesAgent(),
              "thompson": ThompsonAgent(), "kg": KGAgent()}
    ks = list(range(6, 4 + args.q2_max_new + 1, 2))
    rows = []
    t0 = time.perf_counter()
    for w in range(args.q2_worlds):
        world = sample_world(pool, "structured", np.random.default_rng(50_000 + w))
        for name, agent in agents.items():
            env_rng = np.random.default_rng(w); agent_rng = np.random.default_rng(w + 7919)
            obs = env.reset(world, env_rng)
            pending = list(ks)
            done = False
            while not done and pending:
                obs, _, done, _ = env.step(agent.act(obs, agent_rng))
                k_now = int(env.entered.sum())
                while pending and k_now >= pending[0]:
                    k = pending.pop(0)
                    lp = {kn: loco_logp(pool.kernels[kn], env.counts, env.expo, env.entered)
                          for kn in Q2_KERNELS}
                    rows.append({"world": w, "agent": name, "k": k, "quarter": env.t,
                                 **{f"lp_{kn}": v for kn, v in lp.items()}})
        if w % 25 == 0:
            print(f"[Q2] world {w} ({time.perf_counter() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    aggs = {"median_quarter_reached": ("quarter", "median"), "n_worlds_reached": ("world", "size")}
    for a, b in Q2_PAIRS:
        df[f"adv_{a}_vs_{b}"] = df[f"lp_{a}"] - df[f"lp_{b}"]
        df[f"win_{a}_vs_{b}"] = df[f"adv_{a}_vs_{b}"] > 0
        aggs[f"p_win_{a}_vs_{b}"] = (f"win_{a}_vs_{b}", "mean")
        aggs[f"mean_adv_{a}_vs_{b}"] = (f"adv_{a}_vs_{b}", "mean")
    summ = df.groupby(["agent", "k"]).agg(**aggs).reset_index()
    summ["frac_worlds_reached"] = summ["n_worlds_reached"] / args.q2_worlds
    return df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-worlds", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ppo-kernels", nargs="*", default=["learned", "independent"])
    ap.add_argument("--ppo-iters", type=int, default=300)
    ap.add_argument("--q2-worlds", type=int, default=150)
    ap.add_argument("--q2-horizon", type=int, default=40)
    ap.add_argument("--q2-max-new", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-q2", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.n_worlds, args.seeds, args.ppo_iters = 6, 1, 3
        args.q2_worlds, args.q2_horizon, args.q2_max_new = 3, 12, 4
    args.tag = "_quick" if args.quick else ""
    torch.set_num_threads(4)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pool = city_pool()

    ppo = {k: train_or_load_ppo(pool, k, args) for k in args.ppo_kernels}
    df = experiment_main(pool, ppo, args)
    df.to_csv(RESULTS / f"expansion_episodes{args.tag}.csv", index=False)
    summ, kv = summarize(df)
    summ.to_csv(RESULTS / f"expansion_summary{args.tag}.csv", index=False)
    kv.to_csv(RESULTS / f"kernel_value{args.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string()); print(kv.round(3).to_string())
    out = {"sigma": SIGMA, "base_rate": BASE_RATE, "n_worlds": args.n_worlds, "seeds": args.seeds,
           "summary": summ.to_dict(orient="records"), "kernel_value": kv.to_dict(orient="records")}
    if not args.skip_q2:
        q2, q2s = experiment_adaptive(pool, args)
        q2.to_csv(RESULTS / f"adaptive_design{args.tag}.csv", index=False)
        q2s.to_csv(RESULTS / f"adaptive_design_summary{args.tag}.csv", index=False)
        print(q2s.round(3).to_string())
        out["adaptive_design"] = q2s.to_dict(orient="records")
    (RESULTS / f"expansion_summary{args.tag}.json").write_text(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
