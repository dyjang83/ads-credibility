"""
Contextual expansion policy trained with PPO.

Each candidate city is a token: its 32-d ODD embedding (the context), posterior
summaries (mean and sd of its log rate, mean rate), entered / candidate flags,
miles, the information value of entering it (total posterior variance it would
remove, CityBelief.var_reduction), and the posterior mean and sd of the offer
value V(k, l) for each loading (agents.offer_values). Global features (quarter,
fraction entered) are appended to every token. A permutation-invariant
transformer encoder maps tokens to per-(city, loading) logits plus a "no offer"
logit and a value estimate; non-candidates are masked.

Training is across randomized worlds (true sigma, true base rate, alpha draw,
cell relativities), so the policy must condition on the context rather than
memorize one world. The training reward is the expected-claims profit
(`exp_profit`, Rao-Blackwellized given the true rates) scaled by 1/10. This is a
simulator-side variance reduction; the agent never observes the true rates.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from .agents import Agent, offer_values
from .config import LOADINGS, SIGMA, BASE_RATE
from .env import ExpansionEnv, run_episode
from .world import sample_world

NL = len(LOADINGS)
N_FEAT = 32 + 3 + 2 + 1 + 1 + 2 * NL + 2
REWARD_SCALE = 0.1


def city_tokens(obs, rng, n_draws: int = 64) -> np.ndarray:
    b = obs.belief
    m, v = b.log_rate_moments()
    sd = np.sqrt(v)
    eta = b.sample_log_rates(n_draws, rng)
    V = offer_values(obs, eta)
    Vm, Vs = V.mean(axis=0), V.std(axis=0)
    info = np.array([b.var_reduction(k, obs.miles[k] * 4).sum() if obs.candidates[k] else 0.0
                     for k in range(obs.pool.K)])
    glob = np.array([obs.t / obs.horizon, obs.entered.mean()])
    X = np.hstack([obs.pool.V, (m - np.log(BASE_RATE))[:, None], sd[:, None],
                   (b.mean_rate() / BASE_RATE - 1.0)[:, None],
                   obs.entered[:, None], obs.candidates[:, None], obs.miles[:, None],
                   info[:, None], Vm / 10.0, Vs / 10.0, np.tile(glob, (obs.pool.K, 1))])
    return X.astype(np.float32)


class ExpansionPolicy(nn.Module):
    def __init__(self, d: int = 64, layers: int = 2, heads: int = 4):
        super().__init__()
        self.inp = nn.Linear(N_FEAT, d)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.offer = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, NL))
        self.none = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, X: torch.Tensor, cand: torch.Tensor):
        """X (B,K,F), cand (B,K) bool -> logits (B, K*NL+1), value (B,)."""
        h = self.enc(self.inp(X))
        pooled = h.mean(1)
        off = self.offer(h).masked_fill(~cand[..., None], -1e9).flatten(1)
        logits = torch.cat([off, self.none(pooled)], 1)
        return logits, self.value(pooled).squeeze(-1)


def decode(a: int, K: int) -> tuple[int, int]:
    return (-1, 0) if a == K * NL else (a // NL, a % NL)


class PPOAgent(Agent):
    name = "ppo"

    def __init__(self, policy: ExpansionPolicy, greedy: bool = True):
        self.policy, self.greedy = policy, greedy

    @torch.no_grad()
    def act(self, obs, rng):
        X = torch.as_tensor(city_tokens(obs, rng))[None]
        cand = torch.as_tensor(obs.candidates)[None]
        logits, _ = self.policy(X, cand)
        if self.greedy:
            a = int(logits.argmax(1))
        else:
            a = int(torch.distributions.Categorical(logits=logits).sample())
        return decode(a, obs.pool.K)


def _random_world(pool, rng, dgp="structured"):
    sigma = float(rng.uniform(0.25, 0.6))
    rate = float(BASE_RATE * np.exp(rng.uniform(-0.3, 0.3)))
    return sample_world(pool, dgp, rng, sigma=sigma, base_rate=rate)


def train_ppo(pool, kernel: str = "learned", iters: int = 150, episodes: int = 32,
              epochs: int = 4, minibatch: int = 128, lr: float = 3e-4, gamma: float = 1.0,
              lam: float = 0.95, clip: float = 0.2, ent_coef: float = 0.01,
              seed: int = 0, dgp: str = "structured", log_every: int = 10):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = ExpansionEnv(pool, kernel)
    policy = ExpansionPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    hist = []
    t0 = time.perf_counter()
    for it in range(iters):
        buf = {k: [] for k in ["X", "cand", "a", "logp", "v", "adv", "ret"]}
        ep_ret = []
        for _ in range(episodes):
            world = _random_world(pool, rng, dgp)
            obs = env.reset(world, np.random.default_rng(rng.integers(1 << 31)))
            Xs, cs, acts, lps, vals, rews = [], [], [], [], [], []
            done = False
            while not done:
                X = torch.as_tensor(city_tokens(obs, rng))
                cand = torch.as_tensor(obs.candidates)
                with torch.no_grad():
                    logits, v = policy(X[None], cand[None])
                dist = torch.distributions.Categorical(logits=logits[0])
                a = int(dist.sample())
                obs, _, done, info = env.step(decode(a, pool.K))
                Xs.append(X); cs.append(cand); acts.append(a)
                lps.append(float(dist.log_prob(torch.tensor(a)))); vals.append(float(v[0]))
                rews.append(info["exp_profit"] * REWARD_SCALE)
            # GAE
            T = len(rews); adv = np.zeros(T); last = 0.0
            for t in reversed(range(T)):
                nv = vals[t + 1] if t + 1 < T else 0.0
                delta = rews[t] + gamma * nv - vals[t]
                last = delta + gamma * lam * last
                adv[t] = last
            buf["X"] += Xs; buf["cand"] += cs; buf["a"] += acts; buf["logp"] += lps
            buf["v"] += vals; buf["adv"] += list(adv); buf["ret"] += list(adv + np.array(vals))
            ep_ret.append(sum(rews) / REWARD_SCALE)
        X = torch.stack(buf["X"]); cand = torch.stack(buf["cand"])
        a = torch.as_tensor(buf["a"]); old_lp = torch.as_tensor(buf["logp"], dtype=torch.float32)
        adv = torch.as_tensor(buf["adv"], dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = torch.as_tensor(buf["ret"], dtype=torch.float32)
        N = len(a)
        for _ in range(epochs):
            perm = torch.randperm(N)
            for s in range(0, N, minibatch):
                idx = perm[s:s + minibatch]
                logits, v = policy(X[idx], cand[idx])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(a[idx])
                ratio = torch.exp(lp - old_lp[idx])
                pg = -torch.min(ratio * adv[idx],
                                torch.clamp(ratio, 1 - clip, 1 + clip) * adv[idx]).mean()
                vl = 0.5 * ((v - ret[idx]) ** 2).mean()
                loss = pg + 0.5 * vl - ent_coef * dist.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
        hist.append({"iter": it, "mean_exp_profit": float(np.mean(ep_ret)),
                     "elapsed_s": time.perf_counter() - t0})
        if log_every and it % log_every == 0:
            print(f"[ppo/{kernel}] iter {it:4d}  mean exp profit {np.mean(ep_ret):8.2f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return policy, hist
