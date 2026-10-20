"""
Learning-guided combinatorial optimization for the geofence problem.

A graph neural network over the H3 adjacency graph builds a geofence one cell at
a time (attention-model style: encode the instance once, decode step by step).

  encoder   3 residual message-passing layers (mean over H3 neighbors) on
            static node features: 32-d cell embedding, exposure share, per-unit
            mean margin, relativity sd, tail-loss loading, z(HDV freq), and the
            instance globals (price, kappa, min_share, size)
  decoder   per step, scores every frontier cell (any cell at step 0) and a
            STOP action (allowed once the service-area constraint holds) from
            the node embedding, its selected/frontier flags and a context vector
            (pooled selected embedding, share / min_share, fraction selected)

The construction mask enforces contiguity, so every output is feasible.
Training: REINFORCE with a POMO-style shared baseline (B rollouts per instance,
advantage = J - mean J over the B rollouts, standardized), on random connected
sub-regions of the TRAINING cities. Evaluation is on a held-out city, mirroring
the leave-one-city-out design of the main paper.

Inference modes
  gnn_greedy      argmax decode
  gnn_sample      best of N sampled rollouts, then local_search polish
  gnn_milp        "learning-guided MILP": the exact MILP restricted to the
                  graph neighborhood (radius 2) of the sampled solutions
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from .geofence import GeofenceInstance, objective, local_search, milp_solve

N_STATIC = 32 + 5 + 4


def node_features(inst: GeofenceInstance) -> np.ndarray:
    reg, sc = inst.region, inst.scen
    rbar, rsd = sc.mean(axis=0), sc.std(axis=0)
    L_full = sc @ reg.m
    S = sc.shape[0]
    tail = np.argsort(-L_full)[: max(1, int(round((1 - inst.q) * S)))]
    tail_load = sc[tail].mean(axis=0) - rbar
    n = reg.n
    glob = np.array([inst.price, inst.kappa, inst.min_share, n / 200.0])
    X = np.hstack([reg.emb, (reg.m * n)[:, None], (inst.price - rbar)[:, None],
                   rsd[:, None], tail_load[:, None], reg.z[:, None],
                   np.tile(glob, (n, 1))])
    return X.astype(np.float32)


def adjacency(inst: GeofenceInstance) -> torch.Tensor:
    """Row-normalized dense adjacency (mean aggregation); regions are small."""
    E = inst.region.edges
    A = np.zeros((inst.n, inst.n), np.float32)
    A[E[:, 0], E[:, 1]] = 1.0
    A[E[:, 1], E[:, 0]] = 1.0
    return torch.as_tensor(A / np.maximum(A.sum(1, keepdims=True), 1.0))


class GeofencePolicy(nn.Module):
    def __init__(self, d: int = 64, layers: int = 3):
        super().__init__()
        self.inp = nn.Linear(N_STATIC, d)
        self.self_w = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.nbr_w = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.ctx = nn.Sequential(nn.Linear(d + 2, d), nn.ReLU())
        self.node_head = nn.Sequential(nn.Linear(2 * d + 2, d), nn.ReLU(), nn.Linear(d, 1))
        self.stop_head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 1))

    def encode(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        h = self.inp(X)
        for ws, wn, ln in zip(self.self_w, self.nbr_w, self.norms):
            h = ln(h + torch.relu(ws(h) + wn(A @ h)))
        return h

    def step_logits(self, h, sel, front, share_ratio, frac):
        """h (n,d); sel, front (B,n) float; share_ratio, frac (B,).
        Returns node logits (B,n) and stop logit (B,)."""
        B, n = sel.shape
        cnt = sel.sum(1, keepdim=True).clamp(min=1.0)
        pooled = (sel @ h) / cnt                                     # (B,d)
        ctx = self.ctx(torch.cat([pooled, share_ratio[:, None], frac[:, None]], 1))
        hn = h.unsqueeze(0).expand(B, n, -1)
        z = torch.cat([hn, ctx.unsqueeze(1).expand(B, n, -1),
                       sel.unsqueeze(-1), front.unsqueeze(-1)], -1)
        node = self.node_head(z).squeeze(-1)
        stop = self.stop_head(torch.cat([ctx, h.mean(0).expand(B, -1)], 1)).squeeze(-1)
        return node, stop


def rollout(policy: GeofencePolicy, inst: GeofenceInstance, B: int,
            greedy: bool = False, gen: torch.Generator | None = None):
    """Construct B geofences. Returns (x (B,n) bool numpy, sum log-prob (B,))."""
    X = torch.as_tensor(node_features(inst))
    A = adjacency(inst)
    h = policy.encode(X, A)
    n = inst.n
    m = torch.as_tensor(inst.m, dtype=torch.float32)
    nbr = A > 0
    sel = torch.zeros(B, n); front = torch.zeros(B, n)
    done = torch.zeros(B, dtype=torch.bool)
    logp = torch.zeros(B)
    for t in range(n + 1):
        share = sel @ m
        feas = share >= inst.min_share - 1e-9
        node_logit, stop_logit = policy.step_logits(
            h, sel, front, share / inst.min_share, sel.sum(1) / n)
        allowed = (front > 0) if t > 0 else torch.ones(B, n, dtype=torch.bool)
        allowed = allowed & (sel == 0)
        stop_ok = feas | ~allowed.any(1)
        logits = torch.cat([node_logit.masked_fill(~allowed, -1e9),
                            stop_logit.masked_fill(~stop_ok, -1e9)[:, None]], 1)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            a = logits.argmax(1)
        else:
            u = torch.rand(B, n + 1, generator=gen).clamp(1e-9, 1 - 1e-9)
            a = (logits - torch.log(-torch.log(u))).argmax(1)       # Gumbel-max
        lp = dist.log_prob(a)
        logp = logp + torch.where(done, torch.zeros_like(lp), lp)
        stop = (a == n) | done
        pick = (~stop).nonzero().squeeze(-1)
        if len(pick):
            cells = a[pick]
            sel = sel.clone(); front = front.clone()          # keep autograd history intact
            sel[pick, cells] = 1.0
            front[pick] = ((front[pick] > 0) | nbr[cells]).float() * (1 - sel[pick])
        done = stop
        if bool(done.all()):
            break
    return sel.numpy() > 0.5, logp


def train(policy: GeofencePolicy, instances: list, epochs: int = 3, B: int = 16,
          lr: float = 3e-4, seed: int = 0, log_every: int = 50, ref_obj=None):
    """REINFORCE with shared (POMO-style) baseline. Returns training history."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    hist = []
    step = 0
    for ep in range(epochs):
        for i in rng.permutation(len(instances)):
            inst = instances[i]
            x, logp = rollout(policy, inst, B, gen=gen)
            J = torch.as_tensor([objective(inst, xi) for xi in x], dtype=torch.float32)
            adv = (J - J.mean()) / (J.std() + 1e-8)
            loss = -(adv * logp).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            rec = {"step": step, "epoch": ep, "J_mean": float(J.mean()), "J_max": float(J.max())}
            if ref_obj is not None:
                rec["gap_vs_ref"] = float(ref_obj[i] - J.max())
            hist.append(rec)
            if log_every and step % log_every == 0:
                tail = hist[-log_every:]
                msg = f"step {step:5d} ep {ep} J_mean {np.mean([r['J_mean'] for r in tail]):+.4f}"
                if ref_obj is not None:
                    msg += f"  gap(best-of-B vs greedy+LS) {np.mean([r['gap_vs_ref'] for r in tail]):+.4f}"
                print(msg, flush=True)
            step += 1
    return hist


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
@torch.no_grad()
def gnn_greedy(policy, inst):
    t0 = time.perf_counter()
    x, _ = rollout(policy, inst, 1, greedy=True)
    return x[0], {"time": time.perf_counter() - t0, "obj": objective(inst, x[0])}


@torch.no_grad()
def gnn_sample(policy, inst, n_samples: int = 64, polish: bool = True, seed: int = 0):
    t0 = time.perf_counter()
    gen = torch.Generator().manual_seed(seed)
    xs, _ = rollout(policy, inst, n_samples, gen=gen)
    xg, _ = rollout(policy, inst, 1, greedy=True)
    xs = np.vstack([xs, xg])
    J = np.array([objective(inst, x) for x in xs])
    x = xs[int(np.argmax(J))]
    if polish:
        x, _ = local_search(inst, x)
    return x, {"time": time.perf_counter() - t0, "obj": objective(inst, x), "pool": xs, "pool_J": J}


def neighborhood(inst: GeofenceInstance, xs: np.ndarray, radius: int = 2) -> np.ndarray:
    mask = np.asarray(xs, bool).any(axis=0) if np.ndim(xs) == 2 else np.asarray(xs, bool)
    mask = mask.copy()
    for _ in range(radius):
        grow = mask.copy()
        for i in np.flatnonzero(mask):
            grow[inst.region.nbrs[i]] = True
        mask = grow
    return mask


def gnn_milp(policy, inst, n_samples: int = 64, top: int = 8, radius: int = 2,
             time_limit: float = 20.0, seed: int = 0):
    """MILP restricted to the radius-`radius` neighborhood of the `top` best
    sampled GNN geofences. Returns the better of the restricted MILP and the
    polished GNN solution."""
    t0 = time.perf_counter()
    x_s, info_s = gnn_sample(policy, inst, n_samples, polish=True, seed=seed)
    order = np.argsort(-info_s["pool_J"])[:top]
    allowed = neighborhood(inst, np.vstack([info_s["pool"][order], x_s[None]]), radius)
    x_m, info_m = milp_solve(inst, time_limit=time_limit, allowed=allowed)
    x = x_m if x_m is not None and info_m["obj"] > info_s["obj"] else x_s
    return x, {"time": time.perf_counter() - t0, "obj": objective(inst, x),
               "n_allowed": int(allowed.sum())}
