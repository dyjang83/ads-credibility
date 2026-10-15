"""
Geofence (ODD) design as a combinatorial optimization problem.

Given a city's candidate service region (H3 cells i = 1..n with exposure shares
m_i, see world.py) choose a CONNECTED set G of cells to serve. Mileage inside G
is allocated in proportion to m_i and the territory is written at a single
premium relativity `price` (the city rate times a loading, in units of the
city rate). Under relativity scenario s (a posterior draw of cell relativities
r_i^s) the expected loss of the geofence is  L^s(x) = sum_i x_i m_i r_i^s.
The objective is mean margin minus a risk charge,

    J(x) = price * sum_i m_i x_i - E_s[L^s] - kappa * (CVaR_q(L) - E_s[L^s]),

subject to   sum_i m_i x_i >= min_share   (a viable service area)
             G connected on the H3 adjacency graph.

This is a prize-collecting connected-subgraph problem with a CVaR term
(NP-hard). Solvers:
  milp_solve     exact MILP: Rockafellar-Uryasev CVaR linearization plus
                 single-commodity-flow contiguity, via scipy.optimize.milp (HiGHS)
  greedy_solve   multi-start contiguous growth
  local_search   add / remove (non-articulation) improvement from any start
All solvers score with the same `objective` on the same scenarios, so gaps are
directly comparable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import coo_matrix

from .world import CellRegion


@dataclass
class GeofenceInstance:
    region: CellRegion
    scen: np.ndarray       # (S, n) relativity scenarios r_i^s
    price: float
    kappa: float
    q: float
    min_share: float

    @property
    def n(self) -> int:
        return self.region.n

    @property
    def m(self) -> np.ndarray:
        return self.region.m


def make_instance(region: CellRegion, rng: np.random.Generator, n_scen: int = 64,
                  price: float = 1.05, kappa: float = 1.0, q: float = 0.9,
                  min_share: float = 0.4) -> GeofenceInstance:
    scen = np.exp(region.sample_log_r(n_scen, rng))
    return GeofenceInstance(region, scen, price, kappa, q, min_share)


def random_instance(region: CellRegion, rng: np.random.Generator,
                    n_scen: int = 64) -> GeofenceInstance:
    """Instance distribution used for training and the CO benchmark."""
    return make_instance(region, rng, n_scen,
                         price=float(rng.uniform(0.95, 1.15)),
                         kappa=float(rng.choice([0.0, 0.5, 1.0, 2.0])),
                         q=0.9, min_share=float(rng.uniform(0.25, 0.6)))


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------
def cvar(L: np.ndarray, q: float) -> np.ndarray:
    """CVaR_q of equally weighted scenarios along axis 0 (exact, fractional
    tail atom). Works for L of shape (S,) or (S, k)."""
    S = L.shape[0]
    k = (1.0 - q) * S
    kf = int(np.floor(k))
    Ls = -np.sort(-L, axis=0)
    top = Ls[:kf].sum(axis=0) if kf > 0 else 0.0
    frac = (k - kf) * Ls[kf] if kf < S else 0.0
    return (top + frac) / k


def _J_from_L(inst: GeofenceInstance, share, L) -> np.ndarray:
    EL = L.mean(axis=0)
    return inst.price * share - EL - inst.kappa * (cvar(L, inst.q) - EL)


def objective(inst: GeofenceInstance, x: np.ndarray) -> float:
    xm = np.asarray(x, float) * inst.m
    return float(_J_from_L(inst, xm.sum(), inst.scen @ xm))


def is_connected(region: CellRegion, x: np.ndarray) -> bool:
    sel = np.flatnonzero(x)
    if len(sel) == 0:
        return False
    seen, stack = {sel[0]}, [sel[0]]
    while stack:
        u = stack.pop()
        for v in region.nbrs[u]:
            if x[v] and v not in seen:
                seen.add(v); stack.append(v)
    return len(seen) == len(sel)


def is_feasible(inst: GeofenceInstance, x: np.ndarray) -> bool:
    return (inst.m[np.asarray(x, bool)].sum() >= inst.min_share - 1e-9
            and is_connected(inst.region, np.asarray(x, bool)))


# --------------------------------------------------------------------------
# Exact MILP
# --------------------------------------------------------------------------
def milp_solve(inst: GeofenceInstance, time_limit: float = 60.0,
               allowed: np.ndarray | None = None, mip_rel_gap: float = 1e-6):
    """Returns (x, info). `allowed` (bool mask) fixes the other cells to 0,
    which is how the GNN-restricted MILP is run."""
    t0 = time.perf_counter()
    n, S = inst.n, inst.scen.shape[0]
    m, kap = inst.m, inst.kappa
    E = inst.region.edges
    arcs = np.vstack([E, E[:, ::-1]])                  # directed arcs (tail, head)
    A = len(arcs)
    # variable blocks
    ix = np.arange(n); iy = n + ix; ig = 2 * n + ix
    iF = 3 * n + np.arange(A); it = 3 * n + A; iu = it + 1 + np.arange(S)
    nv = it + 1 + S
    rbar = inst.scen.mean(axis=0)
    c = np.zeros(nv)
    c[ix] = -m * (inst.price - (1.0 - kap) * rbar)
    c[it] = kap
    c[iu] = kap / ((1.0 - inst.q) * S)

    rows, cols, vals, lb, ub = [], [], [], [], []
    r = 0

    def add(cs, vs, lo, hi):
        nonlocal r
        rows.extend([r] * len(cs)); cols.extend(cs); vals.extend(vs)
        lb.append(lo); ub.append(hi); r += 1

    for s in range(S):                                  # CVaR epigraph
        add(list(ix) + [it, iu[s]], list(m * inst.scen[s]) + [-1.0, -1.0], -np.inf, 0.0)
    add(list(ix), list(m), inst.min_share, np.inf)      # viable service area
    add(list(iy), [1.0] * n, 1.0, 1.0)                  # exactly one root
    for i in range(n):
        add([iy[i], ix[i]], [1.0, -1.0], -np.inf, 0.0)  # root is selected
        add([ig[i], iy[i]], [1.0, -float(n)], -np.inf, 0.0)
    inflow = [[] for _ in range(n)]; outflow = [[] for _ in range(n)]
    for a, (u, v) in enumerate(arcs):
        outflow[u].append(iF[a]); inflow[v].append(iF[a])
        add([iF[a], ix[u]], [1.0, -(n - 1.0)], -np.inf, 0.0)
        add([iF[a], ix[v]], [1.0, -(n - 1.0)], -np.inf, 0.0)
    for i in range(n):                                  # flow conservation
        add(inflow[i] + outflow[i] + [ig[i], ix[i]],
            [1.0] * len(inflow[i]) + [-1.0] * len(outflow[i]) + [1.0, -1.0], 0.0, 0.0)
    Acon = coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()

    lo = np.zeros(nv); hi = np.full(nv, np.inf)
    hi[ix] = 1; hi[iy] = 1; hi[ig] = n; hi[iF] = n - 1
    hi[it] = float((inst.scen * m).sum(axis=1).max())
    if allowed is not None:
        hi[ix[~allowed]] = 0; hi[iy[~allowed]] = 0
    integ = np.zeros(nv); integ[ix] = 1; integ[iy] = 1
    from scipy.optimize import Bounds
    res = milp(c, integrality=integ, bounds=Bounds(lo, hi),
               constraints=LinearConstraint(Acon, lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap,
                        "disp": False})
    elapsed = time.perf_counter() - t0
    if res.x is None:
        return None, {"time": elapsed, "status": res.status, "obj": -np.inf,
                      "bound": np.nan, "gap": np.nan}
    x = res.x[ix] > 0.5
    bound = -getattr(res, "mip_dual_bound", np.nan)
    return x, {"time": elapsed, "status": int(res.status), "obj": objective(inst, x),
               "bound": float(bound), "gap": float(getattr(res, "mip_gap", np.nan))}


# --------------------------------------------------------------------------
# Heuristics
# --------------------------------------------------------------------------
def _grow(inst: GeofenceInstance, seed: int, score_bias: np.ndarray | None = None):
    n, m, sc = inst.n, inst.m, inst.scen
    x = np.zeros(n, bool); x[seed] = True
    L = sc[:, seed] * m[seed]; share = m[seed]
    J = float(_J_from_L(inst, share, L[:, None])[0])
    frontier = set(inst.region.nbrs[seed])
    best_x, best_J = (x.copy(), J) if share >= inst.min_share else (None, -np.inf)
    while frontier:
        F = np.fromiter(frontier, int)
        Lc = L[:, None] + sc[:, F] * m[F]
        Jc = _J_from_L(inst, share + m[F], Lc)
        sel = Jc + (score_bias[F] if score_bias is not None else 0.0)
        k = int(np.argmax(sel))
        if share >= inst.min_share and Jc[k] <= J:
            break
        i = F[k]
        x[i] = True; L = Lc[:, k]; share += m[i]; J = float(Jc[k])
        frontier.discard(i)
        frontier.update(v for v in inst.region.nbrs[i] if not x[v])
        if share >= inst.min_share and J > best_J:
            best_x, best_J = x.copy(), J
    return best_x, best_J


def greedy_solve(inst: GeofenceInstance, n_starts: int = 5):
    t0 = time.perf_counter()
    single = inst.price * inst.m - inst.m * inst.scen.mean(axis=0)
    margin_rate = single / inst.m                       # per-unit margin
    seeds = list(np.argsort(-margin_rate)[: n_starts - 1]) + [int(np.argmax(inst.m))]
    best_x, best_J = None, -np.inf
    for s in seeds:
        x, J = _grow(inst, int(s))
        if x is not None and J > best_J:
            best_x, best_J = x, J
    return best_x, {"time": time.perf_counter() - t0, "obj": best_J}


def local_search(inst: GeofenceInstance, x0: np.ndarray, max_iter: int = 400):
    """Best-improvement add/remove moves that keep the geofence connected and
    feasible."""
    t0 = time.perf_counter()
    x = np.asarray(x0, bool).copy()
    m, sc, nbrs = inst.m, inst.scen, inst.region.nbrs
    xm = x * m
    L = sc @ xm; share = xm.sum()
    J = float(_J_from_L(inst, share, L[:, None])[0])
    for _ in range(max_iter):
        sel = np.flatnonzero(x)
        add = np.array(sorted({v for u in sel for v in nbrs[u] if not x[v]}), int)
        G = nx.Graph()
        G.add_nodes_from(sel.tolist())
        G.add_edges_from((u, v) for u in sel for v in nbrs[u] if x[v] and u < v)
        art = set(nx.articulation_points(G)) if len(sel) > 2 else set(sel.tolist())
        rem = np.array([i for i in sel if i not in art
                        and share - m[i] >= inst.min_share - 1e-12], int)
        best = (0.0, None, None)
        if len(add):
            Lc = L[:, None] + sc[:, add] * m[add]
            dJ = _J_from_L(inst, share + m[add], Lc) - J
            k = int(np.argmax(dJ))
            if dJ[k] > best[0]:
                best = (dJ[k], add[k], +1)
        if len(rem) and len(sel) > 1:
            Lc = L[:, None] - sc[:, rem] * m[rem]
            dJ = _J_from_L(inst, share - m[rem], Lc) - J
            k = int(np.argmax(dJ))
            if dJ[k] > best[0]:
                best = (dJ[k], rem[k], -1)
        if best[1] is None or best[0] < 1e-12:
            break
        i, sgn = best[1], best[2]
        x[i] = sgn > 0
        L = L + sgn * sc[:, i] * m[i]; share += sgn * m[i]
        J = float(_J_from_L(inst, share, L[:, None])[0])
    return x, {"time": time.perf_counter() - t0, "obj": J}


def greedy_ls_solve(inst: GeofenceInstance):
    x, i1 = greedy_solve(inst)
    x, i2 = local_search(inst, x)
    return x, {"time": i1["time"] + i2["time"], "obj": i2["obj"]}


def brute_force(inst: GeofenceInstance):
    """Exhaustive search (n <= ~16), for tests."""
    best_x, best_J = None, -np.inf
    for b in range(1, 2 ** inst.n):
        x = np.array([(b >> i) & 1 for i in range(inst.n)], bool)
        if not is_feasible(inst, x):
            continue
        J = objective(inst, x)
        if J > best_J:
            best_x, best_J = x, J
    return best_x, best_J
