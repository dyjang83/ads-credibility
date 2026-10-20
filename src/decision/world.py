"""
Simulated worlds for the decision layer.

City level
----------
The candidate pool is the 7 real cities (unit-normalized mean cell embeddings)
followed by synthetic cities drawn by `_embedding_sampler` from
power_analysis_ncities.py, i.e. matched to the learned embedding geometry. The
three kernels the insurer may adopt are
    learned      S = V V^T            (cosine of city embeddings)
    euclidean    `_euclidean_kernel`  (median-heuristic RBF on the embeddings)
    independent  I
    *_nugget     the first two plus NUGGET * I (see config.NUGGET)
Ground truth: alpha ~ N(0, sigma^2 S_true), with S_true = S_learned under the
"structured" DGP and I under the "identity_null" DGP (specificity check, as in
power_analysis.py).

Cell level (MODELLING ASSUMPTION, not observed)
-----------------------------------------------
No public source reports ADS exposure or claims below the city level, so the
within-city geography is modelled, not estimated:
  * a city's candidate service area is a connected ball of REGION_SIZE H3 cells
    grown from its highest-HDV-exposure cell;
  * ADS mileage inside a geofence is allocated in proportion to HDV exposure
    (data/cell_exposure.csv);
  * cell risk relativities  log r_i = CELL_HDV_COEF * z(log HDV freq_i) + eps_i,
    eps ~ GP(0, CELL_TAU^2 * exp((cos(e_i, e_j) - 1) / CELL_BANDWIDTH)) over the
    32-d cell embeddings, centered so the exposure-weighted region mean of
    log r is ~0 (the city level is carried by alpha).
Synthetic cities borrow the cell template of their most similar real city.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache

import h3
import numpy as np
import pandas as pd

from .config import (DATA, EMBED, SIGMA, BASE_RATE, JITTER, CELL_HDV_COEF,
                     CELL_TAU, CELL_BANDWIDTH, REGION_SIZE, POOL_SIZE, NUGGET)
from data_generator import DEPLOYED_CITIES, HYPOTHETICAL_CITIES  # noqa: E402
from power_analysis_ncities import _embedding_sampler, _euclidean_kernel  # noqa: E402

REAL_CITIES = DEPLOYED_CITIES + HYPOTHETICAL_CITIES


# --------------------------------------------------------------------------
# Cell data
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_cells() -> pd.DataFrame:
    emb = pd.read_csv(EMBED / "cell_embeddings.csv")
    feat = pd.read_csv(DATA / "cell_features.csv")[
        ["cell_id", "hdv_claim_freq_per_million_miles"]]
    expo = pd.read_csv(DATA / "cell_exposure.csv")[["cell_id", "hdv_exposure_million_miles"]]
    df = emb.merge(feat, on="cell_id").merge(expo, on="cell_id")
    lf = np.log(df["hdv_claim_freq_per_million_miles"].clip(lower=1e-3))
    df["z_hdv"] = (lf - lf.mean()) / lf.std()
    return df


def embedding_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]


@lru_cache(maxsize=None)
def city_cells(city: str) -> tuple[pd.DataFrame, dict]:
    """Cells of one real city restricted to its largest connected component,
    plus an adjacency dict cell_id -> set(neighbor ids)."""
    df = load_cells()
    d = df[df.city == city].reset_index(drop=True)
    ids = set(d.cell_id)
    adj = {u: {v for v in h3.grid_ring(u, 1) if v in ids} for u in ids}
    seen, best = set(), set()
    for s in ids:
        if s in seen:
            continue
        comp, q = {s}, deque([s])
        while q:
            u = q.popleft()
            for v in adj[u] - comp:
                comp.add(v); q.append(v)
        seen |= comp
        if len(comp) > len(best):
            best = comp
    d = d[d.cell_id.isin(best)].reset_index(drop=True)
    adj = {u: adj[u] & best for u in best}
    return d, adj


# --------------------------------------------------------------------------
# Cell regions (the geofence CO instances live on these)
# --------------------------------------------------------------------------
@dataclass
class CellRegion:
    city: str
    cell_ids: list
    emb: np.ndarray          # (n, 32)
    m: np.ndarray            # exposure share, sums to 1
    z: np.ndarray            # z(log HDV frequency)
    log_r_mean: np.ndarray   # prior mean of log relativity
    L: np.ndarray            # Cholesky of prior cov of log relativity
    edges: np.ndarray        # (E, 2) undirected, i < j
    nbrs: list = field(repr=False, default_factory=list)

    @property
    def n(self) -> int:
        return len(self.cell_ids)

    def sample_log_r(self, n_draws: int, rng: np.random.Generator) -> np.ndarray:
        return self.log_r_mean + rng.standard_normal((n_draws, self.n)) @ self.L.T


def _grow_region(adj: dict, start: str, size: int, rng=None) -> list:
    """BFS ball from `start`; with an rng, the frontier is shuffled so random
    regions have irregular shapes."""
    region, order, q = {start}, [start], deque([start])
    while q and len(order) < size:
        u = q.popleft()
        nb = sorted(adj[u] - region)      # sort first: set order depends on the hash seed
        if rng is not None:
            rng.shuffle(nb)
        for v in nb:
            if len(order) >= size:
                break
            region.add(v); order.append(v); q.append(v)
    return order


def make_region(city: str, cell_order: list) -> CellRegion:
    d, adj = city_cells(city)
    d = d.set_index("cell_id").loc[cell_order]
    ecols = embedding_cols(d)
    emb = d[ecols].to_numpy()
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    w = d["hdv_exposure_million_miles"].to_numpy()
    m = w / w.sum()
    z = d["z_hdv"].to_numpy()
    K = CELL_TAU ** 2 * np.exp((emb @ emb.T - 1.0) / CELL_BANDWIDTH)
    L = np.linalg.cholesky(K + 1e-8 * np.eye(len(m)))
    mean = CELL_HDV_COEF * (z - m @ z) - 0.5 * CELL_TAU ** 2
    idx = {c: i for i, c in enumerate(cell_order)}
    edges = sorted({(min(idx[u], idx[v]), max(idx[u], idx[v]))
                    for u in cell_order for v in adj[u] if v in idx})
    nbrs = [[] for _ in cell_order]
    for i, j in edges:
        nbrs[i].append(j); nbrs[j].append(i)
    return CellRegion(city, list(cell_order), emb, m, z, mean, L,
                      np.array(edges, dtype=int).reshape(-1, 2), nbrs)


@lru_cache(maxsize=None)
def service_region(city: str, size: int = REGION_SIZE) -> CellRegion:
    """Deterministic candidate service area: BFS ball around the city's
    highest-HDV-exposure cell."""
    d, adj = city_cells(city)
    start = d.loc[d.hdv_exposure_million_miles.idxmax(), "cell_id"]
    return make_region(city, _grow_region(adj, start, size))


def random_region(city: str, size: int, rng: np.random.Generator) -> CellRegion:
    d, adj = city_cells(city)
    start = d.cell_id.iloc[rng.integers(len(d))]
    return make_region(city, _grow_region(adj, start, size, rng))


# --------------------------------------------------------------------------
# City pool and worlds
# --------------------------------------------------------------------------
@dataclass
class CityPool:
    names: list
    V: np.ndarray            # (K, 32) unit embeddings
    template: list           # real city whose cell region each city uses
    kernels: dict            # name -> (K, K)

    @property
    def K(self) -> int:
        return len(self.names)


def real_city_embeddings() -> np.ndarray:
    df = load_cells()
    ecols = embedding_cols(df)
    C = np.vstack([df.loc[df.city == c, ecols].to_numpy().mean(axis=0) for c in REAL_CITIES])
    return C / np.linalg.norm(C, axis=1, keepdims=True)


@lru_cache(maxsize=None)
def city_pool(size: int = POOL_SIZE, seed: int = 0) -> CityPool:
    C = real_city_embeddings()
    n_extra = size - len(REAL_CITIES)
    names = list(REAL_CITIES)
    V = C
    if n_extra > 0:
        sampler = _embedding_sampler(seed=seed, max_k=len(DEPLOYED_CITIES) + n_extra)
        extra = sampler(len(DEPLOYED_CITIES) + n_extra)[len(DEPLOYED_CITIES):]
        V = np.vstack([C, extra])
        names += [f"Synthetic-{i:02d}" for i in range(n_extra)]
    template = [REAL_CITIES[int(np.argmax(C @ v))] for v in V]
    S_learned = V @ V.T
    S_eu = _euclidean_kernel(V)
    I = np.eye(len(V))
    kernels = {"learned": S_learned, "euclidean": S_eu, "independent": I,
               "learned_nugget": S_learned + NUGGET * I,
               "euclidean_nugget": S_eu + NUGGET * I}
    return CityPool(names, V, template, kernels)


@dataclass
class World:
    pool: CityPool
    dgp: str
    sigma: float
    base_rate: float
    mu: float
    alpha: np.ndarray        # (K,)
    log_r: dict              # template city -> true cell log relativities

    def city_rate(self, k: int, geofence_mask: np.ndarray | None = None) -> float:
        """True claim frequency of city k when served on `geofence_mask` over
        its template region (exposure-weighted mean cell relativity)."""
        reg = service_region(self.pool.template[k])
        r = np.exp(self.log_r[self.pool.template[k]][k])
        x = np.ones(reg.n, bool) if geofence_mask is None else geofence_mask
        R = float(reg.m[x] @ r[x] / reg.m[x].sum())
        return float(np.exp(self.mu + self.alpha[k]) * R)


def sample_world(pool: CityPool, dgp: str, rng: np.random.Generator,
                 sigma: float = SIGMA, base_rate: float = BASE_RATE) -> World:
    S_true = {"structured": pool.kernels["learned"],
              "identity_null": np.eye(pool.K),
              "euclidean_null": pool.kernels["euclidean"]}[dgp]
    Lc = np.linalg.cholesky(sigma ** 2 * S_true + JITTER * np.eye(pool.K))
    alpha = Lc @ rng.standard_normal(pool.K)
    # Independent cell relativities per CITY (synthetic cities share a
    # template's geography but not its realized risk surface).
    log_r = {}
    for t in sorted(set(pool.template)):
        reg = service_region(t)
        draws = reg.sample_log_r(pool.K, rng)          # row k used by city k
        log_r[t] = draws
    return World(pool, dgp, sigma, base_rate, float(np.log(base_rate)), alpha, log_r)
