"""
OSM road-network feature fetcher.

Produces, per H3 cell, six road-network features:
    road_length_arterial_km
    road_length_collector_km
    road_length_local_km
    intersection_density_per_km2
    signalized_intersection_fraction
    betweenness_centrality_mean

Data source: the OpenStreetMap Overpass API. We query, for each city bounding
box, (a) all highway ways of the functional classes we model, and (b) all
nodes tagged highway=traffic_signals. We then:

  1. Split each way into segments between consecutive nodes, assign each
     segment to the H3 cell containing its midpoint, and accumulate segment
     length into the way's functional-class bucket for that cell.
  2. Detect intersections as OSM nodes shared by two or more distinct ways;
     assign each to its H3 cell and divide by cell area for density.
  3. Compute the signalized fraction as (traffic-signal nodes in cell) /
     (intersection nodes in cell), clamped to [0, 1].
  4. Build a graph per city from the way topology and compute node betweenness
     centrality, then average over the intersection nodes in each cell. For
     large cities we approximate betweenness with k-source sampling (networkx
     `k` parameter) to keep runtime tractable.

Network access: required (Overpass). Once the cache is warm, reruns are offline.

Dependencies: requests, h3 (>=4), networkx, numpy, pandas. Deliberately avoids
geopandas/osmnx so it installs without a GDAL toolchain.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import h3
import networkx as nx
import numpy as np
import pandas as pd

from .config import (
    City, FetchConfig, H3_RESOLUTION, OSM_HIGHWAY_BUCKETS,
    H3_RES8_AVG_AREA_KM2,
)
from .http_cache import CachedSession


# ---------------------------------------------------------------------------
# Overpass query construction
# ---------------------------------------------------------------------------
def build_overpass_query(bbox: tuple[float, float, float, float]) -> str:
    """Overpass QL for the drivable road network + traffic signals in a bbox.

    bbox is (south, west, north, east). We request way geometry inline
    (`out geom`) so we get node coordinates without a second round-trip, and
    request signal nodes separately.
    """
    s, w, n, e = bbox
    classes = "|".join(sorted(set(OSM_HIGHWAY_BUCKETS.keys())))
    return f"""
[out:json][timeout:180];
(
  way["highway"~"^({classes})$"]({s},{w},{n},{e});
);
out geom;
(
  node["highway"="traffic_signals"]({s},{w},{n},{e});
);
out body;
""".strip()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def cell_of(lat: float, lon: float) -> str:
    return h3.latlng_to_cell(lat, lon, H3_RESOLUTION)


def cell_area_km2(cell: str) -> float:
    try:
        return h3.cell_area(cell, unit="km^2")
    except Exception:
        return H3_RES8_AVG_AREA_KM2


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------
@dataclass
class OSMParseResult:
    # cell -> {arterial, collector, local} -> km
    road_km: dict[str, dict[str, float]]
    # cell -> set of intersection node ids
    intersection_nodes: dict[str, set[int]]
    # cell -> count of traffic-signal nodes
    signal_count: dict[str, int]
    # cell -> mean betweenness over its intersection nodes
    betweenness: dict[str, float]


def _node_use_counts(elements: list[dict]) -> dict[int, int]:
    """Count, for every node id, how many distinct ways reference it.

    A node referenced by >= 2 ways is an intersection. `out geom` gives us node
    ids in the way's `nodes` array alongside `geometry` coordinates.
    """
    counts: dict[int, int] = defaultdict(int)
    for el in elements:
        if el.get("type") != "way":
            continue
        node_ids = el.get("nodes", [])
        for nid in node_ids:
            counts[nid] += 1
    return counts


def parse_osm(elements: list[dict], *, betweenness_k: int = 200) -> OSMParseResult:
    road_km: dict[str, dict[str, float]] = defaultdict(
        lambda: {"arterial": 0.0, "collector": 0.0, "local": 0.0}
    )
    intersection_nodes: dict[str, set[int]] = defaultdict(set)
    signal_count: dict[str, int] = defaultdict(int)

    # node id -> (lat, lon), harvested from way geometry
    node_coord: dict[int, tuple[float, float]] = {}

    node_uses = _node_use_counts(elements)
    graph = nx.Graph()

    for el in elements:
        etype = el.get("type")
        if etype == "way":
            hclass = el.get("tags", {}).get("highway")
            bucket = OSM_HIGHWAY_BUCKETS.get(hclass)
            if bucket is None:
                continue
            geom = el.get("geometry", [])
            node_ids = el.get("nodes", [])
            # geometry and nodes are parallel arrays under `out geom`
            for i in range(len(geom) - 1):
                lat1, lon1 = geom[i]["lat"], geom[i]["lon"]
                lat2, lon2 = geom[i + 1]["lat"], geom[i + 1]["lon"]
                seg_km = haversine_km(lat1, lon1, lat2, lon2)
                mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
                c = cell_of(mid_lat, mid_lon)
                road_km[c][bucket] += seg_km
                if i < len(node_ids):
                    node_coord[node_ids[i]] = (lat1, lon1)
                # graph edge between consecutive nodes for betweenness
                if i < len(node_ids) - 1:
                    graph.add_edge(node_ids[i], node_ids[i + 1], weight=seg_km)
            if node_ids:
                last = node_ids[-1]
                if geom:
                    node_coord.setdefault(last, (geom[-1]["lat"], geom[-1]["lon"]))

            # intersection nodes within this way
            for nid in node_ids:
                if node_uses.get(nid, 0) >= 2 and nid in node_coord:
                    lat, lon = node_coord[nid]
                    intersection_nodes[cell_of(lat, lon)].add(nid)

        elif etype == "node":
            if el.get("tags", {}).get("highway") == "traffic_signals":
                c = cell_of(el["lat"], el["lon"])
                signal_count[c] += 1

    # ---- betweenness centrality (approximate for large graphs) ----
    betweenness: dict[str, float] = {}
    if graph.number_of_nodes() > 0:
        k = min(betweenness_k, graph.number_of_nodes())
        bc = nx.betweenness_centrality(graph, k=k, weight="weight", seed=20260525)
        per_cell_vals: dict[str, list[float]] = defaultdict(list)
        for nid, val in bc.items():
            if nid in node_coord:
                lat, lon = node_coord[nid]
                per_cell_vals[cell_of(lat, lon)].append(val)
        for c, vals in per_cell_vals.items():
            betweenness[c] = float(np.mean(vals)) if vals else 0.0

    return OSMParseResult(
        road_km=dict(road_km),
        intersection_nodes=dict(intersection_nodes),
        signal_count=dict(signal_count),
        betweenness=betweenness,
    )


def result_to_frame(city: City, parsed: OSMParseResult) -> pd.DataFrame:
    """Collapse the per-cell dicts into the city's OSM feature rows."""
    cells = set(parsed.road_km) | set(parsed.intersection_nodes) | set(parsed.signal_count)
    rows = []
    for c in sorted(cells):
        rk = parsed.road_km.get(c, {"arterial": 0.0, "collector": 0.0, "local": 0.0})
        n_intersections = len(parsed.intersection_nodes.get(c, set()))
        n_signals = parsed.signal_count.get(c, 0)
        area = cell_area_km2(c)
        sig_frac = (n_signals / n_intersections) if n_intersections > 0 else 0.0
        rows.append({
            "cell_id": c,
            "city": city.name,
            "road_length_arterial_km": rk["arterial"],
            "road_length_collector_km": rk["collector"],
            "road_length_local_km": rk["local"],
            "intersection_density_per_km2": n_intersections / area,
            "signalized_intersection_fraction": min(sig_frac, 1.0),
            "betweenness_centrality_mean": parsed.betweenness.get(c, 0.0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def fetch_osm_features(cfg: FetchConfig) -> pd.DataFrame:
    """Fetch OSM road features for all cities in the config. Requires network."""
    session = CachedSession(
        cfg.cache_dir, user_agent=cfg.user_agent,
        request_pause_s=cfg.request_pause_s, max_retries=cfg.max_retries,
        namespace="overpass",
    )
    frames = []
    for city in cfg.cities:
        query = build_overpass_query(city.bbox)
        print(f"[osm] {city.name}: querying Overpass ...")
        data = session.post_json(cfg.overpass_url, data=query,
                                 timeout_s=cfg.overpass_timeout_s)
        elements = data.get("elements", [])
        print(f"[osm] {city.name}: {len(elements)} elements")
        parsed = parse_osm(elements)
        frames.append(result_to_frame(city, parsed))
    return pd.concat(frames, ignore_index=True)
