"""
FARS (Fatality Analysis Reporting System) crash-density feature fetcher.

Produces, per H3 cell, one feature:
    historical_crash_density_per_mile

and contributes to the HDV-frequency target used for contrastive positive pairs.

Data source: NHTSA's FARS annual data files. Each year ships an `accident.csv`
(one row per fatal crash) carrying LATITUDE / LONGITUD columns in decimal
degrees. We download the requested year range, filter to crashes whose
coordinates fall inside a city bbox, assign each to its H3 cell, and divide the
count by the cell's road-mileage (supplied by the OSM layer) to get a
per-mile crash density.

FARS coordinate hygiene: FARS uses sentinel values (e.g. 77.7777, 88.8888,
99.9999 and 0.0) for unknown coordinates. We drop these before binning.

Network access: required (NHTSA static file server). Cache makes reruns offline.

NOTE on transport: the annual files are ZIP archives, not JSON. This module
fetches them with a plain streaming download (not the JSON CachedSession) and
caches the extracted accident.csv to disk.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import requests

from .config import City, FetchConfig, H3_RESOLUTION

# NHTSA file-server layout. The directory naming has shifted slightly over the
# years; this pattern covers 2015+ which is the relevant window for ADS-era
# spatial texture. Adjust FARS_URL_TEMPLATE if NHTSA reorganizes.
FARS_URL_TEMPLATE = (
    "https://static.nhtsa.gov/nhtsa/downloads/FARS/"
    "{year}/National/FARS{year}NationalCSV.zip"
)

# Sentinel coordinate values FARS uses for missing/unknown.
FARS_BAD_COORDS = {77.7777, 88.8888, 99.9999, 0.0, 77.0, 88.0, 99.0}


def _download_accident_csv(
    year: int, cache_dir: Path, user_agent: str
) -> pd.DataFrame:
    """Download and cache the year's accident.csv as a parquet for fast reruns."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"fars_{year}_accident.parquet"
    if cached.exists():
        return pd.read_parquet(cached)

    url = FARS_URL_TEMPLATE.format(year=year)
    print(f"[fars] downloading {url}")
    resp = requests.get(url, headers={"User-Agent": user_agent}, timeout=300)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        # find accident.csv case-insensitively, anywhere in the archive
        name = next(
            (n for n in zf.namelist() if n.lower().endswith("accident.csv")),
            None,
        )
        if name is None:
            raise RuntimeError(f"accident.csv not found in FARS {year} archive")
        with zf.open(name) as fh:
            df = pd.read_csv(fh, encoding="latin-1", low_memory=False)

    # Normalize the coordinate column names (they are stable but case varies).
    cols = {c.upper(): c for c in df.columns}
    lat_col = cols.get("LATITUDE")
    lon_col = cols.get("LONGITUD") or cols.get("LONGITUDE")
    if lat_col is None or lon_col is None:
        raise RuntimeError(f"FARS {year}: no lat/lon columns in {list(df.columns)}")
    out = df[[lat_col, lon_col]].rename(
        columns={lat_col: "lat", lon_col: "lon"}
    )
    out.to_parquet(cached)
    return out


def _clean_coords(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    bad = df["lat"].isin(FARS_BAD_COORDS) | df["lon"].isin(FARS_BAD_COORDS)
    df = df[~bad]
    # Plausible CONUS + HI/AK envelope; drops obvious geocoding errors.
    df = df[(df["lat"].between(17.0, 72.0)) & (df["lon"].between(-180.0, -64.0))]
    return df


def crashes_in_city(df: pd.DataFrame, city: City) -> pd.DataFrame:
    s, w, n, e = city.bbox
    return df[
        df["lat"].between(s, n) & df["lon"].between(w, e)
    ].copy()


def crash_density_per_cell(
    crashes: pd.DataFrame,
    road_km_by_cell: dict[str, float],
) -> pd.DataFrame:
    """Crashes per cell / road miles per cell. Cells with no road get density 0."""
    counts: dict[str, int] = {}
    for lat, lon in zip(crashes["lat"], crashes["lon"]):
        c = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        counts[c] = counts.get(c, 0) + 1

    rows = []
    all_cells = set(counts) | set(road_km_by_cell)
    for c in all_cells:
        n_crash = counts.get(c, 0)
        road_km = road_km_by_cell.get(c, 0.0)
        road_miles = road_km * 0.621371
        density = (n_crash / road_miles) if road_miles > 0.1 else 0.0
        rows.append({
            "cell_id": c,
            "historical_crash_density_per_mile": density,
            "_fars_crash_count": n_crash,
        })
    return pd.DataFrame(rows)


def fetch_fars_features(
    cfg: FetchConfig,
    osm_features: pd.DataFrame,
    years: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022),
) -> pd.DataFrame:
    """Fetch FARS crash density for all cities, normalized by OSM road mileage.

    osm_features must already be computed (it supplies per-cell road mileage).
    Requires network for the first run.
    """
    cache_dir = Path(cfg.cache_dir) / "fars"
    # accumulate accident coordinates across the year range, once nationally
    frames = [
        _download_accident_csv(y, cache_dir, cfg.user_agent) for y in years
    ]
    national = _clean_coords(pd.concat(frames, ignore_index=True))
    print(f"[fars] {len(national)} clean fatal-crash coordinates over {years}")

    # per-cell road mileage from OSM (sum of the three buckets)
    osm = osm_features.copy()
    osm["_road_km_total"] = (
        osm["road_length_arterial_km"]
        + osm["road_length_collector_km"]
        + osm["road_length_local_km"]
    )

    out_frames = []
    for city in cfg.cities:
        city_crashes = crashes_in_city(national, city)
        road_by_cell = (
            osm.loc[osm["city"] == city.name]
            .set_index("cell_id")["_road_km_total"]
            .to_dict()
        )
        frame = crash_density_per_cell(city_crashes, road_by_cell)
        frame["city"] = city.name
        out_frames.append(frame)
        print(f"[fars] {city.name}: {len(city_crashes)} crashes in bbox")
    return pd.concat(out_frames, ignore_index=True)
