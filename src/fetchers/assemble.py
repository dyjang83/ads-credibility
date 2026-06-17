"""
Assemble real OSM + ACS + FARS features into the cell_features.csv schema that
src/embedding.py consumes, plus the calibrated HDV-frequency target.

Usage:
    python -m src.fetchers.assemble --out data
    python -m src.fetchers.assemble --out data --cities "San Francisco,Boston"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CELL_FEATURE_COLUMNS, CITIES, CITY_BY_NAME, FetchConfig,
    FEATURE_PROVENANCE, HDV_FREQUENCY_COLUMN,
)
from .osm_fetcher import fetch_osm_features
from .acs_fetcher import fetch_acs_features
from .fars_fetcher import fetch_fars_features
from .fhwa_fetcher import fetch_fhwa_vmt

# Published HDV benchmark frequencies (claims per million miles) per city,
# used purely to set the absolute level of the per-cell frequency target. 
HDV_BENCHMARK_FREQ = {
    "San Francisco": 4.06,
    "Phoenix": 2.49,
    "Los Angeles": 2.79,
    "Austin": 2.48,
    "Boston": 3.90,
    "Denver": 2.65,
    "Miami": 4.20,
}


def _impute_missing(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """Fill NaNs with the per-city median; track how many cells were imputed."""
    report: dict[str, int] = {}
    df = df.copy()
    for col in columns:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            report[col] = n_missing
            df[col] = df.groupby("city")[col].transform(
                lambda s: s.fillna(s.median())
            )
        # any residual NaN (whole-city missing) -> global median
        df[col] = df[col].fillna(df[col].median())
    return df, report


def _calibrate_hdv_frequency(
    df: pd.DataFrame, cell_exposure: "np.ndarray | None" = None
) -> pd.DataFrame:
    """Map crash density -> claims-frequency proxy, rescaled per city to anchor.

    Transform: freq_raw = log1p(crash_density). Monotone, compresses the heavy
    right tail of crash density, and is 0 where there are no crashes. Then, per
    city, multiply by a constant so a reference mean equals the city's HDV
    benchmark frequency.

    Reference mean: if ``cell_exposure`` is provided as a plain NumPy array
    (real per-cell VMT derived from FHWA urbanized-area totals, see
    fetch_fhwa_vmt), the per-city *exposure-weighted* cell mean is anchored --
    the statistically correct target for a frequency. If it is None (no FHWA
    data), we fall back to the unweighted cell mean, since per-cell exposure is
    otherwise unobserved in the public data. The two agree when exposure is
    uniform across cells.

    ``cell_exposure`` must be a plain ndarray (not a pd.Series) so that there
    is no index to misalign when it is assigned to df["_expo"]. The caller is
    responsible for passing .to_numpy() if it holds a Series.
    """
    df = df.copy()
    df["_freq_raw"] = np.log1p(df["historical_crash_density_per_mile"].clip(lower=0))
    if cell_exposure is not None:
        # cell_exposure is already a plain ndarray — assign directly.
        df["_expo"] = np.asarray(cell_exposure, dtype=float)
    out = []
    for city, g in df.groupby("city"):
        anchor = HDV_BENCHMARK_FREQ.get(city)
        if cell_exposure is not None and g["_expo"].notna().any() and g["_expo"].sum() > 0:
            w = g["_expo"].fillna(0.0)
            raw_mean = float((g["_freq_raw"] * w).sum() / w.sum())
        else:
            raw_mean = float(g["_freq_raw"].mean())
        if anchor is None or raw_mean <= 0:
            base = anchor if anchor is not None else 0.0
            g[HDV_FREQUENCY_COLUMN] = base
        else:
            scale = anchor / raw_mean
            g[HDV_FREQUENCY_COLUMN] = g["_freq_raw"] * scale
            # guard against degenerate zeros: floor at 10% of the anchor so that
            # contrastive pairs are well-defined even in crash-free cells
            g[HDV_FREQUENCY_COLUMN] = g[HDV_FREQUENCY_COLUMN].clip(lower=0.1 * anchor)
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    return res.drop(columns=[c for c in ("_freq_raw", "_expo") if c in res.columns])


def _stub_acs(osm: "pd.DataFrame") -> "pd.DataFrame":
    """Return a zero-filled ACS frame with the correct schema.

    Used when --skip-acs is requested. The four demographic features will be
    zero throughout; the embedding will rely entirely on OSM road-network
    geometry. The provenance report will flag all four columns as stubbed.
    """
    import numpy as np
    df = osm[["cell_id", "city"]].copy()
    df["population_density_per_km2"]  = np.nan
    df["pedestrian_commute_share"]     = np.nan
    df["land_use_residential_share"]   = np.nan
    df["land_use_commercial_share"]    = np.nan
    return df


def _road_mileage_per_cell(df: pd.DataFrame) -> pd.Series:
    """Total road kilometres per cell from the three OSM road-length columns.

    Used as the weight for distributing a city's FHWA VMT across its cells:
    a cell with more road carries proportionally more vehicle-miles.
    """
    cols = ["road_length_arterial_km", "road_length_collector_km", "road_length_local_km"]
    return df[cols].sum(axis=1)


def _distribute_vmt_to_cells(
    df: pd.DataFrame, city_vmt: pd.DataFrame
) -> pd.Series:
    """Per-cell HDV exposure (million miles), summing to each city's FHWA VMT.

    Within a city, the urbanized-area total VMT is split across cells in
    proportion to road mileage. Cities absent from ``city_vmt`` (NaN VMT) get
    NaN exposure, and the caller falls back to the unweighted calibration for
    them. This is the real exposure denominator described in Section 6.1.

    Implemented with groupby-transform (not label indexing) so it is correct
    even when ``df`` has a non-unique index.
    """
    vmt_by_city = dict(zip(city_vmt["city"], city_vmt["annual_vmt_millions"]))
    road = _road_mileage_per_cell(df).astype(float)
    city = df["city"]
    total_vmt = city.map(vmt_by_city).astype(float)         # NaN where unmatched
    road_sum = road.groupby(city).transform("sum")
    cell_count = city.groupby(city).transform("size")
    # proportional split where the city has positive road mileage,
    # equal split where it does not, NaN where the city has no VMT.
    with np.errstate(divide="ignore", invalid="ignore"):
        prop = np.where(road_sum > 0, road / road_sum, 1.0 / cell_count)
    expo = total_vmt.to_numpy() * prop
    return pd.Series(expo, index=df.index)


def _fetch_city_vmt(cfg: FetchConfig, skip_fhwa: bool) -> "pd.DataFrame | None":
    """Fetch FHWA urbanized-area VMT, or None if skipped/unavailable.

    Failures are non-fatal: the cell pipeline does not depend on VMT, so a
    network or parse problem degrades gracefully to the unweighted calibration
    rather than aborting the whole assembly.
    """
    if skip_fhwa:
        print("[fhwa] SKIPPED — exposure denominators will not be fetched; "
              "HDV calibration falls back to the unweighted cell mean.")
        return None
    try:
        return fetch_fhwa_vmt(cfg)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        print(f"[fhwa] WARNING: VMT fetch failed ({exc}). Falling back to the "
              f"unweighted HDV calibration. Re-run without --skip-fhwa once the "
              f"FHWA source is reachable.")
        return None


def assemble(
    cfg: FetchConfig,
    out_dir: Path,
    skip_acs: bool = False,
    skip_fhwa: bool = False,
    exposure_weighted_hdv: bool = False,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    osm = fetch_osm_features(cfg)

    if skip_acs:
        print("[acs] SKIPPED — demographic features will be zero-filled.")
        print("[acs] Re-run without --skip-acs once a valid CENSUS_API_KEY is set.")
        acs = _stub_acs(osm)
    else:
        acs = fetch_acs_features(cfg)

    fars = fetch_fars_features(cfg, osm)

    # Real exposure denominators (FHWA urbanized-area VMT). Non-fatal if absent.
    city_vmt = _fetch_city_vmt(cfg, skip_fhwa)

    # Join on (cell_id, city). OSM is the spine: a cell exists if it has roads.
    df = osm.merge(
        acs.drop_duplicates(subset=["cell_id", "city"]),
        on=["cell_id", "city"], how="left",
    ).merge(
        fars.drop_duplicates(subset=["cell_id", "city"]),
        on=["cell_id", "city"], how="left",
    )

    df, impute_report = _impute_missing(df, list(CELL_FEATURE_COLUMNS))

    # Per-cell HDV exposure from FHWA VMT (million miles), if available.
    # Attach as a column BEFORE calibration so it travels with the rows through
    # the groupby/concat reorder inside _calibrate_hdv_frequency (which returns
    # rows sorted by city). Without this it would misalign with the side file.
    cell_exposure = None
    if city_vmt is not None:
        cell_exposure = _distribute_vmt_to_cells(df, city_vmt)
        df["_hdv_exposure"] = cell_exposure.to_numpy()

    # Calibrate the HDV-frequency target. Exposure-weighting is opt-in so that
    # the default output is byte-for-byte the same as before FHWA was wired in;
    # pass --exposure-weighted-hdv to use the real VMT denominator.
    # Pass .to_numpy() explicitly — not the Series — so the callee receives a
    # plain array with no index; this makes alignment unambiguous even if
    # _calibrate_hdv_frequency ever reorders df before consuming the array.
    use_expo = (df["_hdv_exposure"].to_numpy()
                if (exposure_weighted_hdv and cell_exposure is not None)
                else None)
    df = _calibrate_hdv_frequency(df, cell_exposure=use_expo)
    hdv_calibration = "exposure_weighted" if use_expo is not None else "unweighted_cell_mean"

    # Pull the (now row-aligned) exposure column off for the side file, then
    # drop it so cell_features.csv keeps its exact schema.
    cell_exposure_aligned = (
        df.pop("_hdv_exposure") if "_hdv_exposure" in df.columns else None
    )

    # Final column order, matching the synthetic cell_features.csv exactly.
    ordered = ["cell_id", "city", *CELL_FEATURE_COLUMNS, HDV_FREQUENCY_COLUMN]
    df = df[ordered]

    # Validate schema against what embedding.py expects.
    _validate_schema(df)

    out_path = out_dir / "cell_features.csv"
    df.to_csv(out_path, index=False)

    # Side outputs for the exposure layer (do not affect cell_features schema).
    if city_vmt is not None:
        city_vmt.to_csv(out_dir / "city_vmt.csv", index=False)
        print(f"[assemble] wrote {out_dir / 'city_vmt.csv'} "
              f"(FHWA HM-72 vintage {cfg.fhwa_year})")
        if cell_exposure_aligned is not None:
            expo_df = df[["cell_id", "city"]].copy()
            expo_df["hdv_exposure_million_miles"] = cell_exposure_aligned.to_numpy()
            expo_df.to_csv(out_dir / "cell_exposure.csv", index=False)
            print(f"[assemble] wrote {out_dir / 'cell_exposure.csv'} "
                  f"(per-cell VMT, road-mileage weighted)")

    provenance = {
        "n_cells_total": int(len(df)),
        "n_cells_by_city": df.groupby("city").size().to_dict(),
        "feature_provenance": FEATURE_PROVENANCE,
        "imputed_cells_by_feature": impute_report,
        "hdv_benchmark_anchors": HDV_BENCHMARK_FREQ,
        "hdv_calibration": hdv_calibration,
        "exposure_source": (
            f"FHWA Highway Statistics HM-72 (vintage {cfg.fhwa_year})"
            if city_vmt is not None else "none (FHWA skipped/unavailable)"
        ),
        "city_vmt_millions": (
            {str(c): (float(v) if pd.notna(v) else None)
             for c, v in zip(city_vmt["city"], city_vmt["annual_vmt_millions"])}
            if city_vmt is not None else {}
        ),
        "schema_columns": list(df.columns),
        "acs_stubbed": skip_acs,
    }
    (out_dir / "cell_features_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=int)
    )
    print(f"[assemble] wrote {out_path} ({len(df)} cells)")
    print(f"[assemble] provenance -> {out_dir / 'cell_features_provenance.json'}")
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    expected = ["cell_id", "city", *CELL_FEATURE_COLUMNS, HDV_FREQUENCY_COLUMN]
    if list(df.columns) != expected:
        raise ValueError(
            f"Schema mismatch.\n  expected: {expected}\n  got:      {list(df.columns)}"
        )
    if df[list(CELL_FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("NaNs remain in feature columns after imputation")


# Absolute path to the repo root (the directory containing src/).
# Using __file__ means the cache location is always the same regardless of
# which directory you run from, so a warm cache is always found.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = _REPO_ROOT / "data" / "cache"
_DEFAULT_OUT   = _REPO_ROOT / "data"


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble real ADS cell features.")
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE,
                    help="Cache directory for raw API responses (default: "
                         "<repo>/data/cache). Using an absolute path means "
                         "reruns from any working directory hit the same cache.")
    ap.add_argument("--cities", type=str, default=None,
                    help="Comma-separated subset of city names (default: all 7).")
    ap.add_argument("--census-year", type=int, default=2022)
    ap.add_argument("--fhwa-year", type=int, default=2022,
                    help="FHWA Highway Statistics vintage for urbanized-area "
                         "VMT (HM-72). Default 2022.")
    ap.add_argument("--skip-fhwa", action="store_true",
                    help="Skip the FHWA VMT fetch. HDV-frequency calibration "
                         "then uses the unweighted cell mean (the pre-FHWA "
                         "behaviour). No city_vmt.csv is written.")
    ap.add_argument("--exposure-weighted-hdv", action="store_true",
                    help="Anchor the HDV-frequency calibration to the "
                         "exposure-weighted cell mean using FHWA VMT as the "
                         "per-cell denominator (the statistically correct "
                         "target). Off by default so the emitted "
                         "cell_features.csv matches the pre-FHWA pipeline; "
                         "turning it on changes the embedding's frequency "
                         "target, so re-run embedding + analysis afterwards.")
    ap.add_argument("--clear-acs-cache", action="store_true",
                    help="Delete cached ACS responses before fetching. Use this "
                         "if a previous run cached Census error pages (e.g. "
                         "'Invalid Key' HTML) because the key was not yet active.")
    ap.add_argument("--skip-acs", action="store_true",
                    help="Skip the Census ACS fetch entirely and fill the four "
                         "demographic features with zeros. Use this if you cannot "
                         "obtain a Census API key right now. The embedding will "
                         "run on OSM + FARS features only. Re-run without this "
                         "flag once CENSUS_API_KEY is available to get the full "
                         "feature set.")
    args = ap.parse_args()

    cities = CITIES
    if args.cities:
        names = [s.strip() for s in args.cities.split(",")]
        cities = tuple(CITY_BY_NAME[n] for n in names)

    # Resolve to absolute so the cache is always at the same path.
    cache_dir = Path(args.cache_dir).resolve()
    out_dir   = Path(args.out).resolve()

    print(f"[assemble] cache dir : {cache_dir}")
    print(f"[assemble] output dir: {out_dir}")

    cfg = FetchConfig(
        cache_dir=str(cache_dir),
        census_year=args.census_year,
        fhwa_year=args.fhwa_year,
        cities=cities,
    )

    if args.clear_acs_cache:
        from .http_cache import CachedSession
        sess = CachedSession(str(cache_dir), user_agent=cfg.user_agent,
                             namespace="census")
        n = sess.clear_namespace()
        print(f"[assemble] cleared {n} ACS cache entries from {cache_dir / 'census'}")

    assemble(
        cfg, out_dir,
        skip_acs=args.skip_acs,
        skip_fhwa=args.skip_fhwa,
        exposure_weighted_hdv=args.exposure_weighted_hdv,
    )


if __name__ == "__main__":
    main()
