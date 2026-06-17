

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants matching paper conventions
# ---------------------------------------------------------------------------

DEPLOYED_CITIES = ["San Francisco", "Phoenix", "Los Angeles", "Austin"]
HYPOTHETICAL_CITIES = ["Miami", "Boston", "Denver"]
ALL_CITIES = DEPLOYED_CITIES + HYPOTHETICAL_CITIES

CITY_EXPOSURE_SHARES = {
    "San Francisco": 0.28,
    "Phoenix": 0.67,
    "Los Angeles": 0.03,
    "Austin": 0.02,
}
TOTAL_ADS_MILES_MILLIONS = 25.3

SOFTWARE_VERSIONS = ["v1.0", "v1.1", "v2.0", "v2.1", "v3.0"]
N_PERIODS = 8  # quarterly observations

# Features at the H3 cell level
CELL_FEATURES = [
    "road_length_arterial_km",
    "road_length_collector_km",
    "road_length_local_km",
    "intersection_density_per_km2",
    "signalized_intersection_fraction",
    "betweenness_centrality_mean",
    "population_density_per_km2",
    "pedestrian_commute_share",
    "land_use_residential_share",
    "land_use_commercial_share",
    "historical_crash_density_per_mile",
]

# City-level fixed-effect covariates
CITY_ODD_FEATURES = [
    "arterial_share",
    "intersection_density",
    "signalized_fraction",
    "weather_rain_fraction",
    "weather_fog_fraction",
    "operating_hours_night_share",
]


# ---------------------------------------------------------------------------
# Ground-truth parameters
# ---------------------------------------------------------------------------

@dataclass
class TrueParameters:
    """
    Ground-truth parameters used by the generator. Stored so the inference
    pipeline can recover them and we can compute calibration metrics.
    """
    beta0: float                 # baseline log-frequency for ADS (per million miles)
    beta: list                   # fixed-effect coefficients on CITY_ODD_FEATURES
    alpha: dict                  # city -> random effect
    gamma: dict                  # version -> random effect
    delta: dict                  # (city, version) -> interaction
    tau_c: float
    tau_v: float
    tau_cv: float
    hdv_multiplier: float        # how much higher HDV frequency is vs. ADS

    def to_json(self, path: Path) -> None:
        payload = asdict(self)
        # convert tuple keys to strings for JSON
        payload["delta"] = {f"{c}||{v}": val for (c, v), val in self.delta.items()}
        path.write_text(json.dumps(payload, indent=2))


def _draw_true_parameters(rng: np.random.Generator) -> TrueParameters:
    """Sample the ground-truth parameters once and freeze them."""

    # ADS baseline ~ 0.4 claims per million miles (rough scale; PD + BI combined)
    # log scale: log(0.4) ~ -0.92
    beta0 = -0.92

    # Coefficients on CITY_ODD_FEATURES (centered / scaled inputs assumed)
    beta = [
        0.10,   # arterial_share -> more arterial = slightly more claims
        0.20,   # intersection density -> denser intersections -> more claims
        -0.10,  # signalized fraction -> protective
        0.15,   # rain fraction
        0.08,   # fog fraction
        0.12,   # night-hours share
    ]

    tau_c, tau_v, tau_cv = 0.25, 0.15, 0.08

    alpha = {c: float(rng.normal(0.0, tau_c)) for c in DEPLOYED_CITIES}
    gamma = {v: float(rng.normal(0.0, tau_v)) for v in SOFTWARE_VERSIONS}
    delta = {
        (c, v): float(rng.normal(0.0, tau_cv))
        for c in DEPLOYED_CITIES
        for v in SOFTWARE_VERSIONS
    }

    # HDV claims are ~ 7x ADS in the matched comparison (the paper reports
    # ~88-92% reductions; 1 / (1 - 0.88) ≈ 8.3, 1 / (1 - 0.92) ≈ 12.5;
    # we pick a middle-ish value in log space)
    hdv_multiplier = 7.5

    return TrueParameters(
        beta0=beta0, beta=beta,
        alpha=alpha, gamma=gamma, delta=delta,
        tau_c=tau_c, tau_v=tau_v, tau_cv=tau_cv,
        hdv_multiplier=hdv_multiplier,
    )


# ---------------------------------------------------------------------------
# City-level ODD covariates
# ---------------------------------------------------------------------------

def _city_odd_features() -> pd.DataFrame:
    """
    City-level ODD covariates. Values are illustrative but chosen to reflect
    qualitative differences described in Section 6.4 of the paper:

      - Boston/SF: high intersection density, high pedestrian
      - Phoenix/Austin: lower density, more arterial
      - LA: arterial-heavy, high VMT
      - Miami: mixed grid + arterial, weather exposure
      - Denver: similar to Phoenix profile
    """
    rows = [
        # city,             arterial, dens, sig,  rain, fog,  night
        ("San Francisco",   0.45,    160,  0.55, 0.20, 0.08, 0.18),
        ("Phoenix",         0.62,     90,  0.45, 0.02, 0.01, 0.22),
        ("Los Angeles",     0.55,    110,  0.50, 0.05, 0.03, 0.20),
        ("Austin",          0.58,    100,  0.42, 0.10, 0.02, 0.20),
        ("Miami",           0.52,    120,  0.48, 0.28, 0.04, 0.22),
        ("Boston",          0.40,    175,  0.60, 0.18, 0.06, 0.18),
        ("Denver",          0.60,     95,  0.46, 0.06, 0.02, 0.21),
    ]
    df = pd.DataFrame(rows, columns=["city"] + CITY_ODD_FEATURES)
    # The Section 4 model assumes covariates have been centered/scaled.
    # We z-score across the union of deployed + hypothetical cities.
    for col in CITY_ODD_FEATURES:
        df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
    return df


# ---------------------------------------------------------------------------
# H3-cell-level features (synthetic OSM + ACS + FARS proxy)
# ---------------------------------------------------------------------------

def _generate_cell_features(rng: np.random.Generator, n_cells_per_city: int = 400) -> pd.DataFrame:
    """
    Synthetic H3-cell-level feature table. Each city has a characteristic
    feature distribution; the contrastive embedding should recover that
    structure from the features alone.

    We also generate a *latent* per-cell HDV claim frequency that is a smooth
    function of the cell features (plus city-level shifts). The embedding is
    later trained to make cells with similar HDV frequencies close in
    embedding space, per Section 5.2.
    """
    # City prototypes in (intersection_density, signalized_fraction,
    # population_density, pedestrian_share, residential_share, crash_density) space
    prototypes = {
        # city: mean vector
        "San Francisco":  [180, 0.58, 7000, 0.30, 0.55, 4.0],
        "Phoenix":        [ 90, 0.42, 1100, 0.04, 0.65, 2.0],
        "Los Angeles":    [120, 0.50, 3200, 0.10, 0.60, 2.8],
        "Austin":         [105, 0.40, 1500, 0.07, 0.62, 2.1],
        "Miami":          [130, 0.50, 4500, 0.12, 0.55, 3.0],
        "Boston":         [200, 0.62, 6200, 0.32, 0.50, 3.8],
        "Denver":         [100, 0.45, 1800, 0.08, 0.60, 2.3],
    }
    # Within-city covariance — cells inside a city span urban/suburban variation
    rows = []
    for city, mu in prototypes.items():
        mu = np.asarray(mu, dtype=float)
        # variance scales with mean to give plausible spread
        sd = np.array([40.0, 0.08, 1500.0, 0.10, 0.10, 0.8])
        for _ in range(n_cells_per_city):
            x = mu + rng.normal(0.0, sd)
            x = np.maximum(x, 1e-3)
            x[1] = float(np.clip(x[1], 0.05, 0.95))
            x[3] = float(np.clip(x[3], 0.01, 0.80))
            x[4] = float(np.clip(x[4], 0.10, 0.90))

            # Build a richer feature vector matching CELL_FEATURES order
            intersection_density = x[0]
            signal_frac = x[1]
            pop_density = x[2]
            ped_share = x[3]
            resid_share = x[4]
            crash_density = x[5]

            # Road lengths: longer arterial in suburban / lower-density cells
            arterial_km = rng.gamma(2.0, 0.5) + (1.0 / (1.0 + pop_density / 2000.0)) * 1.5
            collector_km = rng.gamma(2.0, 0.4)
            local_km = rng.gamma(2.5, 0.6) + (pop_density / 5000.0)

            betweenness = rng.beta(2.0, 5.0)
            commercial_share = float(np.clip(1.0 - resid_share - rng.uniform(0.1, 0.3), 0.05, 0.85))

            rows.append({
                "cell_id": f"{city.replace(' ', '_')}_{len(rows):06d}",
                "city": city,
                "road_length_arterial_km": arterial_km,
                "road_length_collector_km": collector_km,
                "road_length_local_km": local_km,
                "intersection_density_per_km2": intersection_density,
                "signalized_intersection_fraction": signal_frac,
                "betweenness_centrality_mean": betweenness,
                "population_density_per_km2": pop_density,
                "pedestrian_commute_share": ped_share,
                "land_use_residential_share": resid_share,
                "land_use_commercial_share": commercial_share,
                "historical_crash_density_per_mile": crash_density,
            })
    df = pd.DataFrame(rows)

    # Latent per-cell HDV log-frequency: smooth function of features
    # We use a hand-picked weight vector that mixes density, pedestrians,
    # signal protection, and historical crashes
    z = (
        0.30 * _z(df["intersection_density_per_km2"])
        + 0.25 * _z(df["pedestrian_commute_share"])
        - 0.15 * _z(df["signalized_intersection_fraction"])
        + 0.20 * _z(df["historical_crash_density_per_mile"])
        + 0.10 * _z(df["population_density_per_km2"])
        + 0.05 * _z(df["land_use_commercial_share"])
    )
    # HDV freq centered around ~3 claims per million miles
    log_lambda_hdv = np.log(3.0) + 0.4 * z + rng.normal(0.0, 0.05, size=len(df))
    df["hdv_claim_freq_per_million_miles"] = np.exp(log_lambda_hdv)

    return df


def _z(s: pd.Series) -> np.ndarray:
    a = s.to_numpy()
    return (a - a.mean()) / (a.std() + 1e-9)


# ---------------------------------------------------------------------------
# ADS event counts (NHTSA SGO-style)
# ---------------------------------------------------------------------------

def _generate_ads_events(
    rng: np.random.Generator,
    params: TrueParameters,
    city_odd: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate Poisson-distributed ADS events per (city, version, period).

    Each city's total exposure is set from CITY_EXPOSURE_SHARES * TOTAL_ADS_MILES.
    Within a city, exposure is allocated across software versions over time so
    that newer versions accumulate more recent exposure (versions roll out
    sequentially). Periods are quarters.
    """
    odd_lookup = city_odd.set_index("city")

    rows = []
    for city in DEPLOYED_CITIES:
        total_miles = CITY_EXPOSURE_SHARES[city] * TOTAL_ADS_MILES_MILLIONS

        # Distribute miles across (version, period). Simple model: each
        # version is active for a contiguous window of periods.
        # Earlier versions get less; later versions accumulate as fleet grows.
        version_period_weights = np.zeros((len(SOFTWARE_VERSIONS), N_PERIODS))
        for vi, _v in enumerate(SOFTWARE_VERSIONS):
            # version vi is active from period vi to N_PERIODS - 1
            for t in range(vi, N_PERIODS):
                version_period_weights[vi, t] = (t - vi + 1)
        version_period_weights /= version_period_weights.sum()

        miles_grid = total_miles * version_period_weights

        for vi, v in enumerate(SOFTWARE_VERSIONS):
            for t in range(N_PERIODS):
                miles = miles_grid[vi, t]
                if miles <= 0:
                    continue

                x = odd_lookup.loc[city, CITY_ODD_FEATURES].to_numpy(dtype=float)
                log_lambda = (
                    params.beta0
                    + x @ np.asarray(params.beta)
                    + params.alpha[city]
                    + params.gamma[v]
                    + params.delta[(city, v)]
                )
                lam = np.exp(log_lambda)
                expected = lam * miles
                claims = rng.poisson(expected)

                rows.append({
                    "city": city,
                    "version": v,
                    "period": t,
                    "exposure_million_miles": miles,
                    "claims": int(claims),
                    "expected_claims_true": float(expected),
                    "lambda_true": float(lam),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HDV claim experience at the cell level
# ---------------------------------------------------------------------------

def _generate_hdv_claims(
    rng: np.random.Generator,
    cell_features: pd.DataFrame,
    params: TrueParameters,
) -> pd.DataFrame:
    """
    HDV claims per cell, Poisson with rate from the latent frequency and a
    reasonable exposure assumption (each cell ~ 100k vehicle-miles per year
    aggregated over a 5-year HDV history window, in millions of miles).
    """
    # Exposure: 5 years * gamma-distributed per-cell intensity
    exposure = 0.5 + rng.gamma(2.0, 0.5, size=len(cell_features))  # ~ 1.5 M miles
    rate = cell_features["hdv_claim_freq_per_million_miles"].to_numpy() * exposure
    claims = rng.poisson(rate)
    out = cell_features[["cell_id", "city"]].copy()
    out["exposure_million_miles"] = exposure
    out["claims"] = claims
    out["hdv_freq_per_million_miles_true"] = cell_features["hdv_claim_freq_per_million_miles"].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_all(out_dir: Path, seed: int = 20260525) -> dict:
    """
    Generate the complete synthetic dataset and write it to out_dir.
    Returns a dict summarizing what was written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    params = _draw_true_parameters(rng)

    city_odd = _city_odd_features()
    cell_features = _generate_cell_features(rng)
    ads_events = _generate_ads_events(rng, params, city_odd)
    hdv_claims = _generate_hdv_claims(rng, cell_features, params)

    city_odd.to_csv(out_dir / "city_odd_features.csv", index=False)
    cell_features.to_csv(out_dir / "cell_features.csv", index=False)
    ads_events.to_csv(out_dir / "ads_events.csv", index=False)
    hdv_claims.to_csv(out_dir / "hdv_claims_by_cell.csv", index=False)
    params.to_json(out_dir / "true_parameters.json")

    summary = {
        "seed": seed,
        "n_cities_deployed": len(DEPLOYED_CITIES),
        "n_cities_hypothetical": len(HYPOTHETICAL_CITIES),
        "n_versions": len(SOFTWARE_VERSIONS),
        "n_periods": N_PERIODS,
        "n_cells": len(cell_features),
        "total_ads_miles_millions": float(ads_events["exposure_million_miles"].sum()),
        "total_ads_claims": int(ads_events["claims"].sum()),
        "total_hdv_miles_millions": float(hdv_claims["exposure_million_miles"].sum()),
        "total_hdv_claims": int(hdv_claims["claims"].sum()),
    }
    (out_dir / "data_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", type=Path)
    parser.add_argument("--seed", default=20260525, type=int)
    args = parser.parse_args()
    summary = generate_all(args.out, seed=args.seed)
    print(json.dumps(summary, indent=2))
