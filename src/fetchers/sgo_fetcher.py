"""
NHTSA Standing General Order (SGO 2021-01) ADS crash-report fetcher.

What SGO is and is not
-----------------------
SGO is a crash-level CSV: every reportable crash involving an ADS-equipped
vehicle since 2021, one row per report (updated reports appear as higher
Report Version numbers under the same Report ID). Each row carries operator,
ADS-engagement status, incident date, city/state, injury severity, and a
coarse automation-version string.

SGO is a *numerator-only* source: it has crash counts but no vehicle-miles.
Exposure denominators come from the operator's published mileage disclosures
(EXPOSURE_WINDOW_MILLION_MILES below). The default values are calibrated to
Waymo's published milestones bracketing the Jun 2025 - Apr 2026 SGO window:
170.7M rider-only miles through Dec 2025 (Waymo Safety Impact hub, Mar 2026)
and 200M+ miles through Feb 2026, with the Q4 2025 per-city distribution
(Phoenix 40%, LA 22%, SF 31%, Austin 6%) applied to the estimated 116M four-
city miles in the window.


Network: required on cold cache (one GET to a static NHTSA URL).
Downloaded CSV bytes are cached so reruns are offline.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEPLOYED_METROS: tuple[str, ...] = ("San Francisco", "Phoenix", "Los Angeles", "Austin")

SGO_ADS_URL = (
    "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/"
    "SGO-2021-01_Incident_Reports_ADS.csv"
)

# Per-metro rider-only miles for the SGO window (Jun 2025 - Apr 2026).
# Derived from Waymo milestones: ~86M cumulative at Jun 2025 (interpolated
# from 100M in Jul 2025), ~215M at Apr 2026 (interpolated from 200M+ in
# Feb 2026). Four-city fraction of national total: ~90%. Per-city shares from
# Waymo's published cumulative rider-only miles through December 2025 (Safety
# Impact hub): San Francisco 53.52M, Phoenix 68.613M, Los Angeles 37.857M,
# Austin 10.722M of 170.712M total -> shares 31.4% / 40.2% / 22.2% / 6.3%.
# Total four-city window miles: ~116M.
EXPOSURE_WINDOW_MILLION_MILES: dict[str, float] = {
    "San Francisco": 36.37,   # 31.4% of 116 M
    "Phoenix":       46.62,   # 40.2% of 116 M
    "Los Angeles":   25.72,   # 22.2% of 116 M
    "Austin":         7.29,   #  6.3% of 116 M
}

# Municipality -> study metro (lower-cased keys, matched case-insensitively).
# Municipalities not listed here (e.g. Atlanta, Dallas, Miami, Washington) are
# intentionally excluded: the deployed set is restricted to the four metros with
# stable full-window coverage and reliable exposure denominators. See Section 6.1.
METRO_BY_MUNICIPALITY: dict[str, str] = {
    # San Francisco Bay Area
    "san francisco":       "San Francisco",
    "mountain view":       "San Francisco",
    "palo alto":           "San Francisco",
    "daly city":           "San Francisco",
    "sunnyvale":           "San Francisco",
    "san jose":            "San Francisco",
    "san mateo":           "San Francisco",
    "redwood city":        "San Francisco",
    "fremont":             "San Francisco",
    "brisbane":            "San Francisco",
    "burlingame":          "San Francisco",
    "san bruno":           "San Francisco",
    "south san francisco": "San Francisco",
    # Phoenix metro
    "phoenix":         "Phoenix",
    "tempe":           "Phoenix",
    "scottsdale":      "Phoenix",
    "mesa":            "Phoenix",
    "chandler":        "Phoenix",
    "gilbert":         "Phoenix",
    "glendale":        "Phoenix",
    "peoria":          "Phoenix",
    "guadalupe":       "Phoenix",
    "paradise valley": "Phoenix",
    # Los Angeles metro
    "los angeles":   "Los Angeles",
    "santa monica":  "Los Angeles",
    "inglewood":     "Los Angeles",
    "culver city":   "Los Angeles",
    "venice":        "Los Angeles",
    "beverly hills": "Los Angeles",
    "lennox":        "Los Angeles",
    "long beach":    "Los Angeles",
    "burbank":       "Los Angeles",
    "marina del rey": "Los Angeles",
    "west hollywood": "Los Angeles",
    # Austin metro
    "austin":       "Austin",
    "round rock":   "Austin",
    "pflugerville": "Austin",
    "cedar park":   "Austin",
    "del valle":    "Austin",
}

# Required state per metro to guard against cross-state municipality collisions
# (e.g. Glendale AZ vs Glendale CA).
METRO_STATE: dict[str, str] = {
    "San Francisco": "CA",
    "Los Angeles":   "CA",
    "Phoenix":       "AZ",
    "Austin":        "TX",
}

_VERSION_REDACTED = re.compile(r"redacted|confidential|^\s*$", re.I)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SGOFetchConfig:
    cache_dir: str = "data/cache"
    url: str = SGO_ADS_URL
    user_agent: str = "ads-credibility-research/1.0 (academic)"
    timeout_s: int = 120
    # Keep only rows where the ADS was engaged at the time of the crash.
    ads_engaged_only: bool = True
    # Filter to a single operator. None keeps all operators; most analyses
    # want a single operator so the exposure denominator is well defined.
    operator: str | None = None


# ---------------------------------------------------------------------------
# Version canonicalization
# ---------------------------------------------------------------------------

def _canonicalize_version(raw: object) -> str:
    """Normalize the SGO Automation Feature Version string.

    The live SGO feed contains many typographic variants of a small number of
    distinct Waymo software labels. This function:
      - Maps redacted/blank/placeholder values to 'unspecified'
      - Removes erroneous leading digit prefixes ('35th' -> '5th')
      - Corrects the 'Genearation' misspelling
      - Inserts a comma before 'Version' when missing ('ADS Version 10')
      - Normalises whitespace around commas
      - Restores missing 'ADS' in 'Nth Generation, Version X'
      - Fills bare 'Nth Generation ADS' (no version number) with Version 10
      - Replaces placeholder dashes ('Nth Generation ADS, -') with Version 10

    After canonicalization the 653 Waymo records in the Jun 2025-Apr 2026
    window collapse to three distinct labels:
      '5th Generation ADS, Version 9'   (rare, <2% of records)
      '5th Generation ADS, Version 10'  (dominant, ~98%)
      '6th Generation ADS, Version 10'  (rare, <1%)
    """
    v = str(raw).strip() if raw is not None else ""
    if not v or _VERSION_REDACTED.search(v):
        return "unspecified"

    # Remove erroneous leading digit ('35th' -> '5th', '45th' -> '5th')
    v = re.sub(r"^[0-9]+(?=\d(?:st|nd|rd|th))", "", v)
    # Fix misspelling
    v = v.replace("Genearation", "Generation")
    # Insert comma before 'Version' when missing ('ADS Version 10')
    v = re.sub(r"(ADS)\s+(Version)", r"\1, \2", v)
    # Normalise whitespace around comma
    v = re.sub(r"\s*,\s*", ", ", v)
    # Insert 'ADS' when missing ('Nth Generation, Version X')
    v = re.sub(r"(Generation),\s*(Version)", r"Generation ADS, \2", v)
    # Fill bare 'Nth Generation ADS' with default Version 10
    v = re.sub(r"^(\d+(?:st|nd|rd|th) Generation ADS)$", r"\1, Version 10", v)
    # Replace placeholder dash
    v = re.sub(r",\s*-\s*$", ", Version 10", v)
    # Collapse runs of whitespace
    v = re.sub(r"\s+", " ", v).strip()
    return v


# ---------------------------------------------------------------------------
# Download (cached)
# ---------------------------------------------------------------------------

def _download_csv_bytes(cfg: SGOFetchConfig) -> bytes:
    cache = Path(cfg.cache_dir) / "sgo"
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / "sgo_ads_incident_reports.csv"
    if cached.exists():
        return cached.read_bytes()
    resp = requests.get(
        cfg.url,
        headers={"User-Agent": cfg.user_agent},
        timeout=cfg.timeout_s,
    )
    resp.raise_for_status()
    cached.write_bytes(resp.content)
    return resp.content


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _clean(s: object) -> str:
    return str(s).strip() if s is not None else ""


def _is_engaged(row: pd.Series) -> bool:
    status = _clean(row.get("Engagement Status", "")).lower()
    if status:
        return "not engaged" not in status and "engaged" in status
    return _clean(row.get("Automation System Engaged?", "")).upper() == "Y"


def _incident_month(row: pd.Series) -> "pd.Timestamp | None":
    """Return the first-of-month Timestamp for the incident date."""
    rm = _clean(row.get("Report Month", ""))
    ry = _clean(row.get("Report Year", ""))
    if rm.isdigit() and ry.isdigit():
        try:
            return pd.Timestamp(int(ry), int(rm), 1)
        except ValueError:
            pass
    inc = _clean(row.get("Incident Date", ""))
    if inc and "PERSONALLY IDENTIFIABLE" not in inc.upper():
        for fmt in ("%b-%Y", "%B-%Y", "%Y-%m", "%m/%Y"):
            try:
                return pd.to_datetime(inc, format=fmt).replace(day=1)
            except (ValueError, TypeError):
                continue
        try:
            return pd.to_datetime(inc).replace(day=1)
        except (ValueError, TypeError):
            pass
    return None


def _map_metro(city_raw: object, state_raw: object) -> "str | None":
    city = _clean(city_raw).lower()
    state = _clean(state_raw).upper()
    metro = METRO_BY_MUNICIPALITY.get(city)
    if metro is None:
        return None
    required_state = METRO_STATE.get(metro)
    if required_state and state and state != required_state:
        return None
    return metro


# ---------------------------------------------------------------------------
# Main fetch / aggregate
# ---------------------------------------------------------------------------

def fetch_sgo_long(cfg: SGOFetchConfig) -> pd.DataFrame:
    """Fetch and parse the SGO ADS CSV; return one clean row per incident.

    Deduplicates to the latest Report Version per Report ID.
    Columns: report_id, operator, engaged, incident_month, municipality,
    state, metro, version (canonicalized), severity.
    """
    raw = _download_csv_bytes(cfg)
    df = pd.read_csv(io.BytesIO(raw), dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Keep latest version per Report ID
    if {"Report ID", "Report Version"}.issubset(df.columns):
        df["_v"] = pd.to_numeric(df["Report Version"], errors="coerce").fillna(0)
        df = (
            df.sort_values("_v")
            .drop_duplicates(subset=["Report ID"], keep="last")
            .drop(columns="_v")
        )

    rows = []
    for _, r in df.iterrows():
        engaged = _is_engaged(r)
        if cfg.ads_engaged_only and not engaged:
            continue
        operator = _clean(r.get("Reporting Entity", ""))
        if cfg.operator and operator.lower() != cfg.operator.lower():
            continue
        metro = _map_metro(r.get("City", ""), r.get("State", ""))
        rows.append({
            "report_id":      _clean(r.get("Report ID", "")),
            "operator":       operator,
            "engaged":        engaged,
            "incident_month": _incident_month(r),
            "municipality":   _clean(r.get("City", "")),
            "state":          _clean(r.get("State", "")),
            "metro":          metro,
            "version":        _canonicalize_version(
                                  r.get("Automation Feature Version", "")),
            "severity":       _clean(r.get("Highest Injury Severity Alleged", "")),
        })
    return pd.DataFrame(rows)


def build_ads_events(
    long_df: pd.DataFrame,
    exposure: "dict[str, float] | None" = None,
    period_freq: str = "Q",
) -> pd.DataFrame:
    """Aggregate the long SGO table into the ads_events.csv schema.

    Output columns: city, version, period, exposure_million_miles, claims,
    claims_source.

    Periods are zero-based integer indices over observed quarters (0 =
    earliest observed quarter). Exposure is the per-metro window total split
    evenly across the (version, period) cells that actually occur for that
    metro -- the only defensible split without per-quarter mileage data.
    """
    exposure = exposure or EXPOSURE_WINDOW_MILLION_MILES
    df = long_df.dropna(subset=["metro", "incident_month"]).copy()
    df = df[df["metro"].isin(DEPLOYED_METROS)]
    if df.empty:
        return pd.DataFrame(columns=[
            "city", "version", "period",
            "exposure_million_miles", "claims", "claims_source",
        ])

    df["quarter"] = df["incident_month"].dt.to_period(period_freq)
    quarters = sorted(df["quarter"].unique())
    qidx = {q: i for i, q in enumerate(quarters)}
    df["period"] = df["quarter"].map(qidx)

    grp = (
        df.groupby(["metro", "version", "period"])
        .size()
        .reset_index(name="claims")
        .rename(columns={"metro": "city"})
    )
    cells_per_metro = grp.groupby("city")["claims"].transform("size")
    grp["exposure_million_miles"] = (
        grp["city"].map(exposure).astype(float) / cells_per_metro
    )
    grp["claims_source"] = "nhtsa_sgo"
    return (
        grp[["city", "version", "period",
             "exposure_million_miles", "claims", "claims_source"]]
        .sort_values(["city", "version", "period"])
        .reset_index(drop=True)
    )


def write_ads_events(
    cfg: SGOFetchConfig,
    out_dir: "str | Path",
    period_freq: str = "Q",
    exposure: "dict[str, float] | None" = None,
) -> dict:
    """End-to-end: fetch -> canonicalize -> aggregate -> write -> return summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long_df = fetch_sgo_long(cfg)
    events = build_ads_events(long_df, exposure=exposure, period_freq=period_freq)

    long_df.to_csv(out_dir / "ads_events_sgo_long.csv", index=False)
    events.to_csv(out_dir / "ads_events.csv", index=False)

    in_metro = long_df[long_df["metro"].isin(DEPLOYED_METROS)]
    return {
        "claims_source":              "nhtsa_sgo",
        "sgo_url":                    cfg.url,
        "operator_filter":            cfg.operator,
        "ads_engaged_only":           cfg.ads_engaged_only,
        "total_reports_kept":         int(len(long_df)),
        "reports_in_deployed_metros": int(len(in_metro)),
        "deployed_metro_claim_counts": (
            in_metro["metro"].value_counts().to_dict()
            if not in_metro.empty else {}
        ),
        "versions_observed":          sorted(
            long_df["version"].dropna().unique().tolist()
        ),
        "ads_events_rows":            int(len(events)),
        "ads_events_total_claims":    int(events["claims"].sum())
                                      if not events.empty else 0,
        "ads_events_total_exposure_million_miles": (
            float(events["exposure_million_miles"].sum())
            if not events.empty else 0.0
        ),
        "exposure_denominators_used": {
            k: round(v, 2)
            for k, v in (exposure or EXPOSURE_WINDOW_MILLION_MILES).items()
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(
        description="Fetch real NHTSA SGO ADS counts and write ads_events.csv."
    )
    p.add_argument("--out",                 default="data", type=Path)
    p.add_argument("--cache-dir",           default="data/cache")
    p.add_argument("--operator",            default="Waymo LLC",
                   help="Reporting entity to filter to. "
                        "Empty string keeps all operators.")
    p.add_argument("--include-not-engaged", action="store_true",
                   help="Keep crashes where the ADS was not engaged.")
    p.add_argument("--period-freq",         default="Q",
                   help="pandas period frequency string (default: Q).")
    args = p.parse_args()

    cfg = SGOFetchConfig(
        cache_dir=args.cache_dir,
        operator=(args.operator or None),
        ads_engaged_only=not args.include_not_engaged,
    )
    summary = write_ads_events(cfg, args.out, period_freq=args.period_freq)
    print(json.dumps(summary, indent=2, default=str))