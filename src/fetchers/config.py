"""
Shared configuration for the real-data feature fetchers.

The fetchers (OSM, ACS, FARS) all produce features at the H3-cell level for a
fixed set of cities, then the assembler joins them into the cell_features.csv
schema that src/embedding.py consumes. This module is the single source of
truth for: which cities, what geographic extent each covers, the H3 resolution,
and the canonical feature column order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# H3 grid
# ---------------------------------------------------------------------------

H3_RESOLUTION = 8

# Approximate edge length and area at res 8, for documentation only.
H3_RES8_AVG_AREA_KM2 = 0.737327598


# ---------------------------------------------------------------------------
# Cities
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class City:
    """A city's identity plus the metadata each fetcher needs.

    bbox is (south_lat, west_lon, north_lat, east_lon) in WGS84 degrees and
    bounds the operational area we model. These are deliberately tight around
    the ADS operating territory rather than the full metro/MSA, because the
    embedding should describe where the vehicles actually drive. Widen them if
    you want to model the full metropolitan footprint.

    state_fips / county_fips identify the counties whose Census tracts and FARS
    records overlap the bbox. A city may span several counties (e.g. NYC); list
    all of them. The ACS fetcher pulls all tracts in these counties and then
    spatially filters to the bbox.
    """
    name: str                         # human-readable, must match across all CSVs
    slug: str                         # filesystem / cell-id prefix, no spaces
    bbox: tuple[float, float, float, float]
    state_fips: str
    county_fips: tuple[str, ...]      # one or more 3-digit county codes
    deployed: bool                    # True = ADS operates here; False = hypothetical


# The four deployed cities match Di Lillo et al. (2024b); the three
# hypothetical cities (Boston, Denver, Miami) are the prospective-pricing
# targets. County FIPS verified against the Census 2020 gazetteer.
CITIES: tuple[City, ...] = (
    City(
        name="San Francisco", slug="San_Francisco",
        bbox=(37.708, -122.515, 37.833, -122.357),
        state_fips="06", county_fips=("075",), deployed=True,
    ),
    City(
        name="Phoenix", slug="Phoenix",
        bbox=(33.290, -112.325, 33.690, -111.925),
        state_fips="04", county_fips=("013",), deployed=True,
    ),
    City(
        name="Los Angeles", slug="Los_Angeles",
        bbox=(33.950, -118.500, 34.180, -118.200),
        state_fips="06", county_fips=("037",), deployed=True,
    ),
    City(
        name="Austin", slug="Austin",
        bbox=(30.150, -97.850, 30.450, -97.650),
        state_fips="48", county_fips=("453",), deployed=True,
    ),
    City(
        name="Boston", slug="Boston",
        bbox=(42.300, -71.150, 42.400, -71.000),
        state_fips="25", county_fips=("025",), deployed=False,
    ),
    City(
        name="Denver", slug="Denver",
        bbox=(39.610, -105.060, 39.800, -104.840),
        state_fips="08", county_fips=("031",), deployed=False,
    ),
    City(
        name="Miami", slug="Miami",
        bbox=(25.700, -80.300, 25.860, -80.130),
        state_fips="12", county_fips=("086",), deployed=False,
    ),
)

CITY_BY_NAME = {c.name: c for c in CITIES}
CITY_BY_SLUG = {c.slug: c for c in CITIES}


# ---------------------------------------------------------------------------
# FHWA urbanized-area mapping
# ---------------------------------------------------------------------------
# FHWA Highway Statistics reports vehicle-miles-traveled (VMT) by Census
# urbanized area, not by city. The urbanized-area names carry state suffixes
# and multi-anchor hyphenation that shift slightly between vintages, so we map
# each of our cities to a list of candidate substrings; the fetcher matches an
# FHWA row if ANY candidate is a case-insensitive substring of the row's area
# name. List the most specific candidate first. Verified against the 2022
# Highway Statistics HM-72 urbanized-area table; if a future vintage renames an
# area, add the new spelling here rather than editing the fetcher.
URBANIZED_AREA_BY_CITY: dict[str, tuple[str, ...]] = {
    "San Francisco": ("san francisco--oakland",),
    "Phoenix":       ("phoenix--mesa",),
    "Los Angeles":   ("los angeles--long beach--anaheim", "los angeles-long beach-anaheim"),
    "Austin":        ("austin, tx",),
    "Boston":        ("boston, ma--nh--ri", "boston, ma-nh-ri", "boston, ma"),
    "Denver":        ("denver--aurora",),
    "Miami":         ("miami, fl",),
}


# ---------------------------------------------------------------------------
# Canonical output schema
# ---------------------------------------------------------------------------
# This MUST stay in sync with the feature_cols list in src/embedding.py.
# The assembler validates against it before writing cell_features.csv.
CELL_FEATURE_COLUMNS: tuple[str, ...] = (
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
)

# Which fetcher is responsible for each feature, for the assembler's provenance
# report and for clear error messages when a source is missing.
FEATURE_PROVENANCE: dict[str, str] = {
    "road_length_arterial_km": "osm",
    "road_length_collector_km": "osm",
    "road_length_local_km": "osm",
    "intersection_density_per_km2": "osm",
    "signalized_intersection_fraction": "osm",
    "betweenness_centrality_mean": "osm",
    "population_density_per_km2": "acs",
    "pedestrian_commute_share": "acs",
    "land_use_residential_share": "acs",
    "land_use_commercial_share": "acs",
    "historical_crash_density_per_mile": "fars",
}

# The frequency target column used to build positive pairs in the contrastive
# objective. Produced by the HDV-claims fetcher (or, in the public-data
# analogue, approximated from FARS fatal-crash density scaled to a claims proxy).
HDV_FREQUENCY_COLUMN = "hdv_claim_freq_per_million_miles"


# ---------------------------------------------------------------------------
# OSM functional-class -> our three road buckets
# ---------------------------------------------------------------------------
# OSM highway tags are many; we collapse them into the arterial/collector/local
# buckets the model uses. Motorways are included under "arterial" since within
# an urban ODD they carry arterial-like through traffic; tune to taste.
OSM_HIGHWAY_BUCKETS: dict[str, str] = {
    "motorway": "arterial",
    "motorway_link": "arterial",
    "trunk": "arterial",
    "trunk_link": "arterial",
    "primary": "arterial",
    "primary_link": "arterial",
    "secondary": "collector",
    "secondary_link": "collector",
    "tertiary": "collector",
    "tertiary_link": "collector",
    "unclassified": "local",
    "residential": "local",
    "living_street": "local",
    "service": "local",
    "road": "local",
}


@dataclass
class FetchConfig:
    """Runtime knobs shared by all fetchers."""
    cache_dir: str = "data/cache"
    # Overpass is rate-limited and occasionally slow; be polite.
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    overpass_timeout_s: int = 180
    request_pause_s: float = 1.5          # between successive remote calls
    max_retries: int = 4
    # Census API key is optional for <500 calls/day but recommended. Set via
    # the CENSUS_API_KEY environment variable; None falls back to keyless mode.
    census_year: int = 2022              # ACS 5-year vintage
    # FHWA Highway Statistics vintage for urbanized-area VMT (HM-72 table).
    # The 5-year ACS vintage and the FHWA vintage need not match; both default
    # to 2022, the latest year for which all sources were finalized at writing.
    fhwa_year: int = 2022
    user_agent: str = "ads-credibility-research/1.0 (academic; contact: author)"
    cities: tuple[City, ...] = field(default_factory=lambda: CITIES)
