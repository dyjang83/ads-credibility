# Real-data feature fetchers

## What it produces

A `data/cell_features.csv` with exactly the schema `src/embedding.py` consumes:

| column | source | meaning |
|---|---|---|
| `road_length_arterial_km` | OSM | arterial road km in the H3 cell |
| `road_length_collector_km` | OSM | collector road km |
| `road_length_local_km` | OSM | local/residential road km |
| `intersection_density_per_km2` | OSM | shared-node intersections / cell area |
| `signalized_intersection_fraction` | OSM | traffic-signal nodes / intersections |
| `betweenness_centrality_mean` | OSM | mean node betweenness over the cell |
| `population_density_per_km2` | ACS | areal-interpolated tract population |
| `pedestrian_commute_share` | ACS | walk-to-work share (B08301) |
| `land_use_residential_share` | ACS | 1-unit structures / all units (B25024) |
| `land_use_commercial_share` | ACS | complement proxy (see caveat) |
| `historical_crash_density_per_mile` | FARS | fatal crashes / road mile, 2018–2022 |
| `hdv_claim_freq_per_million_miles` | FARS + anchor | calibrated frequency target |

Cities (4 deployed + 3 hypothetical): San Francisco, Phoenix, Los Angeles,
Austin, Boston, Denver, Miami. Defined with bounding boxes and county FIPS in
`config.py`; edit there to change extent or add cities.

## Data sources and access

| Fetcher | Source | Endpoint | Auth |
|---|---|---|---|
| `osm_fetcher` | OpenStreetMap | Overpass API | none |
| `acs_fetcher` | Census ACS 5-yr + TIGERweb | api.census.gov, tigerweb.geo.census.gov | optional `CENSUS_API_KEY` |
| `fars_fetcher` | NHTSA FARS | static.nhtsa.gov | none |
| `fhwa_fetcher` | FHWA Highway Statistics HM-72 | fhwa.dot.gov | none |

The first three build the per-cell embedding features; `fhwa_fetcher` builds the
per-city **exposure denominator** (urbanized-area VMT) used to turn HDV claim
*counts* into *frequencies*. FHWA gives real exposure, not a claim count — the
absolute HDV claim level is still anchored to the published aggregate. All fetchers require outbound
network on the **first** run and cache to `data/cache/` thereafter.

> If you set a Census API key: `export CENSUS_API_KEY=your_key_here`. Without
> one, the keyless tier (500 calls/day) is more than enough for 7 cities.

## How to run

```bash
pip install -r src/fetchers/requirements.txt

# all seven cities (cold run hits the network; ~10–20 min, mostly Overpass)
python -m src.fetchers.assemble --out data

# a subset while iterating
python -m src.fetchers.assemble --out data --cities "San Francisco,Boston"

# fetch the FHWA exposure denominator on its own
python -m src.fetchers.fhwa_fetcher --year 2022 --out data/city_vmt.csv
```

`assemble` writes three files: `cell_features.csv` (the embedding input,
unchanged schema), `city_vmt.csv` (per-city FHWA VMT), and `cell_exposure.csv`
(per-cell VMT, split by road mileage). By default the HDV-frequency target is
anchored to the **unweighted** cell mean, so the emitted `cell_features.csv` is
identical to the pre-FHWA pipeline and existing results stand. Pass
`--exposure-weighted-hdv` to instead anchor to the exposure-weighted mean using
the real VMT denominator (the statistically correct target); this changes the
embedding's frequency target, so re-run `embedding.py` and `run_analysis.py`
afterwards. Use `--skip-fhwa` to skip the VMT fetch entirely.

Then run the existing pipeline:

```bash
python src/embedding.py    --data-dir data --out results/embedding --epochs 30
python src/run_analysis.py --data-dir data --embed-dir results/embedding --out results
python src/make_figures.py --data-dir data --embed-dir results/embedding --results-dir results --out figures
```
