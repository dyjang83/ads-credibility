"""Real-data feature fetchers for the ADS credibility pipeline.

Modules:
    config        shared cities, H3 settings, canonical schema
    http_cache    cached retrying HTTP client
    osm_fetcher   road-network features (Overpass)
    acs_fetcher   demographic features (Census ACS + TIGERweb)
    fars_fetcher  crash-density feature (NHTSA FARS)
    fhwa_fetcher  urbanized-area VMT exposure denominators (FHWA Highway Statistics)
    assemble      joins the cell sources into cell_features.csv and writes the
                  FHWA exposure layer (city_vmt.csv, cell_exposure.csv)

Run `python -m src.fetchers.assemble --out data` to build the real dataset
(requires network on first run; cached thereafter).
"""
