# ads-credibility

Reproducible code base for the paper **"Credibility-Weighted Pricing of Autonomous Vehicle Liability Under ODD Shifts"**.

The paper proposes a hierarchical Bayesian credibility framework for pricing autonomous driving system (ADS) liability when an operator deploys into a new city or releases a new software version. Two ideas drive the framework:

1. A hierarchical Poisson GLM with random effects on city, software version, and city×version interactions, fit by NUTS in NumPyro.
2. A learned ODD-similarity matrix that replaces the implicit independence assumption on city random effects with a multivariate-normal prior whose covariance is a Gaussian-process kernel over a contrastive embedding of H3 cells.

## Repo layout

```
ads_credibility/
├── README.md                          this file
├── data/                              CSVs produced by the fetcher pipeline
│   ├── ads_events.csv                 NHTSA-SGO ADS counts: city × version × period
│   ├── hdv_claims_by_cell.csv         HLDI-style HDV claim experience per H3 cell
│   ├── city_odd_features.csv          City-level covariates (z-scored)
│   ├── cell_features.csv              OSM/ACS/FARS features per H3 cell
│   ├── city_vmt.csv                   FHWA HM-72 urbanized-area VMT by city
│   ├── cell_exposure.csv              Cell-level exposure weights
│   └── cell_features_provenance.json  source/imputation report per feature
├── src/
│   ├── embedding.py                   PyTorch contrastive encoder + training loop
│   ├── models.py                      NumPyro hierarchical models + BS closed form
│   ├── run_analysis.py                end-to-end driver
│   ├── make_figures.py                figure generation
│   ├── ablation_hdv_kernel.py         §6.4 HDV-only-kernel baseline + embedding dimensionality
│   ├── prior_sensitivity.py           random-effect-scale prior sweep
│   ├── loco_seed_robustness.py        across-seed summary
│   ├── power_analysis.py              §6.5 claim-volume and number-of-cities power curves
│   └── fetchers/                      public-data pipeline (OSM / ACS / FARS / FHWA)
│       ├── README.md                  fetcher docs, sources, run instructions
│       ├── config.py                  cities, H3 settings, canonical schema
│       ├── http_cache.py              cached retrying HTTP client
│       ├── osm_fetcher.py             road-network features (Overpass)
│       ├── acs_fetcher.py             demographic features (Census ACS + TIGERweb)
│       ├── fars_fetcher.py            crash-density feature (NHTSA FARS)
│       ├── fhwa_fetcher.py            VMT exposure denominator (FHWA HM-72)
│       ├── sgo_fetcher.py             ADS crash counts (NHTSA SGO 2021-01)
│       ├── assemble.py                joins sources into cell_features.csv
│       └── requirements.txt           light deps (no GDAL/geopandas)
├── tests/
│   ├── fixtures.py                    hand-built API-response fixtures
│   ├── test_fetchers.py               18 unit tests on parse/transform logic
│   └── test_assemble_integration.py    3 end-to-end + cache tests
├── results/                           all JSON/CSV outputs cited in §6 and §7
├── figures/                           9 PNGs referenced in the paper
└── notebooks/
    └── empirical_walkthrough.ipynb    narrated walkthrough of §6 results
```

## How to reproduce

```bash
# 1. Install requirements
pip install numpyro torch jax arviz pandas numpy matplotlib scikit-learn

# 2. Fetch real data and assemble cell features + FHWA exposure (~10–20 min cold cache)
pip install -r src/fetchers/requirements.txt
python -m src.fetchers.assemble --out data

# 3. Fetch SGO crash counts
python -m src.fetchers.sgo_fetcher --out data --operator 'Waymo LLC'

# 4. Train the contrastive embedding (~10 s)
python src/embedding.py --data-dir data --out results/embedding --epochs 30

# 5. Run the full Bayesian analysis (~5–8 min on a 4-core CPU)
python src/run_analysis.py --data-dir data --embed-dir results/embedding --out results

# 6. Reviewer-requested ablations (§6.4 and §7)
python src/ablation_hdv_kernel.py
python src/loco_seed_robustness.py
python src/prior_sensitivity.py

# 7. Power analysis (§6.5)
python src/power_analysis.py

# 8. Generate figures (~5 s)
python src/make_figures.py --data-dir data --embed-dir results/embedding \
    --results-dir results --out figures
```

See [`src/fetchers/README.md`](src/fetchers/README.md) for data sources, an
optional Census API key, per-feature provenance, and methodological caveats.

## Tests

```bash
python tests/test_fetchers.py              # 18 unit tests
python tests/test_assemble_integration.py  #  3 integration + cache tests
```

The integration suite monkeypatches the network fetchers with fixtures and
proves the assembled output is schema- and dtype-identical across runs.

Reproduce end-to-end:
```bash
python -m src.fetchers.assemble          
python -m src.fetchers.sgo_fetcher --out data --operator 'Waymo LLC'
python src/embedding.py --data-dir data --out results/embedding --epochs 30
python src/run_analysis.py     
python src/ablation_hdv_kernel.py       
python src/loco_seed_robustness.py     
python src/power_analysis.py           
python src/prior_sensitivity.py               
python src/make_figures.py                
```# ads-credibility
