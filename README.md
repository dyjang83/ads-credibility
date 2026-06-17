# ads-credibility

Reproducible code base for the paper **"Credibility-Weighted Pricing of Autonomous Vehicle Liability Under ODD Shifts"**.

The paper proposes a hierarchical Bayesian credibility framework for pricing autonomous driving system (ADS) liability when an operator deploys into a new city or releases a new software version. Two ideas drive the framework:

1. A hierarchical Poisson GLM with random effects on city, software version, and city×version interactions, fit by NUTS in NumPyro.
2. A learned ODD-similarity matrix that replaces the implicit independence assumption on city random effects with a multivariate-normal prior whose covariance is a Gaussian-process kernel over a contrastive embedding of H3 cells.

This repo implements both, calibrates them on real NHTSA SGO data, and reproduces every empirical result in Section 6 of the paper end-to-end.

## What the paper proves and what this repo verifies

| Paper section | What it claims | What the code verifies |
|---|---|---|
| §3 (motivation)   | Classical Bühlmann–Straub gives moderate credibility weight on ADS city-aggregate data (0.12–0.46) but near-zero weight at the per-(city, version, period) cell level | `buhlmann_straub_closed_form()` in `models.py`; output in `results/buhlmann_straub_closed_form.csv` |
| §4 (model)        | Hierarchical Bayesian Poisson GLM with non-centered RE parameterization yields well-mixed posteriors | `ads_credibility_model()` in `models.py`; R-hat in `results/diagnostics_indep.csv` |
| §4.3 (limit case) | The hierarchical model reduces to Bühlmann–Straub when ODD-similarity matrix S → I and covariates → 0 | Sanity check in `run_analysis.py` §4.3 block; closed form in Appendix B |
| §5.2 (embedding)  | A contrastive embedding over H3 cells with HDV-frequency-based positive pairs separates urban vs. suburban arterial profiles; participation ratio 4.75, PC1+PC2 = 53.5% of between-city variance | `embedding.py`; cosine similarity matrix in `results/embedding/city_similarity.csv` |
| §5.3 (GP prior)   | Replacing independent city RE with α ~ N(0, σ² S) tightens predictions for new cities with high similarity to deployed ones | `ads_credibility_gp_model()` in `models.py`; `predict_new_city_with_S()` for new-city predictive draws |
| §6.2 (posterior estimates) | Observed ADS SGO frequency is ~5.59 per million miles; city-aggregate BS weights range from 0.12 (Austin) to 0.46 (Phoenix); both models converge (R-hat → 1.00); β covariate coefficients and τ_v are weakly identified | `results/benchmark_reproduction.json`; `results/diagnostics_indep.csv`, `results/diagnostics_gp.csv` |
| §6.3 (prospective) | Posterior CIs for new cities are tightest when ODD similarity to deployed cities is highest; Denver (primary neighbor Austin, S = 0.76) has the tightest interval; framework updates coherently as local experience accumulates | `results/prospective_new_cities.csv`, `results/prospective_after_first_million.csv` |
| §6.4 (comparison)  | Leave-one-city-out predictive log-likelihood across pooled, independent-RE, GP-Euclidean, GP-HDV-only, and GP-learned models; single pool dominates at current volume, RE model rankings are seed-dependent | `results/loco_comparison.csv`, `results/loco_seed_robustness_full.csv`, `results/loco_table6_summary.csv` |
| §6.4 / §5.2 (ablation) | The learned embedding has participation ratio 4.75, is not equivalent to an HDV-only kernel (Pearson r = 0.58), and is not equivalent to the Euclidean kernel (r = 0.64); HDV-only and Euclidean are nearly collinear (r = 0.95) | `results/ablation_summary.json` |
| §6.5 (power analysis) | Claim-volume power curve: at 648 claims median LOCO advantage reaches +1.65 nats vs. independent-RE; number-of-cities power curve: 80% detection power vs. Euclidean kernel requires ~12 deployed cities; learned kernel never wins under null DGP | `src/power_analysis.py`; `results/power_analysis.csv`, `results/power_analysis_summary.json` |
| §7 (sensitivity)  | Prospective medians are robust to the random-effect scale prior over {0.3, 0.5, 1.0} | `results/prior_sensitivity.csv` |

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
│   ├── prior_sensitivity.py           §7 random-effect-scale prior sweep
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
python tests/test_fetchers.py              # 18 unit tests, offline
python tests/test_assemble_integration.py  #  3 integration + cache tests, offline
```

The integration suite monkeypatches the network fetchers with fixtures and
proves the assembled output is schema- and dtype-identical across runs.

## Headline empirical numbers (NHTSA SGO data, June 2025–April 2026)

| Quantity | Value | Paper reference |
|---|---|---|
| SGO window | June 2025–April 2026 (5 quarterly periods) | §6.1 |
| Total SGO crashes (4 metros) | 648 | §6.1, Table 1 |
| Software versions observed | 3 (5G-v9, 5G-v10, 6G-v10) | §6.1 |
| Total ADS exposure (4 metros) | ~116 M rider-only miles | §6.1 |
| Overall ADS SGO frequency | 5.59 crashes / M miles | §6.1, Table 1 |
| ADS vs. HDV reduction (Di Lillo benchmark) | 88–92% | §6.2 |
| BS credibility Z range (city aggregate) | 0.12 (Austin) – 0.46 (Phoenix) | §6.2, Table 3 |
| Worst R-hat (both models) | 1.00 | §6.2 |
| Min bulk ESS (GP model) | 1,735 of 3,000 draws | §6.2 |
| Prospective median λ — Miami | 0.680 / M miles (95% CI: 0.045–11.345) | §6.3, Table 4 |
| Prospective median λ — Boston | 1.176 / M miles (95% CI: 0.143–8.276) | §6.3, Table 4 |
| Prospective median λ — Denver | 0.981 / M miles (95% CI: 0.203–3.935) | §6.3, Table 4 |
| Embedding participation ratio | 4.75 | §5.2 |
| Embedding PC1+PC2 variance share | 53.5% | §5.2 |
| Learned vs. HDV-only kernel correlation | r = 0.58 | §5.2 |
| Learned vs. Euclidean kernel correlation | r = 0.64 | §5.2 |
| LOCO total log-lik (pool / indep / GP-eu / GP-hdv / GP-learned) | −133.16 / −28.19 / −30.23 / −29.57 / −30.17 | §6.4, Table 6 |
| Power analysis — median LOCO advantage at 648 claims | +1.65 nats vs. indep-RE; +0.75 nats vs. Euclidean | §6.5 |
| Number-of-cities for 80% detection power (vs. Euclidean) | ~12 deployed cities | §6.5, Fig. 9 |





Reproduce end-to-end:
```bash
python -m src.fetchers.assemble                # real features + FHWA exposure
python -m src.fetchers.sgo_fetcher --out data --operator 'Waymo LLC'
python src/embedding.py --data-dir data --out results/embedding --epochs 30
python src/run_analysis.py                     # refit + prospective + canonical LOCO
python src/ablation_hdv_kernel.py              # HDV-only kernel + ablation summary
python src/loco_seed_robustness.py             # Table 6 across-seed summary
python src/power_analysis.py                   # Section 6.5 power analysis
python src/prior_sensitivity.py                # Section 7 sensitivity
python src/make_figures.py                     # all figures, incl. fig_08, fig_09
```# ads-credibility
