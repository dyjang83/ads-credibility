"""
Calibration constants and paths for the decision layer.

Everything in src/decision/ is a SIMULATION calibrated to the fitted credibility
posterior: only four cities are actually deployed, so no policy is trained or
evaluated on real outcomes. The two calibration constants below are read from
the current data/results at import time rather than copied, so they track the
specification the main paper reports.

  SIGMA      posterior mean of the city random-effect scale sigma_c
             (results/diagnostics_gp.csv).
  BASE_RATE  pooled SGO crash frequency per million miles
             (data/ads_events.csv: total claims / total exposure).

Both values come from the same files src/power_analysis.py now reads, so the
decision layer and the power analysis share one calibration.

Every lambda here is an SGO-reportable-crash rate, matching Section 6 of the
paper. LIABILITY_RATE (Di Lillo et al. 2024b, ~0.43 per million miles) is kept
only as a reference scale for users who want to rescale to liability claims.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA = ROOT / "data"
RESULTS = ROOT / "results" / "decision"
FIGURES = ROOT / "figures" / "decision"
EMBED = ROOT / "results" / "embedding"


def _calibrate() -> tuple[float, float]:
    diag = pd.read_csv(ROOT / "results" / "diagnostics_gp.csv", index_col=0)
    sigma = float(diag.loc["sigma_c", "mean"])
    ads = pd.read_csv(DATA / "ads_events.csv")
    rate = float(ads["claims"].sum() / ads["exposure_million_miles"].sum())
    return sigma, rate


SIGMA, BASE_RATE = _calibrate()
MU_PRIOR_SD = 1.0            # prior sd of the log base rate, as in power_analysis
LIABILITY_RATE = 0.43        # reference only (Di Lillo et al. 2024b)
JITTER = 1e-6

# ---- cell-level modelling assumptions (NOT observed; see world.py) ----------
# log r_cell = CELL_HDV_COEF * z(log HDV cell frequency) + GP residual over the
# 32-d cell embedding with scale CELL_TAU and cosine bandwidth CELL_BANDWIDTH.
CELL_HDV_COEF = 0.30
CELL_TAU = 0.35
CELL_BANDWIDTH = 0.25
REGION_SIZE = 200            # candidate service-area cells per city (BFS ball)
# Kernel nugget for the "+nugget" belief kernels, S + NUGGET * I. The geofence's
# cell relativities add a city-level component independent across cities with
# sd ~0.22, i.e. variance ~0.28 sigma^2 (measured in the simulator); a kernel
# without it is overconfident for near-duplicate cities (cosine > 0.9).
NUGGET = 0.28

# ---- expansion environment defaults -----------------------------------------
POOL_SIZE = 24               # 7 real cities + 17 synthetic (embedding-matched)
HORIZON = 16                 # quarters
MILES_PER_QUARTER = 2.0      # million miles per active city per quarter at full region
LOADINGS = (0.0, 0.1, 0.25, 0.5)
RERATE_LOADING = 0.10        # credibility re-rating rule for active cities
COST_OF_CAPITAL = 0.06       # per quarter, on (VaR_99.5 - mean) of quarterly loss
CAPITAL_Q = 0.995
# Operator acceptance: P(accept) = sigmoid(ACCEPT_A + ACCEPT_B * (log lam_true - log rate)).
# The operator knows its own safety record better than the insurer, so an
# underpriced offer is accepted more often (adverse selection).
ACCEPT_A = 0.0
ACCEPT_B = 4.0
