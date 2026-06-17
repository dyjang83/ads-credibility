
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fetchers import config, assemble as assemble_mod
from src.fetchers.config import FetchConfig, CELL_FEATURE_COLUMNS, HDV_FREQUENCY_COLUMN
from src.fetchers.http_cache import CachedSession


def _synthetic_osm(cfg):
    rng = np.random.default_rng(1)
    rows = []
    for city in cfg.cities:
        # 12 fake cells per city, using that city's real centroid for valid H3
        import h3
        s, w, n, e = city.bbox
        clat, clon = (s + n) / 2, (w + e) / 2
        base = h3.latlng_to_cell(clat, clon, config.H3_RESOLUTION)
        disk = list(h3.grid_disk(base, 2))[:12]
        for c in disk:
            rows.append({
                "cell_id": c, "city": city.name,
                "road_length_arterial_km": rng.uniform(0.5, 4),
                "road_length_collector_km": rng.uniform(0.5, 4),
                "road_length_local_km": rng.uniform(1, 8),
                "intersection_density_per_km2": rng.uniform(20, 200),
                "signalized_intersection_fraction": rng.uniform(0, 1),
                "betweenness_centrality_mean": rng.uniform(0, 0.3),
            })
    return pd.DataFrame(rows)


def _synthetic_acs(osm):
    rng = np.random.default_rng(2)
    df = osm[["cell_id", "city"]].copy()
    df["population_density_per_km2"] = rng.uniform(500, 9000, len(df))
    df["pedestrian_commute_share"] = rng.uniform(0, 0.4, len(df))
    df["land_use_residential_share"] = rng.uniform(0.2, 0.9, len(df))
    df["land_use_commercial_share"] = rng.uniform(0, 0.4, len(df))
    return df


def _synthetic_fars(cfg, osm):
    rng = np.random.default_rng(3)
    df = osm[["cell_id", "city"]].copy()
    df["historical_crash_density_per_mile"] = rng.exponential(1.5, len(df))
    df["_fars_crash_count"] = rng.integers(0, 5, len(df))
    return df


def _synthetic_city_vmt(cfg):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "city": [c.name for c in cfg.cities],
        "urbanized_area": [f"{c.name} UA" for c in cfg.cities],
        "annual_vmt_millions": rng.uniform(12000, 90000, len(cfg.cities)),
        "fhwa_year": cfg.fhwa_year,
    })


def test_full_assemble_offline(monkeypatch_like=None):
    # manual monkeypatch (no pytest dependency)
    orig = (assemble_mod.fetch_osm_features,
            assemble_mod.fetch_acs_features,
            assemble_mod.fetch_fars_features,
            assemble_mod.fetch_fhwa_vmt)
    assemble_mod.fetch_osm_features = lambda cfg: _synthetic_osm(cfg)
    assemble_mod.fetch_acs_features = lambda cfg: _synthetic_acs(_synthetic_osm(cfg))
    assemble_mod.fetch_fars_features = lambda cfg, osm: _synthetic_fars(cfg, osm)
    assemble_mod.fetch_fhwa_vmt = lambda cfg: _synthetic_city_vmt(cfg)
    try:
        import json as _json
        with tempfile.TemporaryDirectory() as td:
            cfg = FetchConfig(cache_dir=str(Path(td) / "cache"))
            df = assemble_mod.assemble(cfg, Path(td))
            # schema exactly matches embedding.py expectation
            expected = ["cell_id", "city", *CELL_FEATURE_COLUMNS, HDV_FREQUENCY_COLUMN]
            assert list(df.columns) == expected, list(df.columns)
            # no NaNs in features
            assert not df[list(CELL_FEATURE_COLUMNS)].isna().any().any()
            # frequency strictly positive
            assert (df[HDV_FREQUENCY_COLUMN] > 0).all()
            # all 7 cities present
            assert df["city"].nunique() == 7
            # provenance written
            assert (Path(td) / "cell_features_provenance.json").exists()
            # FHWA side outputs written, but cell_features schema unchanged
            assert (Path(td) / "city_vmt.csv").exists()
            assert (Path(td) / "cell_exposure.csv").exists()
            prov = _json.loads((Path(td) / "cell_features_provenance.json").read_text())
            # default keeps the pre-FHWA calibration so existing results stand
            assert prov["hdv_calibration"] == "unweighted_cell_mean", prov["hdv_calibration"]
            assert "FHWA" in prov["exposure_source"]
            # per-cell exposure sums to each city's VMT
            expo = pd.read_csv(Path(td) / "cell_exposure.csv")
            vmt = _synthetic_city_vmt(cfg).set_index("city")["annual_vmt_millions"]
            by_city = expo.groupby("city")["hdv_exposure_million_miles"].sum()
            for city in by_city.index:
                assert abs(by_city[city] - vmt[city]) < 1e-3, (city, by_city[city], vmt[city])
            # output csv readable and same shape
            reread = pd.read_csv(Path(td) / "cell_features.csv")
            assert list(reread.columns) == expected
            assert len(reread) == len(df)
    finally:
        (assemble_mod.fetch_osm_features,
         assemble_mod.fetch_acs_features,
         assemble_mod.fetch_fars_features,
         assemble_mod.fetch_fhwa_vmt) = orig


def test_exposure_weighted_opt_in_changes_calibration_flag():
    """--exposure-weighted-hdv flips the recorded calibration; default does not."""
    orig = (assemble_mod.fetch_osm_features,
            assemble_mod.fetch_acs_features,
            assemble_mod.fetch_fars_features,
            assemble_mod.fetch_fhwa_vmt)
    assemble_mod.fetch_osm_features = lambda cfg: _synthetic_osm(cfg)
    assemble_mod.fetch_acs_features = lambda cfg: _synthetic_acs(_synthetic_osm(cfg))
    assemble_mod.fetch_fars_features = lambda cfg, osm: _synthetic_fars(cfg, osm)
    assemble_mod.fetch_fhwa_vmt = lambda cfg: _synthetic_city_vmt(cfg)
    try:
        import json as _json
        with tempfile.TemporaryDirectory() as td:
            cfg = FetchConfig(cache_dir=str(Path(td) / "cache"))
            df = assemble_mod.assemble(cfg, Path(td), exposure_weighted_hdv=True)
            prov = _json.loads((Path(td) / "cell_features_provenance.json").read_text())
            assert prov["hdv_calibration"] == "exposure_weighted", prov["hdv_calibration"]
            assert (df[HDV_FREQUENCY_COLUMN] > 0).all()
            assert df["city"].nunique() == 7
        # skip_fhwa path: no VMT, unweighted, no side files
        with tempfile.TemporaryDirectory() as td:
            cfg = FetchConfig(cache_dir=str(Path(td) / "cache"))
            assemble_mod.assemble(cfg, Path(td), skip_fhwa=True)
            prov = _json.loads((Path(td) / "cell_features_provenance.json").read_text())
            assert prov["hdv_calibration"] == "unweighted_cell_mean"
            assert not (Path(td) / "city_vmt.csv").exists()
    finally:
        (assemble_mod.fetch_osm_features,
         assemble_mod.fetch_acs_features,
         assemble_mod.fetch_fars_features,
         assemble_mod.fetch_fhwa_vmt) = orig


def test_cache_roundtrip_no_second_call():
    """CachedSession returns cached value without re-invoking transport."""
    with tempfile.TemporaryDirectory() as td:
        sess = CachedSession(td, user_agent="test", request_pause_s=0.0)
        calls = {"n": 0}

        # monkeypatch the underlying session.request to count invocations
        class FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"ok": True, "n": calls["n"]}

        def fake_request(method, url, params=None, data=None, timeout=None):
            calls["n"] += 1
            return FakeResp()

        sess._session.request = fake_request
        a = sess.get_json("https://example.org/x", params={"q": 1})
        b = sess.get_json("https://example.org/x", params={"q": 1})
        assert a == b
        assert calls["n"] == 1, f"expected 1 transport call, got {calls['n']}"
        # a different request key triggers a new call
        sess.get_json("https://example.org/x", params={"q": 2})
        assert calls["n"] == 2


def test_schema_matches_embedding_module():
    """The canonical feature list must equal embedding.py's feature_cols."""
    emb_path = Path(__file__).resolve().parents[1] / "src" / "embedding.py"
    text = emb_path.read_text()
    # crude but effective: every canonical feature appears in embedding.py
    for col in CELL_FEATURE_COLUMNS:
        assert col in text, f"{col} missing from embedding.py feature list"
    assert HDV_FREQUENCY_COLUMN in text


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
