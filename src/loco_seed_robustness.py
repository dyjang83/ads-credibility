"""
Seed-robustness of the leave-one-city-out comparison.

At seven claims the per-seed LOCO totals are noisy, so a single MCMC seed is
not a sound basis for ranking the kernels. This script repeats the full
leave-one-city-out evaluation across N seeds for all five models and reports,
per model, the mean total predictive log-likelihood, its Monte-Carlo standard
deviation, and how often each is the best of the four random-effects models.
These are the numbers reported in Table 6 and the robustness paragraph.

Usage:
    python src/loco_seed_robustness.py --seeds 101 202 303 404 505 606 707 808 909
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, jax.numpy as jnp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_generator import DEPLOYED_CITIES, SOFTWARE_VERSIONS, CITY_ODD_FEATURES  # noqa
from models import (ads_credibility_model, ads_credibility_gp_model, InferenceConfig,  # noqa
                    run_inference, buhlmann_straub_closed_form, leave_one_city_out_predictive_logp)
from run_analysis import euclidean_similarity_from_features, _poisson_logp, _predict_indep_re_held  # noqa
import data_generator as dg


def loco_one_seed(ads, odd, S_emb, S_eu, S_hdv, seed, versions):
    tot = {"pool": 0.0, "indep": 0.0, "eu": 0.0, "hdv_only": 0.0, "learned": 0.0}
    for held in DEPLOYED_CITIES:
        train = [c for c in DEPLOYED_CITIES if c != held]
        a_tr = ads[ads.city.isin(train)].copy(); a_h = ads[ads.city == held]
        hx = odd.set_index("city").loc[held, CITY_ODD_FEATURES].to_numpy(float)
        he = float(a_h.exposure_million_miles.sum()); hc = int(a_h.claims.sum())
        c2i = {c: i for i, c in enumerate(train)}; v2i = {v: i for i, v in enumerate(versions)}
        a_tr["ci"] = a_tr.city.map(c2i); a_tr["vi"] = a_tr.version.map(v2i)
        d = a_tr.merge(odd, on="city"); X = jnp.asarray(d[CITY_ODD_FEATURES].to_numpy(float))
        td = dict(city_idx=jnp.asarray(d.ci.to_numpy()), version_idx=jnp.asarray(d.vi.to_numpy()), X=X,
                  exposure=jnp.asarray(d.exposure_million_miles.to_numpy(float)),
                  claims=jnp.asarray(d.claims.to_numpy(int)), n_cities=len(train),
                  n_versions=len(versions), n_features=X.shape[1])
        agg = a_tr.groupby("city").agg(claims=("claims", "sum"), miles=("exposure_million_miles", "sum")).reindex(train)
        bs = buhlmann_straub_closed_form(agg.claims.to_numpy(), agg.miles.to_numpy())
        tot["pool"] += _poisson_logp(hc, bs["X_bar"] * he)
        cfg = InferenceConfig(n_warmup=500, n_samples=600, n_chains=1, seed=seed)
        tot["indep"] += _predict_indep_re_held(run_inference(ads_credibility_model, td, cfg), hx, hc, he)
        for key, Smat in [("eu", S_eu), ("hdv_only", S_hdv), ("learned", S_emb)]:
            sub = Smat.loc[train + [held], train + [held]].to_numpy()
            Stt = sub[:len(train), :len(train)]; s2t = sub[-1, :len(train)]
            tdk = dict(td); tdk["L_chol"] = jnp.asarray(np.linalg.cholesky(Stt + 1e-6 * np.eye(len(train))))
            sk = run_inference(ads_credibility_gp_model, tdk, cfg)
            tot[key] += leave_one_city_out_predictive_logp(sk, hc, he, hx, similarity_to_train=s2t,
                                                           S_train_train=Stt, s_self=float(sub[-1, -1]))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[101, 202, 303, 404, 505, 606, 707, 808, 909])
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    ads = pd.read_csv(ROOT / "data/ads_events.csv")
    odd = pd.read_csv(ROOT / "data/city_odd_features.csv")
    S_emb = pd.read_csv(ROOT / "results/embedding/city_similarity.csv", index_col=0)
    S_eu = euclidean_similarity_from_features(odd, dg.ALL_CITIES)
    S_hdv = pd.read_csv(ROOT / "results/city_similarity_hdv_only.csv", index_col=0)

    # Derive the version vocabulary from the data so real NHTSA SGO labels work
    # without code edits (matches run_analysis.resolve_versions).
    versions = sorted(ads[ads.city.isin(DEPLOYED_CITIES)].version.dropna().unique().tolist())

    rows = []
    for sd in args.seeds:
        t = loco_one_seed(ads, odd, S_emb, S_eu, S_hdv, sd, versions); t["seed"] = sd
        rows.append(t)
        print(f"seed {sd:4d}: pool {t['pool']:.3f}  indep {t['indep']:.3f}  "
              f"eu {t['eu']:.3f}  hdv {t['hdv_only']:.3f}  learned {t['learned']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "loco_seed_robustness_full.csv", index=False)

    labels = {"pool": "Single pool", "indep": "Independent RE", "eu": "GP Euclidean",
              "hdv_only": "GP HDV-only", "learned": "GP learned"}
    re_keys = ["indep", "eu", "hdv_only", "learned"]
    best = df[re_keys].idxmax(axis=1).value_counts()
    summary = pd.DataFrame([
        {"model": labels[k], "mean_total_logp": df[k].mean(),
         "mc_sd": df[k].std(ddof=1),
         "best_among_RE_of_%d" % len(df): (None if k == "pool" else int(best.get(k, 0)))}
        for k in ["pool", "indep", "eu", "hdv_only", "learned"]
    ])
    summary.to_csv(args.out / "loco_table6_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
