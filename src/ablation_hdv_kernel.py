"""
Ablation requested by reviewer (T5/M2): compare the learned 32-d contrastive
embedding kernel against a one-dimensional kernel built from HDV frequency
alone. If the embedding adds value over a 1-D HDV kernel, learned-vs-HDV
gap in LOCO log-likelihood and in pairwise similarity correlation should
be visible.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALL_CITIES = ["San Francisco", "Phoenix", "Los Angeles", "Austin",
              "Boston", "Denver", "Miami"]
DEPLOYED = ["San Francisco", "Phoenix", "Los Angeles", "Austin"]


def hdv_only_similarity(hdv_by_city: dict, all_cities: list) -> pd.DataFrame:
    """Squared-exponential kernel on a single feature: log HDV frequency.

    The bandwidth is set by the median heuristic, matching how the
    Euclidean and learned-embedding kernels are constructed in run_analysis.
    """
    f = np.array([np.log(hdv_by_city[c]) for c in all_cities]).reshape(-1, 1)
    diffs = f[:, None, :] - f[None, :, :]
    d2 = (diffs ** 2).sum(axis=-1)
    pair_d2 = d2[np.triu_indices_from(d2, k=1)]
    ell2 = float(np.median(pair_d2)) / (2.0 * np.log(2.0))
    S = np.exp(-d2 / (2.0 * max(ell2, 1e-6)))
    return pd.DataFrame(S, index=all_cities, columns=all_cities)


def main():
    repo = Path(__file__).resolve().parents[1]  # src/ -> repo root
    out = repo / "results"

    # ---- Build the HDV-only kernel using deployed-cell HDV frequencies ----
    df_hdv = pd.read_csv(repo / "data" / "hdv_claims_by_cell.csv")
    hdv_city = df_hdv.groupby("city").agg(
        claims=("claims", "sum"),
        exposure=("exposure_million_miles", "sum"),
    )
    hdv_city["freq"] = hdv_city["claims"] / hdv_city["exposure"]
    hdv_by_city = hdv_city["freq"].to_dict()

    S_hdv = hdv_only_similarity(hdv_by_city, ALL_CITIES)
    S_hdv.to_csv(out / "city_similarity_hdv_only.csv")
    print("HDV-only similarity matrix (full 7x7):")
    print(S_hdv.round(3))

    # ---- Compare against learned and Euclidean kernels ----
    S_learned = pd.read_csv(out / "embedding" / "city_similarity.csv", index_col=0)
    S_eu = pd.read_csv(out / "city_similarity_euclidean.csv", index_col=0)

    # Pairwise correlation of off-diagonals.
    # Reindex all matrices to the same canonical city order before extracting
    # the upper triangle — the learned embedding CSV may have a different row/
    # column order (alphabetical from groupby) than S_eu and S_hdv (which
    # follow ALL_CITIES order), so numeric triu indexing without alignment
    # would pair wrong cities.
    canonical = sorted(ALL_CITIES)
    S_learned_aligned = S_learned.reindex(index=canonical, columns=canonical)
    S_eu_aligned      = S_eu.reindex(index=canonical, columns=canonical)
    S_hdv_aligned     = S_hdv.reindex(index=canonical, columns=canonical)

    iu = np.triu_indices(len(canonical), k=1)
    learned_pairs = S_learned_aligned.to_numpy()[iu]
    eu_pairs      = S_eu_aligned.to_numpy()[iu]
    hdv_pairs     = S_hdv_aligned.to_numpy()[iu]

    corr_learned_hdv = float(np.corrcoef(learned_pairs, hdv_pairs)[0, 1])
    corr_learned_eu = float(np.corrcoef(learned_pairs, eu_pairs)[0, 1])
    corr_hdv_eu = float(np.corrcoef(hdv_pairs, eu_pairs)[0, 1])
    print()
    print(f"Pearson correlation (off-diagonal pairwise similarities):")
    print(f"  learned vs HDV-only kernel: {corr_learned_hdv:.3f}")
    print(f"  learned vs Euclidean:       {corr_learned_eu:.3f}")
    print(f"  HDV-only vs Euclidean:      {corr_hdv_eu:.3f}")

    # ---- LOCO log-likelihood for HDV-only kernel ----
    # We re-use the existing inference pipeline from run_analysis.py.
    import sys
    sys.path.insert(0, str(repo / "src"))
    import jax.numpy as jnp
    from models import ads_credibility_gp_model
    from run_analysis import (
        run_inference, InferenceConfig,
        leave_one_city_out_predictive_logp,
        CITY_ODD_FEATURES,
    )

    ads = pd.read_csv(repo / "data" / "ads_events.csv")
    city_odd = pd.read_csv(repo / "data" / "city_odd_features.csv")

    # Derive the version vocabulary from the data so real NHTSA SGO labels
    # ("5th Generation ADS, Version 10", ...) map correctly.  Importing the
    # synthetic SOFTWARE_VERSIONS constant would make every real row return
    # NaN, which then becomes float32 — the source of the TypeError.
    _versions = sorted(ads["version"].dropna().unique().tolist())

    rows = []
    for held in DEPLOYED:
        train_cities = [c for c in DEPLOYED if c != held]
        ads_train = ads[ads["city"].isin(train_cities)].copy()
        ads_held = ads[ads["city"] == held]
        held_exposure = float(ads_held["exposure_million_miles"].sum())
        held_claims = int(ads_held["claims"].sum())
        held_x = city_odd.set_index("city").loc[held, CITY_ODD_FEATURES].to_numpy(dtype=float)

        city_to_idx = {c: i for i, c in enumerate(train_cities)}
        ads_train["city_idx"] = ads_train["city"].map(city_to_idx)
        version_to_idx = {v: i for i, v in enumerate(_versions)}
        ads_train["version_idx"] = ads_train["version"].map(version_to_idx)

        df_train = ads_train.merge(city_odd, on="city")
        X = jnp.asarray(df_train[CITY_ODD_FEATURES].to_numpy(dtype=float))

        S_sub = S_hdv.loc[train_cities + [held], train_cities + [held]].to_numpy()
        S_train_train = S_sub[:len(train_cities), :len(train_cities)]
        s_to_train = S_sub[-1, :len(train_cities)]
        L = np.linalg.cholesky(S_train_train + 1e-6 * np.eye(len(train_cities)))

        train_data = {
            "city_idx": jnp.asarray(df_train["city_idx"].to_numpy(), dtype=jnp.int32),
            "version_idx": jnp.asarray(df_train["version_idx"].to_numpy(), dtype=jnp.int32),
            "X": X,
            "exposure": jnp.asarray(df_train["exposure_million_miles"].to_numpy(dtype=float)),
            "claims": jnp.asarray(df_train["claims"].to_numpy(dtype=int)),
            "n_cities": len(train_cities),
            "n_versions": len(_versions),
            "n_features": X.shape[1],
            "L_chol": jnp.asarray(L),
        }
        cfg = InferenceConfig(n_warmup=600, n_samples=800, n_chains=1)
        samples = run_inference(ads_credibility_gp_model, train_data, cfg)
        logp = leave_one_city_out_predictive_logp(
            samples, held_claims, held_exposure, held_x,
            similarity_to_train=s_to_train,
            S_train_train=S_train_train,
            s_self=float(S_sub[-1, -1]),
        )
        rows.append({"held_city": held, "logp_gp_hdv_only": logp})
        print(f"Held out {held}: logp_gp_hdv_only = {logp:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "loco_hdv_only.csv", index=False)
    print()
    print(f"HDV-only kernel total LOCO logp: {df['logp_gp_hdv_only'].sum():.4f}")

    # ---- Effective dimensionality of the learned embedding ----
    emb = np.load(out / "embedding" / "cell_embeddings.npy")
    emb_c = emb - emb.mean(axis=0)
    cov = emb_c.T @ emb_c / (emb_c.shape[0] - 1)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = eigvals[eigvals > 1e-12]
    pr = float((eigvals.sum() ** 2) / (eigvals ** 2).sum())
    cum = np.cumsum(eigvals) / eigvals.sum()
    print()
    print(f"Embedding effective dimensionality (participation ratio): {pr:.3f}")
    print(f"PC1 cumulative variance: {cum[0]*100:.1f}%")
    print(f"PC1+PC2 cumulative variance: {cum[1]*100:.1f}%")
    print(f"PC1..PC5 cumulative variance: {cum[4]*100:.1f}%")

    summary = {
        "embedding_participation_ratio": pr,
        "embedding_cum_var_pc1": float(cum[0]),
        "embedding_cum_var_pc2": float(cum[1]),
        "embedding_cum_var_pc5": float(cum[4]),
        "pearson_learned_hdv": corr_learned_hdv,
        "pearson_learned_eu": corr_learned_eu,
        "pearson_hdv_eu": corr_hdv_eu,
        "loco_hdv_only_total": float(df["logp_gp_hdv_only"].sum()),
        "loco_hdv_only_per_city": {
            r["held_city"]: float(r["logp_gp_hdv_only"]) for _, r in df.iterrows()
        },
    }
    with open(out / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Wrote {out / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
