"""
Experiment 1: geofence CO benchmark (learning-guided vs exact vs heuristics).

Train the GNN constructive policy on random connected sub-regions of six cities,
then evaluate every solver on 200-cell regions of the held-out city:

  greedy        multi-start contiguous growth
  greedy_ls     greedy + add/remove local search
  gnn_greedy    argmax GNN construction
  gnn_sample    best of 64 GNN samples + local search
  gnn_milp      MILP restricted to the GNN neighborhood (learning-guided MILP)
  ls_milp       control for gnn_milp: the same restricted MILP around the
                greedy+LS solution instead of the GNN samples
  milp          full MILP with a time limit (gives the dual bound)

Gaps are reported in percent of the region's premium at full service
(100 * (bound - J) / price), against both the MILP dual bound and the best
solution any method found.

Outputs (results/decision/):
  geofence_benchmark.csv         per (instance, method)
  geofence_summary.json          mean gaps / times by method
  geofence_gnn.pt                trained policy
  geofence_gnn_history.csv       training curve
  geofence_maps.csv              solutions on the held-out city's service region
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision.config import RESULTS  # noqa: E402
from decision.world import REAL_CITIES, random_region, service_region  # noqa: E402
from decision.geofence import (random_instance, make_instance, greedy_solve,  # noqa: E402
                               greedy_ls_solve, milp_solve, is_feasible)
from decision.geofence_gnn import (GeofencePolicy, train, gnn_greedy, gnn_sample,  # noqa: E402
                                   gnn_milp, neighborhood)


KAPPAS = (0.0, 0.5, 1.0, 2.0)


def ls_milp(inst, time_limit, radius=2):
    """Restricted-MILP control: neighborhood of the greedy+LS geofence."""
    import time
    t0 = time.perf_counter()
    x0, i0 = greedy_ls_solve(inst)
    x, info = milp_solve(inst, time_limit=time_limit, allowed=neighborhood(inst, x0, radius))
    if x is None or info["obj"] < i0["obj"]:
        x, info = x0, i0
    return x, {"time": time.perf_counter() - t0, "obj": info["obj"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="Denver")
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=24)
    ap.add_argument("--milp-time", type=float, default=60.0)
    ap.add_argument("--gnn-milp-time", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="run only these methods and merge into the existing CSV")
    args = ap.parse_args()
    if args.quick:
        args.n_train, args.epochs, args.n_test = 48, 1, 4
        args.milp_time, args.gnn_milp_time = 10.0, 5.0
    torch.set_num_threads(4)
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = "_quick" if args.quick else ""
    rng = np.random.default_rng(args.seed)

    train_cities = [c for c in REAL_CITIES if c != args.holdout]
    ckpt = RESULTS / f"geofence_gnn{tag}.pt"
    policy = GeofencePolicy()
    if ckpt.exists() and not args.retrain:
        policy.load_state_dict(torch.load(ckpt))
        print(f"loaded {ckpt}")
    else:
        train_insts = []
        for i in range(args.n_train):
            c = train_cities[i % len(train_cities)]
            reg = random_region(c, int(rng.integers(60, 201)), rng)
            train_insts.append(random_instance(reg, rng))
        ref = [greedy_ls_solve(inst)[1]["obj"] for inst in train_insts]
        hist = train(policy, train_insts, epochs=args.epochs, seed=args.seed, ref_obj=ref)
        torch.save(policy.state_dict(), ckpt)
        pd.DataFrame(hist).to_csv(RESULTS / f"geofence_gnn_history{tag}.csv", index=False)
    policy.eval()

    test_rng = np.random.default_rng(args.seed + 1)
    tests = [random_instance(random_region(args.holdout, 200, test_rng), test_rng)
             for _ in range(args.n_test)]
    for k, inst in enumerate(tests):              # stratify risk aversion
        inst.kappa = KAPPAS[k % len(KAPPAS)]
    methods = {
        "greedy": greedy_solve,
        "greedy_ls": greedy_ls_solve,
        "gnn_greedy": lambda inst: gnn_greedy(policy, inst),
        "gnn_sample": lambda inst: gnn_sample(policy, inst, 64),
        "gnn_milp": lambda inst: gnn_milp(policy, inst, time_limit=args.gnn_milp_time),
        "ls_milp": lambda inst: ls_milp(inst, args.gnn_milp_time),
        "milp": lambda inst: milp_solve(inst, time_limit=args.milp_time),
    }
    order = list(methods)
    if args.methods:
        methods = {k: v for k, v in methods.items() if k in args.methods}
    rows = []
    for k, inst in enumerate(tests):
        bound = np.nan
        for name, fn in methods.items():
            x, info = fn(inst)
            feas = x is not None and is_feasible(inst, x)
            if name == "milp":
                bound = info.get("bound", np.nan)
            rows.append({"instance": k, "method": name, "obj": info["obj"] if feas else np.nan,
                         "time_s": info["time"], "feasible": feas, "kappa": inst.kappa,
                         "price": inst.price, "min_share": inst.min_share})
        if "milp" in methods:
            for r in rows[-len(methods):]:
                r["milp_bound"] = bound
        print(f"[{k:2d}] kappa={inst.kappa:.1f} " + "  ".join(
            f"{r['method']}={r['obj']:+.4f}" for r in rows[-len(methods):]), flush=True)
    df = pd.DataFrame(rows)
    csv = RESULTS / f"geofence_benchmark{tag}.csv"
    if args.methods and csv.exists():
        old = pd.read_csv(csv)
        bounds = old[old.method == "milp"].set_index("instance")["milp_bound"]
        df["milp_bound"] = df["instance"].map(bounds)
        df = pd.concat([old[~old.method.isin(df.method.unique())], df], ignore_index=True)
        df = df[[c for c in df.columns if not c.startswith("gap_")]]
    best = df.groupby("instance")["obj"].transform("max")
    ub = np.maximum(df["milp_bound"].fillna(best), best)
    df["gap_best_pct"] = 100 * (best - df["obj"]) / df["price"]
    df["gap_bound_pct"] = 100 * (ub - df["obj"]) / df["price"]
    df.to_csv(csv, index=False)
    summ = (df.groupby("method")
              .agg(gap_best_pct=("gap_best_pct", "mean"), gap_bound_pct=("gap_bound_pct", "mean"),
                   time_s=("time_s", "mean"), feasible=("feasible", "mean"),
                   wins=("gap_best_pct", lambda g: int((g < 1e-6).sum())))
              .reindex([m for m in order if m in set(df.method)]))
    print(summ.round(4).to_string())
    out = {"holdout_city": args.holdout, "train_cities": train_cities,
           "n_test": args.n_test, "milp_time_limit_s": args.milp_time,
           "by_method": summ.reset_index().to_dict(orient="records")}
    (RESULTS / f"geofence_summary{tag}.json").write_text(json.dumps(out, indent=2))

    if args.methods:
        return
    # Solutions on the held-out city's default service region (for the map figure)
    reg = service_region(args.holdout)
    inst = make_instance(reg, np.random.default_rng(0))
    maps = {"cell_id": reg.cell_ids, "m": reg.m, "rbar": inst.scen.mean(0)}
    for name in ["greedy_ls", "gnn_sample", "milp"]:
        x, info = methods[name](inst)
        maps[name] = x.astype(int)
        print(f"map {name}: J={info['obj']:+.4f}")
    pd.DataFrame(maps).to_csv(RESULTS / f"geofence_maps{tag}.csv", index=False)


if __name__ == "__main__":
    main()
