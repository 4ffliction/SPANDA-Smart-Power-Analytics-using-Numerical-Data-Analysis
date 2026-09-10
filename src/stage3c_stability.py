"""
SPANDA Stage 3c - CLUSTER STABILITY VALIDATION (read-only w.r.t. raw data,
Stage-2 artifacts, Stage-3b artifacts, and the official K=2 model; every
decision logged to spanda_out/audit/stability_log.txt)

Purpose
-------
Determine whether the K=2 / K=3 / K=4 clustering structures are REPRODUCIBLE
across random K-Means initializations, rather than artifacts of one seed.
ONLY the K-Means init/fit varies. Everything upstream is frozen:

  1. building-day observation unit      (Stage-2 table, unchanged)
  2. feature engineering                (spanda.model.prepare_modelling_matrix)
  3. transformations                    (spanda.model.apply_transforms)
  4. StandardScaler                     (frozen scaler.joblib)
  5. PCA representation                 (3 PCs, re-derived exactly as Stage 3)
  6. number of retained PCs             (3, from modelling_report.json)

What this stage does NOT do
---------------------------
* does not modify raw data, spanda/model.py, preprocessing decisions,
  clustered_building_days.csv, scaler.joblib, PCA artifacts, or any Stage-3b
  output (frozen artifacts are hash-verified before and after the run)
* does not change the official model (K=2, random_state=42, n_init=10)
* does not promote K=3/K=4
* no causal claims anywhere; ARI (permutation-invariant) is used, never raw
  label equality

Outputs (all NEW files; nothing overwritten)
--------------------------------------------
spanda_out/modelling/cluster_stability_summary.csv
spanda_out/modelling/cluster_stability_runs.csv
spanda_out/modelling/cluster_stability_report.json
spanda_out/figures/cluster_stability_ari.png
spanda_out/figures/cluster_stability_silhouette.png
spanda_out/figures/cluster_stability_comparison.png
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from spanda.config import (
    AUDIT_DIR, FIGURES_DIR, MODELLING_DIR, PROCESSED_DIR, PROJECT_ROOT,
    get_logger,
)
import spanda.model as _spanda_model
from spanda.model import apply_transforms, prepare_modelling_matrix

log = get_logger("spanda.stability", AUDIT_DIR / "stability_log.txt")

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
SEEDS = [0, 7, 21, 42, 100, 123, 999, 2026]
K_LIST = [2, 3, 4]
N_INIT = 10                      # identical to Stage 3
INIT = "k-means++"               # identical to Stage 3 (sklearn default)
REFERENCE_SEED = 42              # Stage-3 official random_state

# Frozen artifacts that must remain byte-identical (verified before/after)
FROZEN_ARTIFACTS = [
    PROCESSED_DIR / "building_day_features.csv",
    PROCESSED_DIR / "hourly_power_clean.parquet",
    MODELLING_DIR / "clustered_building_days.csv",
    MODELLING_DIR / "scaler.joblib",
    MODELLING_DIR / "pca_features.parquet",
    MODELLING_DIR / "pca_loadings.csv",
    MODELLING_DIR / "pca_interpretation.json",
    MODELLING_DIR / "modelling_report.json",
    MODELLING_DIR / "cluster_validation_report.json",
    MODELLING_DIR / "cluster_interpretation.csv",
    MODELLING_DIR / "building_cluster_distribution.csv",
    MODELLING_DIR / "cluster_profile.csv",
    MODELLING_DIR / "potential_anomalies.csv",
    MODELLING_DIR / "lecture_dominance.json",
    MODELLING_DIR / "cluster_labels.json",
    MODELLING_DIR / "k_sensitivity.csv",
    MODELLING_DIR / "cluster_by_meter_by_date.csv",
    MODELLING_DIR / "cluster_centroids.csv",
    MODELLING_DIR / "standardized_features.parquet",
    MODELLING_DIR / "feature_names.json",
    # the code package lives in the agent workspace; hash the imported module
    Path(_spanda_model.__file__),
]
RAW_DATA_FILES = [
    PROJECT_ROOT / f for f in
    ["acad_build_mains.csv", "boys_hostel_mains.csv", "boys_hostel_ups.csv",
     "facilities_build_mains.csv", "girls_hostel_mains.csv",
     "girls_hostel_ups.csv", "lecture_build_mains.csv",
     "library_build_mains.csv", "mess_build_mains.csv"]]


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_frozen_artifacts() -> dict[str, str]:
    """Hash raw data, frozen modelling artifacts, and spanda/model.py."""
    out = {}
    for p in RAW_DATA_FILES + FROZEN_ARTIFACTS:
        out[str(p)] = _sha256(p)
    return out


def verify_frozen(before: dict[str, str], after: dict[str, str]) -> None:
    diff = [k for k, v in before.items() if after.get(k) != v]
    log.info("=" * 66)
    log.info("FINAL VERIFICATION - frozen artifacts byte-identical")
    log.info(f"  files hash-verified: {len(before)}")
    if diff:
        for k in diff:
            log.error(f"  MODIFIED DURING RUN: {k}")
        raise RuntimeError(f"{len(diff)} frozen artifacts were modified: "
                           f"{diff}")
    log.info("  result: ALL frozen artifacts byte-identical "
             "(raw data, Stage-2 outputs, Stage-3b outputs, spanda/model.py)")


# ---------------------------------------------------------------------------
# 1. recreate the frozen Stage-3 PCA representation
# ---------------------------------------------------------------------------
def recreate_representation() -> tuple[dict, pd.DataFrame, dict]:
    log.info("=" * 66)
    log.info("STEP 1 - recreate the frozen Stage-3 PCA representation")

    clustered = pd.read_csv(MODELLING_DIR / "clustered_building_days.csv")
    report = json.loads((MODELLING_DIR / "modelling_report.json")
                        .read_text(encoding="utf-8"))

    # same feature engineering + transformations as Stage 3 (imported functions)
    X, ids, feat_names = prepare_modelling_matrix()
    Zt = apply_transforms(X)

    # explicit row-alignment proof against the frozen clustered table
    ok_meter = (ids["meter"].to_numpy() == clustered["meter"].to_numpy()).all()
    ok_date = (ids["date_iso"].to_numpy()
               == clustered["obs_date"].to_numpy()).all()
    log.info(f"  row alignment vs clustered_building_days.csv "
             f"(meter + obs_date): {bool(ok_meter and ok_date)}")
    if not (ok_meter and ok_date):
        raise RuntimeError("row alignment FAILED - stopping rather than "
                           "continuing with a misaligned matrix")

    # checks demanded by the brief
    n_expected = int(report["n_observations"])
    log.info(f"  observations: {len(X):,} (expected {n_expected:,})")
    if len(X) != n_expected:
        raise RuntimeError("observation count mismatch vs Stage-3 report")

    scaler = joblib.load(MODELLING_DIR / "scaler.joblib")     # frozen scaler
    Z = scaler.transform(Zt.to_numpy(dtype="float64"))

    n_pc = int(report["pca"]["retained_components"])
    pca = PCA(n_components=n_pc, svd_solver="full",
              random_state=int(report["random_state"])).fit(Z)
    Pc = pca.transform(Z)
    cum = float(np.sum(pca.explained_variance_ratio_))
    log.info(f"  retained PCs: {n_pc} (expected {n_pc})")
    log.info(f"  cumulative explained variance: {cum:.4%} "
             f"(Stage-3 reported {report['pca']['variance_retained']:.4%})")
    if abs(cum - float(report["pca"]["variance_retained"])) > 5e-4:
        raise RuntimeError("PCA cumulative variance does not reproduce "
                           "the Stage-3 value")
    log.info("  PCA representation reproduces Stage 3 exactly "
             "(frozen scaler; same transforms; same retained components)")
    return {"Pc": Pc, "n_pc": n_pc}, clustered, report


# ---------------------------------------------------------------------------
# 3-4. run the K x seed grid (ONLY the K-Means init varies)
# ---------------------------------------------------------------------------
def run_grid(Pc: np.ndarray) -> dict[int, dict[int, dict]]:
    log.info("=" * 66)
    log.info(f"STEP 2 - K-Means runs: K={K_LIST}, seeds={SEEDS} "
             f"(n_init={N_INIT}, init='{INIT}'); "
             f"ONLY the initialization/fit varies")
    runs: dict[int, dict[int, dict]] = {}
    for k in K_LIST:
        runs[k] = {}
        for seed in SEEDS:
            km = KMeans(n_clusters=k, random_state=seed, n_init=N_INIT,
                        init=INIT)
            labels = km.fit_predict(Pc)
            sil = float(silhouette_score(Pc, labels))
            sizes = np.bincount(labels, minlength=k)
            runs[k][seed] = {"labels": labels, "inertia": float(km.inertia_),
                             "silhouette": sil, "sizes": sizes}
            log.info(f"  K={k} seed={seed:4d}: inertia={km.inertia_:12.1f} "
                     f"silhouette={sil:.4f} sizes={sizes.tolist()}")
    return runs


def ari_metrics(runs: dict[int, dict[int, dict]]) -> dict[int, dict]:
    """ARI vs seed-42 reference + all pairwise ARIs (permutation-invariant)."""
    log.info("=" * 66)
    log.info("STEP 3 - stability metrics (adjusted Rand index; label "
             "permutations handled by ARI, never raw label equality)")
    out: dict[int, dict] = {}
    for k in K_LIST:
        ref = runs[k][REFERENCE_SEED]["labels"]
        vs42 = {s: float(adjusted_rand_score(ref, runs[k][s]["labels"]))
                for s in SEEDS}
        pairs = {}
        for a, b in itertools.combinations(SEEDS, 2):
            pairs[(a, b)] = float(adjusted_rand_score(runs[k][a]["labels"],
                                                      runs[k][b]["labels"]))
        arr = np.array(list(pairs.values()), dtype="float64")
        out[k] = {"ari_vs_seed42": vs42,
                  "pairwise": pairs,
                  "pairwise_stats": {
                      "mean": float(arr.mean()),
                      "median": float(np.median(arr)),
                      "min": float(arr.min()),
                      "max": float(arr.max())}}
        log.info(f"  K={k}: ARI vs seed42 = "
                 f"{[{s: v for s, v in vs42.items()}]}")
        log.info(f"  K={k}: pairwise ARI mean={arr.mean():.4f} "
                 f"median={np.median(arr):.4f} min={arr.min():.4f} "
                 f"max={arr.max():.4f} "
                 f"({len(pairs)} pairs)")
    return out


# ---------------------------------------------------------------------------
# 5. centroid consistency (Hungarian alignment; supplementary evidence)
# ---------------------------------------------------------------------------
def centroid_consistency(Pc: np.ndarray,
                         runs: dict[int, dict[int, dict]]) -> dict[int, dict]:
    log.info("=" * 66)
    log.info("STEP 4 - centroid consistency (supplementary; Hungarian "
             "assignment on Euclidean distance to the seed-42 centroids)")
    out: dict[int, dict] = {}
    for k in K_LIST:
        cents_ref = (np.vstack([Pc[runs[k][REFERENCE_SEED]["labels"] == c].mean(axis=0)
                                for c in range(k)]))
        dists_all = []
        for s in SEEDS:
            if s == REFERENCE_SEED:
                dists_all.append(0.0)
                continue
            cents_s = np.vstack([Pc[runs[k][s]["labels"] == c].mean(axis=0)
                                 for c in range(k)])
            d = np.linalg.norm(cents_s[:, None, :] - cents_ref[None, :, :],
                               axis=-1)
            ri, ci = linear_sum_assignment(d)
            dists_all.append(float(d[ri, ci].mean()))
        arr = np.array(dists_all, dtype="float64")
        out[k] = {"mean_matching_distance": float(arr.mean()),
                  "max_matching_distance": float(arr.max()),
                  "per_seed": {str(s): round(v, 4)
                               for s, v in zip(SEEDS, dists_all)}}
        log.info(f"  K={k}: centroid matching distance vs seed 42 "
                 f"mean={arr.mean():.4f} max={arr.max():.4f}")
    log.info("  note: ARI remains the primary stability metric; centroid "
             "distances are supplementary evidence only")
    return out


# ---------------------------------------------------------------------------
# 6. summary + runs CSVs
# ---------------------------------------------------------------------------
def export_tables(runs: dict, ari: dict, cent: dict) -> tuple[pd.DataFrame,
                                                              pd.DataFrame]:
    log.info("=" * 66)
    log.info("STEP 5 - export cluster_stability_summary.csv + "
             "cluster_stability_runs.csv")

    rows_runs = []
    for k in K_LIST:
        total = sum(int(runs[k][s]["sizes"].sum()) for s in SEEDS)
        for s in SEEDS:
            r = runs[k][s]
            sizes = r["sizes"]
            rows_runs.append({
                "K": k, "seed": s,
                "inertia": round(r["inertia"], 1),
                "silhouette": round(r["silhouette"], 4),
                "smallest_cluster_pct":
                    round(100.0 * sizes.min() / sizes.sum(), 2),
                **{f"cluster_{c}_size": int(sizes[c]) for c in range(k)},
                **{f"cluster_{c}_pct":
                   round(100.0 * int(sizes[c]) / int(sizes.sum()), 2)
                   for c in range(k)},
                "ari_vs_seed42": round(ari[k]["ari_vs_seed42"][s], 4),
            })
    runs_df = pd.DataFrame(rows_runs)
    runs_path = MODELLING_DIR / "cluster_stability_runs.csv"
    runs_df.to_csv(runs_path, index=False)

    rows_sum = []
    for k in K_LIST:
        sil = np.array([runs[k][s]["silhouette"] for s in SEEDS])
        smallest = np.array([100.0 * runs[k][s]["sizes"].min()
                             / runs[k][s]["sizes"].sum() for s in SEEDS])
        vs42 = np.array([ari[k]["ari_vs_seed42"][s] for s in SEEDS])
        ps = ari[k]["pairwise_stats"]
        rows_sum.append({
            "K": k,
            "mean_silhouette": round(float(sil.mean()), 4),
            "std_silhouette": round(float(sil.std(ddof=1)), 4),
            "min_silhouette": round(float(sil.min()), 4),
            "max_silhouette": round(float(sil.max()), 4),
            "mean_pairwise_ARI": round(ps["mean"], 4),
            "median_pairwise_ARI": round(ps["median"], 4),
            "min_pairwise_ARI": round(ps["min"], 4),
            "max_pairwise_ARI": round(ps["max"], 4),
            "mean_ARI_vs_seed42": round(float(vs42.mean()), 4),
            "min_ARI_vs_seed42": round(float(vs42.min()), 4),
            "max_ARI_vs_seed42": round(float(vs42.max()), 4),
            "mean_smallest_cluster_pct": round(float(smallest.mean()), 2),
            "min_smallest_cluster_pct": round(float(smallest.min()), 2),
            "max_smallest_cluster_pct": round(float(smallest.max()), 2),
            "mean_centroid_matching_distance":
                round(cent[k]["mean_matching_distance"], 4),
            "max_centroid_matching_distance":
                round(cent[k]["max_matching_distance"], 4),
        })
    sum_df = pd.DataFrame(rows_sum)
    sum_path = MODELLING_DIR / "cluster_stability_summary.csv"
    sum_df.to_csv(sum_path, index=False)

    log.info(f"  saved {runs_path} ({len(runs_df)} rows = K x seed runs)")
    log.info(f"  saved {sum_path} ({len(sum_df)} rows, one per K)")
    log.info("\n" + sum_df.to_string(index=False))
    return sum_df, runs_df


# ---------------------------------------------------------------------------
# 7. figures (project conventions: matplotlib Agg, dpi=150, tab10 colours)
# ---------------------------------------------------------------------------
def plot_figures(runs: dict, ari: dict) -> None:
    log.info("=" * 66)
    log.info("STEP 6 - figures")

    # ARI distributions across K (vs seed42 + pairwise box)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    data = [[ari[k]["ari_vs_seed42"][s] for s in SEEDS] for k in K_LIST]
    bp = ax.boxplot(data, tick_labels=[f"K={k}" for k in K_LIST],
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#aec7e8")
    for i, k in enumerate(K_LIST):
        ax.scatter(np.full(len(SEEDS), i + 1),
                   [ari[k]["ari_vs_seed42"][s] for s in SEEDS],
                   zorder=3, s=18, c="tab:blue", alpha=0.8)
    ax.set_ylabel("Adjusted Rand Index vs seed-42 solution")
    ax.set_title("ARI vs seed 42 across random seeds")
    ax.set_ylim(-0.05, 1.05)
    ax = axes[1]
    pw = [[ari[k]["pairwise"][p] for p in itertools.combinations(SEEDS, 2)]
          for k in K_LIST]
    bp = ax.boxplot(pw, tick_labels=[f"K={k}" for k in K_LIST],
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#ffbb78")
    ax.set_ylabel("Pairwise ARI between seeds")
    ax.set_title(f"Pairwise ARI across all seed pairs "
                 f"({len(SEEDS)} seeds -> {len(SEEDS)*(len(SEEDS)-1)//2} pairs per K)")
    ax.set_ylim(-0.05, 1.05)
    fig.suptitle("Cluster stability across K-Means random initializations")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_stability_ari.png", dpi=150)
    plt.close(fig)

    # silhouette distributions across seeds
    fig, ax = plt.subplots(figsize=(8, 5))
    sil_data = [[runs[k][s]["silhouette"] for s in SEEDS] for k in K_LIST]
    bp = ax.boxplot(sil_data, tick_labels=[f"K={k}" for k in K_LIST],
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#98df8a")
    for i, k in enumerate(K_LIST):
        ax.scatter(np.full(len(SEEDS), i + 1),
                   [runs[k][s]["silhouette"] for s in SEEDS],
                   zorder=3, s=18, c="tab:green", alpha=0.8)
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette distribution across seeds (K=2, 3, 4)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_stability_silhouette.png", dpi=150)
    plt.close(fig)

    # comparison: silhouette (left) + pairwise ARI (right), scatter by seed
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for i, k in enumerate(K_LIST):
        ax.scatter([k] * len(SEEDS), [runs[k][s]["silhouette"] for s in SEEDS],
                   s=28, alpha=0.85, label=f"K={k}")
    ax.set_xticks(K_LIST)
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette by K (each point = one seed)")
    ax = axes[1]
    for i, k in enumerate(K_LIST):
        vals = list(ari[k]["pairwise"].values())
        ax.scatter([k] * len(vals), vals, s=14, alpha=0.55, label=f"K={k}")
        m = ari[k]["pairwise_stats"]["mean"]
        ax.hlines(m, k - 0.15, k + 0.15, color="black", lw=2)
    ax.set_xticks(K_LIST)
    ax.set_xlabel("K")
    ax.set_ylabel("Pairwise ARI")
    ax.set_title("Pairwise ARI by K (black bar = mean)")
    axes[0].legend()
    fig.suptitle("Silhouette vs stability trade-off across K")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_stability_comparison.png", dpi=150)
    plt.close(fig)
    log.info("  figures saved: cluster_stability_ari.png, "
             "cluster_stability_silhouette.png, "
             "cluster_stability_comparison.png")


# ---------------------------------------------------------------------------
# 8-9. interpretation + report (metrics-driven, no auto K change)
# ---------------------------------------------------------------------------
def build_report(Pc: np.ndarray, runs: dict, ari: dict, cent: dict,
                 sum_df: pd.DataFrame, report3: dict,
                 clustered: pd.DataFrame) -> dict:
    log.info("=" * 66)
    log.info("STEP 7 - interpretation (metrics-driven; official model "
             "UNCHANGED regardless of outcome)")

    def row(k: int) -> pd.Series:
        return sum_df[sum_df["K"] == k].iloc[0]

    def stable(k: int) -> bool:
        r = row(k)
        return bool(r["mean_pairwise_ARI"] >= 0.99
                    and r["min_pairwise_ARI"] >= 0.95)

    r2, r3, r4 = row(2), row(3), row(4)

    q1 = {"mean_pairwise_ARI": round(float(r2["mean_pairwise_ARI"]), 4),
          "min_pairwise_ARI": round(float(r2["min_pairwise_ARI"]), 4),
          "max_pairwise_ARI": round(float(r2["max_pairwise_ARI"]), 4),
          "mean_silhouette": round(float(r2["mean_silhouette"]), 4),
          "std_silhouette": round(float(r2["std_silhouette"]), 4),
          "smallest_cluster_pct_range":
              [round(float(r2["min_smallest_cluster_pct"]), 2),
               round(float(r2["max_smallest_cluster_pct"]), 2)],
          "stable": stable(2)}
    q1["answer"] = ("K=2 is statistically stable and reproducible across "
                    "random initializations" if q1["stable"] else
                    "K=2 shows seed sensitivity (see min pairwise ARI)")

    q2 = {"mean_pairwise_ARI": round(float(r3["mean_pairwise_ARI"]), 4),
          "min_pairwise_ARI": round(float(r3["min_pairwise_ARI"]), 4),
          "max_pairwise_ARI": round(float(r3["max_pairwise_ARI"]), 4),
          "mean_silhouette": round(float(r3["mean_silhouette"]), 4),
          "std_silhouette": round(float(r3["std_silhouette"]), 4),
          "stable": stable(3)}
    q2["answer"] = ("K=3 is stable across random initializations "
                    "(reproducible solution)" if q2["stable"] else
                    "K=3 is NOT stable across seeds (seed-dependent solution)")

    q3 = {"mean_pairwise_ARI": round(float(r4["mean_pairwise_ARI"]), 4),
          "min_pairwise_ARI": round(float(r4["mean_pairwise_ARI"]), 4),
          "max_pairwise_ARI": round(float(r4["max_pairwise_ARI"]), 4),
          "mean_silhouette": round(float(r4["mean_silhouette"]), 4),
          "std_silhouette": round(float(r4["std_silhouette"]), 4),
          "stable": stable(4)}
    q3["answer"] = ("K=4 is stable across random initializations "
                    "(reproducible solution)" if q3["stable"] else
                    "K=4 is NOT stable across seeds (seed-dependent solution)")

    k2_leads = bool(r2["mean_silhouette"]
                    > max(r3["mean_silhouette"], r4["mean_silhouette"]))
    q4 = {"official_model": {"K": int(report3["kmeans"]["selected_k"]),
                             "random_state":
                                 int(report3["kmeans"]["random_state"]),
                             "n_init": int(report3["kmeans"]["n_init"])},
          "k2_stable": q1["stable"],
          "k2_silhouette_leads": k2_leads}
    if q1["stable"] and k2_leads:
        q4["answer"] = ("K=2 is statistically stable and reproducible across "
                        "random initializations, strengthening its validity "
                        "as the official clustering solution.")
        q4["supports_retaining_k2"] = True
    elif not q1["stable"]:
        q4["answer"] = ("K=2 itself is UNSTABLE across seeds. The official "
                        "model is NOT changed by this read-only stage; a "
                        "documented model-selection review is recommended.")
        q4["supports_retaining_k2"] = False
    else:
        q4["answer"] = ("K=2 is stable but does not lead on silhouette; a "
                        "documented model-selection review is recommended. "
                        "Official model unchanged by this stage.")
        q4["supports_retaining_k2"] = False

    # Q5: K=3 secondary structure - decided only by measured criteria:
    # stability + recomputed sub-cluster profile from THIS run's data.
    lab3 = runs[3][REFERENCE_SEED]["labels"]
    sub = {}
    for c in range(3):
        m = lab3 == c
        sub[f"cluster_{c}"] = {
            "n": int(m.sum()),
            "mean_daily_kW": round(float(clustered.loc[m, "daily_mean_kW"]
                                         .mean()), 3),
            "mean_base_kW": round(float(clustered.loc[m, "base_kW"].mean()),
                                  3),
            "top_meters": {str(a): int(b) for a, b in
                           clustered.loc[m, "meter"].value_counts()
                           .head(2).items()},
        }
    sizes3 = runs[3][REFERENCE_SEED]["sizes"]
    interpretable_substructure = (
        # distinct load magnitudes between sub-clusters (order of magnitude)
        max(s["mean_daily_kW"] for s in sub.values())
        / max(min(s["mean_daily_kW"] for s in sub.values()), 1e-9) >= 3.0
        # and a coherent meter composition (top meter >= 70% of a sub-cluster)
        and any(max(mv.values()) / sum(mv.values()) >= 0.70
                for mv in (s["top_meters"] for s in sub.values())))
    q5 = {"k3_stable": q2["stable"],
          "k3_mean_silhouette": round(float(r3["mean_silhouette"]), 4),
          "k3_silhouette_lower_than_k2":
              bool(r3["mean_silhouette"] < r2["mean_silhouette"]),
          "k3_reference_subcluster_profile": sub,
          "interpretable_substructure_detected":
              bool(interpretable_substructure)}
    if q2["stable"] and interpretable_substructure:
        q5["answer"] = ("K=3 represents a stable secondary subdivision of the "
                        "data with interpretable sub-structure, but its lower "
                        "separation quality (silhouette) means it remains a "
                        "sensitivity analysis rather than a replacement for "
                        "the official K=2 model.")
    elif q2["stable"]:
        q5["answer"] = ("K=3 is stable but its sub-clusters do not show a "
                        "distinct, interpretable structure in this re-test; "
                        "no secondary-structure claim is made.")
    else:
        q5["answer"] = ("K=3 is not stable across seeds; it does not "
                        "represent a reproducible secondary structure.")

    report = {
        "stage": "SPANDA Stage 3c - cluster stability validation (read-only)",
        "observation_unit": "building-day",
        "design": {
            "seeds": SEEDS,
            "K_values": K_LIST,
            "n_init": N_INIT,
            "init": INIT,
            "reference_seed": REFERENCE_SEED,
            "varies": "K-Means initialization/fit ONLY",
            "frozen": ["observation unit", "feature engineering",
                       "transforms", "StandardScaler", "PCA representation",
                       "retained PCs (3)"],
            "pca_representation": {
                "n_observations": int(Pc.shape[0]),
                "retained_pcs": int(Pc.shape[1]),
                "note": "reproduces Stage-3 cumulative variance 0.9085; "
                        "row alignment verified on meter + obs_date"},
        },
        "metrics_definitions": {
            "ARI": "sklearn.metrics.adjusted_rand_score (permutation-"
                   "invariant; raw label equality never used)",
            "centroid_consistency": "Hungarian (optimal) assignment on "
                                    "Euclidean centroid distance vs the "
                                    "seed-42 solution; supplementary only",
            "stability_rule": "mean pairwise ARI >= 0.99 AND min pairwise "
                              "ARI >= 0.95",
        },
        "k2_stability": q1,
        "k3_stability": q2,
        "k4_stability": q3,
        "question_4_supports_k2": q4,
        "question_5_k3_secondary_structure": q5,
        "decision": ("Official model remains K=2 (random_state=42, "
                     "n_init=10). This stage changes nothing; it only "
                     "measures reproducibility."),
        "caveats": [
            "Stability here = reproducibility across K-Means initializations "
            "on one frozen representation; it does not validate the "
            "representation itself or imply causal structure.",
            "ARI is inflation-biased toward high values for small K; the "
            "K=2/K=3/K=4 ARI values are compared within-K only.",
            "Centroid matching distances are supplementary evidence; ARI is "
            "the primary metric.",
            "No monetary, causal, or consumer-level claims; unit is "
            "building-day.",
        ],
        "figures": ["cluster_stability_ari.png",
                    "cluster_stability_silhouette.png",
                    "cluster_stability_comparison.png"],
        "tables": ["cluster_stability_summary.csv",
                   "cluster_stability_runs.csv"],
    }
    path = MODELLING_DIR / "cluster_stability_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info(f"  saved {path}")
    log.info(f"  Q1 (K=2 stable): {q1['stable']} - {q1['answer']}")
    log.info(f"  Q2 (K=3 stable): {q2['stable']} - {q2['answer']}")
    log.info(f"  Q3 (K=4 stable): {q3['stable']} - {q3['answer']}")
    log.info(f"  Q4: {q4['answer']}")
    log.info(f"  Q5: {q5['answer']}")
    return report


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("#" * 70)
    log.info("SPANDA Stage 3c - CLUSTER STABILITY VALIDATION (read-only)")
    log.info("#" * 70)

    before = hash_frozen_artifacts()
    log.info(f"pre-run hash verification: {len(before)} frozen files hashed")

    rep, clustered, report3 = recreate_representation()
    Pc = rep["Pc"]

    runs = run_grid(Pc)
    ari = ari_metrics(runs)
    cent = centroid_consistency(Pc, runs)
    sum_df, runs_df = export_tables(runs, ari, cent)
    plot_figures(runs, ari)
    build_report(Pc, runs, ari, cent, sum_df, report3, clustered)

    # final verification: nothing frozen was modified
    verify_frozen(before, hash_frozen_artifacts())

    # concise summary
    r2 = sum_df[sum_df["K"] == 2].iloc[0]
    r3 = sum_df[sum_df["K"] == 3].iloc[0]
    r4 = sum_df[sum_df["K"] == 4].iloc[0]
    log.info("=" * 66)
    log.info("SUMMARY")
    log.info(f"  1. K=2 mean silhouette: {r2['mean_silhouette']:.4f} "
             f"+/- {r2['std_silhouette']:.4f}")
    log.info(f"  2. K=2 mean pairwise ARI: {r2['mean_pairwise_ARI']:.4f}")
    log.info(f"  3. K=3 mean silhouette: {r3['mean_silhouette']:.4f} "
             f"+/- {r3['std_silhouette']:.4f}")
    log.info(f"  4. K=3 mean pairwise ARI: {r3['mean_pairwise_ARI']:.4f}")
    log.info(f"  5. K=4 mean silhouette: {r4['mean_silhouette']:.4f} "
             f"+/- {r4['std_silhouette']:.4f}")
    log.info(f"  6. K=4 mean pairwise ARI: {r4['mean_pairwise_ARI']:.4f}")
    log.info(f"  7. Official model: K=2 RETAINED UNCHANGED "
             f"(support verdict in cluster_stability_report.json, "
             f"question_4)")
    log.info("Stage 3c complete.")


if __name__ == "__main__":
    main()
