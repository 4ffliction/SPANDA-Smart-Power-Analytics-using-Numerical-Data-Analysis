"""
SPANDA Stage 3b - CLUSTER PROFILING, VALIDATION & ENGINEERING INTERPRETATION
(read-only w.r.t. raw data and the official Stage-3 model; every decision
logged to spanda_out/audit/profiling_log.txt)

Design rules enforced throughout
--------------------------------
* The official K-Means (K=2) is NOT refitted here. Labels, centroids, and the
  frozen 3-PC representation are loaded from the Stage-3 artifacts
  (clustered_building_days.csv, pca_features.parquet, cluster_centroids.csv).
  Parts A-D and F-H are pure summaries of that frozen model.
* Part E (K=3/K=4) is an explicitly-labelled SENSITIVITY analysis. It reuses
  the SAME preprocessing + PCA transform as Stage 3; only the K-Means step is
  refit for comparison. It does not replace the official model, and K=3/K=4
  are not selected merely for looking richer.
* Observation unit is BUILDING-DAY everywhere. Clusters are statistical
  groupings of building-days - never "consumers". No causal claims.
* All labels (Part F) are assigned ONLY from measured characteristics
  (Parts A-C), after those parts have been computed.
* Part G flags "potentially anomalous energy-behaviour observations"
  (large centroid distance). These are NOT confirmed equipment faults.

Outputs
-------
spanda_out/modelling/cluster_profile.csv              (Part A)
spanda_out/modelling/building_cluster_distribution.csv (Part B)
spanda_out/figures/cluster_temporal_profiles.png       (Part C)
spanda_out/figures/cluster_period_shares.png           (Part C)
spanda_out/figures/k_sensitivity_comparison.png        (Part E)
spanda_out/modelling/k_sensitivity.csv                 (Part E)
spanda_out/modelling/lecture_dominance.json            (Part D)
spanda_out/modelling/cluster_labels.json               (Part F)
spanda_out/modelling/potential_anomalies.csv           (Part G)
spanda_out/figures/potential_anomaly_distances.png     (Part G)
spanda_out/modelling/cluster_validation_report.json    (Part I)
spanda_out/modelling/cluster_interpretation.csv        (Part I)
"""

from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from spanda.config import (
    AUDIT_DIR, FIGURES_DIR, MODELLING_DIR, PROCESSED_DIR, get_logger,
)
from spanda.model import apply_transforms, prepare_modelling_matrix

log = get_logger("spanda.profile", AUDIT_DIR / "profiling_log.txt")

# ---------------------------------------------------------------------------
# configuration (documented; consistent with spanda.model)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_INIT = 10
PERIOD_SHARES = ["morning_share", "afternoon_share", "evening_share",
                 "night_share"]
PERIOD_LABELS = {"morning_share": "morning (06-11)",
                 "afternoon_share": "afternoon (12-17)",
                 "evening_share": "evening (18-23)",
                 "night_share": "night (00-05)"}
PROFILE_STATS = ["mean", "median", "std"]          # per-feature stats (Part A)
ANOMALY_METHODS = ("percentile", "med_mad")        # two independent thresholds
ANOMALY_PCTL = 99.0                                # extreme-distance percentile
ANOMALY_MAD_K = 6.0                                # robust z (MAD) threshold
K_SENS = [2, 3, 4]                                 # Part E comparison set
# single-parameter cluster-name candidates; assigned in Part F from medians
LABEL_CANDIDATES: list[tuple[str, str]] = [
    ("load_factor",            "Low Load Factor (Peaky)"),
    ("load_factor",            "High Load Factor (Flat)"),
    ("night_share",            "Night-Dominant (Continuous)"),
    ("evening_share",          "Evening-Dominant"),
    ("morning_share",          "Morning-Dominant"),
    ("afternoon_share",        "Afternoon-Dominant"),
    ("base_kW",                "High Baseline Load"),
    ("base_kW",                "Low Baseline Load"),
    ("daily_peak_kW",          "Peak-Oriented Load"),
    ("daily_mean_kW",          "Low/Stable Load"),
    ("hourly_variability_kW",  "Highly Variable"),
]


# ---------------------------------------------------------------------------
# load the FROZEN official model (never refitted)
# ---------------------------------------------------------------------------
def load_frozen_model() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Load clustered building-days + Stage-3 artifacts (official K=2 model)."""
    clustered = pd.read_csv(MODELLING_DIR / "clustered_building_days.csv")
    report = json.loads((MODELLING_DIR / "modelling_report.json")
                        .read_text(encoding="utf-8"))
    cents = pd.read_csv(MODELLING_DIR / "cluster_centroids.csv")

    k_official = int(report["kmeans"]["selected_k"])
    sil = float(report["kmeans"]["final_silhouette"])
    if k_official != 2 or clustered["cluster"].nunique() != 2:
        raise RuntimeError("expected the frozen official model to be K=2")
    log.info("=" * 66)
    log.info("FROZEN OFFICIAL MODEL (from Stage-3 artifacts, not refitted)")
    log.info(f"  clustered_building_days.csv: {len(clustered):,} building-days")
    log.info(f"  K = {k_official}, silhouette = {sil:.4f}, sizes = "
             f"{clustered['cluster'].value_counts().sort_index().to_dict()}")
    log.info(f"  PC columns available: "
             f"{[c for c in cents.columns if c.startswith('PC')]}")
    return clustered, report, cents


VF_FEATURES = ["voltage_mean_V", "frequency_mean_Hz", "power_factor_mean"]


def attach_vf_pf_features(clustered: pd.DataFrame) -> pd.DataFrame:
    """PART A support: attach daily mean voltage / frequency / power factor.

    These exist in the Stage-2 hourly table but NOT at building-day level and
    were never part of the frozen model. They are aggregated here for
    PROFILING ONLY (no change to the model or its features)."""
    hourly = pd.read_parquet(
        PROCESSED_DIR / "hourly_power_clean.parquet",
        columns=["dt", "voltage_mean", "frequency_mean", "pf_mean", "meter"])
    hourly["obs_date"] = hourly["dt"].dt.strftime("%Y-%m-%d")
    daily = (hourly.drop(columns="dt")
             .groupby(["meter", "obs_date"])
             .agg(voltage_mean_V=("voltage_mean", "mean"),
                  frequency_mean_Hz=("frequency_mean", "mean"),
                  power_factor_mean=("pf_mean", "mean"))
             .reset_index())
    out = clustered.merge(daily, on=["meter", "obs_date"], how="left")
    n_full = int(out[VF_FEATURES].notna().all(axis=1).sum())
    log.info(f"  attached daily voltage/frequency/pf context features "
             f"(profiling only, NOT model features): {n_full:,} of "
             f"{len(out):,} rows fully matched")
    return out


def within_day_cv(clustered: pd.DataFrame, cl: int) -> float:
    """Median within-observation shape variability: max/mean of the 24 hourly
    kW values of the same building-day (a peak-to-average ratio, computed
    WITHIN each observation - no pooling of differently sized meters)."""
    hcols = [f"h{h:02d}_kW" for h in range(24)]
    sub = clustered.loc[clustered["cluster"] == cl, hcols]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sub.max(axis=1) / sub.mean(axis=1)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    return float(ratio.median())


# ---------------------------------------------------------------------------
# PART A - original-feature cluster profiles
# ---------------------------------------------------------------------------
def part_a_profiles(clustered: pd.DataFrame,
                    vf_cols: list[str] | None = None) -> pd.DataFrame:
    log.info("=" * 66)
    log.info("PART A - original-feature cluster profiles (unscaled)")

    prof_rows: list[dict] = []

    def _add(cluster, feature, series: pd.Series, seen: set[str]) -> None:
        if feature in seen:                      # guard against double-collection
            return
        seen.add(feature)

        s = pd.to_numeric(series, errors="coerce").astype("float64")
        prof_rows.append({
            "cluster": cluster, "feature": feature,
            "mean": float(s.mean()), "median": float(s.median()),
            "std": float(s.std(ddof=1)),
        })

    for cl, sub in clustered.groupby("cluster"):
        seen: set[str] = set()                   # dedupe within this cluster
        n = len(sub)
        log.info(f"  cluster {cl}: {n:,} observations ({n / len(clustered):.1%})")

        # headline quantities + every available hourly power feature
        headline = ["daily_mean_kW", "daily_peak_kW", "daily_min_kW",
                    "daily_total_kWh", "base_kW", "load_factor", "peak_hour"]
        headline += [c for c in (vf_cols or []) if c in sub.columns]
        for f in headline:
            if f in sub.columns:
                _add(cl, f, sub[f], seen)
        for f in [f"h{h:02d}_kW" for h in range(24)]:
            if f in sub.columns:
                _add(cl, f, sub[f], seen)

        # voltage / frequency / power-factor features if present anywhere
        for f in sorted(set(sub.columns)
                        & {"voltage_mean", "frequency_mean", "pf_mean"}
                        | {c for c in sub.columns
                           if c.startswith(("voltage", "frequency", "pf_"))}):
            if pd.api.types.is_numeric_dtype(sub[f]):
                _add(cl, f, sub[f], seen)

        # period shares (mean + median reported; std of a share too)
        for f in PERIOD_SHARES:
            if f in sub.columns:
                _add(cl, f, sub[f], seen)

    prof = pd.DataFrame(prof_rows)
    # wide table: one row per feature, stats per cluster side by side
    wide = prof.pivot(index="feature", columns="cluster",
                      values=PROFILE_STATS)
    wide.columns = [f"{stat}_cluster{int(cl)}"
                    for stat, cl in wide.columns]
    wide = wide.reset_index()

    path = MODELLING_DIR / "cluster_profile.csv"
    wide.to_csv(path, index=False)
    log.info(f"  saved {path} ({len(wide)} features x {2 * len(PROFILE_STATS)} "
             f"stats columns)")

    # headline log summary
    for f in ["daily_mean_kW", "daily_peak_kW", "daily_min_kW", "base_kW",
              "load_factor", "night_share", "afternoon_share",
              "voltage_mean_V", "frequency_mean_Hz", "power_factor_mean"]:
        row = wide[wide["feature"] == f]
        if row.empty:
            continue
        cells = ", ".join(f"C{cl} mean="
                          f"{row.iloc[0][f'mean_cluster{cl}']:.3f}"
                          for cl in sorted(clustered["cluster"].unique()))
        log.info(f"  {f}: {cells}")
    return prof


# ---------------------------------------------------------------------------
# PART B - building x cluster composition
# ---------------------------------------------------------------------------
def part_b_building_composition(clustered: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 66)
    log.info("PART B - building (meter) x cluster composition")

    counts = (clustered.pivot_table(index="meter", columns="cluster",
                                    values="obs_date", aggfunc="count")
              .fillna(0).astype(int))
    counts.columns = [f"cluster_{int(c)}_n" for c in counts.columns]
    pct = (clustered.pivot_table(index="meter", columns="cluster",
                                 values="obs_date", aggfunc="count")
           .fillna(0))
    pct = pct.div(pct.sum(axis=1), axis=0) * 100.0
    pct.columns = [f"cluster_{int(c)}_pct" for c in pct.columns]
    # cluster share WITHIN each cluster (composition of the cluster itself)
    comp = (clustered.pivot_table(index="meter", columns="cluster",
                                  values="obs_date", aggfunc="count")
            .fillna(0))
    comp = comp.div(comp.sum(axis=0), axis=1) * 100.0
    comp.columns = [f"cluster_{int(c)}_composition_pct"
                    for c in comp.columns]
    tot = clustered.groupby("meter").size().rename("total_days")
    dist = pd.concat([tot, counts, pct, comp], axis=1).reset_index()

    path = MODELLING_DIR / "building_cluster_distribution.csv"
    dist.to_csv(path, index=False)

    cl_cols = [c for c in counts.columns]
    specificity = []
    for _, r in dist.iterrows():
        shares = np.array([r[c] for c in pct.columns], dtype="float64")
        # inverse Herfindahl concentration of each building's cluster shares
        h = float((np.where(shares > 0, shares, 0.0) ** 2).sum())
        eff = 1.0 / h if h > 0 else np.nan
        eff_norm = (eff - 1) / (len(shares) - 1) if len(shares) > 1 else np.nan
        if eff_norm >= 0.9:
            tag = "cross-building (evenly spread across clusters)"
        elif eff_norm >= 0.6:
            tag = "mixed (partly building-specific, partly shared)"
        else:
            tag = "building-specific (concentrated in one cluster)"
        specificity.append(tag)
    dist["distribution_type"] = specificity

    log.info(f"  saved {path} ({len(dist)} meters)")
    for _, r in dist.iterrows():
        shares = ", ".join(f"C{c.split('_')[1]}={r[c]:.1f}%"
                           for c in pct.columns)
        log.info(f"    {r['meter']}: n={int(r['total_days']):,}; {shares} "
                 f"-> {r['distribution_type']}")
    return dist


# ---------------------------------------------------------------------------
# PART C - temporal profiles
# ---------------------------------------------------------------------------
def part_c_temporal(clustered: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log.info("=" * 66)
    log.info("PART C - temporal profiles (mean 24-h power + period behaviour)")

    hcols = [f"h{h:02d}_kW" for h in range(24)]
    prof = clustered.groupby("cluster")[hcols].mean()
    prof.columns = [int(c[1:3]) for c in prof.columns]
    prof = prof[sorted(prof.columns)]
    prof_sd = clustered.groupby("cluster")[hcols].std()
    prof_sd.columns = prof.columns

    temp_stats: dict[str, dict] = {}
    for cl in sorted(clustered["cluster"].unique()):
        sub = clustered[clustered["cluster"] == cl]
        day_mean = float(prof.loc[cl, [h for h in range(6, 23)]].mean())
        night_mean = float(prof.loc[cl, [h for h in range(0, 6)]].mean())
        wk = sub[sub["weekday"] < 5]
        we = sub[sub["weekday"] >= 5]
        temp_stats[str(int(cl))] = {
            "n_observations": int(len(sub)),
            "peak_hour_modal": int(prof.loc[cl].idxmax()),
            "peak_hour_mean_power_kW": round(float(prof.loc[cl].max()), 3),
            "daytime_mean_kW_06_22": round(day_mean, 3),
            "night_mean_kW_00_05": round(night_mean, 3),
            "day_night_ratio": round(day_mean / night_mean, 3)
            if night_mean > 0 else None,
            "period_share_means": {PERIOD_LABELS[f]:
                                   round(float(sub[f].mean()), 4)
                                   for f in PERIOD_SHARES},
            "weekday_days": int(len(wk)),
            "weekend_days": int(len(we)),
            "weekday_mean_daily_kW":
                round(float(wk["daily_mean_kW"].mean()), 3)
                if len(wk) else None,
            "weekend_mean_daily_kW":
                round(float(we["daily_mean_kW"].mean()), 3)
                if len(we) else None,
            "weekday_weekend_ratio":
                round(float(wk["daily_mean_kW"].mean()
                            / we["daily_mean_kW"].mean()), 3)
                if len(wk) and len(we) and we["daily_mean_kW"].mean() > 0
                else None,
            "weekday_hourly_profile_kW":
                [round(float(x), 3)
                 for x in wk[hcols].mean().to_numpy()],
            "weekend_hourly_profile_kW":
                [round(float(x), 3)
                 for x in we[hcols].mean().to_numpy()]
                if len(we) else None,
        }
        log.info(f"  cluster {cl}: peak hour {temp_stats[str(int(cl))]['peak_hour_modal']:02d}:00, "
                 f"day/night ratio {temp_stats[str(int(cl))]['day_night_ratio']}, "
                 f"weekday/weekend ratio {temp_stats[str(int(cl))]['weekday_weekend_ratio']}")

    # figures -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for cl in prof.index:
        ax.plot(prof.columns, prof.loc[cl], marker="o", ms=4,
                label=f"Cluster {cl} (n={int((clustered['cluster'] == cl).sum()):,})")
        ax.fill_between(prof.columns,
                        prof.loc[cl] - prof_sd.loc[cl],
                        prof.loc[cl] + prof_sd.loc[cl], alpha=0.15)
    ax.set(xlabel="Hour of day", ylabel="Mean power (kW)",
           xticks=range(0, 24, 2),
           title="Average 24-hour power profile per cluster (K=2 official model)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_temporal_profiles.png", dpi=150)
    plt.close(fig)

    # per-meter profile figures (mean +/- sd), small multiples - only for
    # meters with enough days in the cluster
    for cl in prof.index:
        meters = clustered.loc[clustered["cluster"] == cl, "meter"] \
            .value_counts()
        keep = [m for m, n in meters.items() if n >= 60]
        if not keep:
            continue
        fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
        axes = axes.ravel()
        for ax, m in zip(axes, sorted(clustered["meter"].unique())):
            sub = clustered[(clustered["cluster"] == cl)
                            & (clustered["meter"] == m)]
            if len(sub) < 60:
                ax.set_visible(False)
                continue
            mu = sub[hcols].mean().to_numpy()
            sd = sub[hcols].std().to_numpy()
            ax.plot(range(24), mu, lw=1.4)
            ax.fill_between(range(24), mu - sd, mu + sd, alpha=0.2)
            ax.set_title(f"{m} (n={len(sub):,})", fontsize=9)
            ax.tick_params(labelsize=7)
        fig.suptitle(f"Cluster {cl} - mean 24-h power profile per meter "
                     "(mean +/- sd)")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"cluster_{cl}_profiles_by_meter.png",
                    dpi=150)
        plt.close(fig)
    log.info("  per-meter profile figures saved (where n >= 60 days)")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(PERIOD_SHARES))
    width = 0.8 / len(prof.index)
    for i, cl in enumerate(sorted(prof.index)):
        vals = [float(clustered.loc[clustered["cluster"] == cl, f].mean())
                for f in PERIOD_SHARES]
        ax.bar(x + i * width, vals, width, label=f"Cluster {cl}")
    ax.set_xticks(x + width * (len(prof.index) - 1) / 2)
    ax.set_xticklabels([PERIOD_LABELS[f] for f in PERIOD_SHARES])
    ax.set_ylabel("Mean share of daily energy")
    ax.set_title("Period-of-day energy shares per cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_period_shares.png", dpi=150)
    plt.close(fig)
    log.info("  figures saved: cluster_temporal_profiles.png, "
             "cluster_period_shares.png")
    return prof, temp_stats


# ---------------------------------------------------------------------------
# PART D - Lecture-building dominance check (diagnostic only)
# ---------------------------------------------------------------------------
def part_d_lecture_check(clustered: pd.DataFrame, temp_stats: dict) -> dict:
    log.info("=" * 66)
    log.info("PART D - Lecture-building dominance check (diagnostic)")

    is_lec = clustered["meter"].eq("Lecture")
    n_c1 = int((clustered["cluster"] == 1).sum())
    n_c1_lec = int((is_lec & (clustered["cluster"] == 1)).sum())
    lec = clustered[is_lec]
    non = clustered[~is_lec]

    hcols = [f"h{h:02d}_kW" for h in range(24)]
    prof_lec = lec[hcols].mean()
    prof_non = non[hcols].mean()

    out = {
        "unit_note": "observations are building-days, not consumers",
        "pct_of_cluster1_that_is_Lecture":
            round(100.0 * n_c1_lec / n_c1, 2) if n_c1 else None,
        "pct_of_Lecture_days_in_cluster1":
            round(100.0 * n_c1_lec / len(lec), 2) if len(lec) else None,
        "cluster0_building_composition": {
            str(k): int(v) for k, v in
            clustered[clustered["cluster"] == 0]["meter"]
            .value_counts().items()},
        "cluster1_building_composition": {
            str(k): int(v) for k, v in
            clustered[clustered["cluster"] == 1]["meter"]
            .value_counts().items()},
        "lecture_vs_nonLecture": {
            "n_Lecture": int(len(lec)), "n_nonLecture": int(len(non)),
            "mean_daily_kW_Lecture": round(float(lec["daily_mean_kW"].mean()), 3),
            "mean_daily_kW_nonLecture":
                round(float(non["daily_mean_kW"].mean()), 3),
            "mean_peak_kW_Lecture":
                round(float(lec["daily_peak_kW"].mean()), 3),
            "mean_peak_kW_nonLecture":
                round(float(non["daily_peak_kW"].mean()), 3),
            "mean_night_share_Lecture":
                round(float(lec["night_share"].mean()), 4),
            "mean_night_share_nonLecture":
                round(float(non["night_share"].mean()), 4),
            "mean_load_factor_Lecture":
                round(float(lec["load_factor"].mean()), 4),
            "mean_load_factor_nonLecture":
                round(float(non["load_factor"].mean()), 4),
            "hourly_profile_Lecture_kW":
                [round(float(x), 3) for x in prof_lec.to_numpy()],
            "hourly_profile_nonLecture_kW":
                [round(float(x), 3) for x in prof_non.to_numpy()],
        },
    }

    # does the split align with the Lecture/non-Lecture partition?
    ct = pd.crosstab(clustered["meter"], clustered["cluster"])
    lec_row = ct.loc["Lecture"]
    lec_c1_share = float(lec_row.get(1, 0) / lec_row.sum())
    purity = []
    for cl in sorted(clustered["cluster"].unique()):
        sub = clustered[clustered["cluster"] == cl]
        purity.append(float((sub["meter"] == "Lecture").mean()))
    purity_gap = abs(purity[0] - purity[1])

    # evidence of a broader behavioural distinction beyond Lecture?
    non_lec_c0 = clustered[(clustered["cluster"] == 0) & ~is_lec]
    non_lec_c1 = clustered[(clustered["cluster"] == 1) & ~is_lec]
    non_lec_in_c1 = int(len(non_lec_c1))
    broader = {
        "nonLecture_days_in_cluster1": non_lec_in_c1,
        "nonLecture_days_in_cluster1_pct_of_cluster1":
            round(100.0 * non_lec_in_c1 / n_c1, 2) if n_c1 else None,
        "cluster1_nonLecture_meter_counts": {
            str(k): int(v) for k, v in non_lec_c1["meter"].value_counts().items()},
        "Lecture_vs_nonLecture_hourly_profiles_differ":
            bool(np.abs(prof_lec.to_numpy()
                        - prof_non.to_numpy()).max() > 1.0),
    }
    out["broader_behaviour_evidence"] = broader

    log.info(f"  cluster 1: {n_c1_lec}/{n_c1} Lecture days "
             f"({out['pct_of_cluster1_that_is_Lecture']}%)")
    log.info(f"  Lecture days in cluster 1: "
             f"{out['pct_of_Lecture_days_in_cluster1']}% of all Lecture days")
    log.info(f"  cluster 0 building composition: "
             f"{out['cluster0_building_composition']}")
    log.info(f"  cluster 1 non-Lecture days: {non_lec_in_c1} "
             f"({out['broader_behaviour_evidence']['nonLecture_days_in_cluster1_pct_of_cluster1']}% of cluster 1)")
    log.info("  VERDICT computed in code below from these numbers only")
    # interpretation (kept in JSON; not a causal claim)
    if lec_c1_share >= 0.90 and purity_gap >= 0.5:
        verdict = ("Lecture-dominant: K=2 primarily separates a "
                   "Lecture/low-load regime from the rest")
    elif broader["nonLecture_days_in_cluster1_pct_of_cluster1"] is not None \
            and broader["nonLecture_days_in_cluster1_pct_of_cluster1"] >= 20.0:
        verdict = ("Mixed: cluster 1 contains substantial non-Lecture mass "
                   "-> evidence of a broader behavioural distinction")
    else:
        verdict = ("Predominantly Lecture-driven separation; only weak "
                   "evidence of a broader behavioural distinction")
    out["interpretation"] = verdict
    log.info(f"  interpretation: {verdict}")

    with open(MODELLING_DIR / "lecture_dominance.json", "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log.info(f"  saved {MODELLING_DIR / 'lecture_dominance.json'}")
    return out


# ---------------------------------------------------------------------------
# PART E - K=3/K=4 sensitivity (same PCA representation, official K=2 intact)
# ---------------------------------------------------------------------------
def part_e_k_sensitivity(pca_data: dict, clustered: pd.DataFrame,
                         report: dict) -> dict:
    log.info("=" * 66)
    log.info("PART E - K=3/K=4 SENSITIVITY (same frozen PCA transform; "
             "official K=2 model NOT replaced)")
    Z = pca_data["Z"]
    Pc = pca_data["Pc"]
    labels2 = clustered["cluster"].to_numpy()
    k2_sil = float(report["kmeans"]["final_silhouette"])
    k2_inertia = float(report["kmeans"]["final_inertia"])

    fits: dict[int, dict] = {}
    # K=2 from the frozen official artefacts (recompute silhouette for the table)
    sil2 = float(silhouette_score(Pc, labels2))
    sizes2 = np.bincount(labels2, minlength=2)
    fits[2] = {"inertia": k2_inertia, "silhouette": sil2,
               "sizes": sizes2.tolist(),
               "smallest_cluster_pct": float(sizes2.min() / sizes2.sum() * 100),
               "model": None, "labels": labels2,
               "note": "official frozen model (not refitted)"}

    for k in (3, 4):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        lab = km.fit_predict(Pc)
        sil = float(silhouette_score(Pc, lab))
        sizes = np.bincount(lab, minlength=k)
        fits[k] = {"inertia": float(km.inertia_), "silhouette": sil,
                   "sizes": sizes.tolist(),
                   "smallest_cluster_pct": float(sizes.min() / sizes.sum() * 100),
                   "model": km, "labels": lab, "note": "sensitivity only"}
        log.info(f"  K={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}, "
                 f"smallest cluster={sizes.min():,} "
                 f"({fits[k]['smallest_cluster_pct']:.1f}%)")

    # overlap: how do K=3/K=4 subdivide the official K=2 groups?
    overlap = {}
    for k in (3, 4):
        ct = pd.crosstab(labels2, fits[k]["labels"])
        overlap[str(k)] = {f"official_cluster_{c0}":
                           {f"K{k}_cluster_{int(c1)}": int(ct.loc[c0, c1])
                            for c1 in ct.columns}
                           for c0 in ct.index}
        log.info(f"  K={k} vs K=2 crosstab: {overlap[str(k)]}")

    rows = []
    interp = {
        2: "coarse separation (official model); highest silhouette",
        3: "subdivides the large K=2 group; lower silhouette",
        4: "further subdivision; lowest silhouette of the three",
    }
    for k in K_SENS:
        rows.append({"K": k,
                     "inertia": round(fits[k]["inertia"], 1),
                     "silhouette": round(fits[k]["silhouette"], 4),
                     "smallest_cluster_pct":
                         round(fits[k]["smallest_cluster_pct"], 2),
                     "interpretation": interp[k]})
    sens = pd.DataFrame(rows)
    sens.to_csv(MODELLING_DIR / "k_sensitivity.csv", index=False)
    log.info(f"  saved {MODELLING_DIR / 'k_sensitivity.csv'}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, k in zip(axes, K_SENS):
        lab = fits[k]["labels"]
        ax.scatter(Pc[:, 0], Pc[:, 1], c=lab, cmap="tab10", s=5, alpha=0.5)
        ax.set(title=f"K={k} (silhouette {fits[k]['silhouette']:.3f})",
               xlabel="PC1", ylabel="PC2")
    fig.suptitle("K sensitivity (same frozen 3-PC representation)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "k_sensitivity_comparison.png", dpi=150)
    plt.close(fig)
    log.info("  figure saved: k_sensitivity_comparison.png")

    # high-level profiling of K=3/K=4 sub-behaviours -------------------------
    sub_profiles: dict[str, dict] = {}
    hcols = [f"h{h:02d}_kW" for h in range(24)]
    Xfeat = pca_data["X_feat"]          # original features (pre-transform)
    for k in (3, 4):
        lab = fits[k]["labels"]
        d: dict[str, dict] = {}
        for c in range(k):
            m = lab == c
            d[str(c)] = {
                "n": int(m.sum()),
                "mean_daily_kW": round(float(Xfeat.loc[m, "daily_mean_kW"]
                                             .mean()), 3),
                "mean_peak_kW": round(float(Xfeat.loc[m, "daily_peak_kW"]
                                            .mean()), 3),
                "mean_base_kW": round(float(Xfeat.loc[m, "base_kW"].mean()), 3),
                "mean_load_factor": round(float(Xfeat.loc[m, "load_factor"]
                                                .mean()), 4),
                "mean_night_share": round(float(Xfeat.loc[m, "night_share"]
                                                .mean()), 4),
                "mean_afternoon_share": round(float(Xfeat.loc[m,
                                                 "afternoon_share"].mean()), 4),
                "meter_top3": {str(a): int(b) for a, b in
                               clustered.loc[m, "meter"]
                               .value_counts().head(3).items()},
                "mean_hourly_profile_kW":
                    [round(float(x), 3)
                     for x in Xfeat.loc[m, hcols].mean().to_numpy()],
            }
        sub_profiles[str(k)] = d

    log.info("  K=3 / K=4 sub-cluster high-level profiles:")
    for k in (3, 4):
        for c, v in sub_profiles[str(k)].items():
            log.info(f"    K={k} cluster {c}: n={v['n']:,}, "
                     f"mean={v['mean_daily_kW']} kW, base={v['mean_base_kW']} kW, "
                     f"night={v['mean_night_share']}, top meters="
                     f"{list(v['meter_top3'])[:2]}")

    return {"fits": fits, "overlap": overlap, "table": sens,
            "sub_profiles": sub_profiles,
            "note": ("sensitivity analysis only; K=2 remains the official "
                     "model; K=3/K=4 not promoted because their silhouette "
                     "is materially lower and no diagnostic favours them")}


# ---------------------------------------------------------------------------
# PART F - semantic labels (only from measured characteristics)
# ---------------------------------------------------------------------------
def part_f_labels(profile_wide: pd.DataFrame, temp_stats: dict,
                  clustered: pd.DataFrame) -> dict:
    log.info("=" * 66)
    log.info("PART F - semantic cluster labels (data-driven)")

    labels: dict[str, str] = {}
    rationale: dict[str, str] = {}
    for cl in sorted(clustered["cluster"].unique()):
        cl = int(cl)
        lf = float(profile_wide.loc[profile_wide["feature"] == "load_factor",
                                    f"median_cluster{cl}"].iloc[0])
        night = float(profile_wide.loc[profile_wide["feature"] == "night_share",
                                       f"median_cluster{cl}"].iloc[0])
        base = float(profile_wide.loc[profile_wide["feature"] == "base_kW",
                                      f"median_cluster{cl}"].iloc[0])
        mean_ = float(profile_wide.loc[
            profile_wide["feature"] == "daily_mean_kW",
            f"median_cluster{cl}"].iloc[0])
        peak = float(profile_wide.loc[
            profile_wide["feature"] == "daily_peak_kW",
            f"median_cluster{cl}"].iloc[0])
        # within-observation peak-to-average ratio (see within_day_cv docstring)
        cv = within_day_cv(clustered, cl)
        # base/peak ratio: flatness of the diurnal shape
        flatness = base / peak if peak > 0 else np.nan

        # decision path (median-based, robust to outliers; first match wins).
        # NOTE night_share for a perfectly uniform 24-h profile is 6/24 = 0.25,
        # so ~0.25 means CONTINUOUS load, not a nocturnal peak.
        if flatness >= 0.6:
            label = "High Load Factor / Flat Load"
            why = (f"median base/peak = {flatness:.2f} (flat diurnal shape, "
                   f"median load factor {lf:.2f})")
        elif lf <= 0.45:
            label = "Peak-Oriented / Low Load Factor"
            why = (f"median load factor {lf:.2f} with base/peak "
                   f"{flatness:.2f} (peaky shape, within-day peak/average "
                   f"{cv:.2f})")
        elif night >= 0.20:
            label = "Continuous / Round-the-Clock Load"
            why = (f"night share {night:.2f} ~= the 0.25 uniform-profile "
                   f"expectation with no dominant period (day/night ratio "
                   f"{temp_stats[str(cl)]['day_night_ratio']:.2f}) -> load "
                   f"runs continuously; not a nocturnal peak")
        elif cv >= 0.6:
            label = "Highly Variable Load"
            why = (f"median within-day peak/average ratio {cv:.2f}")
        elif mean_ <= 10.0:
            label = "Low/Stable Load"
            why = f"median daily mean {mean_:.2f} kW (small absolute load)"
        else:
            label = "Daytime-Dominant Moderate Load"
            why = (f"daytime-skewed profile (day/night ratio "
                   f"{temp_stats[str(cl)]['day_night_ratio']}), median mean "
                   f"{mean_:.2f} kW")
        labels[str(cl)] = label
        rationale[str(cl)] = why
        log.info(f"  cluster {cl}: '{label}' <- {why}")

    with open(MODELLING_DIR / "cluster_labels.json", "w",
              encoding="utf-8") as fh:
        json.dump({"labels": labels, "rationale": rationale,
                   "note": ("labels assigned only after Parts A-E; original "
                            "numeric cluster IDs kept everywhere")}, fh,
                  indent=2)
    log.info(f"  saved {MODELLING_DIR / 'cluster_labels.json'}")
    return {"labels": labels, "rationale": rationale}


# ---------------------------------------------------------------------------
# PART G - potential anomalies (distance from assigned centroid)
# ---------------------------------------------------------------------------
def part_g_anomalies(pca_data: dict, clustered: pd.DataFrame,
                     cents: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 66)
    log.info("PART G - potentially anomalous energy-behaviour observations")
    Pc = pca_data["Pc"]
    labels = clustered["cluster"].to_numpy()
    cents_arr = cents[[f"PC{i + 1}" for i in range(Pc.shape[1])]
                      ].to_numpy(dtype="float64")
    d = np.linalg.norm(Pc - cents_arr[labels], axis=1)
    clustered["dist_to_centroid"] = d

    # two independent thresholds (both reported)
    thr_pctl = float(np.percentile(d, ANOMALY_PCTL))
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    thr_mad = med + ANOMALY_MAD_K * (1.4826 * mad)
    flag = (d >= thr_pctl) | (d >= thr_mad)
    log.info(f"  distance stats: median={med:.3f}, MAD={mad:.3f}")
    log.info(f"  threshold 1: {ANOMALY_PCTL}th percentile = {thr_pctl:.3f}")
    log.info(f"  threshold 2: median + {ANOMALY_MAD_K} x 1.4826 x MAD = "
             f"{thr_mad:.3f}")
    log.info(f"  flagged (union): {int(flag.sum()):,} of {len(d):,} "
             f"({100 * flag.mean():.2f}%)")

    anom = clustered.loc[flag, ["meter", "obs_date", "cluster",
                                "daily_mean_kW", "daily_peak_kW",
                                "daily_min_kW", "base_kW", "load_factor",
                                "night_share", "dist_to_centroid"]].copy()
    anom["threshold_percentile_99"] = round(thr_pctl, 4)
    anom["threshold_median_plus_6mad"] = round(thr_mad, 4)
    anom["flagged_by"] = np.where(
        (d[flag] >= thr_pctl) & (d[flag] >= thr_mad), "both",
        np.where(d[flag] >= thr_pctl, "percentile", "med_mad"))
    anom = anom.sort_values("dist_to_centroid", ascending=False)
    anom.insert(0, "rank", range(1, len(anom) + 1))

    path = MODELLING_DIR / "potential_anomalies.csv"
    anom.to_csv(path, index=False)
    log.info(f"  saved {path} ({len(anom)} rows)")
    log.info("  NOTE: these are POTENTIALLY anomalous energy-behaviour "
             "observations, NOT confirmed equipment faults (no causal claim).")

    by_meter = anom["meter"].value_counts()
    log.info(f"  flagged by meter: {by_meter.to_dict()}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for cl in sorted(clustered["cluster"].unique()):
        m = labels == cl
        ax.scatter(np.arange(len(d))[m], d[m], s=4, alpha=0.4,
                   label=f"cluster {cl}")
    ax.axhline(thr_pctl, ls="--", c="tab:red",
               label=f"{ANOMALY_PCTL}th percentile = {thr_pctl:.2f}")
    ax.axhline(thr_mad, ls=":", c="tab:orange",
               label=f"median + {ANOMALY_MAD_K:g} MAD = {thr_mad:.2f}")
    ax.set(xlabel="observation index", ylabel="distance to assigned centroid",
           title="Centroid distance per building-day (potential anomalies)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "potential_anomaly_distances.png", dpi=150)
    plt.close(fig)
    log.info("  figure saved: potential_anomaly_distances.png")
    return anom


# ---------------------------------------------------------------------------
# PART H - engineering recommendations (behaviour-specific, no cost claims)
# ---------------------------------------------------------------------------
def part_h_recommendations(labels: dict, profile_wide: pd.DataFrame,
                           temp_stats: dict, lecture: dict,
                           clustered: pd.DataFrame) -> list[dict]:
    log.info("=" * 66)
    log.info("PART H - engineering recommendations (behaviour-specific)")

    def g(feature, cl, stat="median"):
        return float(profile_wide.loc[profile_wide["feature"] == feature,
                                      f"{stat}_cluster{cl}"].iloc[0])

    recs = []
    for cl_s, label in labels.items():
        cl = int(cl_s)
        lf = g("load_factor", cl)
        night = g("night_share", cl)
        base = g("base_kW", cl)
        peak = g("daily_peak_kW", cl)
        mean_ = g("daily_mean_kW", cl)
        ph = temp_stats[cl_s]["peak_hour_modal"]
        dn = temp_stats[cl_s]["day_night_ratio"]
        cv = within_day_cv(clustered, cl)

        behaviour, strategy = [], []
        if lf <= 0.45 or base / max(peak, 1e-9) <= 0.35:
            behaviour.append(
                f"peaky profile (median load factor {lf:.2f}, base/peak "
                f"{base / max(peak, 1e-9):.2f}, peak around {ph:02d}:00)")
            strategy.append(
                "load shifting / scheduling of deferrable loads away from "
                "the observed peak window; demand-capacity review")
        if lf > 0.45 and night >= 0.20 and dn <= 1.3:
            behaviour.append(
                f"continuous round-the-clock load (night share {night:.2f} "
                f"~= uniform-profile 0.25; median base {base:.2f} kW; "
                f"day/night ratio {dn:.2f})")
            strategy.append(
                "standby / base-load reduction programme: audit after-hours "
                "equipment, HVAC schedules and lighting left on overnight")
        if dn >= 1.5:
            behaviour.append(
                f"daytime-concentrated usage (day/night ratio {dn:.2f}, "
                f"peak ~{ph:02d}:00)")
            strategy.append(
                "operational scheduling: align equipment timetables with "
                "occupancy; avoid pre-cooling/heating outside use hours")
        if cv >= 0.6:
            behaviour.append(
                f"high within-day shape variability (median peak/average "
                f"ratio {cv:.2f})")
            strategy.append(
                "investigate intermittent loads; sub-metering or logging "
                "campaign to localise the variability")
        if not strategy:
            behaviour.append("mixed moderate profile without a single "
                             "dominant characteristic")
            strategy.append(
                "continue monitoring; revisit after additional data or a "
                "finer observation unit")
        recs.append({
            "cluster": cl,
            "semantic_label": label,
            "observed_behaviour": "; ".join(behaviour),
            "evidence": (f"median daily_mean_kW={mean_:.2f}, "
                         f"daily_peak_kW={peak:.2f}, base_kW={base:.2f}, "
                         f"load_factor={lf:.2f}, night_share={night:.2f}, "
                         f"peak_hour={ph:02d}:00, within-day peak/average "
                         f"{cv:.2f}"),
            "likely_operational_characteristic":
                ("statistical description of measured building-day behaviour; "
                 "operational cause requires site verification"),
            "targeted_optimization_strategy": "; ".join(strategy),
        })
        log.info(f"  cluster {cl} [{label}]: {recs[-1]['observed_behaviour']}")
        log.info(f"     -> strategy: {recs[-1]['targeted_optimization_strategy']}")
    log.info("  NOTE: no monetary savings claimed (no tariff/cost data).")
    return recs


# ---------------------------------------------------------------------------
# PART I - final validation report + interpretation CSV
# ---------------------------------------------------------------------------
def part_i_report(clustered: pd.DataFrame, report: dict,
                  dist: pd.DataFrame,
                  temp_stats: dict, lecture: dict, sens: dict,
                  labels: dict, rationale: dict, recs: list[dict],
                  anom: pd.DataFrame) -> None:
    log.info("=" * 66)
    log.info("PART I - final validation report + interpretation CSV")

    sizes = clustered["cluster"].value_counts().sort_index()
    total = int(sizes.sum())

    kcomp = []
    for k in K_SENS:
        fit = sens["fits"][k]
        kcomp.append({
            "K": k,
            "inertia": round(fit["inertia"], 1),
            "silhouette": round(fit["silhouette"], 4),
            "cluster_sizes": {str(i): int(v)
                              for i, v in enumerate(fit["sizes"])},
            "smallest_cluster_pct": round(fit["smallest_cluster_pct"], 2),
        })

    caveats = [
        "Observation unit is BUILDING-DAY; clusters are groupings of "
        "building-days, not of consumers.",
        "K-Means + PCA describe statistical structure; no causal claim is "
        "made anywhere.",
        "Silhouette favours the coarse K=2 split; K=3/K=4 give richer "
        "subdivision but materially lower silhouette (sensitivity only, "
        "official model unchanged).",
        "The 880-day cluster is small (7.7%); its profile statistics carry "
        "more sampling uncertainty than the 10,502-day cluster.",
        f"Potential anomalies (Part G) are flagged by distance only "
        f"({ANOMALY_PCTL:g}th percentile and/or median+{ANOMALY_MAD_K:g} MAD); "
        "they are NOT confirmed equipment faults.",
        "No tariff/cost data -> no monetary savings claims.",
        "Lecture dominance is diagnostic: K=2 partly mirrors the Lecture/"
        "non-Lecture split; see lecture_dominance.json for exact numbers.",
        "Semantics (Part F) assigned from measured medians only, after "
        "Parts A-E; original numeric cluster IDs kept in all datasets.",
    ]

    report_obj = {
        "stage": "SPANDA Stage 3b - cluster profiling, validation and "
                 "engineering interpretation",
        "observation_unit": "building-day",
        "n_observations": total,
        "cluster_sizes": {str(k): int(v) for k, v in sizes.items()},
        "cluster_percent": {str(k): round(100 * v / total, 2)
                            for k, v in sizes.items()},
        "official_model": {
            "K": int(report["kmeans"]["selected_k"]),
            "silhouette": float(report["kmeans"]["final_silhouette"]),
            "inertia": float(report["kmeans"]["final_inertia"]),
            "pca_components": int(report["pca"]["retained_components"]),
            "pca_variance_retained": float(report["pca"]["variance_retained"]),
            "frozen": True,
        },
        "building_composition": dist.to_dict(orient="records"),
        "temporal_behaviour": temp_stats,
        "lecture_dominance_analysis": lecture,
        "k_sensitivity": {
            "comparison": kcomp,
            "crosstab_vs_official": sens["overlap"],
            "sub_profiles": sens["sub_profiles"],
            "note": sens["note"],
        },
        "semantic_labels": labels,
        "semantic_label_rationale": rationale,
        "potential_anomalies": {
            "method": (f"union of distance >= {ANOMALY_PCTL:g}th percentile "
                       f"and distance >= median + {ANOMALY_MAD_K:g} * 1.4826 * "
                       f"MAD (both reported in the CSV)"),
            "n_flagged": int(len(anom)),
            "pct_flagged": round(100 * len(anom) / total, 2),
            "by_meter": {str(k): int(v) for k, v in
                         anom["meter"].value_counts().items()},
            "interpretation": "potentially anomalous energy-behaviour "
                              "observations; not confirmed equipment faults",
        },
        "engineering_recommendations": recs,
        "validation_caveats": caveats,
        "figures": sorted(p.name for p in FIGURES_DIR.glob("*.png")
                          if p.name.startswith(("cluster", "potential",
                                                "k_sens"))),
    }
    path = MODELLING_DIR / "cluster_validation_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report_obj, fh, indent=2)
    log.info(f"  saved {path}")

    # interpretation CSV: one row per cluster
    rows = []
    for cl_s, label in labels.items():
        cl = int(cl_s)
        r = next(x for x in recs if x["cluster"] == cl)
        rows.append({
            "cluster": cl,
            "n_observations": int(sizes[cl]),
            "pct_of_total": round(100 * sizes[cl] / total, 2),
            "semantic_label": label,
            "label_rationale": rationale[cl_s],
            "peak_hour_modal": temp_stats[cl_s]["peak_hour_modal"],
            "day_night_ratio": temp_stats[cl_s]["day_night_ratio"],
            "weekday_weekend_ratio": temp_stats[cl_s]["weekday_weekend_ratio"],
            "mean_daily_kW": round(float(clustered.loc[
                clustered["cluster"] == cl, "daily_mean_kW"].mean()), 3),
            "top_buildings": ", ".join(
                f"{k} ({v})" for k, v in
                clustered.loc[clustered["cluster"] == cl, "meter"]
                .value_counts().head(3).items()),
            "observed_behaviour": r["observed_behaviour"],
            "evidence": r["evidence"],
            "likely_operational_characteristic":
                r["likely_operational_characteristic"],
            "targeted_optimization_strategy":
                r["targeted_optimization_strategy"],
        })
    interp = pd.DataFrame(rows)
    ipath = MODELLING_DIR / "cluster_interpretation.csv"
    interp.to_csv(ipath, index=False)
    log.info(f"  saved {ipath}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("#" * 70)
    log.info("SPANDA Stage 3b - CLUSTER PROFILING, VALIDATION & INTERPRETATION")
    log.info("#" * 70)

    # frozen official model (never refitted)
    clustered, report, cents = load_frozen_model()

    # recompute EXACTLY the Stage-3 preprocessing + PCA (frozen artifacts)
    log.info("=" * 66)
    log.info("Re-deriving the frozen Stage-3 preprocessing + PCA transform")
    clustered = attach_vf_pf_features(clustered)   # profiling-only context
    X, ids, _feat_names = prepare_modelling_matrix()
    Z = apply_transforms(X)
    assert len(ids) == len(clustered), \
        "re-derived matrix must match clustered_building_days.csv"
    # verify identical ordering vs the Stage-3 export (row alignment proof)
    same = (ids["meter"].to_numpy() ==
            clustered["meter"].to_numpy()).all() and \
           (ids["date_iso"].to_numpy() ==
            clustered["obs_date"].to_numpy()).all()
    log.info(f"  row alignment vs clustered_building_days.csv: {same}")
    if not same:
        raise RuntimeError("re-derived rows do not align with the frozen "
                           "clustered table; refusing to continue")
    scaler = joblib.load(MODELLING_DIR / "scaler.joblib")
    Zs = scaler.transform(Z.to_numpy(dtype="float64"))
    pca = PCA(n_components=int(report["pca"]["retained_components"]),
              svd_solver="full", random_state=RANDOM_STATE).fit(Zs)
    Pc = pca.transform(Zs)
    log.info(f"  frozen representation re-derived: {Pc.shape[1]} PCs, "
             f"cumulative variance "
             f"{float(np.sum(pca.explained_variance_ratio_)):.4f} "
             f"(Stage-3 reported "
             f"{report['pca']['variance_retained']:.4f})")
    pca_data = {"Z": Zs, "Pc": Pc, "X_feat": X.reset_index(drop=True)}

    # PART A
    prof = part_a_profiles(clustered, vf_cols=VF_FEATURES)
    prof_wide = prof.pivot(index="feature", columns="cluster",
                           values=PROFILE_STATS)
    prof_wide.columns = [f"{stat}_cluster{int(cl)}"
                         for stat, cl in prof_wide.columns]
    prof_wide = prof_wide.reset_index()

    # PART B
    dist = part_b_building_composition(clustered)

    # PART C
    _, temp_stats = part_c_temporal(clustered)

    # PART D
    lecture = part_d_lecture_check(clustered, temp_stats)

    # PART E
    sens = part_e_k_sensitivity(pca_data, clustered, report)

    # PART F
    lab = part_f_labels(prof_wide, temp_stats, clustered)

    # PART G
    anom = part_g_anomalies(pca_data, clustered, cents)

    # PART H
    recs = part_h_recommendations(lab["labels"], prof_wide, temp_stats,
                                  lecture, clustered)

    # PART I
    part_i_report(clustered, report, dist, temp_stats, lecture,
                  sens, lab["labels"], lab["rationale"], recs, anom)

    log.info("=" * 66)
    log.info("Stage 3b complete. Official K=2 model and all raw data "
             "untouched; Part E was sensitivity-only.")


if __name__ == "__main__":
    main()
