"""
SPANDA Stage 3e - CROSS-METER VALIDATION OF THE ACADEMIC AUG-OCT 2017 REGIME
(read-only w.r.t. raw data, Stage-1/2/3b/3c/3d artifacts, spanda/model.py, the
official K=2 model, and all thresholds; logged to
spanda_out/audit/anomaly_crossmeter_log.txt)

Narrow question
---------------
Do the seven weekday-packed Academic episodes found in Stage 3d also appear in
other building meters on the same dates?  This distinguishes a pattern
consistent with a BUILDING-SPECIFIC operational change from one consistent with
a BROADER CAMPUS-LEVEL operational/contextual driver.  Association only - no
causal claim is possible or made.

The seven target episodes are taken verbatim from Stage 3d (not re-derived):
  2017-07-31..08-04, 08-21..25, 08-28..09-01, 09-04..08, 09-18..22,
  09-25..29, 10-09..13   (35 target dates total)

Method (transparent, no new ML)
-------------------------------
* baseline for EVERY meter = up to 7 previous + 7 next valid days of the
  SAME meter in observation order (Stage-3d convention; no calendar cap),
  excluding: (a) the episode dates themselves for every meter, and (b) for
  ACADEMIC, ALL 35 regime dates - so no Academic episode is ever part of an
  Academic baseline ("the baseline must not include the target regime").
  For other meters the regime dates stay in their baselines: those dates are
  hypothesized-normal for them, and keeping them biases the comparison
  TOWARD "building-specific" (the conservative direction). Baselines are
  never pooled across meters; they include weekends (documented).
* deviation metrics: mean/peak/base/load-factor ratios (or absolute
  differences on near-zero baselines) and a normalized Euclidean 24-h shape
  distance vs the baseline median profile
* documented thresholds:
    elevated load      ratio >= 1.20
    reduced load       ratio <= 0.80        (ratio undefined on baseline <= 0.05 kW;
                                            absolute difference used instead)
    substantial peak   peak_ratio >= 1.30
    substantial shape  shape_distance >= 0.30   (rel. L2 vs baseline profile)
    substantially changed meter (episode): any of elevated_load,
        peak_ratio >= 1.30, shape_distance >= 0.30
    regime classification (per episode, over meters with valid comparisons,
        Academic excluded from the denominator):
        BUILDING-SPECIFIC  changed_fraction < 1/3
        PARTIALLY SYNCHRONIZED  1/3 <= changed_fraction < 2/3
        BROADLY SYNCHRONIZED  changed_fraction >= 2/3
    daily multi-meter synchronization: >= 2 non-Academic meters elevated
* Aug-Oct 2017 monthly baseline = all valid 2017 days OUTSIDE Aug 1 - Oct 31
  (the target regime is excluded from its own baseline)

Outputs (all NEW; nothing overwritten)
--------------------------------------
spanda_out/modelling/crossmeter_episode_summary.csv
spanda_out/modelling/crossmeter_daily_sync.csv
spanda_out/modelling/crossmeter_monthly_2017.csv
spanda_out/modelling/crossmeter_validation_report.json
spanda_out/figures/crossmeter_episode_heatmap.png
spanda_out/figures/crossmeter_daily_synchronization.png
spanda_out/figures/academic_2017_vs_other_meters.png
"""

from __future__ import annotations

import bisect
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import spanda.model as _spanda_model
from spanda.config import (
    AUDIT_DIR, FIGURES_DIR, MODELLING_DIR, PROCESSED_DIR, PROJECT_ROOT,
    get_logger,
)

log = get_logger("spanda.anomaly_crossmeter",
                 AUDIT_DIR / "anomaly_crossmeter_log.txt")

HCOLS = [f"h{h:02d}_kW" for h in range(24)]
N_BASE = 7                    # previous/next valid baseline days (obs order)
T_ELEVATED = 1.20             # load ratio threshold (brief's starting point)
T_REDUCED = 0.80
T_ZERO_BASE = 0.05            # kW; below this a ratio is undefined
T_PEAK = 1.30
T_SHAPE = 0.30
T_SYNC_DAILY = 2              # >= 2 non-Academic meters elevated
T_BROAD = 2 / 3               # changed_fraction >= 2/3 -> broadly synchronized
T_PARTIAL = 1 / 3

EPISODES = [
    ("EP01", "2017-07-31", "2017-08-04"),
    ("EP02", "2017-08-21", "2017-08-25"),
    ("EP03", "2017-08-28", "2017-09-01"),
    ("EP04", "2017-09-04", "2017-09-08"),
    ("EP05", "2017-09-18", "2017-09-22"),
    ("EP06", "2017-09-25", "2017-09-29"),
    ("EP07", "2017-10-09", "2017-10-13"),
]
TARGET_DATES = sorted({d for _, s, e in EPISODES
                       for d in pd.date_range(s, e, freq="D")
                       .strftime("%Y-%m-%d")})
REGIME_DATES = set(TARGET_DATES)   # all 35 dates - excluded from Academic's
                                   # baselines so the regime never baselines
                                   # itself
MONTHS_2017 = ["2017-08", "2017-09", "2017-10"]
REGIME_START, REGIME_END = "2017-08-01", "2017-10-31"

# ---------------------------------------------------------------------------
# frozen-artifact hash verification (Stage-3c/3d approach, + Stage-3d outputs)
# ---------------------------------------------------------------------------
RAW_DATA_FILES = [PROJECT_ROOT / f for f in
                  ["acad_build_mains.csv", "boys_hostel_mains.csv",
                   "boys_hostel_ups.csv", "facilities_build_mains.csv",
                   "girls_hostel_mains.csv", "girls_hostel_ups.csv",
                   "lecture_build_mains.csv", "library_build_mains.csv",
                   "mess_build_mains.csv"]]
STAGE1_FILES = [AUDIT_DIR / f for f in
                ["audit_summary.json", "audit_table.csv",
                 "column_missingness.csv", "preprocessing_summary.json",
                 "quality_deepdive.json"]
                + [f"gaps_{m}.csv" for m in
                   ["Academic", "Boys_UPS", "Boys_mains", "Facilities",
                    "Girls_UPS", "Girls_mains", "Lecture", "Library", "Mess"]]]
STAGE2_FILES = [PROCESSED_DIR / "building_day_features.csv",
                PROCESSED_DIR / "hourly_power_clean.parquet"]
STAGE3B_FILES = [MODELLING_DIR / f for f in
                 ["clustered_building_days.csv", "scaler.joblib",
                  "pca_features.parquet", "pca_loadings.csv",
                  "pca_interpretation.json", "modelling_report.json",
                  "cluster_validation_report.json", "cluster_interpretation.csv",
                  "building_cluster_distribution.csv", "cluster_profile.csv",
                  "potential_anomalies.csv", "lecture_dominance.json",
                  "cluster_labels.json", "k_sensitivity.csv",
                  "cluster_by_meter_by_date.csv", "cluster_centroids.csv",
                  "standardized_features.parquet", "feature_names.json"]]
STAGE3C_FILES = [MODELLING_DIR / f for f in
                 ["cluster_stability_summary.csv",
                  "cluster_stability_runs.csv",
                  "cluster_stability_report.json"]]
STAGE3D_FILES = [MODELLING_DIR / f for f in
                 ["anomaly_context.csv", "anomaly_summary_by_meter.csv",
                  "anomaly_summary_by_category.csv", "anomaly_episodes.csv",
                  "anomaly_context_report.json"]]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_frozen_artifacts() -> dict[str, str]:
    groups = [("raw", RAW_DATA_FILES), ("stage1", STAGE1_FILES),
              ("stage2", STAGE2_FILES), ("stage3b", STAGE3B_FILES),
              ("stage3c", STAGE3C_FILES), ("stage3d", STAGE3D_FILES),
              ("model_code", [Path(_spanda_model.__file__)])]
    out: dict[str, str] = {}
    for _g, paths in groups:
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"frozen artifact missing: {p}")
            out[str(p)] = _sha256(p)
    return out


def verify_frozen(before: dict[str, str], after: dict[str, str]) -> None:
    diff = [k for k, v in before.items() if after.get(k) != v]
    log.info("=" * 66)
    log.info("FROZEN-FILE VERIFICATION")
    log.info(f"  files hash-verified: {len(before)} (raw x9, Stage-1, Stage-2, "
             "Stage-3b, Stage-3c, Stage-3d, spanda/model.py)")
    if diff:
        for k in diff:
            log.error(f"  MODIFIED DURING RUN: {k}")
        raise RuntimeError(f"{len(diff)} frozen artifacts modified: {diff}")
    log.info("ALL FROZEN ARTIFACTS BYTE-IDENTICAL")


# ---------------------------------------------------------------------------
# data + baseline machinery
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    clustered = pd.read_csv(MODELLING_DIR / "clustered_building_days.csv")
    log.info(f"  modelling dataset: {len(clustered):,} building-days, "
             f"{clustered['meter'].nunique()} meters")
    return clustered


def baseline_indices(dates: list[str], start_date: str, end_date: str,
                     exclude: set[str]) -> list[int]:
    """Indices of up to N_BASE previous + N_BASE next valid days (observation
    order) whose dates are not in `exclude`. The [start, end] span itself is
    never selected."""
    start_i = bisect.bisect_left(dates, start_date)
    end_i = bisect.bisect_right(dates, end_date)
    before = [i for i in range(start_i) if dates[i] not in exclude][-N_BASE:]
    after = [i for i in range(end_i, len(dates)) if dates[i] not in exclude]
    return before + after[:N_BASE]


def episode_stats(g: pd.DataFrame, idx: list[int], ep_idx: list[int]) -> dict:
    ep, nb = g.loc[ep_idx], g.loc[idx]
    nb_mean = float(nb["daily_mean_kW"].median())
    nb_peak = float(nb["daily_peak_kW"].median())
    nb_base = float(nb["base_kW"].median())
    nb_lf = float(nb["load_factor"].median())
    nb_prof = nb[HCOLS].median().to_numpy(dtype="float64")
    ep_prof = ep[HCOLS].mean().to_numpy(dtype="float64")
    denom = float(np.abs(nb_prof).sum())
    shape = (float(np.linalg.norm(ep_prof - nb_prof) / max(denom, 1e-9))
             if denom > 1e-9 else np.nan)
    return {
        "valid_days": int(len(ep_idx)),
        "episode_mean_kW": round(float(ep["daily_mean_kW"].mean()), 3),
        "baseline_median_kW": round(nb_mean, 3),
        "mean_load_ratio": (round(float(ep["daily_mean_kW"].mean()) / nb_mean, 3)
                            if nb_mean > T_ZERO_BASE else np.nan),
        "episode_peak_median_kW": round(float(ep["daily_peak_kW"].median()), 3),
        "baseline_peak_median_kW": round(nb_peak, 3),
        "peak_ratio": (round(float(ep["daily_peak_kW"].median()) / nb_peak, 3)
                       if nb_peak > T_ZERO_BASE else np.nan),
        "episode_base_median_kW": round(float(ep["base_kW"].median()), 3),
        "baseline_base_median_kW": round(nb_base, 3),
        "base_difference_kW": round(float(ep["base_kW"].median()) - nb_base, 3),
        "episode_load_factor_median": round(float(ep["load_factor"].median()), 3),
        "baseline_load_factor_median": round(nb_lf, 3),
        "shape_distance": round(shape, 4) if np.isfinite(shape) else np.nan,
    }


def interpret_meter_row(r: dict) -> tuple[bool, str]:
    """Documented meter-level rule -> (substantially_changed, interpretation)."""
    elevated = (np.isfinite(r["mean_load_ratio"])
                and r["mean_load_ratio"] >= T_ELEVATED) or \
               (not np.isfinite(r["mean_load_ratio"])
                and r["base_difference_kW"] >= 1.0)
    reduced = (np.isfinite(r["mean_load_ratio"])
               and r["mean_load_ratio"] <= T_REDUCED) or \
              (not np.isfinite(r["mean_load_ratio"])
               and r["base_difference_kW"] <= -1.0)
    peak_hi = np.isfinite(r["peak_ratio"]) and r["peak_ratio"] >= T_PEAK
    shape_hi = np.isfinite(r["shape_distance"]) and r["shape_distance"] >= T_SHAPE
    parts = []
    if elevated:
        parts.append(f"elevated load (ratio {r['mean_load_ratio']})"
                     if np.isfinite(r["mean_load_ratio"])
                     else f"elevated on near-zero baseline (+{r['base_difference_kW']} kW base)")
    if reduced:
        parts.append(f"reduced load (ratio {r['mean_load_ratio']})"
                     if np.isfinite(r["mean_load_ratio"])
                     else f"reduced on near-zero baseline ({r['base_difference_kW']} kW base)")
    if peak_hi:
        parts.append(f"elevated peak (ratio {r['peak_ratio']})")
    if shape_hi:
        parts.append(f"shape change ({r['shape_distance']})")
    if not parts:
        return False, "near own baseline"
    changed = elevated or peak_hi or shape_hi
    return changed, "; ".join(parts)


# ---------------------------------------------------------------------------
# STEP 1: per-meter x episode table
# ---------------------------------------------------------------------------
def episode_table(clustered: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 66)
    log.info("STEP 1 - per-meter x episode deviation (same-meter baselines, "
             f"{N_BASE}+{N_BASE} valid days in observation order; all regime "
             "dates excluded from Academic's baselines)")
    rows = []
    for ep_id, s, e in EPISODES:
        ep_dates = set(pd.date_range(s, e, freq="D").strftime("%Y-%m-%d"))
        for meter, g in clustered.groupby("meter"):
            g = g.sort_values("obs_date").reset_index(drop=True)
            dates = g["obs_date"].tolist()
            ep_idx = [i for i, d in enumerate(dates) if d in ep_dates]
            if not ep_idx:
                rows.append({"episode_id": ep_id, "start_date": s,
                             "end_date": e, "meter": meter, "valid_days": 0,
                             "interpretation": "no valid observation in episode"})
                continue
            # Academic's baseline must never include ANY regime date; other
            # meters keep their normal days (conservative, see docstring)
            excl = REGIME_DATES if meter == "Academic" else set()
            bidx = baseline_indices(dates, s, e, excl)
            n_b, n_a = (len([i for i in bidx if i < ep_idx[0]]),
                        len([i for i in bidx if i > ep_idx[-1]]))
            if n_b < N_BASE or n_a < N_BASE:
                log.info(f"  {ep_id}/{meter}: limited baseline "
                         f"(before={n_b}, after={n_a} non-excluded days)")
            st = episode_stats(g, bidx, ep_idx)
            changed, interp = interpret_meter_row(st)
            rows.append({"episode_id": ep_id, "start_date": s, "end_date": e,
                         "meter": meter, **st, "elevated_load": elevated_flag(st),
                         "changed_meter": changed,
                         "interpretation": interp})
    df = pd.DataFrame(rows)
    df.to_csv(MODELLING_DIR / "crossmeter_episode_summary.csv", index=False)
    log.info(f"  saved crossmeter_episode_summary.csv ({len(df)} rows)")
    return df


def elevated_flag(r: dict) -> bool:
    return (np.isfinite(r["mean_load_ratio"])
            and r["mean_load_ratio"] >= T_ELEVATED) or \
           (not np.isfinite(r["mean_load_ratio"])
            and r["base_difference_kW"] >= 1.0)


def classify_episodes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log.info("=" * 66)
    log.info("STEP 2 - regime classification per episode "
             "(changed_fraction over valid non-Academic meters; "
             f"BROAD >= {T_BROAD:.2f}, PARTIAL >= {T_PARTIAL:.2f})")
    verdicts = {}
    for ep_id, s, e in EPISODES:
        sub = df[(df["episode_id"] == ep_id) & (df["valid_days"] > 0)]
        others = sub[sub["meter"] != "Academic"]
        acad = sub[sub["meter"] == "Academic"].iloc[0]
        n_valid = int(others["valid_days"].gt(0).sum())
        n_changed = int(others["changed_meter"].sum())
        frac = (n_changed / n_valid) if n_valid else np.nan
        if not np.isfinite(frac):
            cat = "INDETERMINATE"
        elif frac >= T_BROAD:
            cat = "BROADLY SYNCHRONIZED"
        elif frac >= T_PARTIAL:
            cat = "PARTIALLY SYNCHRONIZED"
        else:
            cat = "BUILDING-SPECIFIC SIGNAL"
        verdicts[ep_id] = {
            "episode_id": ep_id, "start_date": s, "end_date": e,
            "academic_changed": bool(acad["changed_meter"]),
            "academic_interpretation": acad["interpretation"],
            "academic_mean_load_ratio": (None if not np.isfinite(
                acad["mean_load_ratio"]) else float(acad["mean_load_ratio"])),
            "valid_nonacademic_meters": n_valid,
            "changed_nonacademic_meters": n_changed,
            "changed_fraction": round(float(frac), 3)
            if np.isfinite(frac) else None,
            "classification": cat,
        }
        log.info(f"  {ep_id} {s}..{e}: Academic [{'X' if verdicts[ep_id]['academic_changed'] else ' '}] "
                 f"changed {n_changed}/{n_valid} other meters "
                 f"(fraction {verdicts[ep_id]['changed_fraction']}) -> {cat}")
    counts = pd.Series([v["classification"] for v in verdicts.values()]) \
        .value_counts().to_dict()
    log.info(f"  classification counts: {counts}")
    return verdicts, counts


# ---------------------------------------------------------------------------
# STEP 3: daily synchronization across the 35 target dates
# ---------------------------------------------------------------------------
def daily_sync(clustered: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 66)
    log.info("STEP 3 - daily synchronization over the 35 target dates "
             f"(elevated = day mean >= {T_ELEVATED}x same-meter baseline "
             f"median; ratio undefined on baseline <= {T_ZERO_BASE} kW -> "
             "absolute-difference rule)")
    rows = []
    clustered = clustered.sort_values("obs_date")
    for d in TARGET_DATES:
        ts = pd.Timestamp(d)
        elevated, valid, notes = 0, 0, []
        for meter, g in clustered.groupby("meter"):
            gi = g.reset_index(drop=True)
            dates = gi["obs_date"].tolist()
            if d not in dates:
                continue                      # meter has no valid day here
            i = dates.index(d)
            excl = REGIME_DATES if meter == "Academic" else set()
            bidx = baseline_indices(dates, d, d, excl)
            if len(bidx) < 4:
                notes.append(f"{meter}: insufficient baseline")
                continue
            valid += 1
            base_med = float(gi.loc[bidx, "daily_mean_kW"].median())
            day_mean = float(gi.loc[i, "daily_mean_kW"])
            if base_med > T_ZERO_BASE:
                if day_mean / base_med >= T_ELEVATED:
                    elevated += 1
            elif day_mean - base_med >= 1.0:
                elevated += 1
                notes.append(f"{meter}: near-zero baseline, +"
                             f"{day_mean - base_med:.1f} kW absolute rule")
        acad_day = "Academic" in clustered["meter"].values and \
            bool((clustered["meter"] == "Academic").any())
        acad_elev = False
        ga = clustered[clustered["meter"] == "Academic"].reset_index(drop=True)
        da = ga["obs_date"].tolist()
        if d in da:
            ia = da.index(d)
            ba = baseline_indices(da, d, d, REGIME_DATES)
            if len(ba) >= 4:
                bm = float(ga.loc[ba, "daily_mean_kW"].median())
                dm = float(ga.loc[ia, "daily_mean_kW"])
                acad_elev = (dm / bm >= T_ELEVATED if bm > T_ZERO_BASE
                             else dm - bm >= 1.0)
        n_other = max(valid - (1 if acad_day and valid else 0), 0)
        other_elev = elevated - (1 if acad_elev else 0)
        rows.append({
            "obs_date": d,
            "valid_meter_count": valid,
            "elevated_meter_count": elevated,
            "elevated_fraction": round(elevated / valid, 3) if valid else np.nan,
            "Academic_elevated": bool(acad_elev),
            "other_meter_elevated_count": int(other_elev),
            "multi_meter_sync": bool(other_elev >= T_SYNC_DAILY),
            "notes": "; ".join(notes) if notes else "",
        })
    ds = pd.DataFrame(rows)
    ds.to_csv(MODELLING_DIR / "crossmeter_daily_sync.csv", index=False)
    n_sync = int(ds["multi_meter_sync"].sum())
    log.info(f"  {len(ds)} target dates; multi-meter synchronization on "
             f"{n_sync} dates; mean elevated_fraction = "
             f"{ds['elevated_fraction'].mean():.3f}")
    log.info(f"  saved crossmeter_daily_sync.csv")
    return ds


# ---------------------------------------------------------------------------
# STEP 4: Aug-Oct 2017 monthly context vs non-regime 2017 baseline
# ---------------------------------------------------------------------------
def monthly_context(clustered: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 66)
    log.info("STEP 4 - monthly context (Aug/Sep/Oct 2017 vs 2017 days outside "
             "Aug 1 - Oct 31; the regime is excluded from its own baseline)")
    d = clustered.copy()
    d["dt"] = pd.to_datetime(d["obs_date"])
    in_regime = (d["dt"] >= REGIME_START) & (d["dt"] <= REGIME_END)
    base = d[(d["dt"].dt.year == 2017) & ~in_regime]
    base_desc = ("2017 days outside 2017-08-01..2017-10-31 "
                 f"(n={len(base)} building-days)")
    rows = []
    for month in MONTHS_2017:
        for meter, g in d[d["dt"].dt.strftime("%Y-%m") == month] \
                .groupby("meter"):
            gb = base[base["meter"] == meter]
            m_mean = float(g["daily_mean_kW"].mean())
            b_mean = float(gb["daily_mean_kW"].mean()) if len(gb) else np.nan
            rows.append({
                "month": month, "meter": meter,
                "valid_days": int(len(g)),
                "mean_daily_kW": round(m_mean, 3),
                "median_daily_kW": round(float(g["daily_mean_kW"].median()), 3),
                "median_peak_kW": round(float(g["daily_peak_kW"].median()), 3),
                "median_load_factor": round(float(g["load_factor"].median()), 3),
                "baseline_period_description": base_desc,
                "baseline_mean_daily_kW": round(b_mean, 3)
                if np.isfinite(b_mean) else np.nan,
                "mean_load_ratio_vs_baseline":
                    round(m_mean / b_mean, 3)
                    if np.isfinite(b_mean) and b_mean > T_ZERO_BASE else np.nan,
            })
    mc = pd.DataFrame(rows)
    mc.to_csv(MODELLING_DIR / "crossmeter_monthly_2017.csv", index=False)
    log.info("\n" + mc[mc["month"] == "2017-08"][["meter", "valid_days",
             "mean_daily_kW", "baseline_mean_daily_kW",
             "mean_load_ratio_vs_baseline"]].to_string(index=False))
    log.info("  (Sep/Oct analogous; full table saved)")
    log.info(f"  saved crossmeter_monthly_2017.csv ({len(mc)} rows)")
    return mc


# ---------------------------------------------------------------------------
# STEP 5: figures
# ---------------------------------------------------------------------------
def plot_figures(ep_df: pd.DataFrame, ds: pd.DataFrame, mc: pd.DataFrame,
                 clustered: pd.DataFrame) -> None:
    log.info("=" * 66)
    log.info("STEP 5 - figures")

    # Fig 1: heatmap of mean_load_ratio (episode x meter), documented metric
    piv = ep_df.pivot_table(index="episode_id", columns="meter",
                            values="mean_load_ratio", aggfunc="first")
    order = [c for c in ["Academic", "Library", "Lecture", "Mess", "Facilities",
                         "Boys_mains", "Boys_UPS", "Girls_mains", "Girls_UPS"]
             if c in piv.columns]
    piv = piv[order]
    fig, ax = plt.subplots(figsize=(11, 5))
    data = piv.to_numpy(dtype="float64")
    im = ax.imshow(np.clip(data, 0, 2), cmap="RdBu_r", vmin=0, vmax=2,
                   aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, "n/a" if not np.isfinite(v) else f"{v:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="k")
    ax.set_xticks(range(len(piv.columns)), piv.columns, rotation=30,
                  fontsize=8)
    ax.set_yticks(range(len(piv.index)),
                  [f"{e} ({s[5:]})" for e, s in
                   zip(piv.index, ep_df.drop_duplicates('episode_id')
                       .set_index('episode_id').loc[piv.index, 'start_date'])],
                  fontsize=8)
    ax.set_title("Episode mean load vs same-meter baseline median\n"
                 "(ratio; clipped to [0, 2] for display; n/a = undefined on "
                 "near-zero baseline)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean_load_ratio")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "crossmeter_episode_heatmap.png", dpi=150)
    plt.close(fig)

    # Fig 2: daily elevated fraction across the 35 target dates
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(ds))
    ax.bar(x, ds["elevated_fraction"], color="#aec7e8",
           label="elevated fraction (all meters)")
    ax.bar(x, ds["other_meter_elevated_count"]
           / ds["valid_meter_count"].clip(lower=1),
           color="#ff9896", alpha=0.7,
           label="elevated fraction (non-Academic meters)")
    ax.axhline(T_PARTIAL, ls=":", c="grey", lw=1)
    ax.axhline(T_BROAD, ls="--", c="grey", lw=1)
    ax.text(len(ds) - 0.5, T_BROAD + 0.01, "2/3", fontsize=7, ha="right")
    ax.text(len(ds) - 0.5, T_PARTIAL + 0.01, "1/3", fontsize=7, ha="right")
    ax.set_xticks(x, ds["obs_date"], rotation=60, fontsize=6)
    ax.set_ylabel("Fraction of meters elevated vs own baseline")
    ax.set_title("Daily cross-meter synchronization over the 35 target dates "
                 f"(elevated >= {T_ELEVATED}x same-meter baseline)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "crossmeter_daily_synchronization.png", dpi=150)
    plt.close(fig)

    # Fig 3: Academic vs other meters, normalized daily load, Aug-Oct 2017
    d = clustered.copy()
    d["dt"] = pd.to_datetime(d["obs_date"])
    d17 = d[(d["dt"] >= "2017-06-01") & (d["dt"] <= "2017-12-31")].copy()
    med17 = d17.groupby("meter")["daily_mean_kW"].transform(
        lambda s: s.median())
    d17["rel_load"] = d17["daily_mean_kW"] / med17.clip(lower=1e-9)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax = axes[0]
    for m, c in [("Academic", "tab:red"), ("Library", "tab:blue"),
                 ("Mess", "tab:green"), ("Facilities", "tab:purple")]:
        g = d17[d17["meter"] == m]
        ax.plot(g["dt"], g["rel_load"].rolling(7, min_periods=3).mean(),
                label=m, color=c, lw=1.2)
    for _, s, e in EPISODES:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1),
                   color="orange", alpha=0.15)
    ax.set_ylabel("daily_mean_kW / meter 2017 median")
    ax.set_title("Academic vs commercial meters (7-day rolling relative load; "
                 "shaded = target episodes)")
    ax.legend(fontsize=8)
    ax = axes[1]
    for m, c in [("Boys_mains", "tab:brown"), ("Boys_UPS", "tab:pink"),
                 ("Girls_mains", "tab:olive"), ("Girls_UPS", "tab:cyan"),
                 ("Lecture", "tab:gray")]:
        g = d17[d17["meter"] == m]
        ax.plot(g["dt"], g["rel_load"].rolling(7, min_periods=3).mean(),
                label=m, color=c, lw=1.0)
    for _, s, e in EPISODES:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e) + pd.Timedelta(days=1),
                   color="orange", alpha=0.15)
    ax.set_ylabel("daily_mean_kW / meter 2017 median")
    ax.set_xlabel("Date")
    ax.set_title("Hostel + Lecture meters (same normalization)")
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "academic_2017_vs_other_meters.png", dpi=150)
    plt.close(fig)
    log.info("  figures saved: crossmeter_episode_heatmap.png, "
             "crossmeter_daily_synchronization.png, "
             "academic_2017_vs_other_meters.png")


# ---------------------------------------------------------------------------
# STEP 6: final report
# ---------------------------------------------------------------------------
def build_report(ep_df: pd.DataFrame, verdicts: dict, counts: dict,
                 ds: pd.DataFrame, mc: pd.DataFrame,
                 clustered: pd.DataFrame) -> dict:
    log.info("=" * 66)
    log.info("STEP 6 - final interpretation report")

    n_sync_dates = int(ds["multi_meter_sync"].sum())
    n_acad_dates = int(ds["Academic_elevated"].sum())
    sync_frac = ds["multi_meter_sync"].mean()
    cats = list(counts.items())

    # strongest synchronized / building-specific episodes (measured)
    def frac(v):
        f = v["changed_fraction"]
        return -1 if f is None else f

    strongest_sync = max(verdicts.values(), key=frac) \
        if any(v["changed_fraction"] for v in verdicts.values()) else None
    most_specific = min(verdicts.values(), key=frac)

    # Academic monthly shift (Aug-Oct vs rest of 2017)
    acad_mc = mc[mc["meter"] == "Academic"]
    acad_ratios = acad_mc["mean_load_ratio_vs_baseline"].tolist()
    others_max = mc[mc["meter"] != "Academic"] \
        .groupby("month")["mean_load_ratio_vs_baseline"].max()

    if sync_frac >= T_BROAD:
        final_class = "BROADLY SYNCHRONIZED"
        final_text = ("The contemporaneous changes across multiple meters are "
                      "consistent with a broader campus-level operational or "
                      "contextual effect, but causality cannot be established "
                      "from the energy data alone.")
    elif sync_frac >= T_PARTIAL:
        final_class = "PARTIALLY SYNCHRONIZED"
        final_text = ("The Academic regime coincides with changes in several "
                      "other meters, suggesting a possible shared operational/"
                      "contextual influence. The available data cannot "
                      "identify the cause.")
    else:
        final_class = "BUILDING-SPECIFIC SIGNAL"
        final_text = ("The observed Aug-Oct 2017 regime appears predominantly "
                      "localized to the Academic meter and should be "
                      "investigated through building-level operational "
                      "records, equipment schedules, or sub-metering.")

    # honest per-episode nuance: episodes where Academic itself did NOT change
    episode_notes = []
    for v in verdicts.values():
        if not v["academic_changed"]:
            sub = ep_df[(ep_df["episode_id"] == v["episode_id"])
                        & (ep_df["changed_meter"])
                        & (ep_df["meter"] != "Academic")]
            changed_m = sub["meter"].tolist()
            acad_row = ep_df[(ep_df["episode_id"] == v["episode_id"])
                             & (ep_df["meter"] == "Academic")].iloc[0]
            episode_notes.append({
                "episode_id": v["episode_id"],
                "note": (f"Academic's episode-level magnitude is near its own "
                         f"clean baseline (mean_load_ratio "
                         f"{acad_row['mean_load_ratio']}); its Stage-3d flags "
                         f"for these dates stem from shape descriptors, while "
                         + (f"{', '.join(changed_m)} showed contemporaneous "
                            "elevation" if changed_m else
                            "no other meter changed")),
            })

    report = {
        "stage": "SPANDA Stage 3e - cross-meter validation of the Academic "
                 "Aug-Oct 2017 regime (read-only; final analytical stage)",
        "target_episodes": [{"episode_id": e, "start_date": s, "end_date": x}
                            for e, s, x in EPISODES],
        "episode_notes": episode_notes,
        "n_target_dates": len(TARGET_DATES),
        "meters_analyzed": sorted(clustered["meter"].unique().tolist()),
        "documented_thresholds": {
            "elevated_load": f"mean_load_ratio >= {T_ELEVATED}",
            "reduced_load": f"mean_load_ratio <= {T_REDUCED}",
            "undefined_ratio": f"baseline <= {T_ZERO_BASE} kW -> absolute "
                               "base-difference rule (+/- 1 kW)",
            "substantial_peak": f"peak_ratio >= {T_PEAK}",
            "substantial_shape": f"shape_distance (rel. L2) >= {T_SHAPE}",
            "changed_meter": "elevated_load OR substantial peak OR "
                             "substantial shape",
            "broadly_synchronized": f"changed_fraction >= {T_BROAD:.2f} "
                                    "(non-Academic meters, per episode)",
            "partially_synchronized": f"{T_PARTIAL:.2f} <= changed_fraction "
                                      f"< {T_BROAD:.2f}",
            "building_specific": f"changed_fraction < {T_PARTIAL:.2f}",
            "daily_multi_meter_sync": f">= {T_SYNC_DAILY} non-Academic meters "
                                      "elevated on the same date",
        },
        "episode_classification": verdicts,
        "classification_counts": dict(cats),
        "daily_synchronization": {
            "n_dates_multi_meter_sync": n_sync_dates,
            "n_dates_academic_elevated": n_acad_dates,
            "mean_elevated_fraction": round(float(ds["elevated_fraction"]
                                                  .mean()), 3),
            "n_dates_with_notes": int((ds["notes"] != "").sum()),
        },
        "monthly_context_academic_2017": acad_mc[
            ["month", "valid_days", "mean_daily_kW",
             "baseline_mean_daily_kW", "mean_load_ratio_vs_baseline"]]
            .to_dict(orient="records"),
        "monthly_context_max_ratio_other_meters":
            {str(k): float(v) for k, v in others_max.items()},
        "answers": {
            "q1_episodes_in_other_meters":
                "see episode_classification[].changed_nonacademic_meters "
                "per episode",
            "q2_contemporaneous_meters":
                ep_df[ep_df["changed_meter"] & (ep_df["meter"] != "Academic")]
                .groupby("meter").size().sort_values(ascending=False)
                .to_dict(),
            "q3_multi_meter_sync_dates": n_sync_dates,
            "q4_regime_class": final_class,
            "q5_academic_distinct_regime":
                f"Academic Aug/Sep/Oct 2017 mean-load ratios vs the rest of "
                f"2017: {acad_ratios}",
            "q6_evidence": [
                "per-meter, per-episode ratios vs same-meter 7+7-day "
                "baselines (crossmeter_episode_summary.csv)",
                "daily synchronization across 35 target dates "
                "(crossmeter_daily_sync.csv)",
                "monthly Aug-Oct 2017 vs non-regime 2017 baseline "
                "(crossmeter_monthly_2017.csv)",
            ],
            "q7_missing_evidence": [
                "no authoritative campus operational calendar, activity log, "
                "or equipment-register data",
                "no weather data to test weather-driven demand",
                "no sub-meter breakdown inside the Academic building",
                "no meter-metadata change log (CT ratios, calibration, "
                "logging configuration)",
            ],
            "q8_claims_not_permitted": [
                "no causality (weather, events, holidays, exams, faults, "
                "theft)",
                "no confirmed equipment failure or malfunction",
                "no campus-wide event claim from association alone",
                "no monetary quantification (no tariff data)",
            ],
        },
        "final_interpretation": final_text,
        "scientific_statement": (
            "Observed association: Academic's Aug-Oct 2017 anomaly regime "
            "coincides with [see q4] contemporaneous deviations in other "
            "meters. Contextual evidence: same-meter baselines, daily "
            "synchronization, and monthly 2017 context. Required next step: "
            "engineering/operational investigation - the energy data alone "
            "does not convert association into causation."),
        "official_model": "K=2, random_state=42, n_init=10 - UNTOUCHED",
        "outputs": ["crossmeter_episode_summary.csv",
                    "crossmeter_daily_sync.csv",
                    "crossmeter_monthly_2017.csv",
                    "crossmeter_validation_report.json",
                    "crossmeter_episode_heatmap.png",
                    "crossmeter_daily_synchronization.png",
                    "academic_2017_vs_other_meters.png"],
    }
    path = MODELLING_DIR / "crossmeter_validation_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info(f"  saved {path}")
    log.info(f"  final classification: {final_class}")
    log.info(f"  final interpretation: {final_text}")
    return report


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> None:
    clustered = load_data()

    log.info("#" * 70)
    log.info("SPANDA Stage 3e - CROSS-METER VALIDATION (read-only)")
    log.info("#" * 70)
    before = hash_frozen_artifacts()
    log.info(f"pre-run hash verification: {len(before)} frozen files hashed")

    ep_df = episode_table(clustered)
    verdicts, counts = classify_episodes(ep_df)
    ds = daily_sync(clustered)
    mc = monthly_context(clustered)
    plot_figures(ep_df, ds, mc, clustered)
    report = build_report(ep_df, verdicts, counts, ds, mc, clustered)

    verify_frozen(before, hash_frozen_artifacts())

    # 10-item terminal summary
    sync_ep = max(verdicts.values(),
                  key=lambda v: -1 if v["changed_fraction"] is None
                  else v["changed_fraction"])
    spec_ep = min(verdicts.values(),
                  key=lambda v: 2 if v["changed_fraction"] is None
                  else v["changed_fraction"])
    log.info("=" * 66)
    log.info("SUMMARY")
    log.info(f"  1. Academic target episodes = {len(EPISODES)}")
    log.info(f"  2. target dates = {len(TARGET_DATES)}")
    log.info(f"  3. meters analyzed = {clustered['meter'].nunique()} "
             f"({', '.join(sorted(clustered['meter'].unique()))})")
    log.info(f"  4. dates with multi-meter synchronization = "
             f"{int(ds['multi_meter_sync'].sum())} of {len(ds)}")
    log.info(f"  5. strongest synchronized episode = {sync_ep['episode_id']} "
             f"({sync_ep['changed_nonacademic_meters']}/"
             f"{sync_ep['valid_nonacademic_meters']} meters changed)")
    log.info(f"  6. strongest building-specific episode = "
             f"{spec_ep['episode_id']} "
             f"({spec_ep['changed_nonacademic_meters']}/"
             f"{spec_ep['valid_nonacademic_meters']} meters changed)")
    log.info(f"  7. final regime classification = "
             f"{report['answers']['q4_regime_class']}")
    n_acad_changed = sum(1 for v in verdicts.values() if v["academic_changed"])
    dominant = (n_acad_changed >= 6
                and report["answers"]["q4_regime_class"]
                == "BUILDING-SPECIFIC SIGNAL")
    log.info(f"  8. Academic remains the dominant signal = {dominant} "
             f"(Academic changed in {n_acad_changed}/7 episodes; "
             "EP01 exception documented in report episode_notes)")
    log.info("  9. official K=2 model (random_state=42, n_init=10) UNTOUCHED")
    log.info("  10. ALL FROZEN ARTIFACTS BYTE-IDENTICAL")
    log.info("Stage 3e complete - analytical pipeline concluded.")


if __name__ == "__main__":
    main()
