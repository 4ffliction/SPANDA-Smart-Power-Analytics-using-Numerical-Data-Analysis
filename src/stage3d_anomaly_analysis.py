"""
SPANDA Stage 3d - POTENTIAL ANOMALY CONTEXT & CALENDAR VALIDATION
(read-only w.r.t. raw data, Stage-1/2/3b/3c artifacts, spanda/model.py, the
official K=2 model, and the existing anomaly thresholds; every decision logged
to spanda_out/audit/anomaly_context_log.txt)

Purpose
-------
NOT a new anomaly search. The 114 building-days flagged in Stage 3b
(potential_anomalies.csv - the authoritative list) are placed in calendar and
same-meter operational context to separate:
  A. CALENDAR / WEEKLY-PATTERN EXPLAINABLE
  B. UNUSUAL BUT PLAUSIBLE
  C. POTENTIALLY UNEXPLAINED - ENERGY-AUDIT CANDIDATE  (candidate, NOT fault)

Hard rules enforced here
------------------------
* anomaly thresholds and the detection method are NEVER recalculated or changed
* no flagged observation is deleted; potential_anomalies.csv is never rewritten
* no refitting of PCA / K-Means; official model stays K=2 (rs=42, n_init=10)
* no confirmed-fault / failure / theft language; screening only
* no invented institutional calendar (exams/holidays/festivals): only generic
  weekday/weekend and month facts; the lack of an authoritative academic
  calendar is stated as a limitation
* frozen artifacts are SHA-256 hash-verified before/after (Stage-3c approach)

Evidence-tag thresholds (all documented, relative to SAME-meter neighbours)
---------------------------------------------------------------------------
HIGH_PEAK           peak_ratio >= 1.5            (anomaly peak / nearby median peak)
HIGH_BASE           base_ratio >= 1.5, or nearby median base <= 0.05 kW and
                    anomaly base >= 1 kW (zero-baseline fallback)
LOW_BASE            base_ratio <= 0.5, or nearby median base >= 0.5 kW and
                    anomaly base <= 0.05 kW
LOW_LOAD_FACTOR     load_factor <= 0.7 x nearby median load factor
HIGH_LOAD_FACTOR    load_factor >= 1.3 x nearby median load factor
PEAK_HOUR_SHIFT     circular |d peak_hour| >= 3 h vs nearby circular-median hour
SHAPE_SHIFT         L1(anomaly 24-h profile, nearby median profile) /
                    L1(nearby median profile) >= 0.35
ISOLATED            episode duration == 1 day
PERSISTENT_EPISODE  episode duration >= 2 consecutive days
Neighbours = up to 7 previous + 7 next NON-ANOMALOUS valid days of the SAME
meter (observation order; gaps/outages handled by position, not calendar days).

Outputs (all NEW; nothing overwritten)
--------------------------------------
spanda_out/modelling/anomaly_context.csv
spanda_out/modelling/anomaly_summary_by_meter.csv
spanda_out/modelling/anomaly_summary_by_category.csv
spanda_out/modelling/anomaly_episodes.csv
spanda_out/modelling/anomaly_context_report.json
spanda_out/figures/anomaly_calendar_distribution.png
spanda_out/figures/anomaly_rate_by_meter.png
spanda_out/figures/anomaly_episode_lengths.png
spanda_out/figures/anomaly_context_comparison.png
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
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

log = get_logger("spanda.anomaly_context",
                 AUDIT_DIR / "anomaly_context_log.txt")

EXPECTED_N_ANOMALIES = 114
HCOLS = [f"h{h:02d}_kW" for h in range(24)]
NEIGHBOUR_WINDOW = 7           # previous/next 7 non-anomalous same-meter days

# documented tag thresholds (see module docstring)
T_PEAK_RATIO = 1.5
T_BASE_RATIO = 1.5
T_BASE_RATIO_LOW = 0.5
T_BASE_ABS_ZERO_HIGH = 0.05    # nearby median base <= this -> zero baseline
T_BASE_ABS_HIGH = 1.0          # anomaly base >= this on a zero baseline
T_BASE_ABS_LOW = 0.05          # anomaly base <= this when nearby base >= 0.5
T_BASE_ABS_NEARBY = 0.5
T_LF_LOW = 0.7
T_LF_HIGH = 1.3
T_PEAK_HOUR_SHIFT_H = 3.0
T_SHAPE_L1 = 0.35
T_SHAPE_L1_STRONG = 0.50       # single-tag route to category C

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ---------------------------------------------------------------------------
# frozen-artifact hash verification (Stage-3c approach)
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
                  "potential_anomalies.csv",           # authoritative, never rewritten
                  "lecture_dominance.json", "cluster_labels.json",
                  "k_sensitivity.csv", "cluster_by_meter_by_date.csv",
                  "cluster_centroids.csv", "standardized_features.parquet",
                  "feature_names.json"]]
STAGE3C_FILES = [MODELLING_DIR / f for f in
                 ["cluster_stability_summary.csv",
                  "cluster_stability_runs.csv",
                  "cluster_stability_report.json"]]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_frozen_artifacts() -> dict[str, str]:
    groups = [("raw", RAW_DATA_FILES), ("stage1", STAGE1_FILES),
              ("stage2", STAGE2_FILES), ("stage3b", STAGE3B_FILES),
              ("stage3c", STAGE3C_FILES),
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
    log.info(f"  files hash-verified: {len(before)} "
             "(raw x9, Stage-1, Stage-2, Stage-3b incl. potential_anomalies.csv, "
             "Stage-3c, spanda/model.py)")
    if diff:
        for k in diff:
            log.error(f"  MODIFIED DURING RUN: {k}")
        raise RuntimeError(f"{len(diff)} frozen artifacts were modified: {diff}")
    log.info("ALL FROZEN ARTIFACTS BYTE-IDENTICAL")


# ---------------------------------------------------------------------------
# 1. load the authoritative 114 anomalies
# ---------------------------------------------------------------------------
def load_anomalies() -> pd.DataFrame:
    log.info("=" * 66)
    log.info("STEP 1 - load authoritative anomaly list (potential_anomalies.csv)")
    anom = pd.read_csv(MODELLING_DIR / "potential_anomalies.csv")
    log.info(f"  loaded {len(anom)} flagged building-days")
    if len(anom) != EXPECTED_N_ANOMALIES:
        raise RuntimeError(f"expected {EXPECTED_N_ANOMALIES} anomalies, found "
                           f"{len(anom)} - stopping for investigation")
    log.info(f"  count check passed: {EXPECTED_N_ANOMALIES}")
    log.info(f"  thresholds preserved from file: "
             f"p99={anom['threshold_percentile_99'].iloc[0]}, "
             f"median+6MAD={anom['threshold_median_plus_6mad'].iloc[0]} "
             f"(NOT recalculated)")
    # merge the full building-day features (peak_hour, hourly shape, weekday)
    clustered = pd.read_csv(MODELLING_DIR / "clustered_building_days.csv")
    anom = anom.merge(
        clustered[["meter", "obs_date", "peak_hour", "weekday", "night_share",
                   "evening_share", "morning_share", "afternoon_share"]
                  + HCOLS],
        on=["meter", "obs_date"], how="left", validate="one_to_one")
    if anom[["peak_hour", "weekday"]].isna().any().any():
        raise RuntimeError("anomaly rows failed to merge with the modelling "
                           "dataset - stopping")
    log.info("  merged building-day features from clustered_building_days.csv "
             "(one-to-one on meter + obs_date)")
    return anom, clustered


# ---------------------------------------------------------------------------
# 2-4. calendar features + composition + concentration tables
# ---------------------------------------------------------------------------
def add_calendar(anom: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(anom["obs_date"])
    anom = anom.copy()
    anom["year"] = d.dt.year
    anom["month"] = d.dt.month
    anom["month_name"] = d.dt.strftime("%b")
    anom["day_of_month"] = d.dt.day
    anom["day_name"] = d.dt.dayofweek.map(
        lambda i: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                   "Saturday", "Sunday"][i])
    anom["is_weekend"] = d.dt.dayofweek >= 5
    anom["week_of_year"] = d.dt.isocalendar().week.astype(int)
    anom["quarter"] = d.dt.quarter
    anom["day_of_year"] = d.dt.dayofyear
    anom["year_month"] = d.dt.strftime("%Y-%m")
    # simple month-based grouping (documented: NOT an Indian-season claim)
    anom["month_group"] = anom["month"].map(
        lambda m: ("Dec-Jan-Feb (winter months)" if m in (12, 1, 2) else
                   "Mar-Apr-May (hot months)" if m in (3, 4, 5) else
                   "Jun-Jul-Aug-Sep (monsoon months)" if m in (6, 7, 8, 9) else
                   "Oct-Nov (post-monsoon months)"))
    return anom


def weekday_weekend_composition(anom: pd.DataFrame,
                                clustered: pd.DataFrame) -> dict:
    log.info("=" * 66)
    log.info("STEP 3 - weekday vs weekend composition")
    n_wk = int((~anom["is_weekend"]).sum())
    n_we = int(anom["is_weekend"].sum())
    full_we = float((pd.to_datetime(clustered["obs_date"]).dt.dayofweek >= 5)
                    .mean())
    out = {
        "anomalies_weekday": n_wk,
        "anomalies_weekend": n_we,
        "anomalies_weekday_pct": round(100 * n_wk / len(anom), 2),
        "anomalies_weekend_pct": round(100 * n_we / len(anom), 2),
        "full_dataset_weekend_pct": round(100 * full_we, 2),
        "full_dataset_weekday_pct": round(100 * (1 - full_we), 2),
        "note": "weekend anomalies are NOT auto-labelled explainable; a "
                "weekend can still carry unusual energy behaviour",
    }
    log.info(f"  anomalies: weekday {n_wk} ({out['anomalies_weekday_pct']}%), "
             f"weekend {n_we} ({out['anomalies_weekend_pct']}%)")
    log.info(f"  full modelling dataset: weekend "
             f"{out['full_dataset_weekend_pct']}%, weekday "
             f"{out['full_dataset_weekday_pct']}%")
    return out


def concentration_tables(anom: pd.DataFrame) -> dict[str, pd.DataFrame]:
    log.info("=" * 66)
    log.info("STEP 4 - temporal/meter/cluster concentration")

    by_meter = (anom["meter"].value_counts()
                .rename("anomaly_count").reset_index())
    by_meter.columns = ["meter", "anomaly_count"]
    by_meter["anomaly_percentage"] = (100 * by_meter["anomaly_count"]
                                      / len(anom)).round(2)

    def simple(group_col: str) -> pd.DataFrame:
        t = (anom[group_col].value_counts().sort_index().rename("anomaly_count")
             .reset_index())
        t.columns = [group_col, "anomaly_count"]
        return t

    tables = {"by_meter": by_meter,
              "by_year": simple("year"),
              "by_year_month": simple("year_month"),
              "by_day_name": simple("day_name"),
              "by_cluster": simple("cluster")}
    for name, t in tables.items():
        log.info(f"  {name}: {t.to_dict(orient='records')}")
    return tables


# ---------------------------------------------------------------------------
# 5. consecutive-day episodes
# ---------------------------------------------------------------------------
def build_episodes(anom: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 66)
    log.info("STEP 5 - consecutive-day anomaly episodes (per meter)")
    episodes = []
    for meter, g in anom.sort_values("obs_date").groupby("meter"):
        g = g.reset_index(drop=True)
        dates = pd.to_datetime(g["obs_date"]).to_numpy()
        ep_id, start = 0, 0
        for i in range(1, len(g) + 1):
            new_ep = (i == len(g)
                      or (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1]))
                      != pd.Timedelta(days=1))
            if new_ep:
                ep_id += 1
                block = g.iloc[start:i]
                episodes.append({
                    "meter": meter,
                    "episode_id": f"{meter}-EP{ep_id:02d}",
                    "start_date": block["obs_date"].iloc[0],
                    "end_date": block["obs_date"].iloc[-1],
                    "duration_days": int((pd.Timestamp(block["obs_date"].iloc[-1])
                                          - pd.Timestamp(block["obs_date"].iloc[0]))
                                         .days + 1),
                    "number_of_flagged_days": int(len(block)),
                    "clusters": ",".join(str(c) for c
                                         in sorted(block["cluster"].unique())),
                    "mean_anomaly_distance": round(float(
                        block["dist_to_centroid"].mean()), 3),
                    "max_anomaly_distance": round(float(
                        block["dist_to_centroid"].max()), 3),
                })
                start = i
    ep = pd.DataFrame(episodes)
    ep.to_csv(MODELLING_DIR / "anomaly_episodes.csv", index=False)
    n_iso = int((ep["duration_days"] == 1).sum())
    n_multi = int((ep["duration_days"] >= 2).sum())
    log.info(f"  {len(ep)} episodes across {anom['meter'].nunique()} meters: "
             f"{n_iso} isolated single days, {n_multi} multi-day episodes")
    log.info(f"  longest episodes: "
             f"{ep.nlargest(5, 'duration_days')[['episode_id', 'duration_days']]}"
             .replace("\n", " | "))
    log.info(f"  saved {MODELLING_DIR / 'anomaly_episodes.csv'}")
    return ep


# ---------------------------------------------------------------------------
# 6-7. same-metre neighbour comparison + evidence tags
# ---------------------------------------------------------------------------
def _circ_median_hour(hours: np.ndarray) -> float:
    ang = np.deg2rad(360.0 * hours / 24.0)
    h = (np.rad2deg(np.arctan2(np.sin(ang).mean(),
                               np.cos(ang).mean()))) % 360.0
    return 24.0 * h / 360.0


def _circ_diff_hours(a: float, b: float) -> float:
    d = abs(a - b) % 24.0
    return min(d, 24.0 - d)


def neighbour_comparison(anom: pd.DataFrame,
                         clustered: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 66)
    log.info("STEP 6 - same-meter neighbour comparison (prev/next "
             f"{NEIGHBOUR_WINDOW} non-anomalous valid days)")

    anomaly_keys = set(zip(anom["meter"], anom["obs_date"]))
    ctx_rows = []
    for meter, g in clustered.groupby("meter"):
        g = g.sort_values("obs_date").reset_index(drop=True)
        is_anom = np.array([(m, d) in anomaly_keys
                            for m, d in zip(g["meter"], g["obs_date"])])
        anom_idx = np.where(is_anom)[0]
        if len(anom_idx) == 0:
            continue
        for i in anom_idx:
            lo = max(0, i - 2 * NEIGHBOUR_WINDOW)   # room to skip anomalies
            hi = min(len(g), i + 1 + 2 * NEIGHBOUR_WINDOW)
            cand = [j for j in range(lo, hi)
                    if j != i and not is_anom[j]]
            prev = [j for j in cand if j < i][-NEIGHBOUR_WINDOW:]
            nxt = [j for j in cand if j > i][:NEIGHBOUR_WINDOW]
            nb_idx = prev + nxt
            row = {"meter": meter, "obs_date": g.loc[i, "obs_date"],
                   "n_nearby_days": len(nb_idx)}
            if nb_idx:
                nb = g.loc[nb_idx]
                row["nearby_median_mean_kW"] = float(nb["daily_mean_kW"].median())
                row["nearby_median_peak_kW"] = float(nb["daily_peak_kW"].median())
                row["nearby_median_base_kW"] = float(nb["base_kW"].median())
                row["nearby_median_load_factor"] = float(
                    nb["load_factor"].median())
                row["nearby_median_peak_hour"] = _circ_median_hour(
                    nb["peak_hour"].to_numpy(dtype="float64"))
                row["nearby_median_profile"] = nb[HCOLS].median().to_numpy()
            else:
                for c in ["nearby_median_mean_kW", "nearby_median_peak_kW",
                          "nearby_median_base_kW", "nearby_median_load_factor",
                          "nearby_median_peak_hour"]:
                    row[c] = np.nan
                row["nearby_median_profile"] = np.full(24, np.nan)
            a = g.loc[i]
            row["daily_mean_kW"] = float(a["daily_mean_kW"])
            row["daily_peak_kW"] = float(a["daily_peak_kW"])
            row["base_kW"] = float(a["base_kW"])
            row["load_factor"] = float(a["load_factor"])
            row["peak_hour"] = float(a["peak_hour"])
            row["profile"] = a[HCOLS].to_numpy(dtype="float64")
            ctx_rows.append(row)
    ctx = pd.DataFrame(ctx_rows)

    # ratios / differences with explicit zero-baseline handling
    eps = 1e-9
    ctx["mean_load_ratio"] = np.where(
        ctx["nearby_median_mean_kW"] > eps,
        ctx["daily_mean_kW"] / ctx["nearby_median_mean_kW"].clip(lower=eps),
        np.nan)
    ctx["peak_ratio"] = np.where(
        ctx["nearby_median_peak_kW"] > eps,
        ctx["daily_peak_kW"] / ctx["nearby_median_peak_kW"].clip(lower=eps),
        np.nan)
    ctx["base_ratio"] = np.where(
        ctx["nearby_median_base_kW"] > eps,
        ctx["base_kW"] / ctx["nearby_median_base_kW"].clip(lower=eps),
        np.nan)
    ctx["zero_baseline_mean"] = ctx["nearby_median_mean_kW"] <= 0.05
    ctx["zero_baseline_base"] = ctx["nearby_median_base_kW"] <= T_BASE_ABS_ZERO_HIGH
    ctx["lf_ratio"] = np.where(
        ctx["nearby_median_load_factor"] > eps,
        ctx["load_factor"] / ctx["nearby_median_load_factor"].clip(lower=eps),
        np.nan)
    ctx["peak_hour_diff_h"] = [
        _circ_diff_hours(ph, nh) if np.isfinite(ph) and np.isfinite(nh)
        else np.nan
        for ph, nh in zip(ctx["peak_hour"], ctx["nearby_median_peak_hour"])]
    ctx["shape_l1_rel"] = [
        (float(np.abs(p - m).sum() / max(float(np.abs(m).sum()), eps))
         if np.isfinite(m).all() and np.abs(m).sum() > eps else np.nan)
        for p, m in zip(ctx["profile"], ctx["nearby_median_profile"])]
    log.info(f"  neighbour context built for {len(ctx)} anomaly days "
             f"(min nearby days = {int(ctx['n_nearby_days'].min())})")
    return ctx


def assign_tags(anom: pd.DataFrame, ctx: pd.DataFrame,
                episodes: pd.DataFrame,
                clustered: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 66)
    log.info("STEP 7 - evidence tags (thresholds in module docstring + log)")

    # meter-level weekly-pattern strength (from the FULL modelling dataset)
    d = clustered.copy()
    d["dow"] = pd.to_datetime(d["obs_date"]).dt.dayofweek
    weekly = {}
    for meter, g in d.groupby("meter"):
        wk_med = float(g.loc[g["dow"] < 5, "daily_mean_kW"].median())
        we_med = float(g.loc[g["dow"] >= 5, "daily_mean_kW"].median())
        # median daily_mean for each weekday name (for repeated-pattern test)
        dow_med = {dw: float(g.loc[g["dow"] == dw, "daily_mean_kW"].median())
                   for dw in range(7)}
        weekly[meter] = {"weekday_median": wk_med, "weekend_median": we_med,
                         "dow_median": dow_med,
                         "strong_weekly_pattern":
                             wk_med > 0 and we_med >= 0
                             and we_med <= 0.8 * wk_med}

    ep_map = {}
    for _, r in episodes.iterrows():
        dates = pd.date_range(r["start_date"], r["end_date"], freq="D")
        for dt in dates:
            ep_map[(r["meter"], dt.strftime("%Y-%m-%d"))] = r

    eps = 1e-9

    ctx_i = ctx.set_index(["meter", "obs_date"])
    tags_col, ev_col, reason_parts = [], [], []
    for _, a in anom.iterrows():
        key = (a["meter"], a["obs_date"])
        c = ctx_i.loc[key]
        tags: list[str] = []
        # neighbour-based tags ------------------------------------------------
        if np.isfinite(c["peak_ratio"]) and c["peak_ratio"] >= T_PEAK_RATIO:
            tags.append("HIGH_PEAK")
        if (np.isfinite(c["base_ratio"]) and c["base_ratio"] >= T_BASE_RATIO) \
                or (c["zero_baseline_base"]
                    and c["base_kW"] >= T_BASE_ABS_HIGH):
            tags.append("HIGH_BASE")
        if (np.isfinite(c["base_ratio"]) and c["base_ratio"] <= T_BASE_RATIO_LOW) \
                or (c["nearby_median_base_kW"] >= T_BASE_ABS_NEARBY
                    and c["base_kW"] <= T_BASE_ABS_LOW):
            tags.append("LOW_BASE")
        if np.isfinite(c["lf_ratio"]) and c["lf_ratio"] <= T_LF_LOW:
            tags.append("LOW_LOAD_FACTOR")
        if np.isfinite(c["lf_ratio"]) and c["lf_ratio"] >= T_LF_HIGH:
            tags.append("HIGH_LOAD_FACTOR")
        if np.isfinite(c["peak_hour_diff_h"]) \
                and c["peak_hour_diff_h"] >= T_PEAK_HOUR_SHIFT_H:
            tags.append("PEAK_HOUR_SHIFT")
        if np.isfinite(c["shape_l1_rel"]) and c["shape_l1_rel"] >= T_SHAPE_L1:
            tags.append("SHAPE_SHIFT")
        # episode tags ---------------------------------------------------------
        e = ep_map.get(key)
        dur = int(e["duration_days"]) if e is not None else 1
        ep_id = e["episode_id"] if e is not None else f"{a['meter']}-UNKNOWN"
        if dur == 1:
            tags.append("ISOLATED")
        else:
            tags.append("PERSISTENT_EPISODE")
        # calendar evidence (conservative; generic weekly facts only) ----------
        w = weekly[a["meter"]]
        is_weekend = bool(pd.Timestamp(a["obs_date"]).dayofweek >= 5)
        dow = pd.Timestamp(a["obs_date"]).dayofweek
        dow_med = w["dow_median"].get(dow, np.nan)
        calendar_match = bool(
            is_weekend and w["strong_weekly_pattern"]
            and np.isfinite(c["mean_load_ratio"])
            and c["mean_load_ratio"] <= 1.25)
        repeated_weekly = bool(
            np.isfinite(dow_med) and dow_med > eps
            and 0.7 <= c["daily_mean_kW"] / dow_med <= 1.3)
        # evidence summary (transparent booleans, no opaque score) -------------
        ev = {
            "calendar_match": calendar_match,
            "repeated_weekly_pattern": repeated_weekly,
            "isolated_vs_episode": ("isolated" if dur == 1
                                    else f"episode_{dur}d"),
            "same_meter_deviation": bool(
                np.isfinite(c["mean_load_ratio"])
                and not (0.7 <= c["mean_load_ratio"] <= 1.5)),
            "extreme_peak": "HIGH_PEAK" in tags,
            "extreme_base": ("HIGH_BASE" in tags) or ("LOW_BASE" in tags),
            "shape_shift": "SHAPE_SHIFT" in tags,
            "n_nearby_days": int(c["n_nearby_days"]),
        }
        tags_col.append("|".join(tags))
        ev_col.append(ev)

    # attach
    anom = anom.copy()
    anom["anomaly_tags"] = tags_col
    anom["evidence"] = ev_col

    # unpack neighbour context columns for the output table
    ctx_i2 = ctx.set_index(["meter", "obs_date"])
    for col in ["nearby_median_mean_kW", "nearby_median_peak_kW",
                "nearby_median_base_kW", "nearby_median_load_factor",
                "mean_load_ratio", "peak_ratio", "base_ratio", "lf_ratio",
                "peak_hour_diff_h", "shape_l1_rel", "n_nearby_days"]:
        anom[col] = [ctx_i2.loc[(r["meter"], r["obs_date"]), col]
                     for _, r in anom.iterrows()]
    anom["episode_id"] = [ep_map.get((r["meter"], r["obs_date"]), {}).get(
        "episode_id") for _, r in anom.iterrows()]
    anom["episode_duration_days"] = [
        ep_map.get((r["meter"], r["obs_date"]), {}).get("duration_days")
        for _, r in anom.iterrows()]
    return anom


# ---------------------------------------------------------------------------
# 10. conservative classification
# ---------------------------------------------------------------------------
STRONG_TAGS = ["HIGH_PEAK", "HIGH_BASE", "LOW_BASE", "PEAK_HOUR_SHIFT",
               "SHAPE_SHIFT"]


def classify(anom: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 66)
    log.info("STEP 10 - conservative interpretation (A/B/C; prefer B when "
             "evidence is insufficient)")

    def reason(a) -> str:
        ev = a["evidence"]
        parts = []
        if ev["calendar_match"]:
            parts.append("weekend day consistent with this meter's recurring "
                         "low weekend load")
        if ev["repeated_weekly_pattern"]:
            parts.append("load level typical for this weekday at this meter")
        if ev["same_meter_deviation"]:
            parts.append(f"mean load {a['mean_load_ratio']:.2f}x nearby "
                         f"same-meter median")
        if "HIGH_PEAK" in a["anomaly_tags"]:
            parts.append(f"peak {a['peak_ratio']:.2f}x nearby median")
        if "HIGH_BASE" in a["anomaly_tags"]:
            parts.append("elevated base load vs nearby days")
        if "LOW_BASE" in a["anomaly_tags"]:
            parts.append("near-zero base load vs nearby days")
        if "PEAK_HOUR_SHIFT" in a["anomaly_tags"]:
            parts.append(f"peak hour shifted {a['peak_hour_diff_h']:.0f} h")
        if "SHAPE_SHIFT" in a["anomaly_tags"]:
            parts.append(f"24-h shape deviation {a['shape_l1_rel']:.2f} "
                         "(relative L1)")
        parts.append("isolated day" if ev["isolated_vs_episode"] == "isolated"
                     else f"{a['episode_duration_days']}-day episode")
        return "; ".join(parts)

    cats, reasons = [], []
    for _, a in anom.iterrows():
        ev = a["evidence"]
        strong = [t for t in a["anomaly_tags"].split("|") if t in STRONG_TAGS]
        strong_shape = ("SHAPE_SHIFT" in strong
                        and a["shape_l1_rel"] >= T_SHAPE_L1_STRONG)
        no_calendar = not (ev["calendar_match"]
                           or ev["repeated_weekly_pattern"])
        if no_calendar and (len(strong) >= 2 or strong_shape):
            cat = ("C. POTENTIALLY UNEXPLAINED - ENERGY-AUDIT CANDIDATE")
        elif ev["calendar_match"] or (ev["repeated_weekly_pattern"]
                                      and len(strong) == 0):
            cat = "A. CALENDAR / WEEKLY-PATTERN EXPLAINABLE"
        else:
            cat = "B. UNUSUAL BUT PLAUSIBLE"
        cats.append(cat)
        reasons.append(reason(a))
    anom["interpretation_category"] = cats
    anom["interpretation_reason"] = reasons
    counts = anom["interpretation_category"].value_counts()
    log.info(f"  categories: {counts.to_dict()}")
    return anom


# ---------------------------------------------------------------------------
# 8, 12, 13. output tables
# ---------------------------------------------------------------------------
def export_tables(anom: pd.DataFrame, clustered: pd.DataFrame) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log.info("=" * 66)
    log.info("STEP 12/13 - output tables")

    out = anom.copy()
    out["flagged_by_distance"] = out["flagged_by"].isin(
        ["percentile", "both"])
    out["flagged_by_mad"] = out["flagged_by"].isin(["med_mad", "both"])
    out["anomaly_distance"] = out["dist_to_centroid"]
    out["peak_kW"] = out["daily_peak_kW"]
    out["anomaly_tags"] = out["anomaly_tags"]
    out["is_weekend"] = out["is_weekend"].map({True: "weekend",
                                               False: "weekday"})
    keep = ["meter", "obs_date", "cluster", "anomaly_distance",
            "flagged_by_distance", "flagged_by_mad", "weekday", "is_weekend",
            "month", "year", "daily_mean_kW", "peak_kW", "base_kW",
            "load_factor", "peak_hour", "nearby_median_mean_kW",
            "nearby_median_peak_kW", "nearby_median_base_kW",
            "nearby_median_load_factor", "mean_load_ratio",
            "episode_id", "episode_duration_days", "anomaly_tags",
            "interpretation_category", "interpretation_reason",
            "day_name", "week_of_year", "quarter", "day_of_year",
            "peak_ratio", "base_ratio", "lf_ratio", "peak_hour_diff_h",
            "shape_l1_rel", "n_nearby_days"]
    out = out[keep]
    out.to_csv(MODELLING_DIR / "anomaly_context.csv", index=False)
    log.info(f"  saved anomaly_context.csv ({len(out)} rows)")

    total_by_meter = clustered.groupby("meter").size()
    by_meter = out.groupby("meter").agg(
        anomaly_days=("obs_date", "count"),
        mean_anomaly_distance=("anomaly_distance", "mean"),
        max_anomaly_distance=("anomaly_distance", "max")).reset_index()
    by_meter["total_modelling_days"] = by_meter["meter"].map(total_by_meter)
    by_meter["anomaly_rate_pct"] = (100 * by_meter["anomaly_days"]
                                    / by_meter["total_modelling_days"]).round(2)
    by_meter["mean_anomaly_distance"] = \
        by_meter["mean_anomaly_distance"].round(3)
    by_meter["max_anomaly_distance"] = by_meter["max_anomaly_distance"].round(3)
    by_meter = by_meter[["meter", "total_modelling_days", "anomaly_days",
                         "anomaly_rate_pct", "mean_anomaly_distance",
                         "max_anomaly_distance"]]
    by_meter.to_csv(MODELLING_DIR / "anomaly_summary_by_meter.csv", index=False)

    cat = (out["interpretation_category"].value_counts().rename("count")
           .reset_index())
    cat.columns = ["interpretation_category", "count"]
    cat["percentage"] = (100 * cat["count"] / len(out)).round(2)
    cat.to_csv(MODELLING_DIR / "anomaly_summary_by_category.csv", index=False)
    log.info(f"  saved anomaly_summary_by_meter.csv + "
             f"anomaly_summary_by_category.csv")
    log.info("\n" + by_meter.to_string(index=False))
    return out, by_meter, cat


# ---------------------------------------------------------------------------
# 14. figures
# ---------------------------------------------------------------------------
def plot_figures(anom: pd.DataFrame, by_meter: pd.DataFrame) -> None:
    log.info("=" * 66)
    log.info("STEP 14 - figures")

    # Fig 1: anomaly counts over time (monthly, stacked by meter)
    t = anom.copy()
    t["dt"] = pd.to_datetime(t["obs_date"])
    t["ym"] = t["dt"].dt.strftime("%Y-%m")
    pv = (t.pivot_table(index="ym", columns="meter", values="obs_date",
                        aggfunc="count").fillna(0))
    fig, ax = plt.subplots(figsize=(13, 5))
    bottom = np.zeros(len(pv))
    for m in pv.columns:
        ax.bar(range(len(pv)), pv[m].to_numpy(), bottom=bottom, label=m)
        bottom += pv[m].to_numpy()
    ax.set_xticks(range(len(pv)), pv.index, rotation=60, fontsize=6)
    ax.set_ylabel("Flagged building-days per month")
    ax.set_title("Potential anomalies over time (114 flagged building-days, "
                 "stacked by meter)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "anomaly_calendar_distribution.png", dpi=150)
    plt.close(fig)

    # Fig 2: anomaly rate by meter (all 9 meters for context)
    fig, ax = plt.subplots(figsize=(9, 5))
    bm = by_meter.sort_values("anomaly_rate_pct")
    colors = ["tab:red" if r > 0 else "tab:grey"
              for r in bm["anomaly_rate_pct"]]
    ax.barh(bm["meter"], bm["anomaly_rate_pct"],
            color=colors if bm["anomaly_rate_pct"].gt(0).any() else None)
    for y, (m, r) in enumerate(zip(bm["meter"], bm["anomaly_rate_pct"])):
        ax.text(r + 0.05, y, f"{r:.2f}%", va="center", fontsize=8)
    ax.set_xlabel("Anomaly rate (% of the meter's modelling days)")
    ax.set_title("Potential-anomaly rate by building meter")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "anomaly_rate_by_meter.png", dpi=150)
    plt.close(fig)

    # Fig 3: episode length distribution
    ep = pd.read_csv(MODELLING_DIR / "anomaly_episodes.csv")
    lens = ep["duration_days"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(lens.index.astype(str), lens.to_numpy(),
           color=["tab:blue" if i == 1 else "tab:orange"
                  for i in lens.index])
    ax.set_xlabel("Episode duration (consecutive flagged days)")
    ax.set_ylabel("Number of episodes")
    ax.set_title(f"Anomaly episode lengths ({len(ep)} episodes; "
                 f"{int((ep['duration_days'] == 1).sum())} isolated, "
                 f"{int((ep['duration_days'] >= 2).sum())} multi-day)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "anomaly_episode_lengths.png", dpi=150)
    plt.close(fig)

    # Fig 4: anomaly day vs nearby same-meter median (4 metrics)
    metrics = [("daily_mean_kW", "nearby_median_mean_kW", "Daily mean (kW)", True),
               ("peak_kW", "nearby_median_peak_kW", "Daily peak (kW)", True),
               ("base_kW", "nearby_median_base_kW", "Base load (kW)", True),
               ("load_factor", "nearby_median_load_factor", "Load factor", False)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (a_col, n_col, label, use_log) in zip(axes, metrics):
        x = pd.to_numeric(anom[n_col], errors="coerce")
        y = pd.to_numeric(anom[a_col], errors="coerce")
        ok = x.notna() & y.notna() & (x > 0 if use_log else x.notna())
        ax.scatter(x[ok], y[ok], s=18, alpha=0.6)
        lims = [max(min(x[ok].min(), y[ok].min()) * 0.8, 1e-4),
                max(x[ok].max(), y[ok].max()) * 1.2]
        ax.plot(lims, lims, "k--", lw=1, label="y = x")
        if use_log:
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"Nearby median (same meter)")
        ax.set_ylabel("Anomaly day")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Anomalous day vs nearby same-meter normal days "
                 "(points far from the diagonal = strong local deviation)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "anomaly_context_comparison.png", dpi=150)
    plt.close(fig)
    log.info("  figures saved: anomaly_calendar_distribution.png, "
             "anomaly_rate_by_meter.png, anomaly_episode_lengths.png, "
             "anomaly_context_comparison.png")


# ---------------------------------------------------------------------------
# 15-16. final report
# ---------------------------------------------------------------------------
def notable_findings(anom: pd.DataFrame,
                     episodes: pd.DataFrame) -> list[dict]:
    """Evidence-based notable patterns (no causal claim, no fault claim).
    Currently: weekday-packed multi-day episodes (5-day runs starting on a
    Monday, i.e. Monday-Friday blocks with weekend gaps) - a sustained,
    calendar-aligned behavioural regime rather than isolated noise."""
    findings = []
    ep = episodes.copy()
    ep["start_dow"] = pd.to_datetime(ep["start_date"]).dt.dayofweek
    packed = ep[(ep["duration_days"] == 5) & (ep["start_dow"] == 0)]
    if len(packed) >= 3:
        meters = packed["meter"].value_counts()
        m = meters.index[0]
        sub = packed[packed["meter"] == m]
        findings.append({
            "finding": (f"{m} has {len(sub)} weekday-packed anomaly episodes "
                        f"(5-day Monday-Friday blocks with weekend gaps): "
                        f"{', '.join(sub['start_date'] + ' to ' + sub['end_date'])}"),
            "evidence": ("episodes span consecutive Mon-Fri dates only; "
                         "all in cluster " + sub["clusters"].mode().iloc[0] +
                         "; tags dominated by " +
                         (anom[(anom["meter"] == m)
                               & (anom["obs_date"].isin(
                                   pd.date_range(sub["start_date"].min(),
                                                 sub["end_date"].max(),
                                                 freq="D").strftime(
                                     "%Y-%m-%d")))]["anomaly_tags"]
                          .str.split("|").explode().value_counts()
                          .head(3).index.tolist()).__str__()),
            "interpretation": ("a sustained, calendar-aligned weekday-only "
                               "operational-behaviour change during this "
                               "period; screening evidence for an operational "
                               "review, NOT a fault finding"),
        })
    return findings


def build_report(anom: pd.DataFrame, comp: dict, tables: dict,
                 by_meter: pd.DataFrame, cat: pd.DataFrame,
                 episodes: pd.DataFrame) -> dict:
    log.info("=" * 66)
    log.info("STEP 15/16 - final report + engineering interpretation")

    cat_A = cat.loc[cat["interpretation_category"].str.startswith("A"),
                    "count"].sum()
    cat_B = cat.loc[cat["interpretation_category"].str.startswith("B"),
                    "count"].sum()
    cat_C = cat.loc[cat["interpretation_category"].str.startswith("C"),
                    "count"].sum()

    c_rows = anom[anom["interpretation_category"].str.startswith("C")]
    c_sorted = c_rows.sort_values("anomaly_distance", ascending=False)
    strongest = [{"meter": r["meter"], "obs_date": r["obs_date"],
                  "anomaly_distance": round(float(r["anomaly_distance"]), 2),
                  "tags": r["anomaly_tags"],
                  "reason": r["interpretation_reason"]}
                 for _, r in c_sorted.head(10).iterrows()]

    tag_counts = Counter(t for tags in anom["anomaly_tags"]
                         for t in tags.split("|"))

    actions = {
        "HIGH_BASE / elevated overnight base": [
            "after-hours equipment audit", "standby-load review",
            "HVAC/lighting scheduling review"],
        "HIGH_PEAK": ["investigate simultaneous high-load operation",
                      "review demand management",
                      "inspect operating schedules"],
        "SHAPE_SHIFT / PEAK_HOUR_SHIFT": [
            "inspect changed operating schedule",
            "identify temporary/deferrable loads",
            "consider additional sub-metering"],
        "PERSISTENT_EPISODE": ["review operational event logs",
                               "compare with known campus activities",
                               "investigate sustained load changes"],
    }

    findings = notable_findings(anom, episodes)
    for f in findings:
        log.info(f"  notable finding: {f['finding']}")

    report = {
        "stage": "SPANDA Stage 3d - potential anomaly context & calendar "
                 "validation (read-only screening)",
        "observation_unit": "building-day (building meter x day)",
        "notable_findings": findings,
        "n_potential_anomalies": int(len(anom)),
        "source_of_flags": ("spanda_out/modelling/potential_anomalies.csv "
                            "(Stage 3b; thresholds 99th pct = 8.07 and "
                            "median+6x1.4826xMAD = 13.93; NOT recalculated)"),
        "weekday_weekend": comp,
        "anomaly_rate_by_meter": by_meter.to_dict(orient="records"),
        "concentration": {
            "by_year": tables["by_year"].to_dict(orient="records"),
            "by_year_month": tables["by_year_month"].to_dict(orient="records"),
            "by_day_name": tables["by_day_name"].to_dict(orient="records"),
            "by_cluster": tables["by_cluster"].to_dict(orient="records"),
            "by_meter": tables["by_meter"].to_dict(orient="records"),
        },
        "episodes": {
            "n_episodes": int(len(episodes)),
            "n_isolated_single_day": int((episodes["duration_days"] == 1).sum()),
            "n_multi_day": int((episodes["duration_days"] >= 2).sum()),
            "longest": episodes.nlargest(5, "duration_days")[
                ["episode_id", "start_date", "end_date", "duration_days",
                 "number_of_flagged_days"]].to_dict(orient="records"),
        },
        "dominant_behaviour_tags": dict(tag_counts.most_common()),
        "interpretation_categories": {
            "A_CALENDAR_WEEKLY_PATTERN_EXPLAINABLE": int(cat_A),
            "B_UNUSUAL_BUT_PLAUSIBLE": int(cat_B),
            "C_POTENTIALLY_UNEXPLAINED_ENERGY_AUDIT_CANDIDATE": int(cat_C),
            "note": "C means candidate for further investigation, NOT "
                    "confirmed equipment failure",
        },
        "strongest_audit_candidates_top10": strongest,
        "documented_thresholds": {
            "HIGH_PEAK": f"peak_ratio >= {T_PEAK_RATIO}",
            "HIGH_BASE": f"base_ratio >= {T_BASE_RATIO}, or nearby median "
                         f"base <= {T_BASE_ABS_ZERO_HIGH} kW and anomaly base "
                         f">= {T_BASE_ABS_HIGH} kW",
            "LOW_BASE": f"base_ratio <= {T_BASE_RATIO_LOW}, or nearby median "
                        f"base >= {T_BASE_ABS_NEARBY} kW and anomaly base "
                        f"<= {T_BASE_ABS_LOW} kW",
            "LOW_LOAD_FACTOR": f"lf_ratio <= {T_LF_LOW}",
            "HIGH_LOAD_FACTOR": f"lf_ratio >= {T_LF_HIGH}",
            "PEAK_HOUR_SHIFT": f"circular diff >= {T_PEAK_HOUR_SHIFT_H} h",
            "SHAPE_SHIFT": f"relative L1 >= {T_SHAPE_L1} (strong: "
                           f">= {T_SHAPE_L1_STRONG})",
            "neighbours": f"up to {NEIGHBOUR_WINDOW} previous + "
                          f"{NEIGHBOUR_WINDOW} next non-anomalous days of the "
                          "SAME meter",
        },
        "engineering_interpretation": actions,
        "answers": {
            "q1_weekday_vs_weekend":
                f"{comp['anomalies_weekday']} weekday vs "
                f"{comp['anomalies_weekend']} weekend anomalies "
                f"(full dataset: {comp['full_dataset_weekend_pct']}% weekend)",
            "q2_highest_anomaly_rate_meters":
                by_meter.nlargest(3, "anomaly_rate_pct")[
                    ["meter", "anomaly_rate_pct"]].to_dict(orient="records"),
            "q3_isolated_or_episodes":
                f"{int((episodes['duration_days'] == 1).sum())} isolated days, "
                f"{int((episodes['duration_days'] >= 2).sum())} multi-day "
                "episodes",
            "q4_month_year_concentration":
                "see concentration.by_year_month (top months: "
                + str(tables["by_year_month"].nlargest(3, "anomaly_count")
                      .to_dict(orient="records")) + ")",
            "q5_dominant_behaviour_types": dict(tag_counts.most_common(5)),
            "q6_category_counts": {"A": int(cat_A), "B": int(cat_B),
                                   "C": int(cat_C)},
            "q7_strongest_candidates": "see strongest_audit_candidates_top10",
            "q8_limitations": [
                "Calendar interpretation is limited because an authoritative "
                "institutional academic/holiday calendar has not been "
                "integrated into this stage.",
                "No tariff/cost data -> no monetary quantification.",
                "Neighbour comparison uses up to 14 nearby same-meter days; "
                "during long outages the window can span very different "
                "seasons.",
                "Categories are screening judgements from measured evidence, "
                "not diagnoses.",
            ],
        },
        "mandated_statements": [
            "The anomaly analysis identifies potentially unusual building-day "
            "energy behaviour. It does not establish equipment faults, "
            "failures, causality, or energy theft.",
            "Calendar interpretation is limited because an authoritative "
            "institutional academic/holiday calendar has not been integrated "
            "into this stage.",
        ],
        "official_model": "K=2, random_state=42, n_init=10 - UNCHANGED",
        "outputs": ["anomaly_context.csv", "anomaly_summary_by_meter.csv",
                    "anomaly_summary_by_category.csv", "anomaly_episodes.csv",
                    "anomaly_calendar_distribution.png",
                    "anomaly_rate_by_meter.png", "anomaly_episode_lengths.png",
                    "anomaly_context_comparison.png"],
    }
    path = MODELLING_DIR / "anomaly_context_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info(f"  saved {path}")
    return report


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("#" * 70)
    log.info("SPANDA Stage 3d - POTENTIAL ANOMALY CONTEXT & CALENDAR "
             "VALIDATION (read-only)")
    log.info("#" * 70)

    before = hash_frozen_artifacts()
    log.info(f"pre-run hash verification: {len(before)} frozen files hashed")

    anom, clustered = load_anomalies()
    anom = add_calendar(anom)
    comp = weekday_weekend_composition(anom, clustered)
    tables = concentration_tables(anom)
    episodes = build_episodes(anom)
    ctx = neighbour_comparison(anom, clustered)
    anom = assign_tags(anom, ctx, episodes, clustered)
    anom = classify(anom)
    out, by_meter, cat = export_tables(anom, clustered)
    plot_figures(out, by_meter)
    report = build_report(out, comp, tables, by_meter, cat, episodes)

    verify_frozen(before, hash_frozen_artifacts())

    # final terminal summary
    n_multi_ep = int((episodes["duration_days"] >= 2).sum())
    n_isolated = int((episodes["duration_days"] == 1).sum())
    top_meter = by_meter.nlargest(1, "anomaly_rate_pct").iloc[0]
    cc = report["interpretation_categories"]
    log.info("=" * 66)
    log.info("SUMMARY")
    log.info(f"  1. total potential anomalies = {len(anom)}")
    log.info(f"  2. weekday anomaly count = {comp['anomalies_weekday']}")
    log.info(f"  3. weekend anomaly count = {comp['anomalies_weekend']}")
    log.info(f"  4. highest anomaly-rate meter = {top_meter['meter']} "
             f"({top_meter['anomaly_rate_pct']:.2f}% of its modelling days)")
    log.info(f"  5. isolated anomalies = {n_isolated}")
    log.info(f"  6. multi-day anomaly episodes = {n_multi_ep}")
    log.info(f"  7. categories: A={cc['A_CALENDAR_WEEKLY_PATTERN_EXPLAINABLE']}, "
             f"B={cc['B_UNUSUAL_BUT_PLAUSIBLE']}, "
             f"C={cc['C_POTENTIALLY_UNEXPLAINED_ENERGY_AUDIT_CANDIDATE']}")
    log.info(f"  8. strongest energy-audit candidates = "
             f"{cc['C_POTENTIALLY_UNEXPLAINED_ENERGY_AUDIT_CANDIDATE']} "
             f"(top 10 listed in anomaly_context_report.json)")
    log.info("  9. official K=2 model (random_state=42, n_init=10) UNCHANGED")
    log.info("  10. ALL FROZEN ARTIFACTS BYTE-IDENTICAL")
    log.info("Stage 3d complete.")


if __name__ == "__main__":
    main()
