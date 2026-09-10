"""
SPANDA Stage 2 - CLEANING & PREPROCESSING PIPELINE
(reproducible; every decision logged to spanda_out/audit/preprocessing_log.txt)

Order of operations per meter
-----------------------------
 1. Load raw CSV (NA string handled as missing, NEVER as 0).
 2. Convert unix -> Asia/Kolkata; verify minute alignment (seconds == 0).
 3. Sort chronologically; assert strictly increasing.
 4. Remove exact-duplicate timestamps (audit found 0; step kept for safety).
 5. Reindex to a strict 1-minute calendar grid spanning the meter's own
    coverage -> previously-missing timestamps appear as NaN (explicit gaps).
 6. Value-sanity masks (FLAG ONLY - nothing is deleted):
      - voltage / frequency outside physical range  -> V/F glitch row
      - |power_factor| > 1.0005                     -> pf glitch row
      - extreme steps: |dP| > 10x median power between adjacent minutes
    Power is NEVER edited by these masks. V/F glitches are excluded from the
    voltage/frequency columns only (power of that minute is kept).
 7. Short-gap interpolation of power ONLY: linear in time for gaps
    <= MAX_INTERP_GAP_MIN (15) minutes that are bracketed by valid readings;
    boundary gaps (start/end) and longer gaps are left NaN. Power-NaN count
    before/after is logged as the audit trail.
 8. Aggregate to hourly means (power_kW = mean of W / 1000) keeping the
    fraction of valid minutes per hour -> hours with < 60% valid minutes
    become NaN (never fabricated).
 9. BUILDING-DAY observation unit (TASK 11/12): 24-hour normalized shape +
    daily summary statistics per meter-day.
10. Export per-meter cleaned hourly parquet + building-day features CSV.

Buildings-day units are kept only if daily valid share >= MIN_VALID_SHARE_PER_DAY.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from spanda.config import (
    AUDIT_DIR, MAX_INTERP_GAP_MIN, METER_FILES, MIN_VALID_SHARE_PER_DAY,
    NA_VALUES, NOMINAL_INTERVAL_MIN, PF_ABS_MAX, PHYSICS_CHECK_TOL,
    PROCESSED_DIR, PROJECT_ROOT, STEP_JUMP_ALERT_RATIO, TIMEZONE,
    VOLTAGE_MAX, VOLTAGE_MIN, FREQ_MAX, FREQ_MIN, get_logger,
)

log = get_logger("spanda.preprocess", AUDIT_DIR / "preprocessing_log.txt")

GRID = f"{NOMINAL_INTERVAL_MIN}min"


# ---------------------------------------------------------------------------
# per-meter cleaning
# ---------------------------------------------------------------------------
def clean_meter(meter: str, fname: str) -> tuple[pd.DataFrame, dict]:
    path = PROJECT_ROOT / fname
    log.info("=" * 66)
    log.info(f"[{meter}] cleaning {fname}")

    # 1-2. load + timezone
    df = pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)
    n_raw = len(df)
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    bad_ts = int(ts.isna().sum())
    ts = ts.astype("int64")
    dt = pd.to_datetime(ts, unit="s", utc=True).dt.tz_convert(TIMEZONE)
    assert (ts % 60 == 0).all(), f"[{meter}] off-grid seconds found"
    df = df.assign(timestamp=ts, dt=dt)

    # 3-4. sort + drop exact duplicate timestamps (defensive; audit found 0)
    df = df.sort_values("timestamp", kind="mergesort").drop_duplicates(
        subset="timestamp", keep="first")
    n_dup = n_raw - len(df)
    assert df["timestamp"].is_unique and df["timestamp"].is_monotonic_increasing

    # 5. reindex to strict 1-minute grid (gaps become explicit NaN rows)
    full_idx = pd.date_range(df["dt"].min(), df["dt"].max(),
                             freq=GRID, inclusive="both")
    df = df.set_index("dt").reindex(full_idx)
    df.index.name = "dt"
    n_grid = len(df)
    n_power_nan_grid = int(df["power"].isna().sum())

    # 6. value-sanity masks (FLAG ONLY)
    v, f, pf, p = df["voltage"], df["frequency"], df["power_factor"], df["power"]
    v_bad = (v < VOLTAGE_MIN) | (v > VOLTAGE_MAX)
    f_bad = (f < FREQ_MIN) | (f > FREQ_MAX)
    pf_bad = pf.abs() > PF_ABS_MAX
    n_v_bad, n_f_bad, n_pf_bad = int(v_bad.sum()), int(f_bad.sum()), int(pf_bad.sum())

    df["v_flag"] = v_bad | f_bad | pf_bad          # glitch row (V/F/pf untrustworthy)
    med_p = float(p.median())
    step = p.diff().abs()
    p_step_alert = step > STEP_JUMP_ALERT_RATIO * med_p
    n_step_alert = int(p_step_alert.sum())

    # glitch minutes: V/F excluded (set NaN) but power untouched
    df.loc[df["v_flag"], ["voltage", "frequency", "power_factor"]] = np.nan
    # pf==0 rows: sign convention is undefined at zero power - keep as-is (flag only)
    n_pf_zero = int((pf == 0).sum())

    # 7. short-gap interpolation of power ONLY
    p_before = int(p.isna().sum())
    # interpolate(method="time", limit_area="inside", limit=15):
    #   * "inside"     -> only gaps bracketed by valid readings (never edges)
    #   * limit=15     -> only runs of <= 15 consecutive missing minutes filled
    # Linear-in-time == time-weighted linear interpolation on the minute grid.
    df["power_clean"] = p.interpolate(method="time", limit=MAX_INTERP_GAP_MIN,
                                      limit_area="inside")
    n_power_nan_after = int(df["power_clean"].isna().sum())
    n_interp = p_before - n_power_nan_after
    df.drop(columns=["power"], inplace=True)
    df.rename(columns={"power_clean": "power_W"}, inplace=True)

    # 8. hourly aggregation (mean W -> kW; validity share per hour)
    agg = df.resample("1h").agg(
        power_W_mean=("power_W", "mean"),
        voltage_mean=("voltage", "mean"),
        frequency_mean=("frequency", "mean"),
        pf_mean=("power_factor", "mean"),
        n_valid_power=("power_W", "count"),
    )
    agg["power_kW"] = agg["power_W_mean"] / 1000.0
    agg["valid_share_min"] = agg["n_valid_power"] / NOMINAL_INTERVAL_MIN
    agg.loc[agg["valid_share_min"] < 0.6, "power_kW"] = np.nan  # unreliable hour
    hourly = agg[["power_kW", "voltage_mean", "frequency_mean", "pf_mean",
                  "valid_share_min"]].copy()

    stats = {
        "meter": meter,
        "rows_raw": n_raw,
        "rows_after_dedup": n_raw - n_dup,
        "duplicates_removed": n_dup,
        "grid_rows": n_grid,
        "grid_missing_power": n_power_nan_grid,
        "grid_missing_power_pct": round(100 * n_power_nan_grid / n_grid, 3),
        "interp_used": n_interp,
        "power_nan_after_interp": n_power_nan_after,
        "voltage_freq_glitch_rows": n_v_bad + n_f_bad,
        "pf_glitch_rows": n_pf_bad,
        "pf_zero_rows": n_pf_zero,
        "power_step_alerts": n_step_alert,
        "hourly_rows": len(hourly),
        "hourly_valid_power": int(hourly["power_kW"].notna().sum()),
        "hourly_valid_pct": round(100 * hourly["power_kW"].notna().mean(), 2),
    }
    log.info(f"  raw={n_raw:,} dup_removed={n_dup} grid={n_grid:,} "
             f"missing_power={n_power_nan_grid:,} ({stats['grid_missing_power_pct']}%)")
    log.info(f"  interpolation: filled {n_interp:,} power NaNs "
             f"(<= {MAX_INTERP_GAP_MIN}-min inside gaps); "
             f"{n_power_nan_after:,} remain NaN")
    log.info(f"  glitches: V/F={n_v_bad + n_f_bad} pf>{PF_ABS_MAX}={n_pf_bad} "
             f"pf==0={n_pf_zero} step_alerts={n_step_alert}")
    log.info(f"  hourly: {stats['hourly_rows']:,} rows, "
             f"valid power hours {stats['hourly_valid_pct']}%")

    return hourly, stats


# ---------------------------------------------------------------------------
# building-day observation unit
# ---------------------------------------------------------------------------
def build_days(hourly: pd.DataFrame, meter: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = hourly.copy()
    d["date"] = d.index.date
    d["hour"] = d.index.hour

    # per-day validity on the hourly series (16/24 h threshold ~ 0.67)
    day_valid = d.groupby("date")["power_kW"].apply(lambda s: s.notna().mean())
    keep_days = day_valid[day_valid >= MIN_VALID_SHARE_PER_DAY].index

    piv = d.pivot_table(index="date", columns="hour", values="power_kW",
                        aggfunc="first")
    piv.columns = [f"h{h:02d}_kW" for h in piv.columns]

    feats = d.groupby("date").agg(
        daily_mean_kW=("power_kW", "mean"),
        daily_peak_kW=("power_kW", "max"),
        daily_min_kW=("power_kW", "min"),
        daily_total_kWh=("power_kW", "sum"),   # 1 h x kW = kWh
        base_kW=("power_kW", lambda s: s.quantile(0.10)),
    )
    feats["load_factor"] = feats["daily_mean_kW"] / feats["daily_peak_kW"]

    def _peak_hour(s: pd.Series):
        return s.idxmax().hour if s.notna().any() else np.nan

    feats["peak_hour"] = d.groupby("date")["power_kW"].apply(_peak_hour)

    def _period_share(dsub: pd.DataFrame) -> pd.Series:
        return dsub.groupby("date")["power_kW"].sum().reindex(feats.index)

    tot = feats["daily_total_kWh"]
    feats["morning_share"] = _period_share(d[d["hour"].between(6, 11)]) / tot
    feats["afternoon_share"] = _period_share(d[d["hour"].between(12, 17)]) / tot
    feats["evening_share"] = _period_share(d[d["hour"].between(18, 23)]) / tot
    feats["night_share"] = _period_share(d[d["hour"] < 6]) / tot

    bd = piv.join(feats)
    bd["meter"] = meter
    bd["day_valid_share"] = day_valid
    bd = bd[(bd["day_valid_share"] >= MIN_VALID_SHARE_PER_DAY)
            & bd["daily_mean_kW"].notna()].reset_index()
    bd = bd.rename(columns={"date": "obs_date"})
    bd["weekday"] = pd.to_datetime(bd["obs_date"]).dt.dayofweek
    # drop leap-day 02-29 (only 2016) for a clean 24-h shape grid
    bd = bd[~((pd.to_datetime(bd["obs_date"]).dt.month == 2) &
              (pd.to_datetime(bd["obs_date"]).dt.day == 29))]

    hourly_export = hourly.copy()
    hourly_export["meter"] = meter
    return hourly_export, bd


# ---------------------------------------------------------------------------
def main() -> None:
    log.info("#" * 70)
    log.info("SPANDA Stage 2 - CLEANING & PREPROCESSING")
    log.info("#" * 70)

    all_hourly, all_days, all_stats = [], [], []
    for meter, fname in METER_FILES.items():
        hourly, stats = clean_meter(meter, fname)
        hexp, bd = build_days(hourly, meter)
        all_hourly.append(hexp)
        all_days.append(bd)
        all_stats.append(stats)
        log.info(f"  building-days kept: {len(bd):,} "
                 f"(valid share >= {MIN_VALID_SHARE_PER_DAY})")

    hourly_all = pd.concat(all_hourly).reset_index().rename(columns={"index": "dt"})
    days_all = pd.concat(all_days, ignore_index=True)

    hourly_path = PROCESSED_DIR / "hourly_power_clean.parquet"
    days_path = PROCESSED_DIR / "building_day_features.csv"
    stats_path = AUDIT_DIR / "preprocessing_summary.json"
    hourly_all.to_parquet(hourly_path, index=False)
    days_all.to_csv(days_path, index=False)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(all_stats, fh, indent=2, default=str)

    log.info("-" * 66)
    log.info(f"Saved {hourly_path} ({len(hourly_all):,} rows)")
    log.info(f"Saved {days_path} ({len(days_all):,} building-days)")
    log.info(f"Saved {stats_path}")


if __name__ == "__main__":
    main()
