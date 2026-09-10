"""
SPANDA Stage 1b - DATA-QUALITY DEEP DIVE (read-only).

Answers, from the files only:
  1. WHEN is data missing?   -> month x meter availability matrix
  2. Lecture-building zeros  -> zero share per year, longest zero runs,
                                minute-of-day histogram of zero-run starts
  3. Zero runs in all meters -> outages vs genuine shutdowns, with dates
  4. Glitch rows (V/F out of physical range) -> counts, overlap, isolation
  5. Minute alignment        -> seconds-of-minute distribution of unix stamps
  6. Yearly availability     -> share of grid minutes available per year

Outputs: spanda_out/audit/deepdive_log.txt, quality_deepdive.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from spanda.config import (
    AUDIT_DIR, FREQ_MAX, FREQ_MIN, METER_FILES, NA_VALUES, PROJECT_ROOT,
    TIMEZONE, VOLTAGE_MAX, VOLTAGE_MIN, get_logger,
)

log = get_logger("spanda.quality", AUDIT_DIR / "deepdive_log.txt")


def load_meter(fname: str) -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / fname, na_values=NA_VALUES)
    ts = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    df = df.assign(timestamp=ts)
    df["dt"] = pd.to_datetime(ts, unit="s", utc=True).dt.tz_convert(TIMEZONE)
    return df.sort_values("timestamp").reset_index(drop=True)


def month_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Share of each month's grid minutes that contain a reading."""
    dt = df["dt"]
    months = pd.period_range(dt.min().to_period("M"), dt.max().to_period("M"), freq="M")
    rows = []
    for m in months:
        start = m.start_time.tz_localize(TIMEZONE)
        end = m.end_time.tz_localize(TIMEZONE)
        total = int((end - start).total_seconds() // 60) + 1
        have = int(df[(dt >= start) & (dt <= end)]["timestamp"].nunique())
        rows.append({"period": str(m), "expected": total, "have": have,
                     "avail_pct": round(100.0 * have / total, 1) if total else np.nan})
    return pd.DataFrame(rows)


def zero_run_analysis(df: pd.DataFrame, top: int = 6) -> dict:
    """Contiguous runs of power==0 (>= 60 min)."""
    is_zero = (df["power"] == 0).to_numpy()
    zero_idx = np.where(is_zero)[0]
    if len(zero_idx) == 0:
        return {"n_runs_ge_60": 0, "runs": [], "zero_share_by_year_pct": {}}
    breaks = np.where(np.diff(zero_idx) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(zero_idx) - 1]))
    runs = []
    for s, e in zip(starts, ends):
        ln = int(e - s + 1)
        if ln >= 60:
            runs.append({
                "start": df["dt"].iloc[zero_idx[s]].strftime("%Y-%m-%d %H:%M"),
                "end": df["dt"].iloc[zero_idx[e]].strftime("%Y-%m-%d %H:%M"),
                "length_min": ln,
            })
    runs = sorted(runs, key=lambda r: -r["length_min"])[:top]
    n_runs = int((np.diff(zero_idx, prepend=-10) > 1).sum())
    yr = df["dt"].dt.year
    zero_by_year = (df.assign(yr=yr).groupby("yr")["power"]
                    .apply(lambda s: round(100.0 * (s == 0).sum() / max(s.notna().sum(), 1), 2))
                    .to_dict())
    return {"n_runs_ge_60": n_runs, "runs": runs, "zero_share_by_year_pct": zero_by_year}


def glitch_analysis(df: pd.DataFrame) -> dict:
    v, f, p = df["voltage"], df["frequency"], df["power"]
    bad_v = (v < VOLTAGE_MIN) | (v > VOLTAGE_MAX)
    bad_f = (f < FREQ_MIN) | (f > FREQ_MAX)
    both = bad_v & bad_f
    flagged = (bad_v | bad_f).to_numpy()
    idx = np.where(flagged)[0]
    n_with_neighbour = int((np.diff(idx) <= 2).sum()) if len(idx) else 0
    return {
        "n_bad_voltage": int(bad_v.sum()),
        "n_bad_freq": int(bad_f.sum()),
        "n_both": int(both.sum()),
        "n_flagged_with_flagged_neighbour": n_with_neighbour,
        "voltage_zero_rows": int((v == 0).sum()),
        "example_rows": [
            {"dt": str(t), "V": float(df["voltage"].iloc[i]),
             "F": float(df["frequency"].iloc[i]), "P": float(df["power"].iloc[i])}
            for i, t in zip(df.index[both][:3], df["dt"][both][:3])
        ],
        "power_median_W_when_V_bad": float(p[bad_v].median()) if bad_v.any() else None,
        "power_median_W_when_F_bad": float(p[bad_f].median()) if bad_f.any() else None,
    }


def minute_alignment(df: pd.DataFrame) -> dict:
    secs = df["timestamp"] % 60
    return {"share_second_0_pct": round(100.0 * (secs == 0).sum() / len(df), 4),
            "n_offgrid_seconds": int((secs != 0).sum())}


def dst_probe(df: pd.DataFrame) -> dict:
    """Minute-of-day histogram of zero-run starts (Lecture anomaly probe)."""
    p = df["power"]
    zero_start = (p == 0) & ~p.shift(1).eq(0)
    starts = df.loc[zero_start, "dt"]
    mod = starts.dt.hour * 60 + starts.dt.minute
    hist = mod.value_counts().sort_index()
    top = hist.sort_values(ascending=False).head(3)
    return {"top_zero_start_minutes_of_day": {int(k): int(v) for k, v in top.items()},
            "n_zero_starts": int(zero_start.sum())}


def yearly_avail(df: pd.DataFrame) -> dict:
    dt = df["dt"]
    out = {}
    for y in sorted(dt.dt.year.unique()):
        s = max(pd.Timestamp(f"{y}-01-01", tz=TIMEZONE), dt.min())
        e = min(pd.Timestamp(f"{y}-12-31 23:59:59", tz=TIMEZONE), dt.max())
        if e < s:
            out[str(y)] = None
            continue
        total = int((e - s).total_seconds() // 60) + 1
        have = int(df[(dt >= s) & (dt <= e)]["timestamp"].nunique())
        out[str(y)] = round(100.0 * have / total, 1)
    return out


def main() -> None:
    log.info("=" * 70)
    log.info("SPANDA Stage 1b - DEEP DIVE")
    log.info("=" * 70)
    all_res: dict = {}

    for meter, fname in METER_FILES.items():
        log.info(f"\n=== {meter} ===")
        df = load_meter(fname)

        ma = month_availability(df)
        low = ma[ma["avail_pct"] < 60]
        log.info("monthly availability: min %.1f%% | months<60%%: %s",
                 ma["avail_pct"].min(),
                 ", ".join(f"{r.period}={r.avail_pct}" for r in low.itertuples()) or "none")

        zr = zero_run_analysis(df)
        log.info("zero runs >=1h: n=%d | longest: %s | zero%% by year: %s",
                 zr["n_runs_ge_60"],
                 zr["runs"][0] if zr["runs"] else "-",
                 zr["zero_share_by_year_pct"])

        gl = glitch_analysis(df)
        log.info("glitches: badV=%d badF=%d both=%d flagged_w_neighbour=%d V=0rows=%d",
                 gl["n_bad_voltage"], gl["n_bad_freq"], gl["n_both"],
                 gl["n_flagged_with_flagged_neighbour"], gl["voltage_zero_rows"])

        al = minute_alignment(df)
        log.info("alignment: %.4f%% at :00s (off-grid rows: %d)",
                 al["share_second_0_pct"], al["n_offgrid_seconds"])

        yr = yearly_avail(df)
        log.info("yearly availability %%: %s", yr)

        all_res[meter] = {"monthly": ma.to_dict("records"), "zero_runs": zr,
                          "glitches": gl, "alignment": al, "yearly": yr}
        del df

    dfL = load_meter(METER_FILES["Lecture"])
    d = dst_probe(dfL)
    log.info("\n=== Lecture zero-start probe ===")
    log.info("zero-start minute-of-day top: %s (n=%d)",
             d["top_zero_start_minutes_of_day"], d["n_zero_starts"])
    all_res["Lecture_zero_start_probe"] = d

    out = AUDIT_DIR / "quality_deepdive.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(all_res, fh, indent=2, default=str)
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
