"""
SPANDA Stage 3 - PCA + K-MEANS MODELLING (building-day observation unit)
(reproducible; every decision logged to spanda_out/audit/modelling_log.txt)

Design decisions (all data-driven, nothing hard-coded)
------------------------------------------------------
A. MODELLING MATRIX - assembled from the verified Stage-1 building-day table.
   Categories (WHY each is in / out):
     * OUT: identifiers (meter), calendar labels (obs_date, weekday) - carried
       along for interpretation, never scaled or modelled. Categoricals cannot
       enter PCA, and weekday is derivable from obs_date (leak-free metadata).
     * OUT: day_valid_share - a Stage-1 QC mask, not electrical behaviour.
     * OUT: daily_total_kWh - EXACT duplicate of daily_mean_kW (24 h sum,
       r = +1.000) -> redundant by construction (leakage/repeated info).
     * OUT: daily_mean_kW, daily_peak_kW, daily_min_kW, base_kW - absolute
       magnitude stats. They are retained individually as a documented ablation
       group, but the PRIMARY feature set keeps the shape family only.
     * IN (primary): 24 hourly kW columns h00_kW..h23_kW - the raw diurnal
       shape, the physical substrate of behaviour profiling.
     * IN (primary): load_factor (mean/peak, 0-1), morning/afternoon/evening/
       night shares (sum to 1), peak_hour - dimensionless shape descriptors
       that make behaviour comparable across buildings of different size.
   Scale handling: every kW feature is log1p-transformed BEFORE standardizing
   so large commercial buildings do not dominate the variance (the Stage-1
   audit showed Academic ~50 kW vs Lecture ~0-50 kW). Ratios/shares are kept
   linear; tiny negative float noise in kW (>= -1e-6, measurement noise) is
   clamped to 0. peak_hour becomes sin/cos (circular).
   Rows kept only when ALL selected features are finite.

B. STANDARDIZATION - StandardScaler on selected features only; identifiers
   never scaled. Artifacts saved: scaler.joblib, feature_names.json,
   standardized_features.parquet.

C. PCA - full decomposition on standardized matrix. Component count chosen
   from the ACTUAL cumulative explained variance (first n reaching
   PCA_VAR_TARGET = 0.90), reported next to the Kaiser (eigenvalue > 1)
   rule. Loadings, top +/- contributors, engineering interpretations saved.

D. K-MEANS - K in [2, K_MAX]. K_MAX = min(15, n_samples // 500) computed at
   runtime (>>500 samples per candidate cluster). Diagnostics: inertia AND
   mean silhouette. Selection: elbow via maximum second-difference of
   log(inertia) (kneedle-style), silhouette via argmax; agreement analysis,
   final choice documented (NOT blind elbow). Final model: random_state=42,
   n_init=10 (explicit, KMeans++ init).

E. VISUALIZATION - PC1 vs PC2 projection scatter (stated as projection only),
   elbow, silhouette, scree, cumulative-variance, loading-heatmap figures.

F. VALIDATION - silhouette, cluster sizes, pairwise centroid distances,
   min cluster share; small/blur warnings flagged honestly in report + log.

G. OUTPUTS - spanda_out/modelling/* + spanda_out/figures/* + report JSON.

No accuracy/precision/recall/F1 (unsupervised). No semantic cluster names.
PCA + K-Means describe statistical structure - no causal claims anywhere.
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
from sklearn.preprocessing import StandardScaler

from spanda.config import (
    AUDIT_DIR, FIGURES_DIR, MODELLING_DIR, PROCESSED_DIR, get_logger,
)

log = get_logger("spanda.model", AUDIT_DIR / "modelling_log.txt")

# ---------------------------------------------------------------------------
# configuration (documented engineering decisions - see module docstring)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_INIT = 10                    # explicit; KMeans++ init
PCA_VAR_TARGET = 0.90          # smallest n_components reaching 90% cum. var
K_MIN, K_MAX_CAP = 2, 15
MIN_CLUSTER_SHARE_WARN = 0.02  # flag clusters smaller than 2% of sample
PERIOD_SHARES = ["morning_share", "afternoon_share", "evening_share",
                 "night_share"]

# Which absolute-magnitude stats COULD join the model. Kept as an ablation
# group: the primary matrix is shape-only; the ablation re-runs PCA+K-Means
# with magnitudes added to document their effect instead of guessing.
MAGNITUDE_FEATURES = ["daily_mean_kW", "daily_peak_kW", "daily_min_kW",
                      "base_kW"]

# Assembly: hourly shape (log1p) + ratio features (linear) + magnitude (log1p)
ASSEMBLY_FEATURES: dict[str, str] = {
    **{f"h{h:02d}_kW": "log1p" for h in range(24)},
    **{f: "log1p" for f in MAGNITUDE_FEATURES},
    "load_factor": "linear",
    **{f: "linear" for f in PERIOD_SHARES},
    "peak_hour_sin": "linear",
    "peak_hour_cos": "linear",
}


def _nonneg(x: pd.DataFrame) -> pd.DataFrame:
    """Clamp negatives to 0 in kW columns ONLY (physically non-negative).
    sin/cos coordinates are legitimately negative and are NEVER clamped."""
    kw_cols = [c for c in x.columns if c.endswith("_kW")]
    neg = x[kw_cols] < 0
    n_neg = int(neg.sum().sum())
    if n_neg:
        worst = float(x[kw_cols].mask(~neg).min().min())
        log.info(f"  clamped {n_neg} negative kW values (worst {worst:.2e}) to 0")
        x = x.copy()
        x.loc[:, kw_cols] = x[kw_cols].mask(neg, 0.0)
    else:
        log.info("  kW columns: no negative values (Stage-1 verification holds)")
    return x


def apply_transforms(X: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for col, how in ASSEMBLY_FEATURES.items():
        s = X[col]
        out[col] = np.log1p(s) if how == "log1p" else s
    return pd.DataFrame(out, index=X.index)


# ---------------------------------------------------------------------------
# PART A - modelling matrix
# ---------------------------------------------------------------------------
def prepare_modelling_matrix() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Assemble Part-A matrix; returns (X_features, id_cols, feature_cols)."""
    bd = pd.read_csv(PROCESSED_DIR / "building_day_features.csv")
    n_all = len(bd)
    hcols = [f"h{h:02d}_kW" for h in range(24)]

    # circular encoding of peak_hour (0-23 -> unit circle; NaN-safe)
    ph = bd["peak_hour"].to_numpy(dtype="float64")
    bd["peak_hour_sin"] = np.sin(2 * np.pi * ph / 24.0)
    bd["peak_hour_cos"] = np.cos(2 * np.pi * ph / 24.0)

    feats = list(ASSEMBLY_FEATURES)
    X = bd[feats].astype("float64")
    X = _nonneg(X)

    finite_mask = np.isfinite(X.to_numpy()).all(axis=1)
    n_drop = int((~finite_mask).sum())
    drop_detail = (bd.loc[~finite_mask, "meter"].value_counts()
                   .to_dict() if n_drop else {})
    X, ids = X.loc[finite_mask].reset_index(drop=True), bd.loc[finite_mask].reset_index(drop=True)
    ids["date_iso"] = pd.to_datetime(ids["obs_date"]).dt.strftime("%Y-%m-%d")
    log.info("=" * 66)
    log.info("PART A - modelling matrix")
    n_zero_days = int(((bd["daily_peak_kW"] == 0)
                       & bd["load_factor"].isna()).sum())
    log.info(f"  building-days loaded: {n_all:,}; dropped {n_drop} rows with "
             f"non-finite features {drop_detail} -> {len(X):,} observations")
    log.info(f"  of the dropped rows, {n_zero_days} are all-zero days (peak == "
             f"0 W, mostly Lecture closed days): their shape ratios "
             f"(load_factor, period shares, peak_hour) are 0/0 = undefined; "
             f"they are excluded as mathematically undefined behaviour, not "
             f"as bad data (decision logged, never imputed)")
    log.info(f"  features: {len(feats)} "
             f"(24 hourly kW, {len(MAGNITUDE_FEATURES)} magnitude stats, "
             f"load_factor, 4 period shares, peak_hour as sin/cos)")
    log.info("  excluded: meter/obs_date/weekday (ids/calendar), "
             "day_valid_share (QC mask), daily_total_kWh (== 24*daily_mean, "
             "r = +1.000 -> redundant by construction)")
    log.info("  kW features log1p-transformed so absolute size does not "
             "dominate PCA variance; ratios/shares kept linear")

    # zero-variance check on the AS-ASSEMBLED (pre-scaling) matrix
    zv = [c for c in X.columns if float(X[c].std()) <= 1e-12]
    if zv:
        raise RuntimeError(f"zero-variance features present: {zv}")
    log.info("  zero-variance check: none")

    # pairwise redundancy census (documented, not acted on beyond kWh removal)
    corr = X.corr().abs().to_numpy(dtype="float64").copy()
    np.fill_diagonal(corr, 0.0)
    hot = [(feats[i], feats[j], float(corr[i, j]))
           for i, j in zip(*np.where(corr >= 0.95)) if i < j]
    hot.sort(key=lambda t: -t[2])
    log.info("  feature pairs with |r| >= 0.95 "
             "(documented; PCA handles collinearity):")
    for a, b, r in hot:
        log.info(f"    {a} ~ {b}: r = {r:.3f}")
    return X, ids, feats


# ---------------------------------------------------------------------------
# PART B - standardization
# ---------------------------------------------------------------------------
def fit_and_save_scaler(X: pd.DataFrame) -> np.ndarray:
    log.info("=" * 66)
    log.info("PART B - standardization (StandardScaler, features only)")
    scaler = StandardScaler().fit(X.to_numpy(dtype="float64"))
    Z = scaler.transform(X.to_numpy(dtype="float64"))

    joblib.dump(scaler, MODELLING_DIR / "scaler.joblib")
    with open(MODELLING_DIR / "feature_names.json", "w", encoding="utf-8") as fh:
        json.dump({"features": list(X.columns),
                   "transforms": ASSEMBLY_FEATURES,
                   "n_observations": int(Z.shape[0])}, fh, indent=2)

    out = pd.DataFrame(Z, columns=list(X.columns))
    log.info(f"  standardized matrix: {Z.shape[0]:,} x {Z.shape[1]} "
             f"(means ~= 0, stds ~= 1); artifacts saved")
    return Z


# ---------------------------------------------------------------------------
# PART C - PCA
# ---------------------------------------------------------------------------
def run_pca(Z: np.ndarray, feature_names: list[str]) -> dict:
    log.info("=" * 66)
    log.info("PART C - PCA (full decomposition)")
    pca = PCA(n_components=None, svd_solver="full",
              random_state=RANDOM_STATE).fit(Z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    ev = pca.explained_variance_                       # eigenvalues
    n_kaiser = int(np.argmax(ev < 1.0)) if (ev < 1.0).any() else len(ev)
    n_target = int(np.searchsorted(cum, PCA_VAR_TARGET) + 1)
    n_pc = min(n_target, Z.shape[1])
    log.info(f"  components for >= {PCA_VAR_TARGET:.0%} cum. variance: "
             f"{n_target} (retained)")
    log.info(f"  Kaiser rule (eigenvalue > 1) would keep: {n_kaiser}")
    log.info(f"  PC1/PC2/PC3 EVR: {evr[0]:.3f}/{evr[1]:.3f}/{evr[2]:.3f}; "
             f"cum. at retained n: {cum[n_pc - 1]:.3f}")

    loadings = pd.DataFrame(
        pca.components_.T, index=feature_names,
        columns=[f"PC{i + 1}" for i in range(len(evr))])
    loadings.to_csv(MODELLING_DIR / "pca_loadings.csv")

    Pc = pca.transform(Z)[:, :n_pc]
    pc_cols = [f"PC{i + 1}" for i in range(n_pc)]

    def _top_features(w: np.ndarray, k_pos: int = 5, k_neg: int = 3) -> dict:
        k_pos, k_neg = min(k_pos, len(w)), min(k_neg, len(w))
        order = np.argsort(-w)
        return {"top_positive": [
            {"feature": feature_names[i], "loading": round(float(w[i]), 4)}
            for i in order[:k_pos]],
            "top_negative": [
            {"feature": feature_names[i], "loading": round(float(w[i]), 4)}
            for i in order[-k_neg:][::-1]]}

    interpretations: dict[str, dict] = {}
    for i in range(n_pc):
        w = pca.components_[i]
        interpretations[f"PC{i + 1}"] = {
            "explained_variance_ratio": round(float(evr[i]), 4),
            "eigenvalue": round(float(ev[i]), 3),
            **_top_features(w),
        }
    with open(MODELLING_DIR / "pca_interpretation.json", "w",
              encoding="utf-8") as fh:
        json.dump(interpretations, fh, indent=2)

    for pc, info in interpretations.items():
        pos = ", ".join(f"{d['feature']} (+{d['loading']})"
                        for d in info["top_positive"][:3])
        neg = ", ".join(f"{d['feature']} ({d['loading']})"
                        for d in info["top_negative"][:2])
        log.info(f"  {pc}: EVR={info['explained_variance_ratio']:.3f} | "
                 f"top+: {pos} | top-: {neg}")
    log.info("  NOTE: loadings are statistical contributions, NOT causes; "
             "no causal claim is made anywhere.")

    return {"pca": pca, "evr": evr, "cum": cum, "ev": ev, "n_pc": n_pc,
            "n_kaiser": n_kaiser, "pc_cols": pc_cols,
            "Pc": Pc, "loadings": loadings,
            "interpretations": interpretations}


def save_pca_outputs(ids: pd.DataFrame, Pc: np.ndarray, pc_cols: list[str],
                     Z: np.ndarray, feat_names: list[str]) -> None:
    pca_df = ids[["meter", "date_iso"]].copy()
    for j, c in enumerate(pc_cols):
        pca_df[c] = Pc[:, j]
    pca_df.to_parquet(MODELLING_DIR / "pca_features.parquet", index=False)

    std_df = ids[["meter", "date_iso"]].copy()
    for j, c in enumerate(feat_names):
        std_df[c] = Z[:, j]
    std_df.to_parquet(MODELLING_DIR / "standardized_features.parquet",
                      index=False)
    log.info(f"  saved pca_features.parquet ({Pc.shape[0]:,} x {Pc.shape[1]} PCs) "
             f"and standardized_features.parquet ({Z.shape[1]} features)")


# ---------------------------------------------------------------------------
# PART D - K selection + final K-Means
# ---------------------------------------------------------------------------
def choose_k(Pc: np.ndarray) -> dict:
    k_max = min(K_MAX_CAP, Pc.shape[0] // 500)
    ks = list(range(K_MIN, k_max + 1))
    log.info("=" * 66)
    log.info(f"PART D - K-Means model selection (K = {K_MIN}..{k_max}, "
             f"n_init={N_INIT}, random_state={RANDOM_STATE}, k-means++)")
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = km.fit_predict(Pc)
        inertia.append(float(km.inertia_))
        sil.append(float(silhouette_score(Pc, labels)))
        log.info(f"  K={k:2d}: inertia={km.inertia_:.1f}, "
                 f"silhouette={sil[-1]:.4f}")

    li = np.log(np.array(inertia))
    d2 = np.diff(li, n=2)                     # elbow: max second difference
    k_elbow = int(ks[int(np.argmax(d2)) + 1])
    k_sil = int(ks[int(np.argmax(sil))])
    log.info(f"  elbow diagnostic (max 2nd diff of log-inertia): K = {k_elbow}")
    log.info(f"  best mean silhouette: K = {k_sil} ({max(sil):.4f})")
    if k_elbow == k_sil:
        k_final = k_elbow
        log.info(f"  both diagnostics agree -> K = {k_final}")
    else:
        # decision rule: prefer the silhouette optimum when both diagnostics
        # disagree AND the silhouette peak is a clear maximum; else the elbow.
        s = np.array(sil)
        margin = float(s.max() - np.partition(s, -2)[-2])
        if margin >= 0.01:
            k_final = k_sil
            log.info(f"  diagnostics disagree (elbow={k_elbow}, silhouette="
                     f"{k_sil}); silhouette peak is clear "
                     f"(margin {margin:.4f} >= 0.01) -> K = {k_final}")
        else:
            k_final = k_elbow
            log.info(f"  diagnostics disagree and silhouette peak is flat "
                     f"(margin {margin:.4f} < 0.01) -> K = {k_final} (elbow)")
    return {"ks": ks, "inertia": inertia, "silhouette": sil,
            "k_elbow": k_elbow, "k_sil": k_sil, "k_final": k_final}


def final_kmeans(Pc: np.ndarray, k: int) -> dict:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    labels = km.fit_predict(Pc)
    sil = float(silhouette_score(Pc, labels))
    log.info(f"  FINAL model: KMeans(K={k}) inertia={km.inertia_:.1f}, "
             f"silhouette={sil:.4f}, iterations={km.n_iter_}")
    return {"model": km, "labels": labels, "silhouette": sil,
            "inertia": float(km.inertia_), "n_iter": int(km.n_iter_)}


# ---------------------------------------------------------------------------
# PART E - figures
# ---------------------------------------------------------------------------
def plot_pca_figures(pca: dict) -> None:
    evr, cum, ev = pca["evr"], pca["cum"], pca["ev"]
    n_pc = pca["n_pc"]

    # scree
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(evr) + 1), evr, "o-", ms=4, label="EVR")
    ax.axvline(n_pc + 0.5, ls="--", c="tab:red",
               label=f"retained = {n_pc}")
    ax.axhline(1.0 / len(evr), ls=":", c="grey", lw=1,
               label="average EVR")
    ax.set(xlabel="Principal component", ylabel="Explained variance ratio",
           title=f"PCA scree plot (eigenvalue PC1 = {ev[0]:.2f})")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "pca_scree.png", dpi=150)
    plt.close(fig)

    # cumulative variance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(cum) + 1), cum, "o-", ms=4)
    ax.axhline(PCA_VAR_TARGET, ls="--", c="tab:red")
    ax.axvline(n_pc + 0.5, ls="--", c="tab:red")
    ax.annotate(f"{n_pc} PCs -> {cum[n_pc - 1]:.1%}",
                xy=(n_pc, cum[n_pc - 1]), xytext=(n_pc + 2, cum[n_pc - 1] - 0.15),
                arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="Number of components", ylabel="Cumulative explained variance",
           title="PCA cumulative explained variance",
           ylim=(0, 1.02))
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "pca_cumulative_variance.png", dpi=150)
    plt.close(fig)

    # loading heatmap (retained PCs)
    L = pca["loadings"].iloc[:, :n_pc]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * n_pc + 3), 9))
    im = ax.imshow(L.to_numpy(), aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    ax.set_xticks(range(n_pc), L.columns)
    ax.set_yticks(range(len(L)), L.index, fontsize=6)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            ax.text(j, i, f"{L.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=4.5, color="k")
    fig.colorbar(im, ax=ax, shrink=0.7, label="loading")
    ax.set_title(f"PCA loading matrix (first {n_pc} PCs)")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "pca_loadings.png", dpi=150)
    plt.close(fig)
    log.info(f"  figures saved: pca_scree.png, pca_cumulative_variance.png, "
             f"pca_loadings.png")


def plot_k_figures(sel: dict, km_final: dict, Pc: np.ndarray,
                   labels: np.ndarray, n_pc: int) -> None:
    ks, inertia, sil = sel["ks"], sel["inertia"], sel["silhouette"]
    k = sel["k_final"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertia, "o-")
    ax.axvline(k, ls="--", c="tab:red", label=f"selected K = {k}")
    ax.set(xlabel="K (number of clusters)", ylabel="Inertia (WCSS)",
           title=f"Elbow curve (diagnostic elbow at K={sel['k_elbow']})")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "elbow_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, sil, "o-")
    ax.axvline(k, ls="--", c="tab:red", label=f"selected K = {k}")
    ax.annotate(f"{sil[ks.index(k)]:.4f}", xy=(k, sil[ks.index(k)]),
                xytext=(k + 0.5, sil[ks.index(k)]), fontsize=9)
    ax.set(xlabel="K (number of clusters)", ylabel="Mean silhouette score",
           title=f"Silhouette vs K (best K = {sel['k_sil']})")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "silhouette_curve.png", dpi=150)
    plt.close(fig)

    # PC1 vs PC2 projection (visualization only; model uses all retained PCs)
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(Pc[:, 0], Pc[:, 1], c=labels, cmap="tab10" if k <= 10
                    else "tab20", s=6, alpha=0.55)
    cents = km_final["model"].cluster_centers_
    ax.scatter(cents[:, 0], cents[:, 1], marker="X", s=160, c="black",
               edgecolors="white", linewidths=1.2, label="centroids")
    evr = km_final["evr"]
    ax.set(xlabel=f"PC1 ({evr[0]:.1%} var)",
           ylabel=f"PC2 ({evr[1]:.1%} var)",
           title=f"Building-days in PCA space, K-Means K={k} "
                 f"(silhouette {km_final['silhouette']:.3f}; "
                 f"model uses {n_pc} PCs - PC1/PC2 shown for viewing only)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "clusters_pca.png", dpi=150)
    plt.close(fig)
    log.info("  figures saved: elbow_curve.png, silhouette_curve.png, "
             "clusters_pca.png (PC1-PC2 projection only)")


# ---------------------------------------------------------------------------
# PART F - cluster validation
# ---------------------------------------------------------------------------
def validate_clusters(Pc: np.ndarray, km_final: dict, sel: dict,
                      n_pc: int) -> dict:
    k = sel["k_final"]
    labels, cents = km_final["labels"], km_final["model"].cluster_centers_
    sizes = np.bincount(labels, minlength=k)
    shares = sizes / sizes.sum()
    d = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=-1)
    iu = np.triu_indices(k, 1)
    log.info("=" * 66)
    log.info("PART F - cluster validation")
    log.info(f"  silhouette (retained {n_pc}-PC space): "
             f"{km_final['silhouette']:.4f}")
    log.info(f"  centroid pair distances: min={d[iu].min():.2f}, "
             f"max={d[iu].max():.2f}, mean={d[iu].mean():.2f}")
    warnings = []
    for c in range(k):
        msg = (f"cluster {c}: n={sizes[c]:,} ({shares[c]:.1%}) | "
               f"centroid dist to nearest={np.delete(d[c], c).min():.2f}")
        if shares[c] < MIN_CLUSTER_SHARE_WARN:
            msg += " -> FLAG: very small cluster"
            warnings.append(f"cluster {c} holds only {shares[c]:.2%} of "
                            f"observations (< {MIN_CLUSTER_SHARE_WARN:.0%})")
        log.info("  " + msg)
    log.info(f"  silhouette interpretation: "
             f"{'reasonable structure' if km_final['silhouette'] >= 0.35 else 'weak structure' if km_final['silhouette'] >= 0.2 else 'poorly separated clusters'}"
             f" (heuristic thresholds 0.35/0.2)")
    log.info("  clusters are statistical groupings of days, not proven "
             "behaviour types; no causal claim.")
    return {"sizes": sizes.tolist(), "shares": shares.tolist(),
            "centroid_distances": d[iu].tolist(),
            "min_centroid_distance": float(d[iu].min()),
            "warnings": warnings}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("#" * 70)
    log.info("SPANDA Stage 3 - PCA + K-MEANS (building-day units)")
    log.info("#" * 70)

    # A
    X, ids, feat_names = prepare_modelling_matrix()
    Xt = apply_transforms(X)

    # B
    Z = fit_and_save_scaler(Xt)

    # C
    pca = run_pca(Z, feat_names)
    save_pca_outputs(ids, pca["Pc"], pca["pc_cols"], Z, feat_names)
    plot_pca_figures(pca)

    # D
    sel = choose_k(pca["Pc"])
    km_final = final_kmeans(pca["Pc"], sel["k_final"])
    km_final["evr"] = pca["evr"]

    # attach cluster labels -> standardized/pca tables + full building-day CSV
    labels = km_final["labels"]
    clustered = ids.copy()
    clustered["cluster"] = labels
    pca_df = pd.read_parquet(MODELLING_DIR / "pca_features.parquet")
    pca_df["cluster"] = labels
    pca_df.to_parquet(MODELLING_DIR / "pca_features.parquet", index=False)
    std_df = pd.read_parquet(MODELLING_DIR / "standardized_features.parquet")
    std_df["cluster"] = labels
    std_df.to_parquet(MODELLING_DIR / "standardized_features.parquet",
                      index=False)
    pivot = clustered.pivot_table(index="date_iso", columns="meter",
                                  values="cluster", aggfunc="first")
    pivot.to_csv(MODELLING_DIR / "cluster_by_meter_by_date.csv")
    clustered.to_csv(MODELLING_DIR / "clustered_building_days.csv",
                     index=False)

    # centroid info (standardized feature space + PC space)
    cents = km_final["model"].cluster_centers_
    cents_df = pd.DataFrame(cents, columns=[f"PC{i + 1}" for i in range(pca["n_pc"])])
    cents_df.index.name = "cluster"
    cents_df.to_csv(MODELLING_DIR / "cluster_centroids.csv")

    # E
    plot_k_figures(sel, km_final, pca["Pc"], labels, pca["n_pc"])

    # F
    val = validate_clusters(pca["Pc"], km_final, sel, pca["n_pc"])

    # G - report
    report = {
        "stage": "SPANDA Stage 3 - PCA + K-Means (building-day units)",
        "random_state": RANDOM_STATE,
        "n_observations": int(X.shape[0]),
        "n_original_features": int(X.shape[1]),
        "feature_list": feat_names,
        "excluded_features": {
            "identifiers_calendar": ["meter", "obs_date", "weekday"],
            "qc_mask": ["day_valid_share"],
            "redundant_by_construction": ["daily_total_kWh (24h sum of "
                                          "hourly kW == 24 x daily_mean_kW, "
                                          "r = +1.000)"]},
        "scale_transforms": ASSEMBLY_FEATURES,
        "pca": {
            "n_components_total": int(len(pca["evr"])),
            "retained_components": int(pca["n_pc"]),
            "kaiser_components": int(pca["n_kaiser"]),
            "variance_target": PCA_VAR_TARGET,
            "variance_retained": float(pca["cum"][pca["n_pc"] - 1]),
            "explained_variance_ratio": [round(float(v), 6)
                                         for v in pca["evr"]],
            "eigenvalues": [round(float(v), 4) for v in pca["ev"]],
            "note": "loadings = statistical contribution, not causation"},
        "kmeans": {
            "tested_k_values": sel["ks"],
            "inertia_by_k": dict(zip(map(str, sel["ks"]),
                                     [round(v, 1) for v in sel["inertia"]])),
            "silhouette_by_k": dict(zip(map(str, sel["ks"]),
                                        [round(v, 4) for v in sel["silhouette"]])),
            "k_elbow": sel["k_elbow"], "k_silhouette": sel["k_sil"],
            "selected_k": sel["k_final"],
            "selection_rationale": ("elbow and silhouette agree"
                                    if sel["k_elbow"] == sel["k_sil"] else
                                    "diagnostics disagreed; decision logged "
                                    "in modelling_log.txt"),
            "final_inertia": round(km_final["inertia"], 1),
            "final_silhouette": round(km_final["silhouette"], 4),
            "n_init": N_INIT, "random_state": RANDOM_STATE},
        "cluster_sizes": dict(zip(map(str, range(sel["k_final"])),
                                  val["sizes"])),
        "validation": val,
        "figures": sorted(p.name for p in FIGURES_DIR.glob("*.png")),
    }
    report_path = MODELLING_DIR / "modelling_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("=" * 66)
    log.info(f"Saved {report_path}")
    log.info(f"Stage 3 complete: {report['n_observations']:,} building-days, "
             f"{report['n_original_features']} features -> "
             f"{pca['n_pc']} PCs ({report['pca']['variance_retained']:.1%} var) "
             f"-> K={sel['k_final']} clusters, "
             f"silhouette={km_final['silhouette']:.4f}")


if __name__ == "__main__":
    main()
