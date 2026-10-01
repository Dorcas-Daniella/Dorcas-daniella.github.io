# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Compound climate events — corrected pipeline (single-notebook version)
#
# > **v2 structure.** Sections 1–4 and 8 (PSU, climate extraction, SPEI) run once and are shared. Sections 5–7, 9–15 run **once per `hw_threshold_mode`** (`calendar`, `annual`), each writing to `PROCESSED/<mode>/`. Section 17 contains **all** figure code (main + SI + null + sensitivities), reading the parquet outputs of each run; `PROCESSED/figures_shared/` holds the cross-mode figures.
#
# **Sub-Saharan Africa, 1985–2024 · ERA5 + CHIRPS + SPEI/SPI · DHS PSUs · 3 axes**
#
# This notebook contains the COMPLETE corrected analysis in one file, for
# local cell-by-cell testing. It is mechanically derived from the modular
# version (same tested code, same defaults), so once it runs on your data the
# switch to the modular submission version is a path change, not a logic change.
#
# **Methodological corrections preserved here** (details in each section):
# 1. in-base/out-of-base percentile inhomogeneity corrected (Zhang et al. 2005, leave-one-out);
# 2. all-NaN months can no longer become fabricated zero-precipitation (coverage masking) — no spurious permanent droughts at coastal PSUs;
# 3. Axis II: Event Coincidence Analysis with per-heatwave indicators (no double counting by construction) + truncated-window exclusion + **seasonal surrogate null**;
# 4. Axis III: drought state taken at the **antecedent month (m−1)** + SPI-3 alongside SPEI-3 (breaks the PET←temperature circularity);
# 5. Axis I: Likelihood Multiplication Factor, naive **and seasonally adjusted**;
# 6. Hamed–Rao (1998) Mann–Kendall actually implemented; moving-block bootstrap CIs; BH-FDR across countries;
# 7. CHIRPS/ERA5 grid-cell ids stored → unique-cell (pseudo-replication) variant; eligibility flags propagate to all denominators.
#
# **Run order**: top to bottom. Heavy cells are marked with expected cost.
#
#

# %% [markdown]
# ## 1. Imports and logging

# %%
import gc
import logging
import shutil
import time
import warnings
from calendar import monthrange
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

# heavier deps used by specific sections (fail early if missing):
import pyarrow                      # parquet IO
import xarray as xr                 # section 4 (extraction) + 8 (SPEI)
import xclim                        # section 8
# pyreadr is imported inside section 3 (only needed there)

log = logging.getLogger("cce")
log.handlers.clear()
_h = logging.StreamHandler()
_h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
log.addHandler(_h)
log.setLevel(logging.INFO)

print("pandas", pd.__version__, "| pyarrow", pyarrow.__version__,
      "| xarray", xr.__version__, "| xclim", xclim.__version__)


# %% [markdown]
# ## 2. Direct paths and configuration  ← **EDIT THIS CELL**
#
# All parameters of the study live here. The `cfg` namespace mirrors the
# modular `config.py` exactly, so the code below is identical to the tested
# modules.
#

# %%
# ------------------------- PATHS — EDIT HERE -------------------------------
BASE_DIR = Path(r"D:\Home\debedinding\Documents\PhD")     # <-- EDIT

RAW_PSU_RDATA = BASE_DIR / "Code_PhD" / "PSU_geocoordinates.RData"   # <-- CHECK: object name 'psu' assumed in section 3
ERA5_NC   = BASE_DIR / "Code_PhD" / "ERA5_SubSahara_Tmax_Tmin_Daily_1985_2024.nc"  # <-- CHECK: vars tmax/tmin, dims valid_time/latitude/longitude
CHIRPS_NC = BASE_DIR / "Code_PhD" / "chirps_v2_1981_2024.nc"          # <-- CHECK: var precip, dims time/latitude/longitude

PROCESSED = BASE_DIR / "Code_PhD" / "processed_3"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ------------------------- CONFIG ------------------------------------------
cfg = SimpleNamespace(
    # study design
    study_start_year=1985, study_end_year=2024,
    ref_start_year=1985, ref_end_year=2014,
    decade_bounds=((1985, 1994), (1995, 2004), (2005, 2014), (2015, 2024)),
    # spatial matching (section 4)
    max_match_km_era5=20.0, max_match_km_chirps=20.0,
    extraction_batch_size=2000,
    # heatwaves (sections 5-6)
    hw_percentile=90.0, hw_min_days=3, hw_window_half=15,
    hw_threshold_modes=("calendar", "annual"),  # BOTH are run -> PROCESSED/<mode>/
    hw_threshold_mode="calendar",   # current mode; set automatically by set_mode()
    #   calendar : ETCCDI doy percentile in a +/-hw_window_half window (as before)
    #   annual   : ONE P90 per PSU over all reference-period days, applied all year
    in_base_correction="none",      # "none" | "loo" | "zhang"  (Zhang 2005)
    zhang_n_replicates=29,
    # EPE (sections 5,7)
    epe_percentile=95.0, wet_day_min_mm=1.0, min_wet_days_ref=100,
    # SPEI/SPI (section 8)
    spei_scale_months=3, drought_threshold=-1.0, spei_clip=3.5,
    spei_dist="fisk", spei_method="PWM",   # PWM->ML fallback is automatic & logged
    min_month_coverage=0.9,
    # Axis I
    axis1_day_tolerance=0,          # 0 strict same-day; 1 = ERA5/CHIRPS day-convention sensitivity
    # Axis II
    axis2_window_days=30, axis2_exclude_truncated=True,
    axis2_attribution_tiebreak="earlier",
    axis2_window_sens=(7, 15, 45, 60),  # window sensitivity (Axis II only; primary W kept)
    # Axis III
    axis3_precond_lag_months=1,     # PRIMARY: antecedent month m-1
    axis3_lags=(1, 0),              # all lags computed in ONE pass; [0] must be the primary
    axis3_indices=("spei3", "spi3"),
    # significance (section 12)
    n_null_permutations=1000,       # QUICK TEST: set 100-200 first
    null_mode="psu_window",         # "psu_window" (strict, PRIMARY) | "region_pool" (legacy)
    null_window_days=15,            # +/- calendar days around the observed onset (psu_window)
    null_seasons=("DJF", "MAM", "JJA", "SON"),   # onset-season decomposition of the null
    null_doy_pool="region",         # legacy pool label (region_pool mode only)
    n_lmf_surrogates=1000,
    rng_seed=42,
    # trends (section 14)
    n_bootstrap=1000, block_length_years=5, fdr_alpha=0.05,
    # figures (section 17)
    regime_quantile=0.75,           # Fig 6 PRIMARY: prominent = above top quartile
    regime_quantile_sens=2/3,     # Fig 6 sensitivity: top tercile
    regime_threshold_reference="pooled",   # decade panels: "pooled" thresholds (emergence) | "per_decade"
    spei_threshold_sens=(-0.5, -1.0, -1.5, -2.0),
    # processing
    psu_block_size=3000,
    # soft expectations (warn only); set None to silence
    expected_n_psu=54615, expected_n_countries=34,
)
cfg.ref_years = list(range(cfg.ref_start_year, cfg.ref_end_year + 1))
cfg.study_years = list(range(cfg.study_start_year, cfg.study_end_year + 1))

# paths namespace consumed by the step functions (identical to modular version)
cfg.paths = SimpleNamespace(
    raw_psu_rdata=RAW_PSU_RDATA, era5_nc=ERA5_NC, chirps_nc=CHIRPS_NC,
    processed=PROCESSED, climate_daily=PROCESSED / "climate_daily",
)
# one output tree per threshold mode; shared inputs (psu, climate_daily, spei3) stay at the root
MODE_DIRS = {m: PROCESSED / m for m in cfg.hw_threshold_modes}
FIG_DIRS = {m: MODE_DIRS[m] / "figures" for m in cfg.hw_threshold_modes}
for _d in list(MODE_DIRS.values()) + list(FIG_DIRS.values()):
    _d.mkdir(parents=True, exist_ok=True)


def set_mode(mode: str) -> Path:
    """Select the Tmax-threshold mode for the per-mode steps; returns its output dir."""
    assert mode in MODE_DIRS, mode
    cfg.hw_threshold_mode = mode
    return MODE_DIRS[mode]

print("Base dir:", BASE_DIR)
print("Mode dirs:", {m: str(d) for m, d in MODE_DIRS.items()})


# %% [markdown]
# ## Core statistical functions
#
# Shared building blocks (run detection, Zhang/LOO percentiles, Theil–Sen, Hamed–Rao MK, moving-block bootstrap, BH-FDR, Event Coincidence Analysis). Identical to the tested `core.py`.

# %%
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


EPOCH = np.datetime64("1970-01-01")


# =============================================================================
# Small utilities
# =============================================================================
def to_daynum(dates) -> np.ndarray:
    """Datetime -> integer days since epoch (vectorised, tz-naive)."""
    return (np.asarray(dates, dtype="datetime64[D]") - EPOCH).astype("int64")


def safe_ratio(numer, denom) -> np.ndarray:
    """Element-wise numer/denom; NaN where denom <= 0 (pairwise deletion)."""
    numer = np.asarray(numer, dtype="float64")
    denom = np.asarray(denom, dtype="float64")
    out = np.full(numer.shape, np.nan)
    m = denom > 0
    out[m] = numer[m] / denom[m]
    return out


def assign_decade(year: pd.Series, bounds) -> pd.Series:
    edges = [b[0] for b in bounds] + [bounds[-1][1] + 1]
    labels = [f"{a}-{b}" for a, b in bounds]
    return pd.cut(year, bins=edges, labels=labels, right=False, ordered=True)


def read_year_partitions(climate_dir: Path, years, columns, psu_set=None
                         ) -> pd.DataFrame:
    """Read selected years/columns of the partitioned daily dataset,
    optionally filtered to a set of psu_idx (pyarrow predicate pushdown)."""
    parts = []
    for yr in years:
        d = climate_dir / f"year={yr}"
        if not d.exists():
            raise FileNotFoundError(f"Missing partition: {d}")
        kw = {}
        if psu_set is not None:
            kw["filters"] = [("psu_idx", "in", set(psu_set))]
        parts.append(pd.read_parquet(d, columns=columns, **kw))
    out = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    return out


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians,
                                 (np.asarray(lat1, float), np.asarray(lon1, float),
                                  np.asarray(lat2, float), np.asarray(lon2, float)))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# =============================================================================
# Run (heatwave) detection
# =============================================================================
def detect_runs(df: pd.DataFrame, flag_col: str, min_days: int,
                value_col: str = "tmax") -> pd.DataFrame:
    """Detect runs of >= min_days consecutive flagged days, per PSU.

    Rules (explicit assumptions):
      * df must contain psu_idx, date (datetime64), flag_col (bool),
        value_col (float). It is sorted internally.
      * A run breaks on: flag False, a calendar gap (> 1 day between
        consecutive rows of the same PSU, i.e. a MISSING day can never
        merge two runs), or a change of PSU.
      * A NaN value with flag True is not possible if the caller computed
        flag = value > threshold (NaN comparisons are False) — callers must
        not bypass this.

    Returns one row per event: psu_idx, start_date, end_date, duration,
    mean_<value>, peak_<value>.
    """
    df = df.sort_values(["psu_idx", "date"], kind="stable").reset_index(drop=True)
    flag = df[flag_col].to_numpy(dtype=bool)
    day_gap = df.groupby("psu_idx")["date"].diff().dt.days
    psu_change = (df["psu_idx"] != df["psu_idx"].shift()).to_numpy()
    is_break = (~flag) | (day_gap.to_numpy() > 1) | psu_change
    run_id = np.cumsum(is_break)

    sub = df.loc[flag, ["psu_idx", "date", value_col]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["psu_idx", "start_date", "end_date",
                                     "duration", f"mean_{value_col}",
                                     f"peak_{value_col}"])
    sub["run_id"] = run_id[flag]
    runs = sub.groupby("run_id").agg(
        psu_idx=("psu_idx", "first"),
        start_date=("date", "first"),
        end_date=("date", "last"),
        duration=("date", "size"),
        **{f"mean_{value_col}": (value_col, "mean"),
           f"peak_{value_col}": (value_col, "max")},
    )
    return (runs.loc[runs["duration"] >= min_days]
                .reset_index(drop=True))


# =============================================================================
# In-base percentile machinery (Zhang et al. 2005)
# =============================================================================
def calendar_percentiles_with_inbase(grid: np.ndarray, window_idx: list,
                                     q: float, correction: str,
                                     rng: np.random.Generator | None = None,
                                     zhang_n: int = 29):
    """Calendar-day percentiles with in-base correction.

    Parameters
    ----------
    grid : float32 array (n_psu, 366, n_ref_years), NaN where no data.
    window_idx : list of index arrays — the +/-w circular window per doy.
    q : percentile (e.g. 90).
    correction : "none" | "loo" | "zhang".
    rng, zhang_n : duplication-year sampling for the full Zhang procedure
        (each in-base year removed and replaced by one of the remaining
        years, repeated zhang_n times; thresholds averaged).

    Returns
    -------
    base : (n_psu, 366) float32 — all-years percentile (applies to
           out-of-base years).
    inbase : None if correction == "none", else
             (n_psu, 366, n_ref_years) float32 — the threshold that applies
             to in-base year y is inbase[:, :, y_slot] (computed without
             year y, optionally with Zhang duplication).
    """
    n_psu, n_doy, n_yr = grid.shape
    base = np.empty((n_psu, n_doy), dtype=np.float32)
    inbase = None
    if correction != "none":
        inbase = np.empty((n_psu, n_doy, n_yr), dtype=np.float32)

    for d in range(n_doy):
        win = grid[:, window_idx[d], :]                  # (psu, w, years)
        flat = win.reshape(n_psu, -1)
        # all-NaN rows (possible at doy edges of sparse PSUs) -> NaN, no warn
        with np.errstate(all="ignore"):
            valid_any = np.isfinite(flat).any(axis=1)
            base[:, d] = np.nan
            if valid_any.any():
                base[valid_any, d] = np.nanpercentile(
                    flat[valid_any], q, axis=1)

        if correction == "none":
            continue

        for y in range(n_yr):
            keep = np.ones(n_yr, dtype=bool)
            keep[y] = False
            sub = win[:, :, keep]                        # (psu, w, n_yr-1)
            if correction == "zhang":
                # Replace the removed year by each remaining year in turn,
                # average the resulting thresholds (Zhang et al. 2005).
                # zhang_n < n_yr-1 subsamples replacement years for speed.
                others = np.flatnonzero(keep)
                if rng is not None and zhang_n < len(others):
                    others = rng.choice(others, size=zhang_n, replace=False)
                acc = np.zeros(n_psu, dtype=np.float64)
                for rep in others:
                    aug = np.concatenate(
                        [sub, win[:, :, rep:rep + 1]], axis=2
                    ).reshape(n_psu, -1)
                    with np.errstate(all="ignore"):
                        acc += np.nanpercentile(aug, q, axis=1)
                inbase[:, d, y] = (acc / len(others)).astype(np.float32)
            else:                                        # "loo"
                flat_sub = sub.reshape(n_psu, -1)
                with np.errstate(all="ignore"):
                    v = np.isfinite(flat_sub).any(axis=1)
                    col = np.full(n_psu, np.nan, dtype=np.float32)
                    if v.any():
                        col[v] = np.nanpercentile(flat_sub[v], q, axis=1)
                inbase[:, d, y] = col
    return base, inbase


# =============================================================================
# Trend statistics
# =============================================================================
def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Median of pairwise slopes (Theil 1950; Sen 1968)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = np.isfinite(x) & np.isfinite(y)
    x, y = x[v], y[v]
    if len(x) < 2:
        return np.nan
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    m = dx != 0
    if not m.any():
        return np.nan
    return float(np.median((y[j] - y[i])[m] / dx[m]))


def mann_kendall_hamed_rao(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mann-Kendall test with Hamed & Rao (1998) variance correction.

    The series is detrended with the Sen slope before estimating the rank
    autocorrelations (as recommended by Hamed & Rao); only autocorrelations
    significant at the 5% level enter the correction factor. Falls back to
    the classical (tie-corrected) variance when n < 10 or the correction
    factor is < 1 (the correction can only inflate the variance).

    Returns (Z, two-sided p).
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    v = np.isfinite(x) & np.isfinite(y)
    x, y = x[v], y[v]
    n = len(y)
    if n < 4:
        return np.nan, np.nan

    i, j = np.triu_indices(n, k=1)
    s = int(np.sum(np.sign(y[j] - y[i])))

    _, counts = np.unique(y, return_counts=True)
    ties = counts[counts > 1]
    var_s = (n * (n - 1) * (2 * n + 5)
             - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0
    if var_s <= 0:
        return 0.0, 1.0

    # Hamed-Rao correction on ranks of the detrended series
    if n >= 10:
        slope = theil_sen_slope(x, y)
        resid = y - slope * x if np.isfinite(slope) else y.copy()
        r = stats.rankdata(resid)
        r = r - r.mean()
        denom = np.sum(r * r)
        corr = 0.0
        crit = stats.norm.ppf(0.975) / np.sqrt(n)        # 5% significance
        for lag in range(1, n - 2):
            rho = np.sum(r[:-lag] * r[lag:]) / denom if denom > 0 else 0.0
            if abs(rho) > crit:
                corr += ((n - lag) * (n - lag - 1) * (n - lag - 2)) * rho
        n_star_ratio = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * corr
        if n_star_ratio > 1.0:
            var_s *= n_star_ratio

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def moving_block_bootstrap_slope_ci(x: np.ndarray, y: np.ndarray,
                                    n_boot: int, block_len: int,
                                    rng: np.random.Generator,
                                    ci: float = 0.95) -> tuple[float, float, float]:
    """Moving-block bootstrap CI on the Theil-Sen slope.

    Resampling whole blocks of consecutive (x, y) pairs preserves the
    serial-dependence structure that an i.i.d. bootstrap destroys
    (Kunsch 1989). x values are kept attached to their y values so the
    slope remains interpretable.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    v = np.isfinite(x) & np.isfinite(y)
    x, y = x[v], y[v]
    n = len(x)
    if n < max(3, block_len):
        return np.nan, np.nan, np.nan
    slope_obs = theil_sen_slope(x, y)

    n_blocks = int(np.ceil(n / block_len))
    starts_max = n - block_len
    boot = np.empty(n_boot)
    offs = np.arange(block_len)
    for k in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = (starts[:, None] + offs[None, :]).ravel()[:n]
        boot[k] = theil_sen_slope(x[idx], y[idx])
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return slope_obs, np.nan, np.nan
    a = (1 - ci) / 2
    return slope_obs, float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def benjamini_hochberg(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """BH-FDR: boolean array of rejected hypotheses (NaNs never rejected)."""
    p = np.asarray(p, float)
    out = np.zeros(p.shape, dtype=bool)
    v = np.isfinite(p)
    pv = p[v]
    m = pv.size
    if m == 0:
        return out
    order = np.argsort(pv)
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = pv[order] <= thresh
    k = np.max(np.nonzero(below)[0]) + 1 if below.any() else 0
    rej = np.zeros(m, dtype=bool)
    rej[order[:k]] = True
    out[v] = rej
    return out


# =============================================================================
# Event Coincidence Analysis (Axis II core)
# =============================================================================
def encode_psu_day(psu_idx: np.ndarray, daynum: np.ndarray,
                   span: int = 1_000_000) -> np.ndarray:
    """Encode (psu, day) into a single sortable int64 key. `span` must
    exceed any day number gap so that windows can never cross PSUs.
    Day numbers since 1970 stay < 40000 for this study; 1e6 is safe."""
    return psu_idx.astype("int64") * span + daynum.astype("int64")


def eca_indicators(hw_psu: np.ndarray, hw_start: np.ndarray,
                   hw_end: np.ndarray, epe_keys_sorted: np.ndarray,
                   window: int, span: int = 1_000_000
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-heatwave binary precursor / trigger / during indicators.

    Parameters
    ----------
    hw_psu, hw_start, hw_end : arrays per heatwave (start/end as int
        day numbers, inclusive).
    epe_keys_sorted : sorted encoded (psu, day) keys of all EPE days.
    window : window length W in days.

    Returns
    -------
    precursor : True if >= 1 EPE in [start-W, start-1]   (same PSU)
    trigger   : True if >= 1 EPE in [end+1,  end+W]
    during    : True if >= 1 EPE in [start,  end]
    """
    def count_in(lo_day, hi_day):
        lo = encode_psu_day(hw_psu, lo_day, span)
        hi = encode_psu_day(hw_psu, hi_day, span)
        a = np.searchsorted(epe_keys_sorted, lo, side="left")
        b = np.searchsorted(epe_keys_sorted, hi, side="right")
        return b > a

    precursor = count_in(hw_start - window, hw_start - 1)
    trigger = count_in(hw_end + 1, hw_end + window)
    during = count_in(hw_start, hw_end)
    return precursor, trigger, during


# =============================================================================
# Unique EPE -> heatwave attribution (Axis II PRIMARY metric)
# Shared by the observed data (step06) and the surrogate null (step07) so that
# both go through IDENTICAL logic.
# =============================================================================
SEASON_OF_MONTH = np.array(["DJF", "DJF", "MAM", "MAM", "MAM", "JJA", "JJA",
                            "JJA", "SON", "SON", "SON", "DJF"])


def season_of_month(month) -> np.ndarray:
    return SEASON_OF_MONTH[np.asarray(month, dtype=int) - 1]


def attribute_unique(hw_psu, hw_start, hw_end, epe_psu, epe_day, window: int,
                     span: int = 1_000_000):
    """Assign every EPE day to at most ONE heatwave of the same PSU: the
    temporally closest one within `window` days (distance 0 if the EPE falls
    inside the heatwave interval). Ties -> earlier heatwave. Fully vectorised
    (searchsorted); heatwaves MAY overlap (surrogates in the null):
        after  : heatwave with the latest END strictly before the EPE;
        before : heatwave with the earliest START strictly after the EPE;
        during : any interval containing the EPE (prefix-max of END along the
                 start-sorted order, per PSU).

    Returns
    -------
    hw_idx   : int64, index into the hw arrays (-1 = unattributed)
    position : int8, 0 none | 1 before | 2 during | 3 after
    offset   : int32, signed days (before < 0, during 0, after > 0)
    """
    hw_psu = np.asarray(hw_psu, np.int64)
    hw_start = np.asarray(hw_start, np.int64)
    hw_end = np.asarray(hw_end, np.int64)
    epe_psu = np.asarray(epe_psu, np.int64)
    epe_day = np.asarray(epe_day, np.int64)
    n_hw, n_e = len(hw_psu), len(epe_psu)
    hw_idx = np.full(n_e, -1, np.int64)
    pos = np.zeros(n_e, np.int8)
    off = np.zeros(n_e, np.int32)
    if n_hw == 0 or n_e == 0:
        return hw_idx, pos, off
    BIG = np.iinfo(np.int64).max // 4
    ekey = epe_psu * span + epe_day

    os_ = np.lexsort((hw_start, hw_psu))              # start-sorted order
    skey = hw_psu[os_] * span + hw_start[os_]
    oe_ = np.lexsort((hw_end, hw_psu))                # end-sorted order
    ekey_hw = hw_psu[oe_] * span + hw_end[oe_]

    # BEFORE candidate: first heatwave with start > d in the same PSU
    j = np.searchsorted(skey, ekey, side="right")
    jc = np.minimum(j, n_hw - 1)
    has_b = (j < n_hw) & (hw_psu[os_[jc]] == epe_psu)
    ib = os_[jc]
    d_b = np.where(has_b, hw_start[ib] - epe_day, BIG)

    # AFTER candidate: heatwave with the latest end < d in the same PSU
    k = np.searchsorted(ekey_hw, ekey, side="left") - 1
    kc = np.maximum(k, 0)
    has_a = (k >= 0) & (hw_psu[oe_[kc]] == epe_psu)
    ia = oe_[kc]
    d_a = np.where(has_a, epe_day - hw_end[ia], BIG)

    # DURING: running max of end along start order (PSU-safe via the key trick)
    psu_s, end_s = hw_psu[os_], hw_end[os_]
    run = np.maximum.accumulate(psu_s * span + end_s)
    run_end = run - psu_s * span
    carrier = np.maximum.accumulate(
        np.where((psu_s * span + end_s) == run, np.arange(n_hw), 0))
    jp = np.maximum(j - 1, 0)
    has_p = (j - 1 >= 0) & (psu_s[jp] == epe_psu)
    dur = has_p & (run_end[jp] >= epe_day)
    idur = os_[carrier[jp]]

    use_b = has_b & (d_b <= window) & (d_b < d_a)     # tie -> earlier (after) HW
    use_a = has_a & (d_a <= window) & ~use_b
    pos[use_b] = 1; hw_idx[use_b] = ib[use_b]; off[use_b] = -d_b[use_b]
    pos[use_a] = 3; hw_idx[use_a] = ia[use_a]; off[use_a] = d_a[use_a]
    pos[dur] = 2;   hw_idx[dur] = idur[dur];   off[dur] = 0
    return hw_idx, pos, off


def before_after_counts(pos: np.ndarray, hw_idx: np.ndarray, hw_mask: np.ndarray):
    """(n_before, n_after) among EPE attributed to heatwaves selected by hw_mask."""
    sel = hw_idx >= 0
    m = np.zeros(len(pos), bool)
    m[sel] = hw_mask[hw_idx[sel]]
    return int(np.sum(m & (pos == 1))), int(np.sum(m & (pos == 3)))



# %% [markdown]
# ## 3. PSU preparation
#
# step01_prepare_psu.py — Prepare DHS PSU coordinates.
#
# Loads the raw RData, removes invalid coordinates, deduplicates, assigns
# UN M49 sub-regions, and writes psu.parquet.
#
# CHANGES vs the audited version
# ------------------------------
# * Exact-coordinate deduplication is kept, but a NEAR-duplicate diagnostic
#   is added: DHS coordinates are randomly displaced (<= 2 km urban,
#   <= 5 km rural, 1% of rural <= 10 km; Burgert et al. 2013), so the same
#   locality sampled in two surveys will NOT share exact coordinates. We
#   cannot undo the jitter, so near-duplicates are *retained* (dropping them
#   would require an arbitrary radius); instead we (a) report their
#   prevalence, and (b) step02 stores the ERA5 / CHIRPS grid-cell ids of
#   every PSU so that all analyses can be replicated on unique grid cells
#   (the spatial-pseudo-replication sensitivity, step08/step10).
# * Hard-coded dataset-shape asserts are now soft checks (warn, don't crash)
#   driven by config.expected_n_psu / expected_n_countries.
#
# FRAMING ASSUMPTION (must be stated in the Methods):
#   The PSU set is a population-weighted sample of inhabited places, not an
#   areal sample of sub-Saharan Africa. All "continental"/regional statistics
#   therefore describe climate exposure of DHS-sampled populations.

# %%
import numpy as np
import pandas as pd


# UN M49 sub-regions (unstats.un.org/unsd/methodology/m49).
# "Central" = M49 "Middle Africa".
REGIONS_M49 = {
    "Western":  ["Benin", "Burkina Faso", "Cote d'Ivoire", "Ghana", "Guinea",
                 "Liberia", "Mali", "Niger", "Nigeria", "Senegal",
                 "Sierra Leone", "Togo"],
    "Central":  ["Angola", "Cameroon", "Central African Republic", "Chad",
                 "Congo Democratic Republic", "Gabon"],
    "Eastern":  ["Burundi", "Comoros", "Ethiopia", "Kenya", "Madagascar",
                 "Malawi", "Mozambique", "Rwanda", "Tanzania", "Uganda",
                 "Zambia", "Zimbabwe"],
    "Southern": ["Eswatini", "Lesotho", "Namibia", "South Africa"],
}
COUNTRY_TO_REGION = {c: r for r, cs in REGIONS_M49.items() for c in cs}


def run_step01_prepare_psu(cfg) -> pd.DataFrame:
    import pyreadr                                    # local import: heavy dep

    paths = cfg.paths
    paths.processed.mkdir(parents=True, exist_ok=True)

    log.info("Reading %s", paths.raw_psu_rdata)
    df = pyreadr.read_r(str(paths.raw_psu_rdata))["psu"]
    log.info("Loaded %s raw records", f"{len(df):,}")

    # ---- type coercion (R stores SurveyYear as text, PSU as float) -------
    df["SurveyYear"] = df["SurveyYear"].astype(int)
    df["PSU"] = df["PSU"].astype(int)

    # ---- coordinate validity ---------------------------------------------
    valid = (
        df["LATNUM"].notna() & df["LONGNUM"].notna()
        & ~((df["LATNUM"].abs() < 0.01) & (df["LONGNUM"].abs() < 0.01))  # (0,0)
        & df["LATNUM"].between(-35, 25)
        & df["LONGNUM"].between(-18, 52)
    )
    df = df.loc[valid]
    log.info("After coordinate filter: %s", f"{len(df):,}")

    # ---- exact deduplication (earliest survey wins, stable order) --------
    df = (df.sort_values(["SurveyYear", "SurveyId", "PSU"], kind="stable")
            .drop_duplicates(subset=["LATNUM", "LONGNUM"], keep="first"))
    log.info("After exact-coordinate dedup: %s", f"{len(df):,}")

    # ---- region assignment ------------------------------------------------
    df["region"] = df["CountryName"].map(COUNTRY_TO_REGION)
    unmapped = df.loc[df["region"].isna(), "CountryName"].unique()
    if len(unmapped):
        raise ValueError(f"Unmapped countries (extend REGIONS_M49): {unmapped}")

    # ---- final schema -------------------------------------------------------
    df = (df.rename(columns={"CountryName": "country", "LATNUM": "lat",
                             "LONGNUM": "lon", "SurveyId": "survey_id",
                             "SurveyYear": "survey_year"})
            .sort_values(["country", "survey_year", "survey_id"], kind="stable")
            .reset_index(drop=True))
    df.insert(0, "psu_idx", np.arange(len(df), dtype=np.int32))
    df["lat"] = df["lat"].astype("float32")
    df["lon"] = df["lon"].astype("float32")
    df = df[["psu_idx", "country", "region", "lat", "lon",
             "survey_id", "survey_year"]]

    # ---- soft shape checks --------------------------------------------------
    if cfg.expected_n_psu and len(df) != cfg.expected_n_psu:
        log.warning("PSU count %s differs from expected %s — Methods numbers "
                    "must be updated.", f"{len(df):,}", f"{cfg.expected_n_psu:,}")
    if cfg.expected_n_countries and df["country"].nunique() != cfg.expected_n_countries:
        log.warning("Country count %d differs from expected %d.",
                    df["country"].nunique(), cfg.expected_n_countries)
    assert df["psu_idx"].is_unique

    # ---- near-duplicate diagnostic (jitter-aware, report only) -------------
    # Count PSU pairs within 10 km of each other from *different* surveys —
    # an upper bound on jitter-induced repeated localities. Grid-bin pass
    # keeps it O(n) instead of O(n^2).
    cell = 0.1                                          # ~11 km bins
    df["_gx"] = np.floor(df["lon"] / cell).astype(int)
    df["_gy"] = np.floor(df["lat"] / cell).astype(int)
    n_near = 0
    for (_, _), g in df.groupby(["_gx", "_gy"]):
        if len(g) < 2:
            continue
        lat = g["lat"].to_numpy(); lon = g["lon"].to_numpy()
        sid = g["survey_id"].to_numpy()
        i, j = np.triu_indices(len(g), k=1)
        d = haversine_km(lat[i], lon[i], lat[j], lon[j])
        n_near += int(np.sum((d <= 10.0) & (sid[i] != sid[j])))
    df = df.drop(columns=["_gx", "_gy"])
    log.info("Near-duplicate diagnostic: %s cross-survey pairs within 10 km "
             "(same-bin approximation, lower bound). These are retained; the "
             "unique-grid-cell sensitivity (step08) controls for them.",
             f"{n_near:,}")

    out = paths.processed / "psu.parquet"
    df.to_parquet(out, index=False)
    log.info("Wrote %s | %s PSU | %d countries | regions: %s",
             out, f"{len(df):,}", df["country"].nunique(),
             df.groupby("region").size().to_dict())
    return df

psu_df = run_step01_prepare_psu(cfg)
psu_df.head()

# %%
# ---- checks: section 3 -----------------------------------------------------
psu_df = pd.read_parquet(PROCESSED / "psu.parquet")
assert psu_df["psu_idx"].is_unique and (psu_df["psu_idx"].values
        == np.arange(len(psu_df))).all()
assert psu_df[["lat", "lon"]].notna().all().all()
print(f"PSU: {len(psu_df):,} | countries: {psu_df.country.nunique()} | "
      f"regions: {psu_df.groupby('region').size().to_dict()}")
print(psu_df.groupby('country').size().sort_values(ascending=False).head(8))


# %%
# ---- PATCH: rebuild CHIRPS eligibility for a 0.25-deg product ---------------
# Half-diagonal of a 0.25-deg cell at the equator is ~19.6 km; 20 km keeps
# every PSU matched to its own cell while still rejecting genuine outliers.
cfg.max_match_km_chirps = 20.0     # document this value in the Methods
gm = pd.read_parquet(PROCESSED / "psu_gridmatch.parquet")
gm["chirps_too_far"] = gm["chirps_dist_km"] > cfg.max_match_km_chirps
gm["chirps_valid"] = ~(gm["chirps_too_far"] | gm["chirps_all_nan"])
gm.to_parquet(PROCESSED / "psu_gridmatch.parquet", index=False)
print("CHIRPS too far:", int(gm.chirps_too_far.sum()),
      "| all-NaN:", int(gm.chirps_all_nan.sum()),
      "| CHIRPS-eligible:", int(gm.chirps_valid.sum()), "/", len(gm))
# expected: eligible ~53,800 (54,615 - 813 ocean/no-data - a handful too far)

# %% [markdown]
# ## 4. Climate extraction (ERA5 + CHIRPS → daily parquet)
#
# step02_extract_climate.py — Extract daily Tmax/Tmin (ERA5) and precip
# (CHIRPS) at every PSU, 1985-2024, into a year-partitioned parquet dataset.
#
# CHANGES vs the audited version
# ------------------------------
# * Kelvin detection no longer relies on a single (possibly NaN/ocean) cell:
#   it samples a spatial subset and asserts a clear decision.
# * Nearest-neighbour matching now records the PSU -> grid-cell distance and
#   the grid-cell ids for BOTH products. PSU matched farther than
#   cfg.max_match_km_* from the nearest cell, or matched to a CHIRPS cell
#   that is all-NaN over the reference period (ocean / no-data), are FLAGGED
#   in psu_gridmatch.parquet. Flags propagate: step04 excludes chirps-invalid
#   PSU from EPE detection, step05 excludes them from SPEI, step08 restricts
#   denominators to eligible PSU.
# * Grid-cell ids enable the unique-grid-cell sensitivity (pseudo-replication
#   control) without re-extraction.
# * Time alignment between ERA5 and CHIRPS is done on calendar DATES
#   (datetime64[D]), not raw timestamps, so a 00:00 vs 12:00 stamp convention
#   cannot silently empty the intersection. The number of dropped days per
#   product is reported.
#
# ASSUMPTION (state in Methods): ERA5 daily aggregates follow UTC days while
# CHIRPS daily fields follow gauge-reporting days; a residual +/-1-day
# mismatch is possible and is addressed by the Axis-I day-tolerance
# sensitivity (cfg.axis1_day_tolerance).
#
# **Cost**: the heaviest IO step (hours at 54,615 PSUs). Safe to interrupt and restart (it rebuilds from scratch).

# %%
import gc
import shutil
import time

import numpy as np
import pandas as pd
import xarray as xr



def _open_and_align(cfg):
    era5 = xr.open_dataset(cfg.paths.era5_nc)
    rename = {k: v for k, v in
              {"valid_time": "time", "latitude": "lat", "longitude": "lon"}.items()
              if k in era5.dims or k in era5.coords}
    era5 = era5.rename(rename)

    chirps = xr.open_dataset(cfg.paths.chirps_nc)
    rename = {k: v for k, v in
              {"latitude": "lat", "longitude": "lon"}.items()
              if k in chirps.dims or k in chirps.coords}
    chirps = chirps.rename(rename)

    start, end = f"{cfg.study_start_year}-01-01", f"{cfg.study_end_year}-12-31"
    era5 = era5.sel(time=slice(start, end))
    chirps = chirps.sel(time=slice(start, end))

    # Align on calendar dates, robust to time-of-day stamp conventions.
    e_dates = era5["time"].values.astype("datetime64[D]")
    c_dates = chirps["time"].values.astype("datetime64[D]")
    common = np.intersect1d(e_dates, c_dates)
    if common.size == 0:
        raise RuntimeError("ERA5/CHIRPS date intersection is empty — check "
                           "the time axes of both files.")
    log.info("Common dates: %s..%s (%s days) | dropped: ERA5 %d, CHIRPS %d",
             common[0], common[-1], f"{common.size:,}",
             e_dates.size - common.size, c_dates.size - common.size)
    era5 = era5.isel(time=np.isin(e_dates, common))
    chirps = chirps.isel(time=np.isin(c_dates, common))
    return era5, chirps, common


def _detect_kelvin(era5) -> bool:
    """Robust unit detection: sample many cells, require a clear verdict."""
    t = era5["tmax"].isel(time=0)
    sample = t.values[::max(1, t.shape[0] // 20), ::max(1, t.shape[1] // 20)]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise RuntimeError("Could not detect ERA5 units: first time slice "
                           "is all-NaN on the sample grid.")
    frac_kelvin = float(np.mean(sample > 150.0))
    if frac_kelvin > 0.99:
        log.info("ERA5 Tmax detected in Kelvin (median sample %.1f).",
                 float(np.median(sample)))
        return True
    if frac_kelvin < 0.01:
        log.info("ERA5 Tmax detected in Celsius (median sample %.1f).",
                 float(np.median(sample)))
        return False
    raise RuntimeError(f"Ambiguous ERA5 units: {frac_kelvin:.0%} of sampled "
                       "cells look like Kelvin. Inspect the file.")


def _nearest_match(ds, lats, lons, max_km: float):
    """Nearest grid cell per PSU + distance + integer cell id."""
    glat = ds["lat"].values
    glon = ds["lon"].values
    ilat = np.abs(glat[None, :] - lats[:, None]).argmin(axis=1)
    ilon = np.abs(glon[None, :] - lons[:, None]).argmin(axis=1)
    dist = haversine_km(lats, lons, glat[ilat], glon[ilon])
    cell_id = (ilat.astype("int64") * len(glon) + ilon).astype("int64")
    too_far = dist > max_km
    return ilat, ilon, dist.astype("float32"), cell_id, too_far


def run_step02_extract_climate(cfg) -> None:
    t0 = time.time()
    paths = cfg.paths
    psu = pd.read_parquet(paths.processed / "psu.parquet")
    n_psu = len(psu)
    log.info("%s PSU", f"{n_psu:,}")

    era5, chirps, common = _open_and_align(cfg)
    n_days = common.size
    years_of_days = pd.DatetimeIndex(common).year.values.astype("int16")
    is_kelvin = _detect_kelvin(era5)

    # ---- one-shot grid matching for ALL PSU (cheap, index arithmetic) ----
    lats = psu["lat"].to_numpy(float)
    lons = psu["lon"].to_numpy(float)
    e_ilat, e_ilon, e_dist, e_cell, e_far = _nearest_match(
        era5, lats, lons, cfg.max_match_km_era5)
    c_ilat, c_ilon, c_dist, c_cell, c_far = _nearest_match(
        chirps, lats, lons, cfg.max_match_km_chirps)

    match = psu[["psu_idx"]].copy()
    match["era5_cell"] = e_cell
    match["era5_dist_km"] = e_dist
    match["era5_too_far"] = e_far
    match["chirps_cell"] = c_cell
    match["chirps_dist_km"] = c_dist
    match["chirps_too_far"] = c_far

    if cfg.paths.climate_daily.exists():
        shutil.rmtree(cfg.paths.climate_daily)
    cfg.paths.climate_daily.mkdir(parents=True, exist_ok=True)

    # ---- batched extraction -------------------------------------------------
    bs = cfg.extraction_batch_size
    n_batches = (n_psu + bs - 1) // bs
    chirps_all_nan = np.zeros(n_psu, dtype=bool)
    log.info("Extracting in %d batches of %d ...", n_batches, bs)

    for b in range(n_batches):
        lo, hi = b * bs, min((b + 1) * bs, n_psu)
        nb = hi - lo
        sel_e = dict(lat=xr.DataArray(era5["lat"].values[e_ilat[lo:hi]], dims="psu"),
                     lon=xr.DataArray(era5["lon"].values[e_ilon[lo:hi]], dims="psu"))
        sel_c = dict(lat=xr.DataArray(chirps["lat"].values[c_ilat[lo:hi]], dims="psu"),
                     lon=xr.DataArray(chirps["lon"].values[c_ilon[lo:hi]], dims="psu"))

        e = era5.sel(**sel_e, method="nearest")
        tmax = e["tmax"].load().values            # (time, psu)
        tmin = e["tmin"].load().values
        del e
        c = chirps.sel(**sel_c, method="nearest")
        precip = c["precip"].load().values
        del c

        if is_kelvin:
            tmax = tmax - 273.15
            tmin = tmin - 273.15

        # Flag PSU whose CHIRPS series is entirely NaN (ocean / no-data).
        chirps_all_nan[lo:hi] = ~np.isfinite(precip).any(axis=0)

        df_batch = pd.DataFrame({
            "psu_idx": np.repeat(psu["psu_idx"].values[lo:hi], n_days).astype("int32"),
            "date":    np.tile(common.astype("datetime64[ns]"), nb),
            "tmax":    tmax.T.reshape(-1).astype("float32"),
            "tmin":    tmin.T.reshape(-1).astype("float32"),
            "precip":  precip.T.reshape(-1).astype("float32"),
            "year":    np.tile(years_of_days, nb),
        })
        df_batch.to_parquet(cfg.paths.climate_daily, partition_cols=["year"],
                            index=False,
                            basename_template=f"batch{b:03d}-{{i}}.parquet")
        del df_batch, tmax, tmin, precip
        gc.collect()
        log.info("  batch %d/%d (%s PSU) | %.1f min",
                 b + 1, n_batches, f"{hi:,}", (time.time() - t0) / 60)

    era5.close(); chirps.close()

    # ---- eligibility flags ---------------------------------------------------
    match["chirps_all_nan"] = chirps_all_nan
    match["chirps_valid"] = ~(match["chirps_too_far"] | match["chirps_all_nan"])
    match["era5_valid"] = ~match["era5_too_far"]
    match.to_parquet(cfg.paths.processed / "psu_gridmatch.parquet", index=False)

    log.info("Grid-match summary: ERA5 too far: %d | CHIRPS too far: %d | "
             "CHIRPS all-NaN: %d | CHIRPS-eligible PSU: %s / %s",
             int(match["era5_too_far"].sum()),
             int(match["chirps_too_far"].sum()),
             int(match["chirps_all_nan"].sum()),
             f"{int(match['chirps_valid'].sum()):,}", f"{n_psu:,}")
    log.info("Done in %.1f min.", (time.time() - t0) / 60)

run_step02_extract_climate(cfg)

# %%
# ---- DIAGNOSTIC: CHIRPS grid resolution & match distances -------------------
import xarray as xr
ch = xr.open_dataset(CHIRPS_NC)
lat_name = "lat" if "lat" in ch.coords else "latitude"
lon_name = "lon" if "lon" in ch.coords else "longitude"
dlat = float(np.median(np.diff(ch[lat_name].values)))
dlon = float(np.median(np.diff(ch[lon_name].values)))
print(f"CHIRPS grid spacing: dlat={dlat:.4f} deg, dlon={dlon:.4f} deg")
print(f"lon range: {float(ch[lon_name].min()):.2f} .. {float(ch[lon_name].max()):.2f}")
ch.close()

gm = pd.read_parquet(PROCESSED / "psu_gridmatch.parquet")
print(gm["chirps_dist_km"].describe().round(2))
# If spacing ~0.25 deg -> distances should be ~uniform 0-20 km (median ~10).
# If spacing ~0.05 deg but distances are LARGE (hundreds of km) -> coordinate
# convention problem (e.g. 0..360 longitudes) — stop and tell me.

# %%
# ---- checks: section 4 -----------------------------------------------------
gm = pd.read_parquet(PROCESSED / "psu_gridmatch.parquet")
print(gm[["era5_dist_km", "chirps_dist_km"]].describe().round(2))
print("ERA5 too far:", int(gm.era5_too_far.sum()),
      "| CHIRPS too far:", int(gm.chirps_too_far.sum()),
      "| CHIRPS all-NaN (ocean/no-data):", int(gm.chirps_all_nan.sum()),
      "| CHIRPS-eligible:", int(gm.chirps_valid.sum()), "/", len(gm))
# spot-check one year-partition
samp = pd.read_parquet(cfg.paths.climate_daily / "year=2000")
print(f"year=2000: {len(samp):,} rows | tmax NaN {samp.tmax.isna().mean():.2%}"
      f" | precip NaN {samp.precip.isna().mean():.2%}")
print("tmax range (should be Celsius):",
      round(float(samp.tmax.min()), 1), "..", round(float(samp.tmax.max()), 1))
assert samp.tmax.max() < 70, "tmax looks like Kelvin — unit detection failed"
del samp; gc.collect()


# %% [markdown]
# ## 5. Percentile thresholds with in-base correction
#
# step03_thresholds.py — Site-specific percentile thresholds (ETCCDI calendar
# method) WITH correction of the in-base / out-of-base inhomogeneity
# (Zhang et al. 2005, J. Climate).
#
# WHY THIS STEP CHANGED
# ---------------------
# The reference period (1985-2014) lies inside the analysis period
# (1985-2024). If a single threshold is computed from all 30 reference years
# and applied everywhere, days inside the base period are compared against
# thresholds computed *with themselves*, while days after 2014 are not.
# This depresses exceedance rates inside the base period relative to outside
# it and inflates every trend in the study. The fix: the threshold applied
# to an in-base year y is computed WITHOUT year y ("loo"), optionally with
# Zhang's duplication of a remaining year to preserve the 30-year sample
# size ("zhang"). Out-of-base years (2015-2024) keep the all-30-years
# threshold.
#
# OUTPUTS (under <out_dir>)
# -------------------------
# thresholds_tmax_base.parquet      psu_idx, doy, thr      (out-of-base years)
# thresholds_tmax_inbase/year=YYYY  psu_idx, doy, thr      (one per ref year;
#                                   only if correction != "none";
#                                   ~n_psu x 366 rows per year — disk-heavy,
#                                   see README)
# thresholds_precip.parquet         psu_idx, p95_base, n_wet_days,
#                                   p95_reliable [, p95_y1985..p95_y2014]
#                                   (in-base wet-day P95, leave-one-out)
#
# The same correction is applied to the precipitation threshold: the wet-day
# P95 used to detect EPE in an in-base year y is computed on wet days of the
# other 29 years.
#
# NOTE on day-of-year: dt.dayofyear shifts post-February days by one between
# leap and non-leap years; with the +/-15-day window this 1-day blur is
# negligible (documented in Methods/SI), so the simpler doy key is kept.
#
# **Cost warning**: `in_base_correction='loo'` ≈ 31× the uncorrected computation (hours at full scale) and the in-base store is ~20M rows × 30 years on disk. For a first end-to-end test you may set `cfg.in_base_correction='none'` above, then switch back for real results.

# %%
import gc
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


DAYS_IN_YEAR = 366


def _window_indices(half: int) -> list[np.ndarray]:
    doy = np.arange(DAYS_IN_YEAR)
    out = []
    for d in doy:
        diff = np.abs(doy - d)
        circ = np.minimum(diff, DAYS_IN_YEAR - diff)
        out.append(np.where(circ <= half)[0])
    return out


def annual_percentiles_with_inbase(grid: np.ndarray, q: float, correction: str):
    """hw_threshold_mode == "annual": ONE percentile per PSU over ALL days of
    the reference period (no calendar window), applied to every day of the
    year. Returned in the same (n_psu, 366[, n_yr]) layout as the calendar
    version so that step04 needs no change. LOO: percentile of the other
    n_yr-1 years. Zhang duplication is not defined for the pooled-annual
    percentile -> falls back to LOO with a warning."""
    n_psu, n_doy, n_yr = grid.shape

    def _pct(a):
        flat = a.reshape(n_psu, -1)
        out = np.full(n_psu, np.nan, dtype=np.float32)
        with np.errstate(all="ignore"):
            v = np.isfinite(flat).any(axis=1)
            if v.any():
                out[v] = np.nanpercentile(flat[v], q, axis=1)
        return out

    base = np.repeat(_pct(grid)[:, None], n_doy, axis=1).astype(np.float32)
    inbase = None
    if correction != "none":
        if correction == "zhang":
            log.warning("annual mode: Zhang duplication not defined for the "
                        "pooled-annual percentile -> leave-one-out used.")
        inbase = np.empty((n_psu, n_doy, n_yr), dtype=np.float32)
        for y in range(n_yr):
            keep = np.ones(n_yr, dtype=bool); keep[y] = False
            inbase[:, :, y] = _pct(grid[:, :, keep])[:, None]
    return base, inbase


def _tmax_thresholds(cfg, out_dir: Path, all_psu: np.ndarray,
                     rng: np.random.Generator) -> None:
    ref_years = cfg.ref_years
    n_years = len(ref_years)
    window_idx = _window_indices(cfg.hw_window_half)

    base_file = out_dir / "thresholds_tmax_base.parquet"
    inbase_dir = out_dir / "thresholds_tmax_inbase"
    if inbase_dir.exists():
        shutil.rmtree(inbase_dir)
    if cfg.in_base_correction != "none":
        inbase_dir.mkdir(parents=True, exist_ok=True)

    bs = cfg.psu_block_size
    n_blocks = (len(all_psu) + bs - 1) // bs
    base_parts = []
    t0 = time.time()
    log.info("Tmax %s P%g, correction=%s, %d blocks of %d PSU "
             "(correction != 'none' multiplies runtime ~x%d) ...",
             cfg.hw_threshold_mode, cfg.hw_percentile, cfg.in_base_correction,
             n_blocks, bs,
             1 if cfg.in_base_correction == "none" else n_years + 1)

    for b in range(n_blocks):
        psu_block = all_psu[b * bs: min((b + 1) * bs, len(all_psu))]
        block_n = len(psu_block)
        psu_pos = {p: i for i, p in enumerate(psu_block)}

        # grid[psu, doy, ref_year] of Tmax
        grid = np.full((block_n, DAYS_IN_YEAR, n_years), np.nan,
                       dtype=np.float32)
        for slot, yr in enumerate(ref_years):
            part = read_year_partitions(cfg.paths.climate_daily, [yr],
                                        ["psu_idx", "date", "tmax"],
                                        psu_set=psu_block)
            pos = part["psu_idx"].map(psu_pos).to_numpy()
            doy = part["date"].dt.dayofyear.to_numpy() - 1
            grid[pos, doy, slot] = part["tmax"].to_numpy(np.float32)
            del part

        if cfg.hw_threshold_mode == "annual":
            base, inbase = annual_percentiles_with_inbase(
                grid, cfg.hw_percentile, cfg.in_base_correction)
        else:
            base, inbase = calendar_percentiles_with_inbase(
                grid, window_idx, cfg.hw_percentile,
                cfg.in_base_correction, rng=rng,
                zhang_n=cfg.zhang_n_replicates)
        del grid
        gc.collect()

        doys = np.tile(np.arange(1, DAYS_IN_YEAR + 1), block_n).astype("int16")
        psus = np.repeat(psu_block, DAYS_IN_YEAR).astype("int32")
        base_parts.append(pd.DataFrame(
            {"psu_idx": psus, "doy": doys, "thr": base.reshape(-1)}))

        if inbase is not None:
            # Manual partition layout (year=YYYY/blockNNN.parquet) so that
            # successive blocks append instead of overwriting.
            for slot, yr in enumerate(ref_years):
                ydir = inbase_dir / f"year={yr}"
                ydir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {"psu_idx": psus, "doy": doys,
                     "thr": inbase[:, :, slot].reshape(-1)}
                ).to_parquet(ydir / f"block{b:03d}.parquet", index=False)
            del inbase
            gc.collect()

        log.info("  block %d/%d | %.1f min", b + 1, n_blocks,
                 (time.time() - t0) / 60)

    base_df = pd.concat(base_parts, ignore_index=True)
    n_nan = int(base_df["thr"].isna().sum())
    if n_nan:
        log.warning("Tmax base thresholds: %d NaN cells (PSU with no data "
                    "in the window) — those days can never be 'hot'.", n_nan)
    base_df.to_parquet(base_file, index=False)
    log.info("Wrote %s (%s rows)", base_file.name, f"{len(base_df):,}")


def _precip_thresholds(cfg, out_dir: Path, all_psu: np.ndarray,
                       chirps_valid: pd.Series) -> None:
    """Annual wet-day P95 (R95p), base + leave-one-out per in-base year.

    Only CHIRPS-eligible PSU receive thresholds; others get NaN and are
    excluded from EPE detection downstream.
    """
    ref_years = cfg.ref_years
    wet_parts = []
    for yr in ref_years:
        part = read_year_partitions(cfg.paths.climate_daily, [yr],
                                    ["psu_idx", "precip"])
        part = part.loc[part["precip"] >= cfg.wet_day_min_mm]
        part["year"] = yr
        wet_parts.append(part)
    wet = pd.concat(wet_parts, ignore_index=True)
    del wet_parts
    gc.collect()
    log.info("Wet-day rows over reference period: %s", f"{len(wet):,}")

    q = cfg.epe_percentile / 100.0
    p95_base = wet.groupby("psu_idx")["precip"].quantile(q).rename("p95_base")
    n_wet = wet.groupby("psu_idx").size().rename("n_wet_days")

    out = pd.DataFrame({"psu_idx": all_psu.astype("int32")})
    out = (out.merge(p95_base, on="psu_idx", how="left")
              .merge(n_wet, on="psu_idx", how="left"))
    out["n_wet_days"] = out["n_wet_days"].fillna(0).astype("int32")
    out["p95_reliable"] = ((out["n_wet_days"] >= cfg.min_wet_days_ref)
                           & out["psu_idx"].map(chirps_valid).fillna(False))

    if cfg.in_base_correction != "none":
        # Leave-one-out wet-day P95 per in-base year (full Zhang duplication
        # is second-order here too and is omitted for the pooled-annual
        # precip percentile; the "no_inbase" sensitivity quantifies it).
        for yr in ref_years:
            loo = (wet.loc[wet["year"] != yr]
                      .groupby("psu_idx")["precip"].quantile(q)
                      .rename(f"p95_y{yr}"))
            out = out.merge(loo, on="psu_idx", how="left")
    del wet
    gc.collect()

    out.to_parquet(out_dir / "thresholds_precip.parquet", index=False)
    log.info("Wrote thresholds_precip.parquet | reliable PSU: %s / %s",
             f"{int(out['p95_reliable'].sum()):,}", f"{len(out):,}")


def run_step03_thresholds(cfg, out_dir: Path | None = None) -> None:
    out_dir = Path(out_dir or cfg.paths.processed)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.rng_seed)

    psu = pd.read_parquet(cfg.paths.processed / "psu.parquet",
                          columns=["psu_idx"])
    all_psu = psu["psu_idx"].to_numpy()

    gm = pd.read_parquet(cfg.paths.processed / "psu_gridmatch.parquet",
                         columns=["psu_idx", "chirps_valid"])
    chirps_valid = gm.set_index("psu_idx")["chirps_valid"]

    _tmax_thresholds(cfg, out_dir, all_psu, rng)
    # precipitation thresholds do not depend on the Tmax mode: computed once,
    # cached at the shared root and copied into each mode directory.
    shared = cfg.paths.processed / "thresholds_precip.parquet"
    if not shared.exists():
        _precip_thresholds(cfg, cfg.paths.processed, all_psu, chirps_valid)
    if out_dir.resolve() != cfg.paths.processed.resolve():
        shutil.copy(shared, out_dir / "thresholds_precip.parquet")
        log.info("thresholds_precip.parquet copied from shared root")

for _mode in cfg.hw_threshold_modes:
    run_step03_thresholds(cfg, out_dir=set_mode(_mode))

# %%
for _mode, _D in MODE_DIRS.items():
    print(f"\n===== hw_threshold_mode = {_mode} =====")
    # ---- checks: section 5 -----------------------------------------------------
    thr = pd.read_parquet(_D / "thresholds_tmax_base.parquet")
    print(f"base thresholds: {len(thr):,} rows | NaN: {thr.thr.isna().mean():.3%}")
    print(thr.thr.describe().round(2))
    if cfg.in_base_correction != "none":
        y0 = cfg.ref_years[0]
        ib = pd.read_parquet(_D / f"thresholds_tmax_inbase/year={y0}")
        j = thr.merge(ib, on=["psu_idx", "doy"], suffixes=("_base", "_loo"))
        d = (j.thr_loo - j.thr_base).abs()
        print(f"LOO vs base (year {y0}): mean |diff| {d.mean():.3f} degC, "
              f"P99 {d.quantile(.99):.3f} degC (must be > 0: correction active)")
        assert d.mean() > 0
    pr = pd.read_parquet(_D / "thresholds_precip.parquet")
    print(f"precip thresholds: reliable PSU {int(pr.p95_reliable.sum()):,}"
          f" / {len(pr):,} | P95 median {pr.p95_base.median():.1f} mm")


# %% [markdown]
# ## 6–7. Heatwave detection + extreme-precipitation detection (single pass)
#
# step04_detect_events.py — Detect heatwaves and extreme precipitation events
# using the (in-base-corrected) thresholds from step 3, and accumulate the
# day-of-year contingency counts needed for the Axis-I dependence statistics
# (step 7).
#
# DEFINITIONS (explicit)
# ----------------------
# Heatwave : run of >= cfg.hw_min_days consecutive days with
#            Tmax > threshold(psu, doy[, year]) at the same PSU. A run breaks
#            on a sub-threshold day, a MISSING day (NaN Tmax — the comparison
#            is False, so a missing day can never bridge two runs), or a
#            calendar gap. Runs crossing 31 Dec are preserved (all years are
#            concatenated before run labelling).
# Hot day   : a day belonging to a qualifying (>= min_days) heatwave run —
#             this is the day-level flag used for co-occurrence statistics,
#             NOT mere single-day exceedance.
# EPE       : day with precip > wet-day-P95(psu[, year]) at a CHIRPS-eligible
#             PSU. One EPE = one day.
#
# In-base correction: within the reference period, the threshold applied to
# year y is the year-specific (leave-one-out / Zhang) threshold from step 3;
# outside it, the all-years base threshold.
#
# OUTPUTS (under <dirs>)
# ----------------------
# heatwaves.parquet  psu_idx, start_date, end_date, duration, mean_tmax,
#                    peak_tmax
# epe.parquet        psu_idx, date, precip
# doy_counts.parquet psu_idx, doy, n_valid, n_hwday, n_epe, n_joint,
#                    n_joint_tol1   (inputs to the LMF in step 7;
#                    n_joint_tol1 = hot day with an EPE on the same day or
#                    +/-1 day, for the ERA5/CHIRPS day-convention sensitivity)
#
# **Why one pass for both**: both detectors read the same daily blocks; a
# single pass halves the IO. Conceptually they remain two independent
# detectors (sections 6 and 7), and their checks are separate below.
#
# **Your question — calendar-day ±15 window for the EPE threshold too?**
# The heatwave threshold *must* be calendar-day-resolved (a fixed annual
# Tmax threshold would only ever fire in the hottest season). For
# precipitation the standard index (R95p, used here as primary) is the
# **annual wet-day P95**: a *calendar-day* wet-day P95 is statistically
# fragile in semi-arid SSA because many PSUs have very few wet days inside
# a ±15-day window outside the rainy season (the percentile would rest on a
# handful of values, or none). The seasonality concern that motivates a
# calendar threshold is instead handled **explicitly and more robustly**
# where it matters: the seasonally adjusted LMF (Axis I) and the surrogate
# onset-DOY null (Axis II) condition the *inference* on the seasonal cycle.
# If you still want a calendar-day EPE variant as a sensitivity, it can be
# added with the same machinery as the Tmax threshold — say so and I will
# wire it in; it is deliberately not silently included here.
#

# %%
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd



def _load_thresholds(cfg, in_dir: Path):
    base = pd.read_parquet(in_dir / "thresholds_tmax_base.parquet")
    base_map = base.set_index(["psu_idx", "doy"])["thr"]

    inbase_dir = in_dir / "thresholds_tmax_inbase"
    inbase_maps = {}
    if cfg.in_base_correction != "none":
        if not inbase_dir.exists():
            raise FileNotFoundError(
                "in_base_correction != 'none' but thresholds_tmax_inbase/ "
                "is missing — rerun step03 with the same configuration.")
        for yr in cfg.ref_years:
            df = pd.read_parquet(inbase_dir / f"year={yr}")
            inbase_maps[yr] = df.set_index(["psu_idx", "doy"])["thr"]

    pr = pd.read_parquet(in_dir / "thresholds_precip.parquet")
    return base_map, inbase_maps, pr


def _attach_tmax_threshold(block: pd.DataFrame, base_map, inbase_maps,
                           cfg) -> pd.DataFrame:
    block["doy"] = block["date"].dt.dayofyear.astype("int16")
    key = pd.MultiIndex.from_arrays([block["psu_idx"], block["doy"]])
    block["thr_tmax"] = key.map(base_map).astype("float32")
    if inbase_maps:
        yrs = block["date"].dt.year
        for yr, m in inbase_maps.items():
            sel = (yrs == yr).to_numpy()
            if sel.any():
                k = pd.MultiIndex.from_arrays(
                    [block.loc[sel, "psu_idx"], block.loc[sel, "doy"]])
                block.loc[sel, "thr_tmax"] = k.map(m).astype("float32")
    return block


def _attach_precip_threshold(block: pd.DataFrame, pr: pd.DataFrame,
                             cfg) -> pd.DataFrame:
    base = pr.set_index("psu_idx")["p95_base"]
    reliable = pr.set_index("psu_idx")["p95_reliable"]
    block["thr_pr"] = block["psu_idx"].map(base).astype("float32")
    block["pr_ok"] = block["psu_idx"].map(reliable).fillna(False).to_numpy(bool)
    if cfg.in_base_correction != "none":
        yrs = block["date"].dt.year
        for yr in cfg.ref_years:
            col = f"p95_y{yr}"
            if col not in pr.columns:
                continue
            m = pr.set_index("psu_idx")[col]
            sel = (yrs == yr).to_numpy()
            if sel.any():
                block.loc[sel, "thr_pr"] = (
                    block.loc[sel, "psu_idx"].map(m).astype("float32"))
    return block


def run_step04_detect_events(cfg, in_dir: Path | None = None,
         out_dir: Path | None = None) -> None:
    in_dir = Path(in_dir or cfg.paths.processed)
    out_dir = Path(out_dir or cfg.paths.processed)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    base_map, inbase_maps, pr = _load_thresholds(cfg, in_dir)
    all_psu = pr["psu_idx"].to_numpy()
    n_psu = len(all_psu)

    bs = cfg.psu_block_size
    n_blocks = (n_psu + bs - 1) // bs
    hw_parts, epe_parts, doy_parts = [], [], []
    log.info("Detecting events: %d blocks of %d PSU ...", n_blocks, bs)

    for b in range(n_blocks):
        psu_block = all_psu[b * bs: min((b + 1) * bs, n_psu)]
        block = read_year_partitions(
            cfg.paths.climate_daily, cfg.study_years,
            ["psu_idx", "date", "tmax", "precip"], psu_set=psu_block)
        block = _attach_tmax_threshold(block, base_map, inbase_maps, cfg)
        block = _attach_precip_threshold(block, pr, cfg)
        block = (block.sort_values(["psu_idx", "date"], kind="stable")
                      .reset_index(drop=True))

        # ---- heatwaves ----------------------------------------------------
        # NaN tmax or NaN threshold -> comparison False -> breaks the run.
        block["hot_exc"] = block["tmax"] > block["thr_tmax"]
        hw = detect_runs(block, "hot_exc", cfg.hw_min_days, value_col="tmax")
        hw_parts.append(hw)

        # Day-level flag: day belongs to a QUALIFYING run. Reconstructed by
        # interval membership (cheap: events are disjoint per PSU).
        block["hwday"] = False
        if len(hw):
            ev = hw[["psu_idx", "start_date", "end_date"]].copy()
            ev["d"] = [pd.date_range(s, e) for s, e in
                       zip(ev["start_date"], ev["end_date"])]
            days = ev.explode("d")[["psu_idx", "d"]].rename(columns={"d": "date"})
            days["hwday"] = True
            block = block.merge(days, on=["psu_idx", "date"], how="left",
                                suffixes=("", "_ev"))
            block["hwday"] = block["hwday_ev"].fillna(False).to_numpy(bool) \
                if "hwday_ev" in block.columns else block["hwday"]
            block = block.drop(columns=[c for c in ("hwday_ev",)
                                        if c in block.columns])

        # ---- EPE ------------------------------------------------------------
        block["epeday"] = (block["precip"] > block["thr_pr"]) & block["pr_ok"]
        epe_parts.append(
            block.loc[block["epeday"], ["psu_idx", "date", "precip"]].copy())

        # ---- doy contingency counts (LMF inputs) ----------------------------
        # +/-1-day EPE neighbourhood within PSU. Rows are contiguous daily per
        # PSU by construction of step02, so a row shift is a day shift; the
        # first/last row of each PSU is guarded by the psu-equality test.
        same_psu_prev = block["psu_idx"].eq(block["psu_idx"].shift(1))
        same_psu_next = block["psu_idx"].eq(block["psu_idx"].shift(-1))
        epe_pm1 = (block["epeday"]
                   | (block["epeday"].shift(1).fillna(False) & same_psu_prev)
                   | (block["epeday"].shift(-1).fillna(False) & same_psu_next))
        block["n_valid"] = np.isfinite(block["tmax"]).astype("int32")
        block["n_joint"] = (block["hwday"] & block["epeday"]).astype("int32")
        block["n_joint_tol1"] = (block["hwday"] & epe_pm1).astype("int32")
        dc = (block.groupby(["psu_idx", "doy"], observed=True)
                   .agg(n_valid=("n_valid", "sum"),
                        n_hwday=("hwday", "sum"),
                        n_epe=("epeday", "sum"),
                        n_joint=("n_joint", "sum"),
                        n_joint_tol1=("n_joint_tol1", "sum"))
                   .reset_index())
        doy_parts.append(dc)

        del block, hw
        gc.collect()
        log.info("  block %d/%d | %.1f min", b + 1, n_blocks,
                 (time.time() - t0) / 60)

    heatwaves = pd.concat(hw_parts, ignore_index=True)
    epe = pd.concat(epe_parts, ignore_index=True)
    doy_counts = pd.concat(doy_parts, ignore_index=True)
    del hw_parts, epe_parts, doy_parts
    gc.collect()

    heatwaves["duration"] = heatwaves["duration"].astype("int16")
    for c in ("mean_tmax", "peak_tmax"):
        heatwaves[c] = heatwaves[c].astype("float32")
    epe["precip"] = epe["precip"].astype("float32")

    # ---- sanity checks ---------------------------------------------------
    assert (heatwaves["duration"] >= cfg.hw_min_days).all()
    assert (heatwaves["end_date"] >= heatwaves["start_date"]).all()
    assert (heatwaves["peak_tmax"] >= heatwaves["mean_tmax"] - 1e-4).all()
    assert (doy_counts["n_joint"] <= doy_counts["n_hwday"]).all()
    assert (doy_counts["n_joint"] <= doy_counts["n_epe"]).all()

    heatwaves.to_parquet(out_dir / "heatwaves.parquet", index=False)
    epe.to_parquet(out_dir / "epe.parquet", index=False)
    doy_counts.to_parquet(out_dir / "doy_counts.parquet", index=False)

    log.info("Heatwaves: %s (mean duration %.1f d, max %d d) | EPE days: %s "
             "| done in %.1f min",
             f"{len(heatwaves):,}", heatwaves["duration"].mean(),
             int(heatwaves["duration"].max()) if len(heatwaves) else 0,
             f"{len(epe):,}", (time.time() - t0) / 60)

for _mode in cfg.hw_threshold_modes:
    _D = set_mode(_mode)
    run_step04_detect_events(cfg, in_dir=_D, out_dir=_D)

# %%
for _mode, _D in MODE_DIRS.items():
    print(f"\n===== hw_threshold_mode = {_mode} =====")
    # ---- checks: section 6 (heatwaves) ------------------------------------------
    hw = pd.read_parquet(_D / "heatwaves.parquet")
    hw["year"] = hw.start_date.dt.year
    print(f"heatwaves: {len(hw):,} | PSUs with >=1 HW: {hw.psu_idx.nunique():,}")
    print(hw.duration.describe().round(2))
    dc = pd.read_parquet(_D / "doy_counts.parquet")
    # Within the reference period the in-base-corrected exceedance-day rate
    # should sit near (100 - hw_percentile)% on average. n_hwday counts only
    # days inside qualifying runs, so it is BELOW that ceiling; this check is
    # on order of magnitude, not equality.
    frac_hw = dc.n_hwday.sum() / dc.n_valid.sum()
    print(f"fraction of valid days inside qualifying heatwave runs: {frac_hw:.3%}")
    per_decade = hw.groupby(pd.cut(hw.year, [1984,1994,2004,2014,2024])).size()
    print("heatwaves per decade:"); print(per_decade)


# %%
for _mode, _D in MODE_DIRS.items():
    print(f"\n===== hw_threshold_mode = {_mode} =====")
    # ---- checks: section 7 (EPE) -------------------------------------------------
    epe = pd.read_parquet(_D / "epe.parquet")
    gm = pd.read_parquet(PROCESSED / "psu_gridmatch.parquet")
    bad = set(gm.loc[~gm.chirps_valid, "psu_idx"])
    assert not epe.psu_idx.isin(bad).any(), "EPE detected at CHIRPS-ineligible PSU!"
    print(f"EPE days: {len(epe):,} | PSUs with >=1 EPE: {epe.psu_idx.nunique():,}")
    print("EPE per calendar month (seasonality preview):")
    print(epe.date.dt.month.value_counts().sort_index())


# %%
pip install -U xclim

# %%
import xclim
print(xclim.__version__)

# %% [markdown]
# ## 8. SPEI-3 and SPI-3 (drought indices)
#
# step05_spei.py — Monthly water balance, SPEI-3 and SPI-3 per PSU.
#
# CHANGES vs the audited version
# ------------------------------
# * BUG FIX (critical): pandas `groupby(...).sum()` returns 0.0 for an
#   all-NaN month, which fabricated zero precipitation (hence permanent
#   spurious extreme drought) at PSU whose nearest CHIRPS cell is ocean /
#   no-data. Monthly aggregates are now masked by a per-month VALID-DAY
#   COVERAGE requirement (cfg.min_month_coverage, default 90%); months below
#   coverage are NaN, and CHIRPS-ineligible PSU are excluded from the water
#   balance entirely (psu_gridmatch.parquet).
# * SPI-3 (precipitation only, gamma distribution) is computed alongside
#   SPEI-3. SPI breaks the PET<-temperature circularity of Axis III: a
#   heatwave inflates Hargreaves PET and therefore depresses SPEI in its own
#   month, which partially manufactures "heatwaves during drought". SPI is
#   the sensitivity that cannot be accused of this.
# * PET (Hargreaves-Samani 1985, Ra from FAO-56) unchanged — it is the
#   defensible data-limited choice (Begueria et al. 2014 discuss PET-choice
#   sensitivity); Penman-Monteith would require humidity/wind/radiation
#   inputs that are not part of this extraction.
#
# CALIBRATION: distribution parameters fitted per calendar month on
# 1985-2014 (cfg.ref_*), applied to 1985-2024. The standardized index is a
# relative anomaly; the in-base issue affecting fixed *percentile exceedance*
# thresholds does not apply in the same way here, but note in Methods that
# post-2014 SPEI values are out-of-calibration extrapolations.
#
# OUTPUT: spei3.parquet  psu_idx, year, month, precip_mm, pet_mm, spei3,
#         spi3, is_drought_spei3, is_drought_spi3

# %%
import gc
import time
from calendar import monthrange
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


GSC = 0.0820  # solar constant, MJ m-2 min-1


# ---------------------------------------------------------------------------
# Extraterrestrial radiation Ra (FAO-56 eq. 21-25), vectorised.
# ---------------------------------------------------------------------------
def extraterrestrial_radiation(lat_deg, doy):
    phi = np.radians(np.clip(lat_deg, -66.5, 66.5))
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    delta = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(delta), -1.0, 1.0)
    omega = np.arccos(x)
    ra = (24.0 * 60.0 / np.pi) * GSC * dr * (
        omega * np.sin(phi) * np.sin(delta)
        + np.cos(phi) * np.cos(delta) * np.sin(omega))
    return np.maximum(ra, 0.0)


_MONTH_STARTS = np.array([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
_MONTH_LENGTHS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])


def monthly_ra(lats: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Mean Ra per (lat, month); non-leap calendar (leap effect < 0.1%)."""
    lats = np.asarray(lats, float)
    months = np.asarray(months, int)
    starts = _MONTH_STARTS[months - 1]
    lengths = _MONTH_LENGTHS[months - 1]
    out = np.empty(len(lats))
    for L in np.unique(lengths):
        m = lengths == L
        doys = starts[m][:, None] + np.arange(L)[None, :]
        latg = np.broadcast_to(lats[m][:, None], doys.shape)
        out[m] = extraterrestrial_radiation(latg, doys).mean(axis=1)
    return out


def _monthly_aggregate(block: pd.DataFrame, cfg) -> pd.DataFrame:
    """Daily -> monthly with explicit coverage masking (the bug fix)."""
    block["year"] = block["date"].dt.year.astype("int16")
    block["month"] = block["date"].dt.month.astype("int8")
    block["t_ok"] = (np.isfinite(block["tmax"])
                     & np.isfinite(block["tmin"])).astype("int16")
    block["p_ok"] = np.isfinite(block["precip"]).astype("int16")

    monthly = (block.groupby(["psu_idx", "year", "month"], observed=True)
                    .agg(tmax_mean=("tmax", "mean"),
                         tmin_mean=("tmin", "mean"),
                         precip_sum=("precip", "sum"),     # skipna sum ...
                         n_t_ok=("t_ok", "sum"),
                         n_p_ok=("p_ok", "sum"))
                    .reset_index())
    n_in_month = monthly.apply(
        lambda r: monthrange(int(r["year"]), int(r["month"]))[1], axis=1)
    cov_t = monthly["n_t_ok"] / n_in_month
    cov_p = monthly["n_p_ok"] / n_in_month
    # ... masked here: insufficient coverage -> NaN (never silent zeros).
    monthly.loc[cov_p < cfg.min_month_coverage, "precip_sum"] = np.nan
    monthly.loc[cov_t < cfg.min_month_coverage,
                ["tmax_mean", "tmin_mean"]] = np.nan
    monthly["n_days"] = n_in_month.astype("int16")
    monthly["tmean"] = (monthly["tmax_mean"] + monthly["tmin_mean"]) / 2.0
    return monthly


def _standardize(wb_wide: pd.DataFrame, cfg, kind: str) -> pd.DataFrame:
    """Run xclim SPEI (fisk on water balance) or SPI (gamma on precip).

    COMPATIBILITY (verified against xclim 0.61): `method="PWM"` is NOT
    implemented for the fisk distribution — if requested it is attempted
    and, on NotImplementedError, the fit falls back to "ML" with a logged
    warning (Begueria et al. 2014 favour PWM/L-moments; ML is the closest
    available alternative and must be reported in the Methods). SPI with
    gamma/APP requires the location parameter fixed at 0 (floc=0), which
    is the standard SPI convention (zero-bounded precipitation).
    """
    da = xr.DataArray(
        wb_wide.to_numpy(), dims=("time", "psu_idx"),
        coords={"time": wb_wide.index.values,
                "psu_idx": wb_wide.columns.to_numpy()},
        attrs={"units": "mm/month"})
    kw = dict(freq="MS", window=cfg.spei_scale_months,
              cal_start=f"{cfg.ref_start_year}-01-01",
              cal_end=f"{cfg.ref_end_year}-12-31")
    name = f"{kind}{cfg.spei_scale_months}"
    try:
        if kind == "spei":
            from xclim.indices import (
                standardized_precipitation_evapotranspiration_index as f)
            try:
                out = f(da, dist=cfg.spei_dist, method=cfg.spei_method, **kw)
            except NotImplementedError as e:
                log.warning("SPEI fit method '%s' unavailable for dist '%s' "
                            "(%s) — falling back to ML. Report ML in the "
                            "Methods.", cfg.spei_method, cfg.spei_dist, e)
                out = f(da, dist=cfg.spei_dist, method="ML", **kw)
        else:
            from xclim.indices import standardized_precipitation_index as f
            out = f(da, dist="gamma", method="APP",
                    fitkwargs={"floc": 0}, **kw)
        out = out.clip(-cfg.spei_clip, cfg.spei_clip)
        df = out.to_dataframe(name=name).reset_index()
    except Exception as e:                               # no silent fallback
        log.error("%s computation FAILED for this block (%s); values left "
                  "NaN. Investigate before trusting Axis III.", name, e)
        df = (wb_wide.rename_axis("time").reset_index()
              .melt(id_vars="time", var_name="psu_idx", value_name=name))
        df[name] = np.nan
    df["year"] = df["time"].dt.year.astype("int16")
    df["month"] = df["time"].dt.month.astype("int8")
    return df[["psu_idx", "year", "month", name]]


def run_step05_spei(cfg, out_dir: Path | None = None) -> None:
    out_dir = Path(out_dir or cfg.paths.processed)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    psu = pd.read_parquet(cfg.paths.processed / "psu.parquet",
                          columns=["psu_idx", "lat"])
    gm = pd.read_parquet(cfg.paths.processed / "psu_gridmatch.parquet",
                         columns=["psu_idx", "chirps_valid"])
    eligible = gm.loc[gm["chirps_valid"], "psu_idx"].to_numpy()
    excluded = len(psu) - len(eligible)
    if excluded:
        log.info("Excluding %d CHIRPS-ineligible PSU from the water balance.",
                 excluded)
    psu = psu[psu["psu_idx"].isin(eligible)].reset_index(drop=True)
    all_psu = psu["psu_idx"].to_numpy()
    lat_map = dict(zip(psu["psu_idx"], psu["lat"].astype(float)))

    time_index = pd.date_range(f"{cfg.study_start_year}-01-01",
                               f"{cfg.study_end_year}-12-01", freq="MS")
    bs = cfg.psu_block_size
    n_blocks = (len(all_psu) + bs - 1) // bs
    parts = []
    sname = f"spei{cfg.spei_scale_months}"
    pname = f"spi{cfg.spei_scale_months}"

    for b in range(n_blocks):
        psu_block = all_psu[b * bs: min((b + 1) * bs, len(all_psu))]
        block = read_year_partitions(
            cfg.paths.climate_daily, cfg.study_years,
            ["psu_idx", "date", "tmax", "tmin", "precip"], psu_set=psu_block)
        monthly = _monthly_aggregate(block, cfg)
        del block
        gc.collect()

        # PET (Hargreaves-Samani 1985); NaN-safe via the coverage masks above
        monthly["lat"] = monthly["psu_idx"].map(lat_map)
        monthly["ra"] = monthly_ra(monthly["lat"].to_numpy(),
                                   monthly["month"].to_numpy())
        dt = np.maximum(monthly["tmax_mean"] - monthly["tmin_mean"], 0.0)
        pet_day = (0.0023 * 0.408 * monthly["ra"]
                   * (monthly["tmean"] + 17.8) * np.sqrt(dt))
        monthly["pet_mm"] = np.maximum(pet_day * monthly["n_days"], 0.0)
        monthly["precip_mm"] = monthly["precip_sum"]

        # Complete monthly grid (gaps -> NaN, propagated by xclim)
        full = pd.MultiIndex.from_product(
            [psu_block, range(cfg.study_start_year, cfg.study_end_year + 1),
             range(1, 13)], names=["psu_idx", "year", "month"])
        monthly = (monthly.set_index(["psu_idx", "year", "month"])
                          .reindex(full).reset_index())
        monthly["time"] = pd.to_datetime(dict(year=monthly["year"],
                                              month=monthly["month"], day=1))

        wb = (monthly["precip_mm"] - monthly["pet_mm"]).astype("float64")
        wb_wide = (monthly.assign(wb=wb)
                          .pivot(index="time", columns="psu_idx", values="wb")
                          .reindex(time_index))
        pr_wide = (monthly.pivot(index="time", columns="psu_idx",
                                 values="precip_mm")
                          .reindex(time_index).astype("float64"))

        spei = _standardize(wb_wide, cfg, "spei")
        spi = _standardize(pr_wide, cfg, "spi")

        out = monthly[["psu_idx", "year", "month",
                       "precip_mm", "pet_mm"]].merge(
            spei, on=["psu_idx", "year", "month"], how="left").merge(
            spi, on=["psu_idx", "year", "month"], how="left")
        for c, t in [("psu_idx", "int32"), ("year", "int16"),
                     ("month", "int8"), (sname, "float32"),
                     (pname, "float32")]:
            out[c] = out[c].astype(t)
        parts.append(out)
        gc.collect()
        log.info("  block %d/%d | %.1f min", b + 1, n_blocks,
                 (time.time() - t0) / 60)

    df = pd.concat(parts, ignore_index=True)
    del parts
    df[f"is_drought_{sname}"] = (df[sname] < cfg.drought_threshold).fillna(False)
    df[f"is_drought_{pname}"] = (df[pname] < cfg.drought_threshold).fillna(False)

    # ---- validation: ~N(0,1) on the calibration period --------------------
    ref = df[(df["year"] >= cfg.ref_start_year) & (df["year"] <= cfg.ref_end_year)]
    for col in (sname, pname):
        v = ref[col].dropna()
        if len(v):
            log.info("%s reference mean %+0.3f (exp ~0), std %.3f (exp ~1), "
                     "NaN rows whole period: %s",
                     col, v.mean(), v.std(), f"{df[col].isna().sum():,}")
            if abs(v.mean()) > 0.1 or abs(v.std() - 1) > 0.1:
                log.warning("%s drifts from N(0,1) on the calibration period "
                            "— inspect distribution fits.", col)

    df.to_parquet(out_dir / "spei3.parquet", index=False)
    log.info("Wrote spei3.parquet (%s rows) in %.1f min",
             f"{len(df):,}", (time.time() - t0) / 60)

run_step05_spei(cfg)

# %%
# ---- checks: section 8 -------------------------------------------------------
spei = pd.read_parquet(PROCESSED / "spei3.parquet")
gm = pd.read_parquet(PROCESSED / "psu_gridmatch.parquet")
assert not spei.psu_idx.isin(set(gm.loc[~gm.chirps_valid, "psu_idx"])).any()
ref = spei[(spei.year >= cfg.ref_start_year) & (spei.year <= cfg.ref_end_year)]
for col in (f"spei{cfg.spei_scale_months}", f"spi{cfg.spei_scale_months}"):
    v = ref[col].dropna()
    print(f"{col}: ref mean {v.mean():+.3f} (exp ~0) | std {v.std():.3f} "
          f"(exp ~1) | NaN whole period {spei[col].isna().mean():.2%}")
    assert abs(v.mean()) < 0.15 and abs(v.std() - 1) < 0.2, f"{col} fit suspect"
dr = spei.groupby("year")[f"is_drought_spei{cfg.spei_scale_months}"].mean()
print("drought-month fraction by year (head/tail):")
print(pd.concat([dr.head(3), dr.tail(3)]).round(3))


# %% [markdown]
# ## 9–11. Compound axes — definitions
#
# step06_compound.py — Build the three compound-event axes.
#
# AXIS I — multivariate co-occurrence (Zscheischler et al. 2020)
#     EPE days falling inside a heatwave interval [start-tol, end+tol]
#     (tol = cfg.axis1_day_tolerance; 0 = strict same-day, primary).
#     Heatwave intervals are disjoint per PSU by construction, so with
#     tol = 0 each EPE day matches at most one heatwave: no double counting.
#     With tol = 1 an EPE day can sit in two adjacent extended intervals;
#     the pair table keeps both (window semantics), the per-day count in
#     step04's doy_counts does not double count.
#
# AXIS II — temporal vicinity, PRIMARY = Event Coincidence Analysis
#     (Donges et al. 2016). Unit of analysis: the HEATWAVE, with binary
#     indicators
#         precursor : >= 1 EPE in [start-W, start-1]
#         trigger   : >= 1 EPE in [end+1,  end+W]
#         during    : >= 1 EPE in [start,  end]
#     OVERLAP RULE: because the indicators are per-heatwave, an EPE lying in
#     the windows of two heatwaves legitimately informs both indicators and
#     is never *counted* twice — the double-counting problem of pair-level
#     counting disappears by construction. Heatwaves whose window leaves the
#     study period are excluded (cfg.axis2_exclude_truncated).
#     For the SI, two secondary representations are also written:
#       (a) the full pair table (every EPE x HW within +/-W, with position
#           and signed offset) — window-content semantics, double counting
#           acknowledged;
#       (b) a unique-attribution table: every EPE assigned to the temporally
#           closest heatwave (ties -> earlier HW, deterministic).
#
# AXIS III — drought preconditioning
#     PRIMARY: SPEI-3 of onset month MINUS cfg.axis3_precond_lag_months
#     (default 1), i.e. a strictly antecedent drought state — the SPEI of the
#     onset month itself integrates water balance concurrent with / posterior
#     to the onset and is inflated by the heatwave's own PET (circularity).
#     SENSITIVITIES: lag 0 (comparability with prior studies) and SPI-3
#     (breaks the PET circularity) — controlled via config / step10.
#
# AXIS II PRIMARY METRIC (v2): the after/before COUNT RATIO under UNIQUE attribution
# (`attribute_unique`, core) — each EPE assigned to its single nearest heatwave; the
# ECA fractions above are the SECONDARY metric. Both come out of the same pass.
# AXIS III (v2): every lag in cfg.axis3_lags is computed in one pass (columns *_lag{L});
# the primary lag keeps the unsuffixed names.
#
# OUTPUTS: axis1.parquet (pair table), axis2_hw.parquet (per-HW ECA table),
# axis2_pairs.parquet, axis2_attrib.parquet, axis3.parquet.
#
# The next cell defines the axis builders; each axis is then RUN AND CHECKED in its own cell (9, 10, 11).

# %%
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd



def _load_events(in_dir: Path):
    hw = pd.read_parquet(in_dir / "heatwaves.parquet")
    hw["start_date"] = pd.to_datetime(hw["start_date"])
    hw["end_date"] = pd.to_datetime(hw["end_date"])
    hw = (hw.sort_values(["psu_idx", "start_date"], kind="stable")
            .reset_index(drop=True))
    hw["hw_id"] = hw.index.astype("int64")
    epe = pd.read_parquet(in_dir / "epe.parquet")
    epe["date"] = pd.to_datetime(epe["date"])
    epe = (epe.sort_values(["psu_idx", "date"], kind="stable")
              .reset_index(drop=True))
    return hw, epe


def _axis1(hw: pd.DataFrame, epe: pd.DataFrame, cfg) -> pd.DataFrame:
    """EPE x HW pairs with EPE inside [start-tol, end+tol], per PSU.

    Built per-PSU-chunk to bound the cartesian merge.
    """
    tol = pd.Timedelta(days=cfg.axis1_day_tolerance)
    parts = []
    psus = np.array(sorted(set(hw["psu_idx"]).intersection(epe["psu_idx"])))
    hw_g = {p: g for p, g in hw.groupby("psu_idx", sort=False)}
    epe_g = {p: g for p, g in epe.groupby("psu_idx", sort=False)}
    chunk = 2000
    for lo in range(0, len(psus), chunk):
        pb = psus[lo: lo + chunk]
        h = pd.concat([hw_g[p] for p in pb], ignore_index=True)
        e = pd.concat([epe_g[p] for p in pb], ignore_index=True)
        j = e.merge(h[["hw_id", "psu_idx", "start_date", "end_date",
                       "duration", "peak_tmax"]], on="psu_idx", how="inner")
        m = (j["date"] >= j["start_date"] - tol) & \
            (j["date"] <= j["end_date"] + tol)
        a1 = j.loc[m].copy()
        a1["day_in_hw"] = (a1["date"] - a1["start_date"]).dt.days + 1
        parts.append(a1.rename(columns={"date": "epe_date",
                                        "precip": "epe_precip"}))
        del j, h, e
        gc.collect()
    if not parts:
        return pd.DataFrame(columns=["hw_id", "psu_idx", "start_date",
                                     "end_date", "duration", "peak_tmax",
                                     "epe_date", "epe_precip", "day_in_hw"])
    out = pd.concat(parts, ignore_index=True)
    return out[["hw_id", "psu_idx", "start_date", "end_date", "duration",
                "peak_tmax", "epe_date", "epe_precip", "day_in_hw"]]


def _axis2(hw: pd.DataFrame, epe: pd.DataFrame, cfg, study_lo: int,
           study_hi: int):
    """Per-HW ECA indicators + SI pair table + SI unique attribution."""
    W = cfg.axis2_window_days

    hw = hw.copy()
    hw["start_day"] = to_daynum(hw["start_date"].values)
    hw["end_day"] = to_daynum(hw["end_date"].values)
    hw["truncated"] = ((hw["start_day"] - W < study_lo)
                       | (hw["end_day"] + W > study_hi))

    epe_keys = np.sort(encode_psu_day(epe["psu_idx"].to_numpy(),
                                      to_daynum(epe["date"].values)))
    prec, trig, dur = eca_indicators(
        hw["psu_idx"].to_numpy(), hw["start_day"].to_numpy(),
        hw["end_day"].to_numpy(), epe_keys, W)
    hw_out = hw[["hw_id", "psu_idx", "start_date", "end_date", "duration",
                 "peak_tmax", "truncated"]].copy()
    hw_out["precursor"] = prec
    hw_out["trigger"] = trig
    hw_out["during"] = dur
    if cfg.axis2_exclude_truncated:
        n_tr = int(hw_out["truncated"].sum())
        log.info("Axis II: %s truncated-window heatwaves flagged "
                 "(excluded from rates downstream).", f"{n_tr:,}")

    # ---- SI pair table (window-content semantics) -------------------------
    parts = []
    hw_g = {p: g for p, g in hw.groupby("psu_idx", sort=False)}
    epe_g = {p: g for p, g in epe.groupby("psu_idx", sort=False)}
    psus = np.array(sorted(set(hw["psu_idx"]).intersection(epe["psu_idx"])))
    for lo in range(0, len(psus), 2000):
        pb = psus[lo: lo + 2000]
        h = pd.concat([hw_g[p] for p in pb], ignore_index=True)
        e = pd.concat([epe_g[p] for p in pb], ignore_index=True)
        j = e.merge(h[["hw_id", "psu_idx", "start_date", "end_date",
                       "truncated"]], on="psu_idx", how="inner")
        in_win = ((j["date"] >= j["start_date"] - pd.Timedelta(days=W))
                  & (j["date"] <= j["end_date"] + pd.Timedelta(days=W)))
        p = j.loc[in_win].copy()
        before = p["date"] < p["start_date"]
        after = p["date"] > p["end_date"]
        p["position"] = "during"
        p.loc[before, "position"] = "before"
        p.loc[after, "position"] = "after"
        off = np.zeros(len(p), dtype="int32")
        off[before.to_numpy()] = (p.loc[before, "date"]
                                  - p.loc[before, "start_date"]).dt.days
        off[after.to_numpy()] = (p.loc[after, "date"]
                                 - p.loc[after, "end_date"]).dt.days
        p["offset_days"] = off
        parts.append(p.rename(columns={"date": "epe_date",
                                       "precip": "epe_precip"}))
        del j, h, e
        gc.collect()
    pairs = (pd.concat(parts, ignore_index=True) if parts else
             pd.DataFrame(columns=["hw_id", "psu_idx", "epe_date",
                                   "epe_precip", "position", "offset_days",
                                   "truncated", "start_date", "end_date"]))
    pairs = pairs[["hw_id", "psu_idx", "start_date", "end_date", "epe_date",
                   "epe_precip", "position", "offset_days", "truncated"]]

    # ---- Unique attribution (PRIMARY Axis-II metric) -----------------------
    # Vectorised `attribute_unique` (core) — the SAME function is applied to
    # the surrogate heatwaves of the null (step07). The legacy pair-table
    # attribution is kept only as a cross-check (assert below).
    epe_psu = epe["psu_idx"].to_numpy()
    epe_day = to_daynum(epe["date"].values)
    idx, pos, off = attribute_unique(hw["psu_idx"].to_numpy(),
                                     hw["start_day"].to_numpy(),
                                     hw["end_day"].to_numpy(),
                                     epe_psu, epe_day, W)
    sel = idx >= 0
    attrib = pd.DataFrame({
        "hw_id": hw["hw_id"].to_numpy()[idx[sel]],
        "psu_idx": epe_psu[sel],
        "start_date": hw["start_date"].to_numpy()[idx[sel]],
        "end_date": hw["end_date"].to_numpy()[idx[sel]],
        "epe_date": epe["date"].to_numpy()[sel],
        "epe_precip": epe["precip"].to_numpy()[sel],
        "position": np.array(["none", "before", "during", "after"])[pos[sel]],
        "offset_days": off[sel].astype("int32"),
        "truncated": hw["truncated"].to_numpy()[idx[sel]],
    })
    if len(pairs):
        # legacy (merge-based) attribution — cross-check, then discarded
        att = pairs.copy()
        att["absoff"] = att["offset_days"].abs()
        att = att.sort_values(["psu_idx", "epe_date", "absoff", "start_date"],
                              kind="stable")
        legacy = att.drop_duplicates(subset=["psu_idx", "epe_date"],
                                     keep="first")
        key = ["psu_idx", "epe_date", "hw_id", "position", "offset_days"]
        a = attrib[key].sort_values(key).reset_index(drop=True)
        b = legacy[key].sort_values(key).reset_index(drop=True)
        a["epe_date"] = pd.to_datetime(a["epe_date"]); b["epe_date"] = pd.to_datetime(b["epe_date"])
        if not (len(a) == len(b) and (a.to_numpy() == b.to_numpy()).all()):
            raise AssertionError(
                f"attribute_unique disagrees with the legacy attribution "
                f"({len(a):,} vs {len(b):,} rows) — inspect before proceeding.")
        log.info("Axis II unique attribution: %s EPE attributed "
                 "(vectorised == legacy: OK)", f"{len(attrib):,}")
    return hw_out, pairs, attrib


def _axis3(hw: pd.DataFrame, spei: pd.DataFrame, cfg) -> pd.DataFrame:
    """Heatwaves x antecedent drought state, for EVERY lag in cfg.axis3_lags
    in one pass. Columns are suffixed `_lag{L}`; the FIRST lag (primary,
    default 1) is also exposed under the unsuffixed legacy names
    (spei3, spi3, has_spei, has_spi, drought_precond, drought_precond_spi,
    year, month) so that downstream code is unchanged."""
    sname = f"spei{cfg.spei_scale_months}"
    pname = f"spi{cfg.spei_scale_months}"
    lags = tuple(cfg.axis3_lags)
    assert lags[0] == cfg.axis3_precond_lag_months, \
        "cfg.axis3_lags[0] must be the primary lag (cfg.axis3_precond_lag_months)"

    a3 = hw[["hw_id", "psu_idx", "start_date", "end_date", "duration",
             "mean_tmax", "peak_tmax"]].copy()
    onset = a3["start_date"].dt.to_period("M")
    cols = ["psu_idx", "year", "month", sname, pname]
    out_cols = []
    for lag in lags:
        lo = onset - lag
        key = pd.DataFrame({"psu_idx": a3["psu_idx"].to_numpy(),
                            "year": lo.dt.year.astype("int16").to_numpy(),
                            "month": lo.dt.month.astype("int8").to_numpy()})
        m = key.merge(spei[cols], on=["psu_idx", "year", "month"], how="left")
        s = f"_lag{lag}"
        a3[f"{sname}{s}"] = m[sname].to_numpy()
        a3[f"{pname}{s}"] = m[pname].to_numpy()
        a3[f"has_spei{s}"] = a3[f"{sname}{s}"].notna()
        a3[f"has_spi{s}"] = a3[f"{pname}{s}"].notna()
        a3[f"drought_precond{s}"] = (a3[f"{sname}{s}"] < cfg.drought_threshold).fillna(False)
        a3[f"drought_precond_spi{s}"] = (a3[f"{pname}{s}"] < cfg.drought_threshold).fillna(False)
        if lag == lags[0]:
            a3["year"] = key["year"].to_numpy()
            a3["month"] = key["month"].to_numpy()
        out_cols += [f"{sname}{s}", f"{pname}{s}", f"has_spei{s}", f"has_spi{s}",
                     f"drought_precond{s}", f"drought_precond_spi{s}"]
    p = f"_lag{lags[0]}"
    for base in (sname, pname, "has_spei", "has_spi", "drought_precond",
                 "drought_precond_spi"):
        a3[base] = a3[f"{base}{p}"]
    return a3[["hw_id", "psu_idx", "start_date", "end_date", "duration",
               "mean_tmax", "peak_tmax", "year", "month", sname, pname,
               "has_spei", "has_spi", "drought_precond",
               "drought_precond_spi"] + out_cols]


def run_step06_compound(cfg, in_dir: Path | None = None,
         out_dir: Path | None = None) -> None:
    in_dir = Path(in_dir or cfg.paths.processed)
    out_dir = Path(out_dir or cfg.paths.processed)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    hw, epe = _load_events(in_dir)
    log.info("Heatwaves: %s | EPE days: %s", f"{len(hw):,}", f"{len(epe):,}")
    study_lo = to_daynum(np.datetime64(f"{cfg.study_start_year}-01-01"))
    study_hi = to_daynum(np.datetime64(f"{cfg.study_end_year}-12-31"))

    axis1 = _axis1(hw, epe, cfg)
    axis1.to_parquet(out_dir / "axis1.parquet", index=False)
    log.info("Axis I pairs: %s | HW with >=1 co-occurring EPE: %s "
             "(%.2f%% of HW)", f"{len(axis1):,}",
             f"{axis1['hw_id'].nunique():,}",
             100 * axis1["hw_id"].nunique() / max(1, len(hw)))

    hw_eca, pairs, attrib = _axis2(hw, epe, cfg, int(study_lo), int(study_hi))
    hw_eca.to_parquet(out_dir / "axis2_hw.parquet", index=False)
    pairs.to_parquet(out_dir / "axis2_pairs.parquet", index=False)
    attrib.to_parquet(out_dir / "axis2_attrib.parquet", index=False)
    ok = ~hw_eca["truncated"] if cfg.axis2_exclude_truncated \
        else np.ones(len(hw_eca), bool)
    log.info("Axis II (non-truncated HW: %s): precursor rate %.3f | "
             "trigger rate %.3f | during rate %.3f",
             f"{int(ok.sum()):,}",
             hw_eca.loc[ok, "precursor"].mean(),
             hw_eca.loc[ok, "trigger"].mean(),
             hw_eca.loc[ok, "during"].mean())

    spei = pd.read_parquet(cfg.paths.processed / "spei3.parquet")   # SHARED
    axis3 = _axis3(hw, spei, cfg)
    axis3.to_parquet(out_dir / "axis3.parquet", index=False)
    sname = f"spei{cfg.spei_scale_months}"
    n_s = int(axis3["has_spei"].sum())
    log.info("Axis III: %s HW | with %s at lag-%d month: %s | "
             "drought-preconditioned: %.2f%% | lags computed: %s",
             f"{len(axis3):,}", sname, cfg.axis3_precond_lag_months,
             f"{n_s:,}",
             100 * axis3.loc[axis3["has_spei"], "drought_precond"].mean()
             if n_s else float("nan"), cfg.axis3_lags)

    # ---- sanity ------------------------------------------------------------
    assert axis1["epe_date"].between(
        axis1["start_date"] - pd.Timedelta(days=cfg.axis1_day_tolerance),
        axis1["end_date"] + pd.Timedelta(days=cfg.axis1_day_tolerance)).all()
    assert len(axis3) == len(hw)
    if len(attrib):
        assert not attrib.duplicated(["psu_idx", "epe_date"]).any(), \
            "unique attribution produced duplicate EPE assignments"
    log.info("Done in %.1f min.", (time.time() - t0) / 60)

# (definitions only — run cells follow)


# %% [markdown]
# ### 9. Axis I — same-day co-occurrence

# %%
for _mode in cfg.hw_threshold_modes:
    _D = set_mode(_mode)
    run_step06_compound(cfg, in_dir=_D, out_dir=_D)


# %% [markdown]
# ### 10. Axis II — EPE within ±W days of heatwaves (ECA)
#
# **Overlap rule (explicit)**: the primary unit is the heatwave with binary precursor/trigger indicators, so an EPE inside two heatwave windows informs both indicators and is never counted twice. The pair table (window contents) and a unique nearest-HW attribution (ties → earlier HW) are written for the SI.

# %%
for _mode, _D in MODE_DIRS.items():
    print(f"\n===== hw_threshold_mode = {_mode} =====")
    hw_eca = pd.read_parquet(_D / "axis2_hw.parquet")
    a2_attrib = pd.read_parquet(_D / "axis2_attrib.parquet")
    epe_all = pd.read_parquet(_D / "epe.parquet")
    ok = hw_eca[~hw_eca.truncated] if cfg.axis2_exclude_truncated else hw_eca
    print(f"non-truncated HW: {len(ok):,} / {len(hw_eca):,}")
    print(f"[secondary] precursor rate {ok.precursor.mean():.4f} | trigger rate "
          f"{ok.trigger.mean():.4f} | during rate {ok['during'].mean():.4f}")
    att = a2_attrib[~a2_attrib.truncated] if cfg.axis2_exclude_truncated else a2_attrib
    nb_, na_ = int((att.position == "before").sum()), int((att.position == "after").sum())
    print(f"[PRIMARY] unique attribution: before {nb_:,} | after {na_:,} | during "
          f"{int((att.position == 'during').sum()):,} | count ratio {na_ / max(1, nb_):.3f}")
    assert not a2_attrib.duplicated(["psu_idx", "epe_date"]).any()
    # brute-force window verification on a random sample of heatwaves
    _e = {p: set(pd.to_datetime(g.date)) for p, g in epe_all.groupby("psu_idx")}
    _s = ok.sample(min(300, len(ok)), random_state=0)
    W = cfg.axis2_window_days
    for _, r in _s.iterrows():
        es = _e.get(r.psu_idx, set())
        pre = any(r.start_date - pd.Timedelta(days=k) in es for k in range(1, W + 1))
        tri = any(r.end_date + pd.Timedelta(days=k) in es for k in range(1, W + 1))
        assert pre == r.precursor and tri == r.trigger
    print("brute-force ECA window verification: OK on", len(_s), "sampled HW")


# %% [markdown]
# ### 11. Axis III — heatwaves during antecedent drought

# %%
for _mode, _D in MODE_DIRS.items():
    print(f"\n===== hw_threshold_mode = {_mode} =====")
    axis3 = pd.read_parquet(_D / "axis3.parquet")
    hw_n = len(pd.read_parquet(_D / "heatwaves.parquet", columns=["psu_idx"]))
    assert len(axis3) == hw_n
    # verify the primary lag: stored (year, month) must equal onset month - lag
    _m = axis3.dropna(subset=[f"spei{cfg.spei_scale_months}"]).iloc[0]
    assert (pd.Period(year=int(_m.year), month=int(_m.month), freq="M")
            == pd.Timestamp(_m.start_date).to_period("M") - cfg.axis3_precond_lag_months)
    for L in cfg.axis3_lags:
        h = axis3[f"has_spei_lag{L}"]
        print(f"lag {L}: HW with SPEI {int(h.sum()):,} | drought-preconditioned "
              f"{100 * axis3.loc[h, f'drought_precond_lag{L}'].mean():.2f}% | (SPI) "
              f"{100 * axis3.loc[axis3[f'has_spi_lag{L}'], f'drought_precond_spi_lag{L}'].mean():.2f}%")


# %% [markdown]
# ## 12. Dependence & seasonality-aware significance
#
# step07_significance.py — Dependence and seasonality-aware significance.
#
# This step answers the two questions a methods reviewer will ask first:
#
# (Q1, Axis I) "Is the heat-rain co-occurrence more frequent than expected
#     from the two marginal frequencies — and from their SHARED SEASONALITY?"
#     Metric: Likelihood Multiplication Factor (Zscheischler & Seneviratne
#     2017),  LMF = P(hot & EPE) / [P(hot) x P(EPE)], in two flavours:
#       lmf_naive    : marginals pooled over the year — confounded by the
#                      fact that both extremes cluster in particular seasons;
#       lmf_seasonal : expected joint days computed day-of-year-wise,
#                      E[joint] = sum_d  n_hot(d) * n_epe(d) / n_valid(d),
#                      i.e. independence CONDITIONAL on the seasonal cycle.
#     lmf_seasonal > 1 is the defensible claim of dependence beyond
#     seasonality. Inputs: doy_counts.parquet from step04 (exact, no
#     permutation needed). Regional CIs by bootstrap over PSU — NOTE: PSU
#     sharing grid cells are not independent; the same statistic is therefore
#     also produced on the unique-grid-cell collapse (era5_cell) and that is
#     the CI to quote.
#
# (Q2, Axis II) "Is the post-heatwave EPE excess more than the local rainy-season
#     calendar alone produces?" STRICT null (cfg.null_mode = "psu_window", PRIMARY):
#     every observed heatwave keeps its PSU and DURATION; the surrogate is placed in
#     another study year (uniform among the other years) at the observed onset
#     day-of-year +/- cfg.null_window_days (15). Local seasonality is preserved,
#     only the fine timing is randomised. Surrogates are pushed through the SAME
#     `attribute_unique` as the observed data, so the null is produced for the
#     PRIMARY metric (after/before count ratio) as well as for the ECA fractions.
#     Results by scale x onset season (ALL, DJF, MAM, JJA, SON). The legacy
#     regional-pool null is kept as cfg.null_mode = "region_pool".
#     Window sensitivity (cfg.axis2_window_sens) recomputes Axis II only.
#
# OUTPUTS: lmf_by_psu.parquet, lmf_by_scale.parquet, axis2_null_region.parquet,
#          axis2_null_continental.parquet, axis2_null_draws.parquet,
#          axis2_window_sensitivity.parquet

# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd



# =============================================================================
# Q1 — LMF
# =============================================================================
def lmf_tables(cfg, in_dir: Path, out_dir: Path) -> None:
    dc = pd.read_parquet(in_dir / "doy_counts.parquet")
    psu = pd.read_parquet(cfg.paths.processed / "psu.parquet",
                          columns=["psu_idx", "country", "region"])
    gm = pd.read_parquet(cfg.paths.processed / "psu_gridmatch.parquet",
                         columns=["psu_idx", "era5_cell", "chirps_valid"])

    joint_col = "n_joint_tol1" if cfg.axis1_day_tolerance == 1 else "n_joint"

    # ---- per-PSU --------------------------------------------------------
    g = dc.groupby("psu_idx", observed=True)
    tot = g.agg(n_valid=("n_valid", "sum"), n_hot=("n_hwday", "sum"),
                n_epe=("n_epe", "sum"), n_joint=(joint_col, "sum"))
    # seasonal expectation: sum_d n_hot(d) n_epe(d) / n_valid(d)
    dc["_e"] = safe_ratio(dc["n_hwday"] * dc["n_epe"], dc["n_valid"])
    tot["e_seasonal"] = dc.groupby("psu_idx", observed=True)["_e"].sum()
    tot["e_naive"] = safe_ratio(tot["n_hot"] * tot["n_epe"], tot["n_valid"])
    tot["lmf_naive"] = safe_ratio(tot["n_joint"], tot["e_naive"])
    tot["lmf_seasonal"] = safe_ratio(tot["n_joint"], tot["e_seasonal"])
    tot = tot.reset_index().merge(psu, on="psu_idx").merge(gm, on="psu_idx")
    tot = tot[tot["chirps_valid"]]                  # eligible PSU only
    tot.to_parquet(out_dir / "lmf_by_psu.parquet", index=False)

    # ---- pooled per scale, PSU-level and cell-collapsed -------------------
    rng = np.random.default_rng(cfg.rng_seed)

    def pooled(df, label, unit_col):
        """Pooled LMF over the group + bootstrap CI resampling `unit_col`."""
        units = df.groupby(unit_col, observed=True).agg(
            n_joint=("n_joint", "sum"), e_seasonal=("e_seasonal", "sum"),
            e_naive=("e_naive", "sum"))
        obs_s = safe_ratio(units["n_joint"].sum(), units["e_seasonal"].sum())
        obs_n = safe_ratio(units["n_joint"].sum(), units["e_naive"].sum())
        nj = units["n_joint"].to_numpy(float)
        es = units["e_seasonal"].to_numpy(float)
        n = len(units)
        boot = np.empty(cfg.n_lmf_surrogates)
        for k in range(cfg.n_lmf_surrogates):
            idx = rng.integers(0, n, n)
            boot[k] = nj[idx].sum() / max(es[idx].sum(), 1e-12)
        return dict(scale=label, unit=unit_col, n_units=n,
                    lmf_naive=float(obs_n), lmf_seasonal=float(obs_s),
                    lmf_seasonal_lo=float(np.quantile(boot, 0.025)),
                    lmf_seasonal_hi=float(np.quantile(boot, 0.975)))

    rows = []
    for unit in ("psu_idx", "era5_cell"):
        rows.append(pooled(tot, "Continental", unit))
        for reg, sub in tot.groupby("region", observed=True):
            rows.append({**pooled(sub, str(reg), unit)})
        for cou, sub in tot.groupby("country", observed=True):
            rows.append({**pooled(sub, str(cou), unit), "level": "country"})
    out = pd.DataFrame(rows)
    out.to_parquet(out_dir / "lmf_by_scale.parquet", index=False)
    cont = out[(out["scale"] == "Continental") & (out["unit"] == "era5_cell")]
    if len(cont):
        r = cont.iloc[0]
        log.info("LMF continental (cell-collapsed): naive %.2f | seasonal "
                 "%.2f [%.2f, %.2f]", r["lmf_naive"], r["lmf_seasonal"],
                 r["lmf_seasonal_lo"], r["lmf_seasonal_hi"])


# =============================================================================
# Q2 — Axis II seasonal surrogate null
# =============================================================================
def _axis2_groups(hw: pd.DataFrame, cfg) -> dict:
    """{(scale, season): boolean mask over heatwaves}. season 'ALL' + the
    calendar seasons of ONSET (cfg.null_seasons)."""
    season = season_of_month(hw["start_date"].dt.month.to_numpy())
    scales = {"Continental": np.ones(len(hw), bool)}
    for reg in pd.unique(hw["region"].dropna()):
        scales[str(reg)] = (hw["region"] == reg).to_numpy()
    groups = {}
    for sc, m in scales.items():
        groups[(sc, "ALL")] = m
        for s in cfg.null_seasons:
            groups[(sc, s)] = m & (season == s)
    return groups


def _load_axis2_for_null(cfg, in_dir: Path):
    hw = pd.read_parquet(in_dir / "axis2_hw.parquet")
    epe = pd.read_parquet(in_dir / "epe.parquet", columns=["psu_idx", "date"])
    psu = pd.read_parquet(cfg.paths.processed / "psu.parquet",
                          columns=["psu_idx", "country", "region"])
    hw = hw.merge(psu, on="psu_idx", how="left")
    hw["start_date"] = pd.to_datetime(hw["start_date"])
    hw["end_date"] = pd.to_datetime(hw["end_date"])
    hw["start_day"] = to_daynum(hw["start_date"].values)
    hw["end_day"] = to_daynum(hw["end_date"].values)
    epe_psu = epe["psu_idx"].to_numpy()
    epe_day = to_daynum(epe["date"].values)
    epe_keys = np.sort(encode_psu_day(epe_psu, epe_day))
    return hw, epe_psu, epe_day, epe_keys


def axis2_null(cfg, in_dir: Path, out_dir: Path) -> None:
    """Surrogate null for Axis II, computed for BOTH metrics:
        * count_ratio = n_after / n_before under UNIQUE attribution (PRIMARY)
        * ECA fractions precursor / trigger (secondary)
    null_mode == "psu_window" (PRIMARY, strict): each observed heatwave keeps
        its PSU and duration; the surrogate year is drawn uniformly among the
        OTHER study years, the surrogate onset among the observed onset
        day-of-year +/- cfg.null_window_days. Local seasonality is preserved;
        only the fine timing is randomised. Surrogates whose +/-W window
        leaves the study period are dropped (as for the observed data).
    null_mode == "region_pool" (legacy): onset doy drawn from the pooled
        onset-doy distribution of the region (same year).
    Results per (scale x onset season): axis2_null_region.parquet;
    Continental/ALL: axis2_null_continental.parquet; raw draws:
    axis2_null_draws.parquet.
    """
    hw, epe_psu, epe_day, epe_keys = _load_axis2_for_null(cfg, in_dir)
    if cfg.axis2_exclude_truncated:
        hw = hw[~hw["truncated"]].reset_index(drop=True)
    W, K = cfg.axis2_window_days, cfg.null_window_days
    rng = np.random.default_rng(cfg.rng_seed)
    study_lo = int(to_daynum(np.datetime64(f"{cfg.study_start_year}-01-01")))
    study_hi = int(to_daynum(np.datetime64(f"{cfg.study_end_year}-12-31")))

    hw_psu = hw["psu_idx"].to_numpy()
    s_obs, e_obs = hw["start_day"].to_numpy(), hw["end_day"].to_numpy()
    durm1 = e_obs - s_obs
    hw_year = hw["start_date"].dt.year.to_numpy()
    hw_doy = hw["start_date"].dt.dayofyear.to_numpy()
    years = np.array(cfg.study_years)
    jan1 = to_daynum(pd.to_datetime({"year": years, "month": 1, "day": 1}).values)
    obs_yi = hw_year - cfg.study_start_year
    n, ny = len(hw), len(years)
    groups = _axis2_groups(hw, cfg)

    # ---- observed ------------------------------------------------------------
    prec_o = hw["precursor"].to_numpy(bool); trig_o = hw["trigger"].to_numpy(bool)
    idx_o, pos_o, _ = attribute_unique(hw_psu, s_obs, e_obs, epe_psu, epe_day, W)
    obs = {}
    for g, m in groups.items():
        nb, na = before_after_counts(pos_o, idx_o, m)
        obs[g] = dict(n_hw=int(m.sum()), precursor_obs=prec_o[m].mean() if m.any() else np.nan,
                      trigger_obs=trig_o[m].mean() if m.any() else np.nan,
                      n_before_obs=nb, n_after_obs=na,
                      count_ratio_obs=na / nb if nb > 0 else np.nan)

    # ---- surrogates ------------------------------------------------------------
    n_perm = cfg.n_null_permutations
    log.info("Axis II null: mode=%s, %d permutations x %s HW, +/-%d d, W=%d ...",
             cfg.null_mode, n_perm, f"{n:,}", K, W)
    null = {g: {k: np.full(n_perm, np.nan) for k in ("precursor", "trigger", "count_ratio")}
            for g in groups}
    if cfg.null_mode == "region_pool":
        pool_lab = hw["region"].astype(str).to_numpy()
        pools = {lab: hw_doy[pool_lab == lab] for lab in np.unique(pool_lab)}
    t0 = time.time()
    for k in range(n_perm):
        if cfg.null_mode == "psu_window":
            yi = rng.integers(0, ny - 1, size=n)
            yi = yi + (yi >= obs_yi)                       # skip the observed year
            shift = rng.integers(-K, K + 1, size=n)
            s_start = jan1[yi] + (hw_doy - 1) + shift
        elif cfg.null_mode == "region_pool":
            sur_doy = np.empty(n, dtype="int64")
            for lab, pool in pools.items():
                mm = pool_lab == lab
                sur_doy[mm] = rng.choice(pool, size=int(mm.sum()), replace=True)
            s_start = jan1[obs_yi] + (sur_doy - 1)
        else:
            raise ValueError(f"unknown null_mode {cfg.null_mode!r}")
        s_end = s_start + durm1
        valid = (s_start - W >= study_lo) & (s_end + W <= study_hi)
        p, t, _ = eca_indicators(hw_psu[valid], s_start[valid], s_end[valid], epe_keys, W)
        idx, pos, _ = attribute_unique(hw_psu[valid], s_start[valid], s_end[valid],
                                       epe_psu, epe_day, W)
        for g, m in groups.items():
            mv = m[valid]
            if not mv.any():
                continue
            nb, na = before_after_counts(pos, idx, mv)
            null[g]["precursor"][k] = p[mv].mean()
            null[g]["trigger"][k] = t[mv].mean()
            null[g]["count_ratio"][k] = na / nb if nb > 0 else np.nan
        if (k + 1) % max(1, n_perm // 10) == 0:
            log.info("  perm %d/%d | %.1f min", k + 1, n_perm, (time.time() - t0) / 60)

    def q(a, p): return float(np.nanquantile(a, p)) if np.isfinite(a).any() else np.nan
    def pval(null_a, obs_v):                               # one-sided, add-one (Davison & Hinkley)
        a = null_a[np.isfinite(null_a)]
        return float((np.sum(a >= obs_v) + 1) / (len(a) + 1)) if len(a) and np.isfinite(obs_v) else np.nan

    rows, draws = [], []
    for (sc, se), m in groups.items():
        o = obs[(sc, se)]; nl = null[(sc, se)]
        d_null = nl["trigger"] - nl["precursor"]
        d_obs = o["trigger_obs"] - o["precursor_obs"]
        rows.append(dict(
            scale=sc, season=se, null_mode=cfg.null_mode, null_window_days=K, **o,
            diff_obs=d_obs,
            precursor_null_lo=q(nl["precursor"], .025), precursor_null_hi=q(nl["precursor"], .975),
            trigger_null_lo=q(nl["trigger"], .025), trigger_null_hi=q(nl["trigger"], .975),
            diff_null_lo=q(d_null, .025), diff_null_hi=q(d_null, .975),
            count_ratio_null_lo=q(nl["count_ratio"], .025),
            count_ratio_null_med=q(nl["count_ratio"], .5),
            count_ratio_null_hi=q(nl["count_ratio"], .975),
            p_trigger_excess=pval(nl["trigger"], o["trigger_obs"]),
            p_diff_excess=pval(d_null, d_obs),
            p_count_ratio_excess=pval(nl["count_ratio"], o["count_ratio_obs"]),
            n_perm_valid=int(np.isfinite(nl["count_ratio"]).sum())))
        draws.append(pd.DataFrame({"scale": sc, "season": se, "perm": np.arange(n_perm),
                                   "precursor": nl["precursor"], "trigger": nl["trigger"],
                                   "count_ratio": nl["count_ratio"]}))
    out = pd.DataFrame(rows)
    out.to_parquet(out_dir / "axis2_null_region.parquet", index=False)
    out[(out["scale"] == "Continental") & (out["season"] == "ALL")].to_parquet(
        out_dir / "axis2_null_continental.parquet", index=False)
    pd.concat(draws, ignore_index=True).to_parquet(out_dir / "axis2_null_draws.parquet", index=False)
    c = out[(out["scale"] == "Continental") & (out["season"] == "ALL")].iloc[0]
    log.info("Axis II continental: count_ratio %.3f vs null [%.3f, %.3f] (p=%.4f) | "
             "trigger %.3f vs [%.3f, %.3f] (p=%.4f) | precursor %.3f vs [%.3f, %.3f]",
             c["count_ratio_obs"], c["count_ratio_null_lo"], c["count_ratio_null_hi"],
             c["p_count_ratio_excess"], c["trigger_obs"], c["trigger_null_lo"],
             c["trigger_null_hi"], c["p_trigger_excess"], c["precursor_obs"],
             c["precursor_null_lo"], c["precursor_null_hi"])


def axis2_window_sensitivity(cfg, in_dir: Path, out_dir: Path) -> None:
    """Observed Axis-II metrics recomputed for W in cfg.axis2_window_sens
    (+ the primary W), Axis II only — no rerun of the rest of the pipeline.
    Truncation is re-evaluated for each W. Output: axis2_window_sensitivity.parquet"""
    hw, epe_psu, epe_day, epe_keys = _load_axis2_for_null(cfg, in_dir)
    study_lo = int(to_daynum(np.datetime64(f"{cfg.study_start_year}-01-01")))
    study_hi = int(to_daynum(np.datetime64(f"{cfg.study_end_year}-12-31")))
    hw_psu = hw["psu_idx"].to_numpy(); s, e = hw["start_day"].to_numpy(), hw["end_day"].to_numpy()
    scales = {"Continental": np.ones(len(hw), bool)}
    for reg in pd.unique(hw["region"].dropna()):
        scales[str(reg)] = (hw["region"] == reg).to_numpy()
    rows = []
    for W in sorted(set(cfg.axis2_window_sens) | {cfg.axis2_window_days}):
        ok = ~((s - W < study_lo) | (e + W > study_hi)) if cfg.axis2_exclude_truncated \
            else np.ones(len(hw), bool)
        p, t, _ = eca_indicators(hw_psu[ok], s[ok], e[ok], epe_keys, W)
        idx, pos, _ = attribute_unique(hw_psu[ok], s[ok], e[ok], epe_psu, epe_day, W)
        for sc, m in scales.items():
            mv = m[ok]
            nb, na = before_after_counts(pos, idx, mv)
            rows.append(dict(window_days=W, scale=sc, primary=(W == cfg.axis2_window_days),
                             n_hw=int(mv.sum()), precursor_rate=p[mv].mean(),
                             trigger_rate=t[mv].mean(), rate_diff=t[mv].mean() - p[mv].mean(),
                             n_before=nb, n_after=na, count_ratio=na / nb if nb > 0 else np.nan))
        log.info("  W=%d: continental count_ratio %.3f", W, rows[-len(scales)]["count_ratio"])
    pd.DataFrame(rows).to_parquet(out_dir / "axis2_window_sensitivity.parquet", index=False)


def run_step07_significance(cfg, in_dir: Path | None = None,
         out_dir: Path | None = None) -> None:
    in_dir = Path(in_dir or cfg.paths.processed)
    out_dir = Path(out_dir or in_dir / "significance")
    out_dir.mkdir(parents=True, exist_ok=True)
    lmf_tables(cfg, in_dir, out_dir)
    axis2_null(cfg, in_dir, out_dir)
    axis2_window_sensitivity(cfg, in_dir, out_dir)

# (definitions only — run cells follow)



# %%
for _mode, _D in MODE_DIRS.items():
    print(f"\n===== hw_threshold_mode = {_mode} =====")
    # ---- 12a. Axis I LMF (fast) ---------------------------------------------------
    SIG = _D / "significance"; SIG.mkdir(exist_ok=True)
    lmf_tables(cfg, _D, SIG)
    lmf = pd.read_parquet(SIG / "lmf_by_scale.parquet")
    print(lmf[lmf.scale == "Continental"][["unit", "lmf_naive", "lmf_seasonal",
          "lmf_seasonal_lo", "lmf_seasonal_hi"]].to_string(index=False))


# %%
for _mode, _D in MODE_DIRS.items():
    print(f"\n===== hw_threshold_mode = {_mode} =====")
    # ---- 12b. Axis II strict null (SLOW: n_perm x [ECA + unique attribution]) ----
    # First run: consider cfg.n_null_permutations = 100-200; full 1000 overnight.
    SIG = _D / "significance"
    set_mode(_mode)
    axis2_null(cfg, _D, SIG)
    axis2_window_sensitivity(cfg, _D, SIG)
    null = pd.read_parquet(SIG / "axis2_null_region.parquet")
    print(null[null.season == "ALL"][["scale", "n_hw", "count_ratio_obs", "count_ratio_null_lo",
          "count_ratio_null_hi", "p_count_ratio_excess", "trigger_obs", "trigger_null_hi",
          "p_diff_excess"]].round(4).to_string(index=False))
    print(pd.read_parquet(SIG / "axis2_window_sensitivity.parquet")
          .query("scale == 'Continental'")[["window_days", "n_hw", "count_ratio", "rate_diff"]]
          .round(4).to_string(index=False))


# %% [markdown]
# ## 13. Aggregations (PSU / country / region / continent × year / decade)
#
# step08_aggregate.py — Aggregations by PSU, country, region, continent,
# year and decade, for the three axes.
#
# CONVENTIONS
# -----------
# * PRIMARY = per-PSU averaging ("method b": compute the metric per PSU,
#   then take the unweighted mean across PSU of the group), consistent with
#   the DHS design (each PSU = one sampled locality) and with compound-event
#   practice (Bevacqua et al. 2022; Mukherjee & Mishra 2021).
# * SECONDARY ("method a", event-weighted pooled metric) is written next to
#   it in the same files for the SI.
# * ELIGIBLE DENOMINATORS: any metric involving EPE uses only CHIRPS-eligible
#   PSU in BOTH numerator and denominator (otherwise arid/coastal regions are
#   mechanically deflated); Axis III percentages use HW with a valid drought
#   index at the antecedent month.
# * CELL COLLAPSE (pseudo-replication sensitivity): the by-year continental
#   and regional series are recomputed with PSU first averaged within their
#   ERA5 grid cell (suffix *_cellcollapse).
# * Axis II rates are computed on NON-TRUNCATED heatwaves only.
#
# OUTPUTS under <out_dir>/aggregations/ :
#   axis1_by_psu, axis1_by_year_{continental,region,country},
#   axis1_by_psu_decade, axis1_by_region_decade, axis1_by_month_region,
#   axis2_by_psu, axis2_by_year_{continental,region,country},
#   axis2_by_psu_decade, axis2_by_onset_month_region, axis2_by_region_decade,
#   axis2_by_country  (n_before / n_after / count_ratio everywhere = PRIMARY),
#   axis3_by_psu, axis3_by_year_{continental,region,country},
#   axis3_by_psu_decade, axis3_by_region_decade, axis3_by_onset_month_region
#   (pct_drought_lag0_* next to the primary lag-1 columns),
#   *_cellcollapse variants of the by-year continental/region series.

# %%
from pathlib import Path

import numpy as np
import pandas as pd



# ---------------------------------------------------------------------------
def _scaffold(cfg, in_dir: Path):
    psu = pd.read_parquet(cfg.paths.processed / "psu.parquet",
                          columns=["psu_idx", "country", "region"])
    gm = pd.read_parquet(cfg.paths.processed / "psu_gridmatch.parquet",
                         columns=["psu_idx", "era5_cell", "chirps_valid"])
    psu = psu.merge(gm, on="psu_idx", how="left")
    psu["scale"] = "Continental"

    hw = pd.read_parquet(in_dir / "axis3.parquet")    # = all heatwaves + SPEI
    hw = hw.merge(psu, on="psu_idx", how="left")
    hw["hw_year"] = hw["start_date"].dt.year.astype("int16")
    hw["decade"] = assign_decade(hw["hw_year"], cfg.decade_bounds)
    return psu, hw


def _per_psu_then_group(per_psu: pd.DataFrame, metric_cols: list,
                        psu_table: pd.DataFrame, group_keys: list,
                        extra_keys: list) -> pd.DataFrame:
    """Method (b): unweighted mean of per-PSU metrics within each group.
    NaN per-PSU metrics (undefined denominators) are excluded pairwise."""
    j = per_psu.merge(psu_table, on="psu_idx", how="left")
    return (j.groupby(group_keys + extra_keys, observed=True)[metric_cols]
             .mean().add_suffix("_mean_psu").reset_index())


def _write(df: pd.DataFrame, agg_dir: Path, name: str) -> None:
    df.to_parquet(agg_dir / f"{name}.parquet", index=False)
    log.info("  -> %s (%s rows)", name, f"{len(df):,}")


def _complete_years(df: pd.DataFrame, keys: list, years: list,
                    zero_cols: list) -> pd.DataFrame:
    """Ensure every (group x study-year) combination has a row.

    Absence of events in a year is information (count = 0), not missing
    data: count columns are filled with 0; rate/percentage columns are
    left NaN (undefined denominator). Without this, annual series silently
    skip empty years and trend estimators see an irregular time axis.
    """
    if not keys:
        full = pd.DataFrame({"year": years})
    else:
        groups = df[keys].drop_duplicates()
        full = groups.merge(pd.DataFrame({"year": years}), how="cross")
    out = full.merge(df, on=keys + ["year"], how="left")
    for c in zero_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0)
    return out.sort_values(keys + ["year"]).reset_index(drop=True)


# =============================================================================
# AXIS I
# =============================================================================
def axis1_aggregations(cfg, in_dir, agg_dir, psu, hw):
    a1 = pd.read_parquet(in_dir / "axis1.parquet",
                         columns=["hw_id", "psu_idx", "epe_date"])
    a1["year"] = a1["epe_date"].dt.year.astype("int16")
    a1["decade"] = assign_decade(a1["year"], cfg.decade_bounds)

    elig = psu[psu["chirps_valid"]].copy()
    hw_e = hw[hw["psu_idx"].isin(elig["psu_idx"])]

    # ---- per PSU (pooled years) ------------------------------------------
    n_years = cfg.study_end_year - cfg.study_start_year + 1
    per = elig[["psu_idx"]].copy()
    per = per.merge(hw_e.groupby("psu_idx").size().rename("n_hw"),
                    on="psu_idx", how="left")
    per = per.merge(a1.groupby("psu_idx").size().rename("n_cooccur_days"),
                    on="psu_idx", how="left")
    per = per.merge(a1.groupby("psu_idx")["hw_id"].nunique()
                      .rename("n_hw_with_cooc"), on="psu_idx", how="left")
    for c in ("n_hw", "n_cooccur_days", "n_hw_with_cooc"):
        per[c] = per[c].fillna(0).astype("int32")
    per["cooccur_days_per_year"] = per["n_cooccur_days"] / n_years
    per["pct_hw_with_cooc"] = safe_ratio(per["n_hw_with_cooc"],
                                         per["n_hw"]) * 100
    _write(per.merge(psu, on="psu_idx"), agg_dir, "axis1_by_psu")

    # ---- by year x scale -----------------------------------------------------
    def by_year(keys, name, collapse=False):
        unit = "era5_cell" if collapse else "psu_idx"
        a1y = (a1.merge(elig, on="psu_idx")
                 .groupby([unit, "year"] if collapse else ["psu_idx", "year"],
                          observed=True)
                 .agg(n_cooccur_days=("epe_date", "size"),
                      n_hw_with_cooc=("hw_id", "nunique")).reset_index())
        hwy = (hw_e.groupby([unit if collapse else "psu_idx", "hw_year"],
                            observed=True)
                   .size().rename("n_hw").reset_index()
                   .rename(columns={"hw_year": "year"}))
        per_unit = hwy.merge(a1y, on=[unit if collapse else "psu_idx", "year"],
                             how="outer")
        # complete over ALL eligible units x ALL study years: a unit-year
        # with no heatwave and no co-occurrence is a genuine zero exposure,
        # and the per-unit mean must average over the full eligible set.
        all_units = (elig[unit].drop_duplicates().to_frame()
                     .merge(pd.DataFrame({"year": cfg.study_years}),
                            how="cross"))
        per_unit = all_units.merge(per_unit, on=[unit, "year"], how="left")
        for c in ("n_hw", "n_cooccur_days", "n_hw_with_cooc"):
            per_unit[c] = per_unit[c].fillna(0).astype("int32")
        per_unit["cooccur_days_unit"] = per_unit["n_cooccur_days"].astype(float)
        per_unit["pct_hw_with_cooc_unit"] = safe_ratio(
            per_unit["n_hw_with_cooc"], per_unit["n_hw"]) * 100

        units_table = (elig.groupby(unit, observed=True).first().reset_index()
                       if collapse else elig)
        j = per_unit.merge(units_table[[unit, "country", "region", "scale"]]
                           if collapse else
                           units_table[["psu_idx", "country", "region",
                                        "scale"]],
                           on=unit if collapse else "psu_idx", how="left")
        # method (b)
        b = (j.groupby(keys + ["year"], observed=True)
              .agg(cooccur_days_per_unit=("cooccur_days_unit", "mean"),
                   pct_hw_with_cooc_b=("pct_hw_with_cooc_unit", "mean"),
                   n_units=(unit if collapse else "psu_idx", "nunique"))
              .reset_index())
        # method (a)
        a = (j.groupby(keys + ["year"], observed=True)
              .agg(n_hw=("n_hw", "sum"),
                   n_cooccur_days=("n_cooccur_days", "sum"),
                   n_hw_with_cooc=("n_hw_with_cooc", "sum")).reset_index())
        a["pct_hw_with_cooc_a"] = safe_ratio(a["n_hw_with_cooc"],
                                             a["n_hw"]) * 100
        out = a.merge(b, on=keys + ["year"])
        out = _complete_years(out, keys, cfg.study_years,
                              ["n_hw", "n_cooccur_days", "n_hw_with_cooc",
                               "cooccur_days_per_unit", "n_units"])
        _write(out, agg_dir, name)

    by_year(["scale"], "axis1_by_year_continental")
    by_year(["region"], "axis1_by_year_region")
    by_year(["country"], "axis1_by_year_country")
    by_year(["scale"], "axis1_by_year_continental_cellcollapse", collapse=True)
    by_year(["region"], "axis1_by_year_region_cellcollapse", collapse=True)

    # ---- per PSU x decade ------------------------------------------------------
    pd_dec = (hw_e.groupby(["psu_idx", "decade"], observed=False).size()
                  .rename("n_hw").reset_index())
    d1 = (a1.merge(elig[["psu_idx"]], on="psu_idx")
            .groupby(["psu_idx", "decade"], observed=False)
            .agg(n_cooccur_days=("epe_date", "size"),
                 n_hw_with_cooc=("hw_id", "nunique")).reset_index())
    out = pd_dec.merge(d1, on=["psu_idx", "decade"], how="outer")
    for c in ("n_hw", "n_cooccur_days", "n_hw_with_cooc"):
        out[c] = out[c].fillna(0).astype("int32")
    out["cooccur_days_per_year"] = out["n_cooccur_days"] / 10.0
    out["pct_hw_with_cooc"] = safe_ratio(out["n_hw_with_cooc"],
                                         out["n_hw"]) * 100
    _write(out, agg_dir, "axis1_by_psu_decade")

    # ---- region x decade & EPE-month x region (Figs S5a, S9) -------------------
    a1e = a1.merge(elig[["psu_idx", "region", "scale"]], on="psu_idx")
    a1e["month"] = a1e["epe_date"].dt.month.astype("int8")
    n_years = cfg.study_end_year - cfg.study_start_year + 1

    def grp(keys, extra):
        r = (a1e.groupby(keys + extra, observed=True)
                .agg(n_cooccur_days=("epe_date", "size"),
                     n_hw_with_cooc=("hw_id", "nunique")).reset_index())
        n_el = elig.groupby(keys, observed=True)["psu_idx"].nunique().rename("n_units").reset_index()
        r = r.merge(n_el, on=keys, how="left")
        span = 10.0 if extra == ["decade"] else float(n_years)
        r["cooccur_days_per_unit_year"] = r["n_cooccur_days"] / (r["n_units"] * span)
        return r
    _write(_with_continental(grp(["region"], ["decade"]), grp(["scale"], ["decade"])),
           agg_dir, "axis1_by_region_decade")
    _write(_with_continental(grp(["region"], ["month"]), grp(["scale"], ["month"])),
           agg_dir, "axis1_by_month_region")


# =============================================================================
# AXIS II — ECA fractions (secondary) + unique-attribution counts (PRIMARY)
# =============================================================================
def _ba_counts(att: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Unique-attribution before/after EPE counts and their ratio per group."""
    t = att[keys].copy()
    t["n_before"] = (att["position"] == "before").astype("int32")
    t["n_after"] = (att["position"] == "after").astype("int32")
    out = t.groupby(keys, observed=True)[["n_before", "n_after"]].sum().reset_index()
    out["count_ratio"] = safe_ratio(out["n_after"], out["n_before"])
    return out


def _with_continental(df_region: pd.DataFrame, df_cont: pd.DataFrame) -> pd.DataFrame:
    c = df_cont.rename(columns={"scale": "region"})
    return pd.concat([df_region, c], ignore_index=True)


def axis2_aggregations(cfg, in_dir, agg_dir, psu, hw):
    a2 = pd.read_parquet(in_dir / "axis2_hw.parquet")
    a2 = a2.merge(psu, on="psu_idx", how="left")
    a2["year"] = a2["start_date"].dt.year.astype("int16")
    a2["onset_month"] = a2["start_date"].dt.month.astype("int8")
    a2["decade"] = assign_decade(a2["year"], cfg.decade_bounds)
    if cfg.axis2_exclude_truncated:
        a2 = a2[~a2["truncated"]]
    a2 = a2[a2["chirps_valid"]]            # rates require EPE-eligible PSU
    for c in ("precursor", "trigger", "during"):
        a2[c] = a2[c].astype(float)

    # ---- unique-attribution EPE table (PRIMARY metric inputs) ------------------
    att = pd.read_parquet(in_dir / "axis2_attrib.parquet")
    att = att.merge(psu, on="psu_idx", how="left")
    if cfg.axis2_exclude_truncated:
        att = att[~att["truncated"]]
    att = att[att["chirps_valid"] & att["position"].isin(["before", "after"])].copy()
    att["start_date"] = pd.to_datetime(att["start_date"])
    att["year"] = att["start_date"].dt.year.astype("int16")       # HW onset year
    att["onset_month"] = att["start_date"].dt.month.astype("int8")
    att["decade"] = assign_decade(att["year"], cfg.decade_bounds)
    elig = psu[psu["chirps_valid"]]

    # ---- per PSU (pooled) -----------------------------------------------------
    per = (a2.groupby("psu_idx", observed=True)
             .agg(n_hw=("hw_id", "size"), precursor_rate=("precursor", "mean"),
                  trigger_rate=("trigger", "mean"),
                  during_rate=("during", "mean")).reset_index())
    per["rate_diff"] = per["trigger_rate"] - per["precursor_rate"]
    per["rate_ratio"] = safe_ratio(per["trigger_rate"], per["precursor_rate"])
    per = per.merge(_ba_counts(att, ["psu_idx"]), on="psu_idx", how="left")
    for c in ("n_before", "n_after"):
        per[c] = per[c].fillna(0).astype("int32")
    _write(per.merge(psu, on="psu_idx"), agg_dir, "axis2_by_psu")

    # ---- per country (pooled; Fig. 4a) -------------------------------------------
    bc = _ba_counts(att, ["country"]).merge(
        elig.groupby("country").size().rename("n_psu").reset_index(), on="country", how="right")
    for c in ("n_before", "n_after"):
        bc[c] = bc[c].fillna(0).astype("int32")
    bc["count_ratio"] = safe_ratio(bc["n_after"], bc["n_before"])
    bc["before_per_psu"] = bc["n_before"] / bc["n_psu"]
    bc["after_per_psu"] = bc["n_after"] / bc["n_psu"]
    _write(bc.merge(elig.groupby("country")["region"].first().reset_index(), on="country"),
           agg_dir, "axis2_by_country")

    # ---- by year x scale ---------------------------------------------------------
    def by_year(keys, name, collapse=False):
        unit = "era5_cell" if collapse else "psu_idx"
        pu = (a2.groupby([unit, "year"], observed=True)
                .agg(n_hw=("hw_id", "size"),
                     precursor_rate=("precursor", "mean"),
                     trigger_rate=("trigger", "mean")).reset_index())
        pu["rate_diff"] = pu["trigger_rate"] - pu["precursor_rate"]
        lab = (a2.groupby(unit, observed=True)
                 .first()[["country", "region", "scale"]].reset_index())
        j = pu.merge(lab, on=unit, how="left")
        b = (j.groupby(keys + ["year"], observed=True)
              .agg(precursor_rate_b=("precursor_rate", "mean"),
                   trigger_rate_b=("trigger_rate", "mean"),
                   rate_diff_b=("rate_diff", "mean"),
                   n_units=(unit, "nunique")).reset_index())
        a = (a2.groupby(keys + ["year"], observed=True)
               .agg(n_hw=("hw_id", "size"),
                    precursor_rate_a=("precursor", "mean"),
                    trigger_rate_a=("trigger", "mean")).reset_index())
        a["rate_diff_a"] = a["trigger_rate_a"] - a["precursor_rate_a"]
        ba = _ba_counts(att, keys + ["year"]).rename(columns={"count_ratio": "count_ratio_a"})
        out = a.merge(b, on=keys + ["year"], how="outer").merge(ba, on=keys + ["year"], how="outer")
        out = _complete_years(out, keys, cfg.study_years, ["n_hw", "n_before", "n_after"])
        # eligible units of the group (constant denominator for per-PSU counts)
        n_el = (elig.groupby(keys, observed=True)[unit].nunique().rename("n_units_eligible")
                .reset_index())
        out = out.merge(n_el, on=keys, how="left")
        _write(out, agg_dir, name)

    by_year(["scale"], "axis2_by_year_continental")
    by_year(["region"], "axis2_by_year_region")
    by_year(["country"], "axis2_by_year_country")
    by_year(["scale"], "axis2_by_year_continental_cellcollapse", collapse=True)
    by_year(["region"], "axis2_by_year_region_cellcollapse", collapse=True)

    # ---- onset month x region (seasonality; Figs S5b, S10) -----------------------
    def om(keys):
        r = (a2.groupby(keys + ["onset_month"], observed=True)
               .agg(n_hw=("hw_id", "size"), precursor_rate=("precursor", "mean"),
                    trigger_rate=("trigger", "mean")).reset_index())
        return r.merge(_ba_counts(att, keys + ["onset_month"]), on=keys + ["onset_month"], how="left")
    _write(_with_continental(om(["region"]), om(["scale"])), agg_dir, "axis2_by_onset_month_region")

    # ---- per PSU x decade ------------------------------------------------------
    pdc = (a2.groupby(["psu_idx", "decade"], observed=False)
             .agg(n_hw=("hw_id", "size"),
                  precursor_rate=("precursor", "mean"),
                  trigger_rate=("trigger", "mean")).reset_index())
    pdc["rate_diff"] = pdc["trigger_rate"] - pdc["precursor_rate"]
    pdc = pdc.merge(_ba_counts(att, ["psu_idx", "decade"]), on=["psu_idx", "decade"], how="left")
    for c in ("n_before", "n_after"):
        pdc[c] = pdc[c].fillna(0).astype("int32")
    _write(pdc, agg_dir, "axis2_by_psu_decade")

    # ---- region x decade (Fig. 4c, S9) -------------------------------------------
    def rd(keys):
        r = (a2.groupby(keys + ["decade"], observed=True)
               .agg(n_hw=("hw_id", "size"), precursor_rate=("precursor", "mean"),
                    trigger_rate=("trigger", "mean")).reset_index())
        r["rate_diff"] = r["trigger_rate"] - r["precursor_rate"]
        return r.merge(_ba_counts(att, keys + ["decade"]), on=keys + ["decade"], how="left")
    _write(_with_continental(rd(["region"]), rd(["scale"])), agg_dir, "axis2_by_region_decade")


# =============================================================================
# AXIS III — antecedent drought (all lags)
# =============================================================================
def axis3_aggregations(cfg, in_dir, agg_dir, psu, hw):
    a3 = hw.copy()
    lags = tuple(cfg.axis3_lags)
    sens_lags = [L for L in lags[1:]]                   # non-primary lags (e.g. 0)
    a3["dpc"] = a3["drought_precond"].astype(float)
    a3["dpc_spi"] = a3["drought_precond_spi"].astype(float)
    for L in sens_lags:
        a3[f"dpc_lag{L}"] = a3[f"drought_precond_lag{L}"].astype(float)
    # axis3.parquet's year/month refer to the LAGGED antecedent month used
    # for the SPEI merge — drop them and aggregate by heatwave-onset year.
    a3 = a3.drop(columns=["year", "month"], errors="ignore")
    a3 = a3.rename(columns={"hw_year": "year"})
    a3["onset_month"] = a3["start_date"].dt.month.astype("int8")

    def per_psu(df, extra):
        g = df.groupby(["psu_idx"] + extra, observed=False)
        out = g.agg(n_hw=("hw_id", "size")).reset_index()
        s = df[df["has_spei"]]
        out = out.merge(
            s.groupby(["psu_idx"] + extra, observed=False)
             .agg(n_hw_spei=("hw_id", "size"), pct_drought_psu=("dpc", "mean"))
             .reset_index(), on=["psu_idx"] + extra, how="left")
        p = df[df["has_spi"]]
        out = out.merge(
            p.groupby(["psu_idx"] + extra, observed=False)
             .agg(pct_drought_spi_psu=("dpc_spi", "mean")).reset_index(),
            on=["psu_idx"] + extra, how="left")
        out["pct_drought_psu"] *= 100
        out["pct_drought_spi_psu"] *= 100
        for L in sens_lags:
            q = df[df[f"has_spei_lag{L}"]]
            out = out.merge(
                q.groupby(["psu_idx"] + extra, observed=False)
                 .agg(**{f"pct_drought_lag{L}_psu": (f"dpc_lag{L}", "mean")}).reset_index(),
                on=["psu_idx"] + extra, how="left")
            out[f"pct_drought_lag{L}_psu"] *= 100
        out["n_hw_spei"] = out["n_hw_spei"].fillna(0).astype("int32")
        return out

    def method_a(df, keys):
        s = df[df["has_spei"]]
        a = (df.groupby(keys, observed=True)
               .agg(n_hw=("hw_id", "size")).reset_index())
        a = a.merge(
            s.groupby(keys, observed=True)
             .agg(n_hw_spei=("hw_id", "size"),
                  n_hw_drought=("drought_precond", "sum")).reset_index(),
            on=keys, how="left")
        a["n_hw_nodrought"] = a["n_hw_spei"] - a["n_hw_drought"]
        a["pct_drought_a"] = safe_ratio(a["n_hw_drought"], a["n_hw_spei"]) * 100
        for L in sens_lags:
            q = df[df[f"has_spei_lag{L}"]]
            a = a.merge(q.groupby(keys, observed=True)
                         .agg(**{f"n_hw_spei_lag{L}": ("hw_id", "size"),
                                 f"n_hw_drought_lag{L}": (f"drought_precond_lag{L}", "sum")})
                         .reset_index(), on=keys, how="left")
            a[f"pct_drought_lag{L}_a"] = safe_ratio(a[f"n_hw_drought_lag{L}"],
                                                    a[f"n_hw_spei_lag{L}"]) * 100
        # severity contrasts (drought vs no-drought HW)
        d = s[s["drought_precond"]]
        n = s[~s["drought_precond"]]
        a = a.merge(d.groupby(keys, observed=True)
                     .agg(dur_drought=("duration", "mean"),
                          peak_drought=("peak_tmax", "mean")).reset_index(),
                    on=keys, how="left")
        a = a.merge(n.groupby(keys, observed=True)
                     .agg(dur_nodrought=("duration", "mean"),
                          peak_nodrought=("peak_tmax", "mean")).reset_index(),
                    on=keys, how="left")
        a["diff_duration"] = a["dur_drought"] - a["dur_nodrought"]
        a["diff_peak"] = a["peak_drought"] - a["peak_nodrought"]
        return a

    pct_cols = ["pct_drought_psu", "pct_drought_spi_psu"] + \
               [f"pct_drought_lag{L}_psu" for L in sens_lags]

    # ---- per PSU pooled ----------------------------------------------------
    _write(per_psu(a3, []).merge(psu, on="psu_idx"), agg_dir, "axis3_by_psu")

    # ---- by year x scale ------------------------------------------------------
    def by_year(keys, name, collapse=False):
        a = method_a(a3, keys + ["year"])
        pp = per_psu(a3, ["year"]).merge(psu, on="psu_idx", how="left")
        unit = "psu_idx"
        if collapse:
            unit = "era5_cell"
            pp = (pp.groupby(["era5_cell", "year"], observed=True)
                    .agg(**{c: (c, "mean") for c in pct_cols},
                         country=("country", "first"),
                         region=("region", "first"),
                         scale=("scale", "first")).reset_index())
        b = (pp.groupby(keys + ["year"], observed=True)
               .agg(**{c.replace("_psu", "_b"): (c, "mean") for c in pct_cols},
                    n_units=(unit, "nunique"))
               .reset_index())
        out = _complete_years(a.merge(b, on=keys + ["year"], how="outer"),
                              keys, cfg.study_years,
                              ["n_hw", "n_hw_spei", "n_hw_drought", "n_hw_nodrought"])
        _write(out, agg_dir, name)

    by_year(["scale"], "axis3_by_year_continental")
    by_year(["region"], "axis3_by_year_region")
    by_year(["country"], "axis3_by_year_country")
    by_year(["scale"], "axis3_by_year_continental_cellcollapse", collapse=True)
    by_year(["region"], "axis3_by_year_region_cellcollapse", collapse=True)

    # ---- decade tables ---------------------------------------------------------
    _write(per_psu(a3, ["decade"]), agg_dir, "axis3_by_psu_decade")

    def rd(keys):
        a = method_a(a3, keys + ["decade"])
        pp = per_psu(a3, ["decade"]).merge(psu, on="psu_idx", how="left")
        b = (pp.groupby(keys + ["decade"], observed=False)
               .agg(pct_drought_b=("pct_drought_psu", "mean")).reset_index())
        return a.merge(b, on=keys + ["decade"], how="outer")
    _write(_with_continental(rd(["region"]), rd(["scale"])), agg_dir, "axis3_by_region_decade")

    # ---- onset month x region (Figs S5c,d) ---------------------------------------
    def om(keys):
        r = method_a(a3, keys + ["onset_month"])
        n_el = psu.groupby(keys, observed=True)["psu_idx"].nunique().rename("n_units").reset_index()
        r = r.merge(n_el, on=keys, how="left")
        r["hw_per_unit_year"] = r["n_hw"] / (r["n_units"] * len(cfg.study_years))
        return r
    _write(_with_continental(om(["region"]), om(["scale"])), agg_dir, "axis3_by_onset_month_region")


def run_step08_aggregate(cfg, in_dir: Path | None = None,
         out_dir: Path | None = None) -> None:
    in_dir = Path(in_dir or cfg.paths.processed)
    out_root = Path(out_dir or cfg.paths.processed)
    agg_dir = out_root / "aggregations"
    agg_dir.mkdir(parents=True, exist_ok=True)

    psu, hw = _scaffold(cfg, in_dir)
    log.info("[Axis I]")
    axis1_aggregations(cfg, in_dir, agg_dir, psu, hw)
    log.info("[Axis II]")
    axis2_aggregations(cfg, in_dir, agg_dir, psu, hw)
    log.info("[Axis III]")
    axis3_aggregations(cfg, in_dir, agg_dir, psu, hw)
    log.info("All aggregations written to %s", agg_dir)

for _mode in cfg.hw_threshold_modes:
    _D = set_mode(_mode)
    run_step08_aggregate(cfg, in_dir=_D, out_dir=_D)

# %%
for _mode, _D in MODE_DIRS.items():
    print(f"\n===== hw_threshold_mode = {_mode} =====")
    # ---- checks: section 13 -------------------------------------------------------
    AGG = _D / "aggregations"
    import os
    files = sorted(p.name for p in AGG.glob("*.parquet"))
    print(f"{len(files)} aggregation files:"); print("\n".join(files))
    a1c = pd.read_parquet(AGG / "axis1_by_year_continental.parquet")
    a2c = pd.read_parquet(AGG / "axis2_by_year_continental.parquet")
    a3c = pd.read_parquet(AGG / "axis3_by_year_continental.parquet")
    assert len(a1c) == len(cfg.study_years) and len(a2c) == len(cfg.study_years)
    print("\nAxis I continental (first/last 3 years):")
    print(pd.concat([a1c.head(3), a1c.tail(3)])[
        ["year", "n_hw", "n_cooccur_days", "pct_hw_with_cooc_a",
         "pct_hw_with_cooc_b"]].to_string(index=False))
    print("\nAxis II continental (first/last 3):")
    print(pd.concat([a2c.head(3), a2c.tail(3)])[
        ["year", "n_hw", "precursor_rate_b", "trigger_rate_b",
         "rate_diff_b"]].round(4).to_string(index=False))
    print("\nAxis III continental (first/last 3):")
    print(pd.concat([a3c.head(3), a3c.tail(3)])[
        ["year", "n_hw", "n_hw_spei", "pct_drought_a",
         "pct_drought_b"]].round(2).to_string(index=False))


# %% [markdown]
# ## 14. Trends (Theil–Sen + Hamed–Rao MK + block bootstrap + BH-FDR)
#
# step09_trends.py — Trends on the annual aggregated series.
#
# For every (scale, metric) pair:
#   - Theil-Sen slope, reported per decade (Sen 1968; Theil 1950);
#   - Mann-Kendall with the Hamed & Rao (1998) autocorrelation-corrected
#     variance (the classical MK was previously cited but not implemented —
#     this version actually implements it; see core.mann_kendall_hamed_rao);
#   - 95% CI on the slope from a MOVING-BLOCK bootstrap (block length
#     cfg.block_length_years), which preserves serial dependence that the
#     i.i.d. bootstrap destroys;
#   - at the COUNTRY level, Benjamini-Hochberg FDR across the 34 countries,
#     per metric (column `mk_significant_fdr`).
#
# Metrics are the method-(b) primaries with method-(a) twins for the SI,
# plus the cell-collapse series.

# %%
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "axis1": ["cooccur_days_per_unit", "pct_hw_with_cooc_b",
              "n_cooccur_days", "pct_hw_with_cooc_a"],
    "axis2": ["count_ratio_a",                       # PRIMARY (unique attribution)
              "precursor_rate_b", "trigger_rate_b", "rate_diff_b",
              "precursor_rate_a", "trigger_rate_a", "rate_diff_a"],
    "axis3": ["pct_drought_b", "pct_drought_spi_b", "pct_drought_a",
              "pct_drought_lag0_b", "pct_drought_lag0_a",   # lag-0 comparison
              "diff_duration", "diff_peak"],
}
SCOPES = [("continental", None), ("region", "region"), ("country", "country"),
          ("continental_cellcollapse", None), ("region_cellcollapse", "region")]


def _trend_rows(df: pd.DataFrame, scale_col, metrics, rng,
                cfg) -> list[dict]:
    rows = []
    groups = ([("Continental", df)] if scale_col is None
              else [(str(k), g) for k, g in df.groupby(scale_col,
                                                       observed=True)])
    for label, sub in groups:
        sub = sub.sort_values("year")
        x = sub["year"].to_numpy(float)
        for m in metrics:
            if m not in sub.columns:
                continue
            y = sub[m].to_numpy(float)
            slope, lo, hi = moving_block_bootstrap_slope_ci(
                x, y, cfg.n_bootstrap, cfg.block_length_years, rng)
            z, p = mann_kendall_hamed_rao(x, y)
            rows.append(dict(
                scale=label, metric=m,
                slope_per_decade=slope * 10 if np.isfinite(slope) else np.nan,
                ci_low_per_decade=lo * 10 if np.isfinite(lo) else np.nan,
                ci_high_per_decade=hi * 10 if np.isfinite(hi) else np.nan,
                mk_z=z, mk_p=p,
                n_years=int(np.isfinite(y).sum())))
    return rows


def run_step09_trends(cfg, in_dir: Path | None = None,
         out_dir: Path | None = None) -> None:
    in_root = Path(in_dir or cfg.paths.processed)
    agg_dir = in_root / "aggregations"
    out = Path(out_dir or cfg.paths.processed) / "trends"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.rng_seed)

    all_rows = []
    for axis, metrics in METRICS.items():
        for scope, scale_col in SCOPES:
            f = agg_dir / f"{axis}_by_year_{scope}.parquet"
            if not f.exists():
                log.warning("missing %s — skipped", f.name)
                continue
            df = pd.read_parquet(f)
            rows = _trend_rows(df, scale_col, metrics, rng, cfg)
            for r in rows:
                r["axis"] = axis
                r["scope"] = scope
            all_rows += rows
            log.info("%s / %s: %d trend rows", axis, scope, len(rows))

    trends = pd.DataFrame(all_rows)

    # ---- FDR across countries, per (axis, metric) -------------------------
    trends["mk_significant_fdr"] = False
    cmask = trends["scope"] == "country"
    for (axis, m), idx in trends[cmask].groupby(["axis", "metric"]).groups.items():
        rej = benjamini_hochberg(trends.loc[idx, "mk_p"].to_numpy(),
                                 cfg.fdr_alpha)
        trends.loc[idx, "mk_significant_fdr"] = rej
    trends.loc[~cmask, "mk_significant_fdr"] = (
        trends.loc[~cmask, "mk_p"] < cfg.fdr_alpha)

    trends = trends[["axis", "scope", "scale", "metric", "slope_per_decade",
                     "ci_low_per_decade", "ci_high_per_decade", "mk_z",
                     "mk_p", "mk_significant_fdr", "n_years"]]
    trends.to_parquet(out / "all_trends_summary.parquet", index=False)
    log.info("Wrote all_trends_summary.parquet (%s rows)", f"{len(trends):,}")

    # quick console digest of the headline series
    head = trends[(trends["scope"] == "continental")
                  & trends["metric"].isin(["cooccur_days_per_unit", "count_ratio_a",
                                           "rate_diff_b", "pct_drought_b"])]
    if len(head):
        log.info("Headline continental trends (per decade):\n%s",
                 head[["axis", "metric", "slope_per_decade",
                       "ci_low_per_decade", "ci_high_per_decade",
                       "mk_p"]].to_string(index=False))

for _mode in cfg.hw_threshold_modes:
    _D = set_mode(_mode)
    run_step09_trends(cfg, in_dir=_D, out_dir=_D)

# %% [markdown]
# ## 15. Global diagnostic summary

# %%
for _mode, _D in MODE_DIRS.items():
    print(f"\n===== hw_threshold_mode = {_mode} =====")
    print("=" * 72)
    print("PIPELINE DIAGNOSTIC SUMMARY")
    print("=" * 72)
    psu_df = pd.read_parquet(PROCESSED / "psu.parquet")
    gm = pd.read_parquet(PROCESSED / "psu_gridmatch.parquet")
    hw = pd.read_parquet(_D / "heatwaves.parquet")
    epe = pd.read_parquet(_D / "epe.parquet")
    a2 = pd.read_parquet(_D / "axis2_hw.parquet")
    a3 = pd.read_parquet(_D / "axis3.parquet")
    print(f"PSU {len(psu_df):,} | CHIRPS-eligible {int(gm.chirps_valid.sum()):,}"
          f" | unique ERA5 cells {gm.era5_cell.nunique():,} "
          f"(pseudo-replication ratio {len(gm)/gm.era5_cell.nunique():.1f} PSU/cell)")
    print(f"Heatwaves {len(hw):,} | EPE days {len(epe):,} | "
          f"truncated-window HW excluded from Axis II: {int(a2.truncated.sum()):,}")
    print(f"Axis III: HW with antecedent SPEI {int(a3.has_spei.sum()):,} "
          f"({a3.has_spei.mean():.1%})")
    tr = pd.read_parquet(_D / "trends" / "all_trends_summary.parquet")
    print("\nHeadline continental trends (per decade, block-bootstrap 95% CI):")
    print(tr[(tr.scope == "continental") & tr.metric.isin(
        ["cooccur_days_per_unit", "pct_hw_with_cooc_b", "count_ratio_a", "rate_diff_b",
         "pct_drought_b", "pct_drought_lag0_b", "pct_drought_spi_b"])][
        ["axis", "metric", "slope_per_decade", "ci_low_per_decade",
         "ci_high_per_decade", "mk_p"]].round(4).to_string(index=False))
    n_sig = tr[(tr.scope == "country")].groupby("axis").mk_significant_fdr.sum()
    print("\nCountry-level trends significant after BH-FDR, by axis:")
    print(n_sig)


# %% [markdown]
# ## 16. Figures — all main, SI, null-test and sensitivity figures
#
# Every figure reads the parquet outputs of the run (`PROCESSED/<mode>/aggregations`, `significance`, `trends`, event tables) and is produced **once per threshold mode** into `PROCESSED/<mode>/figures/`. Cross-mode figures (calendar vs annual) and the shared drought baseline go to `PROCESSED/figures_shared/`.
#
# | Cell | Figures |
# |---|---|
# | 16.0 | helpers (style, maps, loaders) |
# | 16.1 | Fig 1 maps, Fig 3 decade maps |
# | 16.2 | Fig 2 annual series + Theil–Sen (from step 14) |
# | 16.3 | Fig 4 Axis-II asymmetry, Fig 5 Axis-III amplification |
# | 16.4 | Fig 6 regimes (q75 primary, tercile sensitivity) with first/last decade panels and transitions |
# | 16.5 | S2–S4 geo panels, S3b/S4b paired |
# | 16.6 | S5 seasonality, S6 comparative, S8 latitude, S9 decadal, S10 monthly stratification, Table S1 |
# | 16.7 | null test (primary + secondary metric, seasonal), sensitivity (W, SPEI threshold, lag, SPI) |
# | 16.8 | calendar vs annual comparison, drought baseline (shared) |
#
# Axis II everywhere = **after/before EPE count ratio under unique attribution** (primary); ECA fractions are shown as secondary where relevant. Axis III everywhere = antecedent drought at **lag 1** (primary); lag 0 appears in the sensitivity figure and trend tables.

# %%
# ---- 16.0 common figure helpers ---------------------------------------------
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    log.warning("cartopy not available — maps drawn without coastlines/borders")
try:
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 300, "legend.frameon": False})

REGIONS = ["Western", "Eastern", "Central", "Southern"]
SCALES = ["Continental"] + REGIONS
REGION_COLORS = {"Continental": "#333333", "Western": "#D95F02", "Eastern": "#1B9E77",
                 "Central": "#7570B3", "Southern": "#E7298A"}
DECADES = [f"{a}-{b}" for a, b in cfg.decade_bounds]
DECADE_COLORS = dict(zip(DECADES, ["#4575B4", "#91BFDB", "#FC8D59", "#D73027"]))
FIRST_DEC, LAST_DEC = DECADES[0], DECADES[-1]
YEARS = np.array(cfg.study_years)
N_YEARS = len(YEARS)
AFRICA_EXTENT = [-19, 52, -36, 26]
MONTH_LABELS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
AXIS_NAMES = {"axis1": "HW–EPE compound events", "axis2": "EPE asymmetry around HW",
              "axis3": "HW during antecedent drought"}
PSU_GEO = pd.read_parquet(PROCESSED / "psu.parquet",
                          columns=["psu_idx", "country", "region", "lat", "lon"])


def agg(D, name):    return pd.read_parquet(D / "aggregations" / f"{name}.parquet")
def sig(D, name):    return pd.read_parquet(D / "significance" / f"{name}.parquet")
def trends(D):       return pd.read_parquet(D / "trends" / "all_trends_summary.parquet")


def savefig(fig, FIG, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("figure -> %s.png/.pdf", FIG / name)


def panel_label(ax, s, x=-0.08, y=1.04, size=12):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=size, fontweight="bold",
            va="bottom", ha="left")


def map_ax(fig, spec):
    if HAS_CARTOPY:
        ax = fig.add_subplot(spec, projection=ccrs.PlateCarree())
        ax.set_extent(AFRICA_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor="#f2f5f9", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f7f6f4", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, color="#888888", zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.15, color="#bbbbbb", zorder=1)
        for s in ax.spines.values():
            s.set_visible(False)
    else:
        ax = fig.add_subplot(spec)
        ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1]); ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
        ax.set_aspect("equal"); ax.set_facecolor("#f4f6f9")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    return ax


def map_scatter(ax, df, col=None, c=None, cmap=None, norm=None, s=0.6, **kw):
    kw.setdefault("edgecolors", "none"); kw.setdefault("rasterized", True); kw.setdefault("zorder", 4)
    if HAS_CARTOPY:
        kw["transform"] = ccrs.PlateCarree()
    if col is not None:
        df = df.dropna(subset=[col])
        o = np.argsort(df[col].to_numpy())
        return ax.scatter(df["lon"].to_numpy()[o], df["lat"].to_numpy()[o],
                          c=df[col].to_numpy()[o], cmap=cmap, norm=norm, s=s, **kw)
    return ax.scatter(df["lon"], df["lat"], c=c, s=s, **kw)


def smooth(y, frac=0.3):
    y = np.asarray(y, float); m = np.isfinite(y)
    if m.sum() < 5:
        return np.full_like(y, np.nan)
    if HAS_SM:
        s = sm.nonparametric.lowess(y[m], YEARS[m], frac=frac)
        return np.interp(YEARS, s[:, 0], s[:, 1])
    return pd.Series(y).rolling(5, center=True, min_periods=3).mean().to_numpy()


def theilsen_line(y):
    y = np.asarray(y, float); m = np.isfinite(y)
    if m.sum() < 10:
        return None
    slope, intercept, lo, hi = stats.theilslopes(y[m], YEARS[m].astype(float))
    return slope, intercept, lo, hi


def slope_row(tr, axis, scope, scale, metric):
    r = tr[(tr["axis"] == axis) & (tr["scope"] == scope) & (tr["scale"] == scale)
           & (tr["metric"] == metric)]
    return None if r.empty else r.iloc[0]


def slope_text(tr, axis, scope, scale, metric, fmt="{:+.2f}"):
    r = slope_row(tr, axis, scope, scale, metric)
    if r is None or not np.isfinite(r["slope_per_decade"]):
        return ""
    star = "*" if bool(r["mk_significant_fdr"]) else ""
    return (fmt.format(r["slope_per_decade"]) + f"/dec{star}\n[" +
            fmt.format(r["ci_low_per_decade"]) + ", " + fmt.format(r["ci_high_per_decade"]) + "]")


def series_by_scale(D, axis, metric):
    """{scale: array over YEARS} from *_by_year_continental / *_by_year_region."""
    out = {}
    c = agg(D, f"{axis}_by_year_continental").sort_values("year")
    out["Continental"] = c.set_index("year")[metric].reindex(YEARS).to_numpy(float)
    r = agg(D, f"{axis}_by_year_region")
    for reg in REGIONS:
        out[reg] = (r[r["region"] == reg].set_index("year")[metric]
                    .reindex(YEARS).to_numpy(float))
    return out


def series_by_country(D, axis, metric):
    r = agg(D, f"{axis}_by_year_country")
    return {c: g.set_index("year")[metric].reindex(YEARS).to_numpy(float)
            for c, g in r.groupby("country")}


def psu_metric_table(D):
    """One row per PSU with the three primary axis metrics (+ geo)."""
    a1 = agg(D, "axis1_by_psu")[["psu_idx", "cooccur_days_per_year", "n_cooccur_days"]]
    a2 = agg(D, "axis2_by_psu")[["psu_idx", "count_ratio", "n_before", "n_after", "rate_diff"]]
    a3 = agg(D, "axis3_by_psu")[["psu_idx", "pct_drought_psu", "n_hw", "n_hw_spei"]]
    return (PSU_GEO.merge(a1, on="psu_idx", how="left").merge(a2, on="psu_idx", how="left")
            .merge(a3, on="psu_idx", how="left"))


AXIS_PSU_COLS = {"axis1": "cooccur_days_per_year", "axis2": "count_ratio", "axis3": "pct_drought_psu"}
AXIS_PSU_LABELS = {"axis1": "HW–EPE co-occurrence days per PSU per year",
                   "axis2": "EPE after / before HW (unique attribution)",
                   "axis3": "% HW with antecedent drought (SPEI-3 < −1, m−1)"}
print("figure helpers ready | cartopy:", HAS_CARTOPY, "| statsmodels:", HAS_SM)


# %%
# ---- 16.1 Figures 1 & 3: spatial distribution (pooled) and by decade ---------
MAP_CFG = {
    "axis1": dict(cmap="YlOrRd", norm=None, extend="max"),
    "axis2": dict(cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=6), extend="both"),
    "axis3": dict(cmap="YlOrBr", norm=mcolors.Normalize(vmin=0, vmax=80), extend="max"),
}


def _norm_for(axis, values):
    n = MAP_CFG[axis]["norm"]
    if n is None:
        v = values[np.isfinite(values) & (values > 0)]
        n = mcolors.Normalize(vmin=0, vmax=np.percentile(v, 95) if v.size else 1)
    return n


def fig01_spatial(D, FIG):
    df = psu_metric_table(D)
    fig = plt.figure(figsize=(14, 5.2), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.05], wspace=0.05, hspace=0.05)
    for j, (axis, lab) in enumerate(zip(("axis1", "axis2", "axis3"), "abc")):
        col = AXIS_PSU_COLS[axis]
        ax = map_ax(fig, gs[0, j])
        sc = map_scatter(ax, df, col, cmap=MAP_CFG[axis]["cmap"], norm=_norm_for(axis, df[col].to_numpy()))
        ax.set_title(AXIS_NAMES[axis], fontsize=9, fontweight="bold", pad=6)
        panel_label(ax, lab, x=0.0, y=1.02)
        cb = fig.colorbar(sc, cax=fig.add_subplot(gs[1, j]), orientation="horizontal",
                          extend=MAP_CFG[axis]["extend"])
        cb.set_label(AXIS_PSU_LABELS[axis], fontsize=7)
        cb.ax.tick_params(labelsize=6)
    savefig(fig, FIG, "fig01_spatial_distribution")


def fig03_decades(D, FIG):
    tabs = {"axis1": agg(D, "axis1_by_psu_decade"), "axis2": agg(D, "axis2_by_psu_decade"),
            "axis3": agg(D, "axis3_by_psu_decade")}
    fig = plt.figure(figsize=(14, 12.5), facecolor="white")
    gs = GridSpec(6, 4, figure=fig, wspace=0.03, hspace=0.08,
                  height_ratios=[1, 0.05, 1, 0.05, 1, 0.05])
    for i, (axis, lab) in enumerate(zip(("axis1", "axis2", "axis3"), "abc")):
        col = AXIS_PSU_COLS[axis]
        t = tabs[axis].copy(); t["decade"] = t["decade"].astype(str)
        t = t.merge(PSU_GEO, on="psu_idx", how="left")
        norm = _norm_for(axis, t[col].to_numpy())
        sc = None
        for j, dec in enumerate(DECADES):
            ax = map_ax(fig, gs[2 * i, j])
            sub = t[t["decade"] == dec]
            if len(sub.dropna(subset=[col])):
                sc = map_scatter(ax, sub, col, cmap=MAP_CFG[axis]["cmap"], norm=norm)
            if i == 0:
                ax.set_title(dec.replace("-", "–"), fontsize=9, fontweight="bold", pad=6)
            if j == 0:
                ax.text(-0.04, 0.5, AXIS_NAMES[axis], transform=ax.transAxes, rotation=90,
                        fontsize=7.5, fontweight="bold", va="center", ha="right")
                panel_label(ax, lab, x=-0.08, y=1.02)
        if sc is not None:
            pos = gs[2 * i + 1, :].get_position(fig)
            cax = fig.add_axes([pos.x0 + pos.width * .15, pos.y0 + pos.height * .3,
                                pos.width * .7, pos.height * .4])
            cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend=MAP_CFG[axis]["extend"])
            cb.set_label(AXIS_PSU_LABELS[axis] + " (per decade)", fontsize=7)
            cb.ax.tick_params(labelsize=6)
    savefig(fig, FIG, "fig03_spatiotemporal_decades")


for _mode, _D in MODE_DIRS.items():
    fig01_spatial(_D, FIG_DIRS[_mode])
    fig03_decades(_D, FIG_DIRS[_mode])


# %%
# ---- 16.2 Figure 2: temporal trends (annual series + Theil–Sen from step09) ----
FIG2_ROWS = [
    ("axis1", "cooccur_days_per_unit", "Co-occurrence days / PSU / year", "HW–EPE compound events", "{:+.3f}"),
    ("axis2", "count_ratio_a", "EPE after / before HW", "EPE asymmetry (count ratio, PRIMARY)", "{:+.2f}"),
    ("axis2", "rate_diff_b", "Trigger − precursor fraction", "EPE asymmetry (ECA fractions, secondary)", "{:+.3f}"),
    ("axis3", "pct_drought_b", "% of HW", "HW with antecedent drought (lag 1)", "{:+.2f}"),
]


def fig02_trends(D, FIG):
    tr = trends(D)
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    gs = GridSpec(len(FIG2_ROWS), 5, figure=fig, hspace=0.5, wspace=0.35)
    for i, (axis, metric, ylab, title, fmt) in enumerate(FIG2_ROWS):
        ser = series_by_scale(D, axis, metric)
        allv = np.concatenate([v[np.isfinite(v)] for v in ser.values()])
        ymax = np.percentile(allv, 99) * 1.1 if allv.size else 1
        ymin = min(0, np.nanmin(allv) * 1.1) if allv.size else 0
        for j, sc in enumerate(SCALES):
            ax = fig.add_subplot(gs[i, j]); col = REGION_COLORS[sc]; y = ser[sc]; m = np.isfinite(y)
            ax.scatter(YEARS[m], y[m], s=6, color=col, alpha=.5, edgecolors="none", zorder=3)
            ax.plot(YEARS, smooth(y), color=col, lw=1.5, zorder=4)
            ts = theilsen_line(y)
            if ts is not None:
                ax.plot(YEARS, ts[0] * YEARS + ts[1], color=col, lw=.7, alpha=.55, zorder=2)
            txt = slope_text(tr, axis, "continental" if sc == "Continental" else "region", sc, metric, fmt)
            ax.text(.97, .93, txt, transform=ax.transAxes, fontsize=5.5, ha="right", va="top", color="#555")
            ax.set_ylim(ymin, ymax); ax.set_xlim(YEARS[0] - 1, YEARS[-1] + 1)
            ax.yaxis.grid(True, lw=.2, color="#ddd", zorder=0)
            if i == 0:
                ax.text(.5, 1.18, sc, transform=ax.transAxes, fontsize=8, fontweight="bold",
                        ha="center", va="bottom", color=col)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=6.5)
                ax.set_title(title, fontsize=7.5, fontweight="bold", pad=4, loc="left")
                panel_label(ax, "abcd"[i], x=-0.32, y=1.08, size=13)
    fig.text(.01, -.01, "Slopes: Theil–Sen per decade, 95% moving-block-bootstrap CI (step 14); "
             "* = Mann–Kendall (Hamed–Rao) p < 0.05.", fontsize=6, color="#888", style="italic")
    savefig(fig, FIG, "fig02_temporal_trends")


for _mode, _D in MODE_DIRS.items():
    fig02_trends(_D, FIG_DIRS[_mode])


# %%
# ---- 16.3 Figures 4 & 5: Axis II asymmetry deep dive, Axis III amplification --
SHORT_NAMES = {"Burkina Faso": "Burkina F.", "Central African Republic": "CAR",
               "Congo Democratic Republic": "DR Congo", "Cote d'Ivoire": "Côte d'Iv.",
               "Sierra Leone": "Sierra L.", "South Africa": "S. Africa"}


def fig04_asymmetry(D, FIG):
    W = cfg.axis2_window_days
    bc = agg(D, "axis2_by_country")
    att = pd.read_parquet(D / "axis2_attrib.parquet")
    att = att[~att["truncated"] & att["position"].isin(["before", "after"])].copy()
    att["decade"] = assign_decade(pd.to_datetime(att["start_date"]).dt.year, cfg.decade_bounds).astype(str)
    rd = agg(D, "axis2_by_region_decade"); rd["decade"] = rd["decade"].astype(str)

    fig = plt.figure(figsize=(14, 5), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    # (a) after vs before per PSU, per country
    ax = fig.add_subplot(gs[0, 0])
    mx = max(bc["before_per_psu"].max(), bc["after_per_psu"].max()) * 1.1
    ax.plot([0, mx], [0, mx], "k--", lw=.5, alpha=.4, label="1:1 (symmetry)")
    for reg in REGIONS:
        s = bc[bc["region"] == reg]
        ax.scatter(s["before_per_psu"], s["after_per_psu"], c=REGION_COLORS[reg], s=30,
                   alpha=.85, edgecolors="white", lw=.3, zorder=3, label=f"{reg} Africa")
    for _, r in bc.iterrows():
        ax.text(r["before_per_psu"], r["after_per_psu"], SHORT_NAMES.get(r["country"], r["country"]),
                fontsize=4.5, color=REGION_COLORS.get(r["region"], "#555"), alpha=.9)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("EPE before HW (count / PSU)"); ax.set_ylabel("EPE after HW (count / PSU)")
    ax.set_title("EPE before vs. after heatwaves (unique attribution)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=5.5, loc="upper left"); panel_label(ax, "a")
    # (b) timing distribution by decade
    ax = fig.add_subplot(gs[0, 1])
    bins = np.arange(-W - .5, W + 1.5, 1)
    for dec in DECADES:
        s = att.loc[att["decade"] == dec, "offset_days"]
        if len(s):
            cnt, edges = np.histogram(s, bins=bins, density=True)
            ax.plot((edges[:-1] + edges[1:]) / 2, cnt, color=DECADE_COLORS[dec], lw=1.2, label=dec.replace("-", "–"))
    ax.axvline(0, color="#888", lw=.5)
    ax.set_xlabel("Days from nearest heatwave boundary"); ax.set_ylabel("Density")
    ax.set_title("EPE timing relative to HW, by decade", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6); panel_label(ax, "b")
    # (c) count ratio region x decade
    ax = fig.add_subplot(gs[0, 2])
    M = np.full((len(SCALES), len(DECADES)), np.nan)
    for i, sc in enumerate(SCALES):
        for j, dec in enumerate(DECADES):
            r = rd[(rd["region"] == sc) & (rd["decade"] == dec)]
            if len(r):
                M[i, j] = r["count_ratio"].iloc[0]
    im = ax.imshow(M, cmap="OrRd", aspect="auto", vmin=1, vmax=max(2, np.nanpercentile(M, 95)))
    ax.set_xticks(range(len(DECADES))); ax.set_xticklabels([d.replace("-", "–") for d in DECADES], rotation=30, ha="right")
    ax.set_yticks(range(len(SCALES))); ax.set_yticklabels(SCALES)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7, fontweight="bold",
                        color="white" if M[i, j] > 0.75 * np.nanmax(M) else "#333")
    ax.set_title("After/before count ratio by region and decade", fontsize=9, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, shrink=.7, pad=.04, extend="both"); cb.set_label("Ratio", fontsize=7)
    panel_label(ax, "c", x=-0.12)
    savefig(fig, FIG, "fig04_axis2_asymmetry")


def _paired_box(ax, d0, d1, pos, width=.32):
    for d, dx, fc, mc in ((d0, -width / 2, "#b3cde3", "#2166ac"), (d1, width / 2, "#e6550d", "#b2182b")):
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        bp = ax.boxplot([d], positions=[pos + dx], widths=width * .7, patch_artist=True,
                        showfliers=False, zorder=3, medianprops=dict(color="#333", lw=1),
                        whiskerprops=dict(lw=.5), capprops=dict(lw=.5))
        bp["boxes"][0].set(facecolor=fc, edgecolor="#333", lw=.5, alpha=.85)
        ax.scatter(pos + dx, d.mean(), marker="D", s=14, color=mc, edgecolor="white", lw=.4, zorder=5)


def fig05_drought(D, FIG):
    a3 = pd.read_parquet(D / "axis3.parquet").merge(PSU_GEO[["psu_idx", "region"]], on="psu_idx", how="left")
    a3 = a3[a3["has_spei"]]
    rd = agg(D, "axis3_by_region_decade"); rd["decade"] = rd["decade"].astype(str)
    fig = plt.figure(figsize=(14, 5), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    leg = [mpatches.Patch(fc="#b3cde3", ec="#333", lw=.5, label="No antecedent drought"),
           mpatches.Patch(fc="#e6550d", ec="#333", lw=.5, label="Antecedent drought (SPEI-3 < −1)"),
           plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#666", markersize=4, label="Mean")]
    for k, (col, ylab, title) in enumerate((("duration", "HW duration (days)", "Heatwave duration"),
                                           ("peak_tmax", "Peak Tmax (°C)", "Heatwave intensity"))):
        ax = fig.add_subplot(gs[0, k])
        for i, reg in enumerate(REGIONS):
            s = a3[a3["region"] == reg]
            _paired_box(ax, s.loc[~s["drought_precond"], col].to_numpy(float),
                        s.loc[s["drought_precond"], col].to_numpy(float), i)
        ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=9, fontweight="bold")
        ax.yaxis.grid(True, lw=.15, color="#e0e0e0", zorder=0)
        ax.legend(handles=leg, fontsize=6, loc="upper right"); panel_label(ax, "ab"[k])
    ax = fig.add_subplot(gs[0, 2])
    M = np.full((len(SCALES), len(DECADES)), np.nan)
    for i, sc in enumerate(SCALES):
        for j, dec in enumerate(DECADES):
            r = rd[(rd["region"] == sc) & (rd["decade"] == dec)]
            if len(r):
                M[i, j] = r["pct_drought_a"].iloc[0]
    im = ax.imshow(M, cmap="YlOrBr", aspect="auto", vmin=0, vmax=50)
    ax.set_xticks(range(len(DECADES))); ax.set_xticklabels([d.replace("-", "–") for d in DECADES], rotation=30, ha="right")
    ax.set_yticks(range(len(SCALES))); ax.set_yticklabels(SCALES)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7.5, fontweight="bold",
                        color="white" if M[i, j] > 35 else "#333")
    ax.set_title("% HW with antecedent drought, by region and decade", fontsize=9, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, shrink=.8, pad=.03, extend="max"); cb.set_label("% of HW", fontsize=7)
    panel_label(ax, "c", x=-0.15)
    savefig(fig, FIG, "fig05_drought_amplification")


for _mode, _D in MODE_DIRS.items():
    fig04_asymmetry(_D, FIG_DIRS[_mode])
    fig05_drought(_D, FIG_DIRS[_mode])


# %%
# ---- 16.4 Figure 6: regimes (single relative rule for the three axes) + first vs last decade
REGIME_ORDER = ["All three", "Post-HW precip. + Drought-heat", "Heat-precip. + Post-HW precip.",
                "Heat-precip. + Drought-heat", "Post-HW precip. only", "Drought-heat only",
                "Heat-precip. only", "None prominent"]
REGIME_COLORS = {"All three": "#2c3e50", "Post-HW precip. + Drought-heat": "#8e44ad",
                 "Heat-precip. + Post-HW precip.": "#e67e22", "Heat-precip. + Drought-heat": "#e74c3c",
                 "Post-HW precip. only": "#3498db", "Drought-heat only": "#1abc9c",
                 "Heat-precip. only": "#f39c12", "None prominent": "#bdc3c7"}


def regime_label(x1, x2, x3):
    if x1 and x2 and x3: return "All three"
    if x2 and x3:        return "Post-HW precip. + Drought-heat"
    if x1 and x2:        return "Heat-precip. + Post-HW precip."
    if x1 and x3:        return "Heat-precip. + Drought-heat"
    if x2:               return "Post-HW precip. only"
    if x3:               return "Drought-heat only"
    if x1:               return "Heat-precip. only"
    return "None prominent"


def classify_regimes(df, q, thresholds=None):
    """ONE relative rule for the three axes: prominent on an axis <=> value
    strictly above the q-quantile of the continental distribution of that axis.
    `thresholds` (dict axis->value) lets a reference period's thresholds be
    applied to another period (emergence). NaN metric -> not prominent."""
    cols = AXIS_PSU_COLS
    thr = thresholds or {a: df[c].quantile(q) for a, c in cols.items()}
    flags = {a: (df[c] > thr[a]).fillna(False).to_numpy() for a, c in cols.items()}
    lab = [regime_label(*t) for t in zip(flags["axis1"], flags["axis2"], flags["axis3"])]
    return pd.Series(lab, index=df.index), thr


def _regime_map(ax, df, title):
    for r in REGIME_ORDER[::-1]:
        s = df[df["regime"] == r]
        if len(s):
            map_scatter(ax, s, c=REGIME_COLORS[r], s=0.8)
    ax.set_title(title, fontsize=9, fontweight="bold")


def fig06_regimes(D, FIG, q, tag):
    pooled = psu_metric_table(D)
    pooled["regime"], thr = classify_regimes(pooled, q)
    dec = {}
    for a, name in (("axis1", "axis1_by_psu_decade"), ("axis2", "axis2_by_psu_decade"),
                    ("axis3", "axis3_by_psu_decade")):
        t = agg(D, name); t["decade"] = t["decade"].astype(str)
        dec[a] = t[["psu_idx", "decade", AXIS_PSU_COLS[a]]]
    by_dec = {}
    for d in DECADES:
        m = PSU_GEO.copy()
        for a in dec:
            m = m.merge(dec[a][dec[a]["decade"] == d].drop(columns="decade"), on="psu_idx", how="left")
        ref = thr if cfg.regime_threshold_reference == "pooled" else None
        m["regime"], _ = classify_regimes(m, q, ref)
        by_dec[d] = m
    shares = pd.DataFrame({d: by_dec[d]["regime"].value_counts(normalize=True).reindex(REGIME_ORDER).fillna(0) * 100
                           for d in DECADES})
    trans = pd.crosstab(by_dec[FIRST_DEC].set_index("psu_idx")["regime"],
                        by_dec[LAST_DEC].set_index("psu_idx")["regime"]).reindex(
        index=REGIME_ORDER, columns=REGIME_ORDER).fillna(0).astype(int)

    fig = plt.figure(figsize=(15, 12), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1.2, 1.2], hspace=0.25, wspace=0.2)
    # (a) shares, pooled
    ax = fig.add_subplot(gs[0, 0])
    counts = pooled["regime"].value_counts().reindex(REGIME_ORDER).fillna(0)
    y = np.arange(len(REGIME_ORDER))[::-1]
    ax.barh(y, counts.to_numpy(), color=[REGIME_COLORS[r] for r in REGIME_ORDER], height=.65)
    for yi, c in zip(y, counts.to_numpy()):
        ax.text(c + counts.max() * .01, yi, f"{100 * c / counts.sum():.1f}%", va="center", fontsize=7, color="#555")
    ax.set_yticks(y); ax.set_yticklabels(REGIME_ORDER, fontsize=7); ax.set_xlabel("Number of PSU")
    ax.set_xlim(0, counts.max() * 1.2); ax.set_title(f"Regimes, {YEARS[0]}–{YEARS[-1]}", fontsize=9, fontweight="bold")
    panel_label(ax, "a", x=-0.55)
    # (b) pooled map
    ax = map_ax(fig, gs[0, 1]); _regime_map(ax, pooled, f"Regimes, {YEARS[0]}–{YEARS[-1]}"); panel_label(ax, "b", x=0)
    ax.legend(handles=[mpatches.Patch(color=REGIME_COLORS[r], label=r) for r in REGIME_ORDER],
              loc="lower left", fontsize=6, frameon=True, framealpha=.9)
    # (c) shares by decade (stacked)
    ax = fig.add_subplot(gs[0, 2])
    bottom = np.zeros(len(DECADES))
    for r in REGIME_ORDER:
        v = shares.loc[r].to_numpy()
        ax.bar(range(len(DECADES)), v, bottom=bottom, color=REGIME_COLORS[r], width=.7, label=r)
        bottom += v
    ax.set_xticks(range(len(DECADES))); ax.set_xticklabels([d.replace("-", "–") for d in DECADES])
    ax.set_ylabel("% of PSU"); ax.set_ylim(0, 100)
    ax.set_title("Regime shares by decade", fontsize=9, fontweight="bold"); panel_label(ax, "c")
    # (d, e) first vs last decade maps
    for k, d in enumerate((FIRST_DEC, LAST_DEC)):
        ax = map_ax(fig, gs[1, k]); _regime_map(ax, by_dec[d], d.replace("-", "–")); panel_label(ax, "de"[k], x=0)
    # (f) transition matrix first -> last
    ax = fig.add_subplot(gs[1, 2])
    P = trans.to_numpy(float); P = 100 * P / max(1, P.sum())
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=max(1, P.max()))
    ax.set_xticks(range(len(REGIME_ORDER))); ax.set_xticklabels(REGIME_ORDER, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(REGIME_ORDER))); ax.set_yticklabels(REGIME_ORDER, fontsize=6)
    ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
    ax.set_xlabel(f"Regime in {LAST_DEC.replace('-', '–')}"); ax.set_ylabel(f"Regime in {FIRST_DEC.replace('-', '–')}")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] >= 0.5:
                ax.text(j, i, f"{P[i, j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if P[i, j] > .6 * P.max() else "#333")
    ax.set_title("Regime transitions, % of PSU (first → last decade)", fontsize=9, fontweight="bold")
    panel_label(ax, "f", x=-0.05)
    ref_txt = ("thresholds from the full-period distribution" if cfg.regime_threshold_reference == "pooled"
               else "thresholds recomputed within each decade")
    fig.suptitle(f"Compound-event regimes — prominent = above the {q:.0%} quantile of the continental "
                 f"distribution of each axis ({ref_txt})", fontsize=10, fontweight="bold", y=.995)
    savefig(fig, FIG, f"fig06_regimes{tag}")
    shares.round(2).to_csv(FIG / f"fig06_regime_shares_by_decade{tag}.csv")
    trans.to_csv(FIG / f"fig06_regime_transitions{tag}.csv")
    pd.Series(thr).rename("threshold").to_csv(FIG / f"fig06_regime_thresholds{tag}.csv")
    return shares


for _mode, _D in MODE_DIRS.items():
    fig06_regimes(_D, FIG_DIRS[_mode], cfg.regime_quantile, "")                        # main (q75)
    fig06_regimes(_D, FIG_DIRS[_mode], cfg.regime_quantile_sens, "_sens_tercile")      # sensitivity


# %%
# ---- 16.5 Figures S2–S4 & S3b/S4b: geo-arranged country panels ----------------
GEO_GRID = {
    "Senegal": (0, 0), "Mali": (1, 0), "Burkina Faso": (2, 0), "Niger": (3, 0), "Chad": (4, 0), "Ethiopia": (6, 0),
    "Guinea": (0, 1), "Cote d'Ivoire": (1, 1), "Ghana": (2, 1), "Benin": (3, 1), "Nigeria": (4, 1),
    "Central African Republic": (5, 1), "Uganda": (6, 1), "Kenya": (7, 1),
    "Sierra Leone": (0, 2), "Liberia": (1, 2), "Togo": (3, 2), "Cameroon": (4, 2),
    "Congo Democratic Republic": (5, 2), "Rwanda": (6, 2), "Tanzania": (7, 2),
    "Angola": (3, 3), "Gabon": (4, 3), "Zambia": (5, 3), "Burundi": (6, 3), "Comoros": (7, 3),
    "Namibia": (3, 4), "Zimbabwe": (5, 4), "Malawi": (6, 4), "Madagascar": (7, 4),
    "South Africa": (4, 5), "Eswatini": (5, 5), "Mozambique": (6, 5), "Lesotho": (4, 6)}
NCOLS, NROWS = 8, 7
TREND_BG = {"Increase (FDR-sig.)": "#fdd7d1", "Increase": "#fef0e6", "No slope": "#f5f5f5",
            "Decrease": "#dae8f5", "Decrease (FDR-sig.)": "#b8d4ea", "Not assessable": "#ffffff"}
COUNTRY_REGION = PSU_GEO.groupby("country")["region"].first().to_dict()


def trend_category(tr, axis, country, metric):
    r = slope_row(tr, axis, "country", country, metric)
    if r is None or not np.isfinite(r["slope_per_decade"]):
        return "Not assessable"
    s, sigf = r["slope_per_decade"], bool(r["mk_significant_fdr"])
    if s > 0:
        return "Increase (FDR-sig.)" if sigf else "Increase"
    if s < 0:
        return "Decrease (FDR-sig.)" if sigf else "Decrease"
    return "No slope"


def _geo_frame(title):
    fig = plt.figure(figsize=(NCOLS * 2 + 1, NROWS * 1.5 + 1.5), facecolor="white")
    gs = GridSpec(NROWS, NCOLS, figure=fig, hspace=0.55, wspace=0.35)
    axes = {}
    for r in range(NROWS):
        for c in range(NCOLS):
            ax = fig.add_subplot(gs[r, c]); ax.axis("off"); axes[(c, r)] = ax
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.0)
    return fig, axes


def _style_small(ax, country, col, row):
    ax.axis("on"); ax.set_xlim(YEARS[0] - 1, YEARS[-1] + 1)
    ax.tick_params(axis="both", labelsize=5, width=.2, length=1.5, pad=1)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(.3)
    if row == max(r for c, r in GEO_GRID.values() if c == col):
        ax.set_xticks([1990, 2010]); ax.set_xticklabels(["'90", "'10"], fontsize=4.5)
    else:
        ax.set_xticklabels([])
    ax.set_title(SHORT_NAMES.get(country, country), fontsize=6, fontweight="bold",
                 color=REGION_COLORS.get(COUNTRY_REGION.get(country), "#333"), pad=2)


def figS_geo_single(D, FIG, axis, metric, title, filename, ylim=None, clip=None):
    tr = trends(D); ser = series_by_country(D, axis, metric)
    fig, axes = _geo_frame(title)
    reg_max = {}
    for c, v in ser.items():
        reg = COUNTRY_REGION.get(c); vv = v if clip is None else np.clip(v, None, clip)
        if np.isfinite(vv).any():
            reg_max[reg] = max(reg_max.get(reg, 0), np.nanmax(vv))
    for country, (col, row) in GEO_GRID.items():
        if country not in ser:
            continue
        ax = axes[(col, row)]; v = ser[country]
        if clip is not None:
            v = np.clip(v, None, clip)
        cat = trend_category(tr, axis, country, metric)
        ax.set_facecolor(TREND_BG[cat])
        color = REGION_COLORS.get(COUNTRY_REGION.get(country), "#333"); m = np.isfinite(v)
        ax.scatter(YEARS[m], v[m], s=3, color=color, alpha=.4, edgecolors="none", zorder=3)
        ax.plot(YEARS, smooth(v, .35), color=color, lw=1.3, zorder=4)
        ts = theilsen_line(v)
        if ts is not None:
            ax.plot(YEARS, ts[0] * YEARS + ts[1], color="#888", lw=.5, alpha=.5, zorder=2)
        _style_small(ax, country, col, row)
        ax.set_ylim(ylim if ylim else (0, reg_max.get(COUNTRY_REGION.get(country), 1) * 1.1))
    handles = [mpatches.Patch(fc="none", ec="none", label=r"$\bf{Background = country\ trend}$")]
    handles += [mpatches.Patch(fc=TREND_BG[k], ec="#aaa", lw=.5, label=k) for k in TREND_BG]
    handles += [mpatches.Patch(fc="none", ec="none", label=r"$\bf{Regions}$")]
    handles += [mpatches.Patch(fc=REGION_COLORS[r], label=r) for r in REGIONS]
    axl = fig.add_axes([0.02, 0.05, 0.14, 0.35]); axl.axis("off")
    axl.legend(handles=handles, loc="center", fontsize=7, frameon=True, edgecolor="#ccc")
    savefig(fig, FIG, filename)


def figS_geo_paired(D, FIG, axis, col_a, col_b, lab_a, lab_b, title, filename):
    t = agg(D, f"{axis}_by_year_country")
    n_units = PSU_GEO.groupby("country").size()               # all PSU of the country
    if "n_units_eligible" in t.columns:                      # Axis II: CHIRPS-eligible PSU only
        n_units = t.groupby("country")["n_units_eligible"].max()
    ser = {}
    for c, g in t.groupby("country"):
        g = g.set_index("year").reindex(YEARS)
        ser[c] = (g[col_a].fillna(0).to_numpy(float) / max(1, n_units[c]),
                  g[col_b].fillna(0).to_numpy(float) / max(1, n_units[c]))
    fig, axes = _geo_frame(title)
    reg_max = {}
    for c, (a, b) in ser.items():
        reg = COUNTRY_REGION.get(c)
        reg_max[reg] = max(reg_max.get(reg, 0), np.nanmax(smooth(a, .35)), np.nanmax(smooth(b, .35)))
    for country, (col, row) in GEO_GRID.items():
        if country not in ser:
            continue
        ax = axes[(col, row)]; a, b = ser[country]
        sa, sb = smooth(a, .35), smooth(b, .35)
        ax.scatter(YEARS, a, s=1.2, color="#4575b4", alpha=.25, edgecolors="none")
        ax.scatter(YEARS, b, s=1.2, color="#d73027", alpha=.25, edgecolors="none")
        ax.plot(YEARS, sa, color="#4575b4", lw=1.2); ax.plot(YEARS, sb, color="#d73027", lw=1.2)
        ax.fill_between(YEARS, sa, sb, alpha=.08, color="#d73027")
        _style_small(ax, country, col, row)
        ax.set_ylim(0, reg_max.get(COUNTRY_REGION.get(country), 1) * 1.15)
        ax.yaxis.grid(True, lw=.1, color="#e8e8e8", zorder=0)
    handles = [mpatches.Patch(fc="#4575b4", label=lab_a), mpatches.Patch(fc="#d73027", label=lab_b)]
    handles += [mpatches.Patch(fc=REGION_COLORS[r], label=r) for r in REGIONS]
    fig.legend(handles=handles, loc="center", ncol=6, fontsize=6, frameon=True, edgecolor="#ccc",
               bbox_to_anchor=(0.25, 0.45))
    savefig(fig, FIG, filename)


for _mode, _D in MODE_DIRS.items():
    _F = FIG_DIRS[_mode]
    figS_geo_single(_D, _F, "axis1", "cooccur_days_per_unit",
                    "Axis I: HW–EPE co-occurrence days per PSU per year, by country", "figS02_geo_axis1")
    figS_geo_single(_D, _F, "axis2", "count_ratio_a",
                    "Axis II: EPE after/before count ratio, by country", "figS03_geo_axis2",
                    ylim=(0, 16), clip=15)
    figS_geo_single(_D, _F, "axis3", "pct_drought_b",
                    "Axis III: % HW with antecedent drought, by country", "figS04_geo_axis3", ylim=(0, 100))
    figS_geo_paired(_D, _F, "axis2", "n_before", "n_after", "EPE before HW", "EPE after HW",
                    "Axis II: EPE before vs. after heatwaves (per PSU, unique attribution)", "figS03b_geo_axis2_paired")
    figS_geo_paired(_D, _F, "axis3", "n_hw_nodrought", "n_hw_drought", "HW without antecedent drought",
                    "HW with antecedent drought", "Axis III: HW without vs. with antecedent drought (per PSU)",
                    "figS04b_geo_axis3_paired")


# %%
# ---- 16.6 Figures S5 (seasonality), S6 (comparative evolution), S8 (latitude), S9 (decades), S10 (month strat.), Table S1
def figS05_seasonality(D, FIG):
    m1 = agg(D, "axis1_by_month_region"); m2 = agg(D, "axis2_by_onset_month_region")
    m3 = agg(D, "axis3_by_onset_month_region")
    fig = plt.figure(figsize=(14, 10), facecolor="white"); gs = GridSpec(2, 2, figure=fig, hspace=.35, wspace=.3)
    spec = [(m1, "month", "cooccur_days_per_unit_year", "Co-occurrence days / PSU / year", "HW–EPE compound events (month of EPE)", None),
            (m2, "onset_month", "count_ratio", "EPE after / before HW", "EPE asymmetry (month of HW onset)", 1.0),
            (m3, "onset_month", "pct_drought_a", "% of HW with antecedent drought", "HW ∩ antecedent drought (month of onset)", None),
            (m3, "onset_month", "hw_per_unit_year", "HW / PSU / year", "All heatwaves (baseline seasonality)", None)]
    for k, (t, mcol, vcol, ylab, title, hline) in enumerate(spec):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        for reg in REGIONS:
            s = t[t["region"] == reg].set_index(mcol)[vcol].reindex(range(1, 13))
            v = s.to_numpy(float)
            if vcol == "count_ratio":
                v = np.clip(v, None, 15)
            ax.plot(range(1, 13), v, color=REGION_COLORS[reg], lw=2, marker="o", ms=4, label=reg)
        if hline is not None:
            ax.axhline(hline, color="#888", lw=.5, ls="--")
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_LABELS); ax.set_xlabel("Month")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=7); ax.yaxis.grid(True, lw=.15, color="#e0e0e0", zorder=0); panel_label(ax, "abcd"[k], x=-0.12)
    savefig(fig, FIG, "figS05_seasonality")


def figS06_comparative(D, FIG):
    a2c, a2r = agg(D, "axis2_by_year_continental"), agg(D, "axis2_by_year_region")
    a3c, a3r = agg(D, "axis3_by_year_continental"), agg(D, "axis3_by_year_region")
    n_all = {"Continental": len(PSU_GEO), **PSU_GEO.groupby("region").size().to_dict()}

    def pick(c, r, sc):
        t = c if sc == "Continental" else r[r["region"] == sc]
        return t.set_index("year").reindex(YEARS)
    fig = plt.figure(figsize=(15, 7), facecolor="white"); gs = GridSpec(2, 5, figure=fig, hspace=.4, wspace=.3)
    rows = [("a", a2c, a2r, "n_before", "n_after", "n_units_eligible", "#4575b4", "#d73027", "EPE before HW", "EPE after HW", "EPE / PSU / year"),
            ("b", a3c, a3r, "n_hw_nodrought", "n_hw_drought", None, "#4575b4", "#e6550d", "HW, no antecedent drought", "HW, antecedent drought", "HW / PSU / year")]
    for i, (lab, c, r, ca, cb, cn, cola, colb, la, lb, ylab) in enumerate(rows):
        data = {}
        for sc in SCALES:
            t = pick(c, r, sc)
            n = t[cn].max() if cn else n_all[sc]
            data[sc] = (t[ca].fillna(0).to_numpy(float) / n, t[cb].fillna(0).to_numpy(float) / n)
        ymax = max(np.nanmax(v) for pair in data.values() for v in pair) * 1.15
        for j, sc in enumerate(SCALES):
            ax = fig.add_subplot(gs[i, j]); a, b = data[sc]
            ax.scatter(YEARS, a, s=5, color=cola, alpha=.35, edgecolors="none"); ax.scatter(YEARS, b, s=5, color=colb, alpha=.35, edgecolors="none")
            sa, sb = smooth(a), smooth(b)
            ax.plot(YEARS, sa, color=cola, lw=1.8, label=la); ax.plot(YEARS, sb, color=colb, lw=1.8, label=lb)
            ax.fill_between(YEARS, sa, sb, alpha=.08, color=colb)
            ax.set_xlim(YEARS[0] - 1, YEARS[-1] + 1); ax.set_ylim(0, ymax)
            ax.set_title(sc, fontsize=9, fontweight="bold", color=REGION_COLORS[sc]); ax.set_xticks([1990, 2000, 2010, 2020])
            ax.yaxis.grid(True, lw=.12, color="#e0e0e0", zorder=0)
            if j == 0:
                ax.set_ylabel(ylab); ax.legend(fontsize=5.5, loc="upper left"); panel_label(ax, lab, x=-0.3, size=13)
    savefig(fig, FIG, "figS06_comparative_evolution")


def figS08_latitude(D, FIG):
    df = psu_metric_table(D); df["lat_bin"] = (df["lat"] // 5) * 5 + 2.5
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), facecolor="white")
    spec = [("axis1", "HW–EPE co-occurrence days / PSU / yr"), ("axis2", "EPE after/before ratio"), ("axis3", "% HW with antecedent drought")]
    for i, (axis, ylab) in enumerate(spec):
        ax = axes.flat[i]; col = AXIS_PSU_COLS[axis]
        for reg in REGIONS:
            s = df[df["region"] == reg]
            ax.scatter(s["lat"], s[col], s=.3, alpha=.12, c=REGION_COLORS[reg], rasterized=True, label=reg)
        g = df.groupby("lat_bin")[col]
        med, q25, q75 = g.median(), g.quantile(.25), g.quantile(.75)
        ax.fill_between(med.index, q25.to_numpy(), q75.to_numpy(), alpha=.15, color="k")
        ax.plot(med.index, med.to_numpy(), "k-", lw=1.5, label="Median (5° bins)")
        ax.set_ylabel(ylab); ax.set_xlabel("Latitude (°N)"); panel_label(ax, "abc"[i])
        if axis == "axis2":
            ax.set_ylim(0, min(np.nanquantile(df[col], .99), 20)); ax.axhline(1, color="grey", ls="--", lw=.5)
        if i == 0:
            ax.legend(fontsize=5.5, markerscale=5)
    ax = axes.flat[3]
    for axis, color in zip(("axis1", "axis2", "axis3"), ("#e67e22", "#3498db", "#e74c3c")):
        med = df.groupby("lat_bin")[AXIS_PSU_COLS[axis]].median(); v = med.to_numpy(float)
        vn = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)) if np.nanmax(v) > np.nanmin(v) else np.zeros_like(v)
        ax.plot(med.index, vn, "o-", color=color, ms=3, lw=1.3, label=AXIS_NAMES[axis])
    ax.set_xlabel("Latitude (°N)"); ax.set_ylabel("Normalised median (0–1)"); ax.legend(fontsize=6); panel_label(ax, "d")
    for a in axes.flat:
        a.axvspan(-5, 5, alpha=.04, color="#FFA500", zorder=0)
    fig.tight_layout(); savefig(fig, FIG, "figS08_latitude_gradient")


def figS09_decadal(D, FIG):
    t1, t2, t3 = agg(D, "axis1_by_region_decade"), agg(D, "axis2_by_region_decade"), agg(D, "axis3_by_region_decade")
    for t in (t1, t2, t3):
        t["decade"] = t["decade"].astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), facecolor="white")
    spec = [(t1, "cooccur_days_per_unit_year", "Co-occurrence days / PSU / yr", "Axis I"),
            (t2, "count_ratio", "EPE after/before ratio", "Axis II"), (t3, "pct_drought_a", "% HW with antecedent drought", "Axis III")]
    x = np.arange(len(DECADES)); w = .2
    for ax, (t, col, ylab, title) in zip(axes, spec):
        for i, reg in enumerate(REGIONS):
            v = [t[(t["region"] == reg) & (t["decade"] == d)][col].squeeze() if len(t[(t["region"] == reg) & (t["decade"] == d)]) else np.nan for d in DECADES]
            v = np.array(v, float)
            ax.bar(x + i * w, v, w, color=REGION_COLORS[reg], label=reg, alpha=.85, edgecolor="white", lw=.3)
            for j in range(1, len(v)):
                if np.isfinite(v[j - 1]) and v[j - 1] > 0 and np.isfinite(v[j]):
                    pc = (v[j] - v[j - 1]) / v[j - 1] * 100
                    if abs(pc) > 15:
                        ax.annotate(f"{pc:+.0f}%", xy=(x[j] + i * w, v[j]), fontsize=4.5, ha="center", va="bottom",
                                    color="#006600" if pc > 0 else "#CC0000", rotation=45)
        ax.set_xticks(x + 1.5 * w); ax.set_xticklabels([d.replace("-", "\n") for d in DECADES], fontsize=6)
        ax.set_ylabel(ylab); ax.set_title(title, fontweight="bold"); panel_label(ax, "abc"[list(axes).index(ax)], x=-0.15)
    axes[0].legend(fontsize=6, ncol=2); fig.tight_layout(); savefig(fig, FIG, "figS09_decadal_change")


def figS10_monthly(D, FIG, min_n=100):
    t = agg(D, "axis2_by_onset_month_region")
    R = np.full((len(SCALES), 12), np.nan); N = np.zeros((len(SCALES), 12))
    for i, sc in enumerate(SCALES):
        for j in range(12):
            r = t[(t["region"] == sc) & (t["onset_month"] == j + 1)]
            if len(r):
                R[i, j] = r["count_ratio"].iloc[0]; N[i, j] = r["n_before"].iloc[0] + r["n_after"].iloc[0]
    reliable = N >= min_n
    fig = plt.figure(figsize=(15, 6.5), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, width_ratios=[3, 1.3], height_ratios=[1, .05], hspace=.3, wspace=.3)
    ax = fig.add_subplot(gs[0, 0])
    cmap = mcolors.LinearSegmentedColormap.from_list("r", ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"])
    im = ax.imshow(np.log2(np.clip(R, .1, 64)), cmap=cmap, norm=mcolors.TwoSlopeNorm(vmin=np.log2(.1), vcenter=0, vmax=np.log2(16)), aspect="auto")
    for i in range(len(SCALES)):
        for j in range(12):
            if not reliable[i, j] or not np.isfinite(R[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#888")
            else:
                ax.text(j, i, f"{R[i, j]:.1f}" if R[i, j] < 10 else f"{R[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        fontweight="bold" if R[i, j] >= 1.5 else "normal", color="white" if abs(np.log2(max(R[i, j], .1))) > 2.5 else "k")
    ax.set_xticks(range(12)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(SCALES))); ax.set_yticklabels(SCALES); ax.set_xlabel("Month of heatwave onset")
    ax.set_title("After/before EPE count ratio by month of HW onset", fontsize=11, fontweight="bold"); panel_label(ax, "a", x=-0.06)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 0]), orientation="horizontal", extend="both")
    tv = [.125, .25, .5, 1, 2, 4, 8, 16]; cb.set_ticks(np.log2(tv)); cb.set_ticklabels([str(v) for v in tv]); cb.set_label("Ratio (log2 scale)")
    ax = fig.add_subplot(gs[0, 1]); frac = []; med = []
    for i in range(len(SCALES)):
        rv = R[i, reliable[i]]; frac.append(100 * np.sum(rv > 1) / len(rv) if len(rv) else 0); med.append(np.nanmedian(rv) if len(rv) else np.nan)
    ax.barh(range(len(SCALES)), frac, color=[REGION_COLORS[s] for s in SCALES], height=.7)
    for i, (f, m) in enumerate(zip(frac, med)):
        ax.text(f + 1.5, i, f"{f:.0f}% (median {m:.1f})", va="center", fontsize=8)
    ax.set_yticks(range(len(SCALES))); ax.set_yticklabels(SCALES); ax.invert_yaxis(); ax.set_xlim(0, 130)
    ax.set_xlabel("Months with ratio > 1 (%)"); ax.axvline(50, color="gray", ls=":", lw=.8)
    ax.set_title("Fraction of months with post-HW EPE excess", fontsize=11, fontweight="bold"); panel_label(ax, "b", x=-0.15)
    fig.text(.02, .0, f"— : fewer than {min_n} attributed EPE. Unique attribution, W = ±{cfg.axis2_window_days} d, non-truncated HW.", fontsize=7, color="#666", style="italic")
    savefig(fig, FIG, "figS10_monthly_stratification")


def tableS01(D, FIG):
    tr = trends(D)
    p1 = agg(D, "axis1_by_psu").groupby("country").agg(N_PSU=("psu_idx", "size"), CCE_days_per_PSU=("n_cooccur_days", "mean"))
    p2 = agg(D, "axis2_by_country").set_index("country")[["n_before", "n_after", "count_ratio"]]
    a3 = agg(D, "axis3_by_year_country").groupby("country")[["n_hw", "n_hw_spei", "n_hw_drought"]].sum()
    a3["Pct_HW_drought"] = 100 * a3["n_hw_drought"] / a3["n_hw_spei"].replace(0, np.nan)
    tab = (p1.join(p2).join(a3)).reset_index()
    tab["Region"] = tab["country"].map(COUNTRY_REGION)
    for axis, metric, name in (("axis1", "cooccur_days_per_unit", "CCE_trend_per_decade"),
                               ("axis2", "count_ratio_a", "Ratio_trend_per_decade"),
                               ("axis3", "pct_drought_b", "Drought_trend_per_decade")):
        rows = tr[(tr["axis"] == axis) & (tr["scope"] == "country") & (tr["metric"] == metric)].set_index("scale")
        tab[name] = tab["country"].map(rows["slope_per_decade"]); tab[name + "_FDRsig"] = tab["country"].map(rows["mk_significant_fdr"])
    tab = tab.rename(columns={"country": "Country", "n_before": "N_EPE_before", "n_after": "N_EPE_after", "count_ratio": "EPE_ratio",
                              "n_hw": "N_HW", "n_hw_spei": "N_HW_with_SPEI", "n_hw_drought": "N_HW_drought"})
    tab["_o"] = tab["Region"].map({r: i for i, r in enumerate(REGIONS)})
    tab = tab.sort_values(["_o", "Country"]).drop(columns="_o")
    FIG.mkdir(parents=True, exist_ok=True); tab.round(4).to_csv(FIG / "tableS01_country_statistics.csv", index=False)
    show = tab[["Country", "Region", "N_PSU", "CCE_days_per_PSU", "CCE_trend_per_decade", "EPE_ratio", "Ratio_trend_per_decade",
                "Pct_HW_drought", "Drought_trend_per_decade"]].copy()
    for c in show.columns[3:]:
        show[c] = show[c].map(lambda v: f"{v:.3g}" if pd.notna(v) else "NA")
    fig, ax = plt.subplots(figsize=(14, 11), facecolor="white"); ax.axis("off")
    tb = ax.table(cellText=show.to_numpy().tolist(), colLabels=list(show.columns), cellLoc="center", loc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(6); tb.scale(1, 1.3)
    for j in range(len(show.columns)):
        tb[0, j].set_facecolor("#4a4a4a"); tb[0, j].set_text_props(color="white", fontweight="bold")
    fig.suptitle("Table S1 — Country-level compound climate event statistics", fontsize=10, fontweight="bold")
    savefig(fig, FIG, "tableS01_country_statistics")
    return tab


for _mode, _D in MODE_DIRS.items():
    _F = FIG_DIRS[_mode]
    figS05_seasonality(_D, _F); figS06_comparative(_D, _F); figS08_latitude(_D, _F)
    figS09_decadal(_D, _F); figS10_monthly(_D, _F); tableS01(_D, _F)


# %%
# ---- 16.7 Null-model figure and sensitivity figure --------------------------------
def fig_null(D, FIG):
    nl = sig(D, "axis2_null_region"); dr = sig(D, "axis2_null_draws")
    seasons = ["ALL"] + list(cfg.null_seasons)
    mode_txt = ("same PSU, other year, onset ±%d d" % cfg.null_window_days if cfg.null_mode == "psu_window"
                else "regional onset-doy pool")
    fig = plt.figure(figsize=(14, 9), facecolor="white"); gs = GridSpec(2, 2, figure=fig, hspace=.4, wspace=.3)
    # (a) PRIMARY: count ratio obs vs null, by scale (ALL seasons)
    ax = fig.add_subplot(gs[0, 0]); t = nl[nl["season"] == "ALL"].set_index("scale").reindex(SCALES)
    x = np.arange(len(SCALES))
    ax.bar(x, t["count_ratio_obs"], .5, color=[REGION_COLORS[s] for s in SCALES], alpha=.85, label="observed")
    ax.errorbar(x, t["count_ratio_null_med"], yerr=[t["count_ratio_null_med"] - t["count_ratio_null_lo"],
                t["count_ratio_null_hi"] - t["count_ratio_null_med"]], fmt="s", color="k", ms=4, capsize=4, lw=1,
                label="null median & 95% envelope")
    for i, p in enumerate(t["p_count_ratio_excess"]):
        ax.text(x[i], max(t["count_ratio_obs"].iloc[i], t["count_ratio_null_hi"].iloc[i]) * 1.04, f"p={p:.3f}", ha="center", fontsize=6)
    ax.axhline(1, color="#888", lw=.5, ls="--"); ax.set_xticks(x); ax.set_xticklabels(SCALES)
    ax.set_ylabel("EPE after / before HW (unique attribution)")
    ax.set_title("PRIMARY metric vs. strict null (%s)" % mode_txt, fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2); panel_label(ax, "a")
    # (b) secondary: ECA fractions
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x - .18, t["precursor_obs"], .36, color="#91bfdb", label="precursor (EPE before HW)")
    ax.bar(x + .18, t["trigger_obs"], .36, color="#d73027", label="trigger (EPE after HW)")
    ax.errorbar(x - .18, (t["precursor_null_lo"] + t["precursor_null_hi"]) / 2, yerr=(t["precursor_null_hi"] - t["precursor_null_lo"]) / 2,
                fmt="none", ecolor="k", capsize=3, lw=1)
    ax.errorbar(x + .18, (t["trigger_null_lo"] + t["trigger_null_hi"]) / 2, yerr=(t["trigger_null_hi"] - t["trigger_null_lo"]) / 2,
                fmt="none", ecolor="k", capsize=3, lw=1, label="null 95% envelope")
    for i, p in enumerate(t["p_diff_excess"]):
        ax.text(x[i], max(t["trigger_obs"].iloc[i], t["trigger_null_hi"].iloc[i]) * 1.04, f"p(diff)={p:.3f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(SCALES); ax.set_ylabel(f"Share of HW with ≥1 EPE within {cfg.axis2_window_days} d")
    ax.set_title("Secondary metric: ECA fractions vs. null", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3); panel_label(ax, "b")
    # (c) seasonal decomposition (onset season) — count ratio
    ax = fig.add_subplot(gs[1, 0]); w = .8 / len(seasons)
    for k, se in enumerate(seasons):
        ts = nl[nl["season"] == se].set_index("scale").reindex(SCALES)
        xx = x - .4 + w * (k + .5)
        ax.bar(xx, ts["count_ratio_obs"], w * .9, color=plt.cm.viridis(k / max(1, len(seasons) - 1)), alpha=.85, label=se)
        ax.errorbar(xx, ts["count_ratio_null_med"], yerr=[(ts["count_ratio_null_med"] - ts["count_ratio_null_lo"]).clip(lower=0),
                    (ts["count_ratio_null_hi"] - ts["count_ratio_null_med"]).clip(lower=0)], fmt="_", color="k", ms=5, capsize=2, lw=.8)
    ax.axhline(1, color="#888", lw=.5, ls="--"); ax.set_xticks(x); ax.set_xticklabels(SCALES)
    ax.set_ylabel("EPE after / before HW"); ax.set_title("By season of HW onset (bars = observed, black = null envelope)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, ncol=5); panel_label(ax, "c")
    # (d) null distribution, continental
    ax = fig.add_subplot(gs[1, 1]); d = dr[(dr["scale"] == "Continental") & (dr["season"] == "ALL")]["count_ratio"].dropna()
    ax.hist(d, bins=40, color="#999", alpha=.8, label=f"null ({len(d)} permutations)")
    obs = t.loc["Continental", "count_ratio_obs"]; ax.axvline(obs, color="#d73027", lw=2, label=f"observed = {obs:.2f}")
    ax.set_xlabel("EPE after / before HW"); ax.set_ylabel("Count"); ax.set_title("Null distribution, Continental", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6); panel_label(ax, "d")
    savefig(fig, FIG, "figS_axis2_null_test")
    nl.round(4).to_csv(FIG / "figS_axis2_null_test_table.csv", index=False)


def fig_sensitivity(D, FIG):
    ws = sig(D, "axis2_window_sensitivity"); a3 = pd.read_parquet(D / "axis3.parquet")
    a3 = a3.merge(PSU_GEO[["psu_idx", "region"]], on="psu_idx", how="left"); a3["year"] = pd.to_datetime(a3["start_date"]).dt.year
    sname = f"spei{cfg.spei_scale_months}"; pname = f"spi{cfg.spei_scale_months}"
    Ws = sorted(ws["window_days"].unique()); thr = list(cfg.spei_threshold_sens)
    fig = plt.figure(figsize=(16, 9), facecolor="white"); gs = GridSpec(2, 6, figure=fig, hspace=.4, wspace=.4)
    # (a) W sensitivity by region — count ratio
    ax = fig.add_subplot(gs[0, 0:3]); x = np.arange(len(Ws)); w = .8 / len(REGIONS)
    for i, reg in enumerate(REGIONS):
        v = ws[ws["scale"] == reg].set_index("window_days")["count_ratio"].reindex(Ws)
        ax.bar(x + i * w, v.to_numpy(), w, color=REGION_COLORS[reg], label=reg, alpha=.85, edgecolor="white", lw=.3)
    ax.set_xticks(x + (len(REGIONS) - 1) * w / 2); ax.set_xticklabels([f"±{W}" for W in Ws]); ax.axhline(1, color="grey", ls="--", lw=.5)
    ax.set_xlabel("Window W (days)"); ax.set_ylabel("EPE after / before HW"); ax.legend(fontsize=7, ncol=4)
    ax.set_title(f"Axis II count ratio by region and window (primary W = ±{cfg.axis2_window_days})", fontsize=9, fontweight="bold"); panel_label(ax, "a", x=-0.07)
    # (b) continental W sensitivity: count ratio + trigger-precursor
    ax = fig.add_subplot(gs[0, 3:6]); c = ws[ws["scale"] == "Continental"].set_index("window_days").reindex(Ws)
    ax.plot(Ws, c["count_ratio"], "o-", color="#2c3e50", lw=1.8, ms=6, label="count ratio (primary)")
    for W, r in zip(Ws, c["count_ratio"]):
        ax.annotate(f"{r:.2f}", (W, r), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7, color="#2c3e50")
    ax2 = ax.twinx(); ax2.plot(Ws, c["rate_diff"], "s--", color="#d73027", lw=1.2, ms=5, label="trigger − precursor (secondary)")
    ax2.set_ylabel("Trigger − precursor fraction", color="#d73027"); ax2.spines["right"].set_visible(True)
    ax.axvline(cfg.axis2_window_days, color="#e74c3c", ls=":", lw=.8); ax.axhline(1, color="grey", ls="--", lw=.5)
    ax.set_xticks(Ws); ax.set_xticklabels([f"±{W}" for W in Ws]); ax.set_xlabel("Window W (days)"); ax.set_ylabel("EPE after / before HW")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.set_title("Continental Axis II metrics by window size", fontsize=9, fontweight="bold"); panel_label(ax, "b", x=-0.07)
    # (c) SPEI threshold: % HW preconditioned and duration ratio (primary lag)
    s = a3[a3["has_spei"]]
    ax = fig.add_subplot(gs[1, 0:2]); cols = ["#4292c6", "#f4a582", "#ef6548", "#67000d"][:len(thr)]
    pct = [100 * (s[sname] < t).mean() for t in thr]
    dur = [s.loc[s[sname] < t, "duration"].mean() / s.loc[~(s[sname] < t), "duration"].mean() for t in thr]
    ax.bar(range(len(thr)), pct, color=cols, edgecolor="white")
    for i, v in enumerate(pct):
        ax.text(i, v + .8, f"{v:.1f}%", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(thr))); ax.set_xticklabels([f"< {t}" for t in thr]); ax.set_xlabel("SPEI-3 threshold (antecedent month)")
    ax.set_ylabel("% HW with antecedent drought"); ax.set_title("Drought threshold sensitivity", fontsize=9, fontweight="bold"); panel_label(ax, "c", x=-0.1)
    ax = fig.add_subplot(gs[1, 2:4]); ax.bar(range(len(thr)), dur, color=cols, edgecolor="white"); ax.axhline(1, color="grey", ls="--", lw=.5)
    for i, v in enumerate(dur):
        ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(thr))); ax.set_xticklabels([f"< {t}" for t in thr]); ax.set_xlabel("SPEI-3 threshold")
    ax.set_ylabel("Duration ratio (drought / no drought)"); ax.set_title("HW duration amplification", fontsize=9, fontweight="bold"); panel_label(ax, "d", x=-0.1)
    # (e) lag 1 vs lag 0, SPEI vs SPI by region (% HW preconditioned)
    ax = fig.add_subplot(gs[1, 4:6]); variants = [("drought_precond", "has_spei", "SPEI-3, lag 1 (primary)", "#2c3e50")]
    for L in list(cfg.axis3_lags)[1:]:
        variants.append((f"drought_precond_lag{L}", f"has_spei_lag{L}", f"SPEI-3, lag {L}", "#7570B3"))
    variants.append(("drought_precond_spi", "has_spi", "SPI-3, lag 1", "#1B9E77"))
    w = .8 / len(variants); x = np.arange(len(SCALES))
    for k, (col, has, lab, color) in enumerate(variants):
        v = [100 * a3.loc[a3[has] & ((a3["region"] == sc) if sc != "Continental" else True), col].mean() for sc in SCALES]
        ax.bar(x + k * w, v, w, color=color, label=lab, alpha=.85, edgecolor="white", lw=.3)
    ax.set_xticks(x + (len(variants) - 1) * w / 2); ax.set_xticklabels(SCALES); ax.set_ylabel("% HW with antecedent drought")
    ax.set_title("Axis III: lag and index sensitivity", fontsize=9, fontweight="bold"); ax.legend(fontsize=6.5); panel_label(ax, "e", x=-0.1)
    fig.suptitle("Sensitivity analyses — Axis II window, Axis III threshold / lag / index", fontsize=11, fontweight="bold", y=.995)
    savefig(fig, FIG, "figS_sensitivity")
    # threshold x trend table (Theil-Sen of the annual continental %), for the text
    rows = []
    for t in thr:
        yr = s.groupby("year")[sname].apply(lambda v: 100 * (v < t).mean()).reindex(YEARS)
        ts = theilsen_line(yr.to_numpy(float))
        rows.append(dict(threshold=t, pct_hw=100 * (s[sname] < t).mean(),
                         duration_ratio=s.loc[s[sname] < t, "duration"].mean() / s.loc[~(s[sname] < t), "duration"].mean(),
                         trend_pct_per_decade=ts[0] * 10 if ts else np.nan,
                         trend_ci_lo=ts[2] * 10 if ts else np.nan, trend_ci_hi=ts[3] * 10 if ts else np.nan))
    pd.DataFrame(rows).round(4).to_csv(FIG / "figS_sensitivity_spei_threshold_table.csv", index=False)
    ws.round(4).to_csv(FIG / "figS_sensitivity_window_table.csv", index=False)


for _mode, _D in MODE_DIRS.items():
    fig_null(_D, FIG_DIRS[_mode]); fig_sensitivity(_D, FIG_DIRS[_mode])


# %%
# ---- 16.8 calendar vs annual threshold comparison + shared drought baseline ----
def fig_mode_comparison(FIG):
    """Headline metrics of each threshold mode side by side (needs both runs)."""
    HEAD = [("axis1", "cooccur_days_per_unit", "Co-occurrence days / PSU / yr"),
            ("axis2", "count_ratio_a", "EPE after / before HW"),
            ("axis2", "rate_diff_b", "Trigger − precursor"),
            ("axis3", "pct_drought_b", "% HW antecedent drought")]
    modes = list(MODE_DIRS); cols = {"calendar": "#2c3e50", "annual": "#e67e22"}
    fig, axes = plt.subplots(2, len(HEAD), figsize=(4 * len(HEAD), 7), facecolor="white")
    rows = []
    for j, (axis, metric, lab) in enumerate(HEAD):
        for m in modes:
            D = MODE_DIRS[m]; tr = trends(D)
            ser = series_by_scale(D, axis, metric); x = np.arange(len(SCALES)); w = .8 / len(modes)
            mean_v = [np.nanmean(ser[sc]) for sc in SCALES]
            axes[0, j].bar(x + modes.index(m) * w, mean_v, w, color=cols.get(m, "#999"), label=m, alpha=.85)
            sl = [slope_row(tr, axis, "continental" if sc == "Continental" else "region", sc, metric) for sc in SCALES]
            s = np.array([r["slope_per_decade"] if r is not None else np.nan for r in sl])
            lo = np.array([r["ci_low_per_decade"] if r is not None else np.nan for r in sl])
            hi = np.array([r["ci_high_per_decade"] if r is not None else np.nan for r in sl])
            axes[1, j].errorbar(x + modes.index(m) * w, s, yerr=[s - lo, hi - s], fmt="o", color=cols.get(m, "#999"), capsize=3, label=m)
            for sc, mv, r in zip(SCALES, mean_v, sl):
                rows.append(dict(mode=m, axis=axis, metric=metric, scale=sc, mean_over_years=mv,
                                 slope_per_decade=r["slope_per_decade"] if r is not None else np.nan,
                                 ci_lo=r["ci_low_per_decade"] if r is not None else np.nan,
                                 ci_hi=r["ci_high_per_decade"] if r is not None else np.nan,
                                 mk_p=r["mk_p"] if r is not None else np.nan))
        for i in range(2):
            axes[i, j].set_xticks(x + (len(modes) - 1) * w / 2); axes[i, j].set_xticklabels(SCALES, rotation=30, ha="right")
        axes[0, j].set_title(f"{AXIS_NAMES[axis]}\n{lab}", fontsize=8, fontweight="bold"); axes[0, j].set_ylabel("Mean over years")
        axes[1, j].set_ylabel("Theil–Sen slope / decade (95% CI)"); axes[1, j].axhline(0, color="grey", lw=.5)
    # HW counts by mode
    txt = " | ".join(f"{m}: {len(pd.read_parquet(MODE_DIRS[m] / 'heatwaves.parquet', columns=['psu_idx'])):,} HW" for m in modes)
    axes[0, 0].legend(fontsize=7); fig.suptitle("Calendar vs annual Tmax threshold — headline metrics\n" + txt, fontsize=10, fontweight="bold")
    fig.tight_layout(); savefig(fig, FIG, "figS_threshold_mode_comparison")
    pd.DataFrame(rows).round(4).to_csv(FIG / "figS_threshold_mode_comparison_table.csv", index=False)


def fig_drought_baseline(FIG):
    spei = pd.read_parquet(PROCESSED / "spei3.parquet"); col = f"spei{cfg.spei_scale_months}"
    d = spei.groupby("psu_idx")[col].apply(lambda v: 100 * (v < cfg.drought_threshold).mean()).rename("pct_months_drought").reset_index()
    d = d.merge(PSU_GEO, on="psu_idx", how="left")
    fig = plt.figure(figsize=(8, 8), facecolor="white"); gs = GridSpec(2, 1, figure=fig, height_ratios=[1, .04], hspace=.05)
    ax = map_ax(fig, gs[0]); sc = map_scatter(ax, d, "pct_months_drought", cmap="YlOrBr", norm=mcolors.Normalize(0, 40), s=1.2)
    ax.set_title(f"Baseline drought frequency: % of months with SPEI-3 < {cfg.drought_threshold}, {YEARS[0]}–{YEARS[-1]}", fontsize=10, fontweight="bold")
    cb = fig.colorbar(sc, cax=fig.add_subplot(gs[1]), orientation="horizontal", extend="max"); cb.set_label("% of months")
    savefig(fig, FIG, "figS_drought_baseline")


_ROOT_FIG = PROCESSED / "figures_shared"
fig_drought_baseline(_ROOT_FIG)
if len(MODE_DIRS) > 1:
    fig_mode_comparison(_ROOT_FIG)

