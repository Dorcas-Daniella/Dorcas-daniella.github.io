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
    is the standard SPI convention (zero-bounded precipitation); xclim fits
    the gamma on the NON-ZERO values and gives zeros a separate probability.

    Any other xclim error propagates (no NaN fallback). Returns (df, info):
    info["method"] = EFFECTIVE fit method (read from the xclim output),
    info["nan_by_psu"] / info["unexplained"] = NaN classification of this
    block (see _classify_si_nans).
    """
    from xclim.indices.stats import preprocess_standardized_index
    da = xr.DataArray(
        wb_wide.to_numpy(), dims=("time", "psu_idx"),
        coords={"time": wb_wide.index.values,
                "psu_idx": wb_wide.columns.to_numpy()},
        attrs={"units": "mm/month"})
    kw = dict(freq="MS", window=cfg.spei_scale_months,
              cal_start=f"{cfg.ref_start_year}-01-01",
              cal_end=f"{cfg.ref_end_year}-12-31")
    name = f"{kind}{cfg.spei_scale_months}"
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
    method = out.attrs.get("method")
    if method is None:
        raise RuntimeError(f"{name}: xclim output carries no 'method' attribute "
                           "— cannot record the effective fit method.")
    out = out.clip(-cfg.spei_clip, cfg.spei_clip)

    # NaN control on the SAME rolled series xclim fitted (rolling mean,
    # skipna=False), so zeros / missing windows are classified exactly.
    rolled, _ = preprocess_standardized_index(da, freq=kw["freq"],
                                              window=kw["window"])
    nan_by_psu, unexplained = _classify_si_nans(
        rolled, out, cfg, zero_inflated=(kind == "spi"))

    df = out.to_dataframe(name=name).reset_index()
    df["year"] = df["time"].dt.year.astype("int16")
    df["month"] = df["time"].dt.month.astype("int8")
    info = dict(method=str(method), nan_by_psu=nan_by_psu,
                unexplained=unexplained)
    return df[["psu_idx", "year", "month", name]], info


def _classify_si_nans(rolled: xr.DataArray, si: xr.DataArray, cfg,
                      zero_inflated: bool):
    """Classify every NaN of a standardized index (SPEI or SPI), per PSU:
      spinup                  : first window-1 months (Jan-Feb 1985 for SI-3);
      missing_input           : >= 1 missing monthly input in the window;
      insufficient_calibration: <= 1 fittable value for that calendar month on
                                the calibration period (xclim returns NaN
                                parameters). SPI: fittable = non-zero (zeros
                                are handled separately: a zero value only
                                needs >= 1 non-null calibration value);
      unexplained             : anything else -> the caller stops the run.
    Returns (per-PSU counts DataFrame, DataFrame of unexplained cells)."""
    R = rolled.transpose("time", "psu_idx").to_numpy()
    V = si.transpose("time", "psu_idx").to_numpy()
    psu = rolled["psu_idx"].to_numpy()
    assert np.array_equal(psu, si["psu_idx"].to_numpy())
    t = pd.DatetimeIndex(rolled["time"].to_numpy())
    T = len(t)
    cal = np.asarray((t >= pd.Timestamp(f"{cfg.ref_start_year}-01-01"))
                     & (t <= pd.Timestamp(f"{cfg.ref_end_year}-12-31")))
    month = np.asarray(t.month)
    finite = np.isfinite(R)
    fittable = finite & (R != 0) if zero_inflated else finite
    insuff = np.zeros_like(finite)
    for m in range(1, 13):
        rows = month == m
        n_fit = fittable[rows & cal].sum(axis=0)              # per PSU
        if zero_inflated:
            n_notnull = finite[rows & cal].sum(axis=0)
            is_zero = finite[rows] & (R[rows] == 0)
            insuff[rows] = np.where(is_zero, n_notnull[None, :] == 0,
                                    n_fit[None, :] <= 1)
        else:
            insuff[rows] = np.broadcast_to(n_fit[None, :] <= 1,
                                           (rows.sum(), len(psu)))
    nan = np.isnan(V)
    spin = np.zeros_like(nan)
    spin[: cfg.spei_scale_months - 1] = True
    c_spin = nan & spin
    c_miss = nan & ~spin & ~finite
    c_cal = nan & ~spin & finite & insuff
    c_unx = nan & ~(c_spin | c_miss | c_cal)
    by_psu = pd.DataFrame({
        "psu_idx": psu.astype("int32"), "n_months": T,
        "n_nan": nan.sum(axis=0), "n_spinup": c_spin.sum(axis=0),
        "n_missing_input": c_miss.sum(axis=0),
        "n_insufficient_calibration": c_cal.sum(axis=0),
        "n_unexplained": c_unx.sum(axis=0)})
    ti, pj = np.nonzero(c_unx)
    unexplained = pd.DataFrame({"psu_idx": psu[pj], "time": t[ti],
                                "rolled_value": R[ti, pj]})
    return by_psu, unexplained


def run_step05_spei(cfg, out_dir: Path | None = None) -> None:
    out_dir = Path(out_dir or cfg.paths.processed)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    psu = pd.read_parquet(cfg.paths.processed / "psu.parquet",
                          columns=["psu_idx", "lat"])
    el = psu_eligibility(cfg)
    eligible = el.loc[el["spei_eligible"], "psu_idx"].to_numpy()
    n_psu_all = len(psu)
    excluded = n_psu_all - len(eligible)
    if excluded:
        log.info("Excluding %d SPEI-ineligible PSU (CHIRPS or ERA5 invalid) "
                 "from the water balance.", excluded)
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
    methods, nan_parts = {}, []
    diag_file = cfg.paths.processed / "spei_nan_diagnostics.parquet"

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

        spei, info_s = _standardize(wb_wide, cfg, "spei")
        spi, info_p = _standardize(pr_wide, cfg, "spi")
        for col, info in ((sname, info_s), (pname, info_p)):
            if methods.setdefault(col, info["method"]) != info["method"]:
                raise RuntimeError(f"{col}: effective fit method changed between "
                                   f"blocks ({methods[col]} -> {info['method']}).")
            nan_parts.append(info["nan_by_psu"].assign(index=col, block=b))
            if len(info["unexplained"]):
                pd.concat(nan_parts, ignore_index=True).to_parquet(diag_file, index=False)
                bad = cfg.paths.processed / "spei_nan_unexplained.parquet"
                info["unexplained"].assign(index=col, block=b).to_parquet(bad, index=False)
                raise RuntimeError(
                    f"{col}: {len(info['unexplained']):,} unexplained NaN in block "
                    f"{b} (not spin-up, missing input or insufficient "
                    f"calibration). Diagnostics: {diag_file.name}, {bad.name}.")

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
    # nullable booleans: <NA> where the index is missing (never 'no drought')
    for col in (sname, pname):
        df[f"is_drought_{col}"] = ((df[col] < cfg.drought_threshold)
                                   .astype("boolean").mask(df[col].isna()))

    # ---- NaN accounting on the FULL PSU x month grid ------------------------
    nan_df = pd.concat(nan_parts, ignore_index=True)
    nan_df.to_parquet(diag_file, index=False)
    n_months = len(time_index)
    nan_summary = {}
    for col in (sname, pname):
        s = nan_df[nan_df["index"] == col]
        n_nan_file = int(df[col].isna().sum())
        if int(s["n_nan"].sum()) != n_nan_file:
            raise RuntimeError(f"{col}: NaN diagnostics ({int(s['n_nan'].sum()):,}) "
                               f"!= NaN in spei3.parquet ({n_nan_file:,}).")
        nan_summary[col] = dict(
            grid_cells=n_psu_all * n_months,
            ineligible_psu_cells=excluded * n_months,
            spinup=int(s["n_spinup"].sum()),
            missing_input=int(s["n_missing_input"].sum()),
            insufficient_calibration=int(s["n_insufficient_calibration"].sum()),
            unexplained=int(s["n_unexplained"].sum()),
            valid=int(df[col].notna().sum()),
            psu_with_any_nan_beyond_spinup=int(
                (s["n_nan"] - s["n_spinup"] > 0).sum()))
        log.info("%s NaN categories (full grid): %s", col, nan_summary[col])
    # pooled calibration-period moments: a DIAGNOSTIC only. They are neither
    # necessary nor sufficient for good PSU x month fits; for SPI a large
    # share of zero 3-month totals legitimately shifts them (zeros receive
    # the upper zero-mass probability: 50% zeros -> mean ~+0.4, std ~0.6).
    _ref = df[(df["year"] >= cfg.ref_start_year) & (df["year"] <= cfg.ref_end_year)]
    ref_moments = {col: dict(mean=float(_ref[col].mean()), std=float(_ref[col].std()))
                   for col in (sname, pname)}
    manifest_update("spei_spi", dict(
        calibration_moments_diagnostic=ref_moments,
        spei=dict(index=sname, dist=cfg.spei_dist, method_requested=cfg.spei_method,
                  method_effective=methods.get(sname)),
        spi=dict(index=pname, dist="gamma (floc=0, zeros handled separately)",
                 method_requested="APP", method_effective=methods.get(pname)),
        xclim_version=xclim.__version__, n_spei_eligible_psu=len(eligible),
        n_ineligible_psu=excluded, nan_categories=nan_summary))

    # ---- validation: ~N(0,1) on the calibration period --------------------
    ref = df[(df["year"] >= cfg.ref_start_year) & (df["year"] <= cfg.ref_end_year)]
    for col in (sname, pname):
        v = ref[col].dropna()
        if len(v):
            log.info("%s reference mean %+0.3f (exp ~0), std %.3f (exp ~1), "
                     "NaN rows whole period: %s",
                     col, v.mean(), v.std(), f"{df[col].isna().sum():,}")
            if abs(v.mean()) > 0.1 or abs(v.std() - 1) > 0.1:
                log.warning("%s pooled moments differ from N(0,1) on the calibration "
                            "period — diagnostic only (expected for SPI with many "
                            "zero totals); inspect the fits.", col)

    df.to_parquet(out_dir / "spei3.parquet", index=False)
    log.info("Wrote spei3.parquet (%s rows) in %.1f min",
             f"{len(df):,}", (time.time() - t0) / 60)

run_step05_spei(cfg)
