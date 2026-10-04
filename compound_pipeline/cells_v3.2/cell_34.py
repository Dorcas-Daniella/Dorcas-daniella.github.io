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
                         columns=["psu_idx", "era5_cell"])
    el = psu_eligibility(cfg, in_dir)[["psu_idx", "axis12_eligible"]]
    # Eligible population = epe_eligible & temp_eligible; marginals, expected
    # counts and co-occurrences all come from doy_counts, whose four counts
    # share ONE day mask (Tmax and precip finite; step04).
    dc = dc[dc["psu_idx"].isin(el.loc[el["axis12_eligible"], "psu_idx"])].copy()

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
    tot = (tot.reset_index().merge(psu, on="psu_idx").merge(gm, on="psu_idx")
              .merge(el, on="psu_idx"))
    assert tot["axis12_eligible"].all()             # eligible PSU only
    tot.to_parquet(out_dir / "lmf_by_psu.parquet", index=False)

    # ---- pooled per scale; bootstrap CI resampling PSU, or ERA5 cells -------
    # (cluster bootstrap: PSU sharing an ERA5 cell are resampled together)
    rng = np.random.default_rng(cfg.rng_seed)
    CI_LABEL = {"era5_cell": "cluster bootstrap CI — ERA5 cell",
                "psu_idx": "bootstrap CI — PSU"}

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
        return dict(scale=label, unit=unit_col, ci_method=CI_LABEL[unit_col],
                    n_units=n,
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
        log.info("LMF continental: naive %.2f | seasonal %.2f [%.2f, %.2f] "
                 "(%s)", r["lmf_naive"], r["lmf_seasonal"],
                 r["lmf_seasonal_lo"], r["lmf_seasonal_hi"], r["ci_method"])


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
    """Observed heatwaves of ELIGIBLE PSU (epe & temp; hw_id order kept) —
    the same population for the null and the W sensitivity."""
    hw = pd.read_parquet(in_dir / "axis2_hw.parquet")
    el = psu_eligibility(cfg, in_dir)
    hw = hw[hw["psu_idx"].isin(el.loc[el["axis12_eligible"], "psu_idx"])
            ].reset_index(drop=True)
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


def _transfer_onset(month, day, year) -> np.ndarray:
    """Day number of (year, month, day), vectorised; 29 Feb -> 28 Feb when
    the target year is not a leap year."""
    year = np.asarray(year, np.int64)
    month = np.asarray(month, np.int64)
    day = np.asarray(day, np.int64)
    leap = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    day = np.where((month == 2) & (day == 29) & ~leap, 28, day)
    first = ((year - 1970) * 12 + (month - 1)).astype("datetime64[M]")
    return (first.astype("datetime64[D]") - EPOCH).astype("int64") + day - 1


def _year_of(daynum) -> np.ndarray:
    d = EPOCH + np.asarray(daynum, np.int64).astype("timedelta64[D]")
    return d.astype("datetime64[Y]").astype(np.int64) + 1970


def _admissible_shifts(a, d, s_obs, e_obs, obs_year, W, K, lo, hi):
    """Admissible integer shifts delta of the transferred centres `a`
    (vectorised over heatwaves). Returns (L, U): delta in [L, U]; empty
    where L > U. Constraints of the psu_window null, with s = a + delta:
      (1) the full simulated window [s-W, s+d+W] lies in [lo, hi]:
          max(-K, lo+W-a) <= delta <= min(K, hi-W-d-a);
      (2) year(s) != observed onset year;
      (3) [s-W, s+d+W] does not overlap the observed heatwave [s_obs, e_obs].
    (2) and (3) each exclude ONE contiguous delta-range. As the target year
    differs from the observed year, |a - s_obs| >= 364 days > K, so each
    excluded range is a tail of [-K, K] and the admissible set stays a
    single interval — asserted below (a 'middle' exclusion raises)."""
    L = np.maximum(-K, lo + W - a)
    U = np.minimum(K, hi - W - d - a)
    j0 = _transfer_onset(1, 1, obs_year)
    j1 = _transfer_onset(12, 31, obs_year)
    for x, y in ((j0 - a, j1 - a),                       # (2) same real year
                 (s_obs - d - W - a, e_obs + W - a)):    # (3) overlap
        lower = x <= L
        upper = ~lower & (y >= U)
        if (~lower & ~upper & (L <= U)).any():
            raise AssertionError("psu_window null: excluded shift range inside "
                                 "[L, U] — K too large for the interval sampler.")
        L = np.where(lower, np.maximum(L, y + 1), L)
        U = np.where(upper, np.minimum(U, x - 1), U)
    return L, U


def axis2_null(cfg, in_dir: Path, out_dir: Path) -> None:
    """Surrogate null for Axis II, computed for BOTH metrics:
        * count_ratio = n_after / n_before under UNIQUE attribution (PRIMARY)
        * ECA fractions precursor / trigger (secondary)
    Population: eligible (epe & temp) and non-truncated observed heatwaves.
    null_mode == "psu_window" (PRIMARY, strict): each observed heatwave keeps
        its PSU and duration (d = duration-1). Its onset month/day is
        transferred to a target year y != observed year (29 Feb -> 28 Feb in
        a non-leap year), giving the centre a_y; the onset is a_y + delta,
        |delta| <= K = cfg.null_window_days. Admissible delta (before any
        draw): the full window [s-W, s+d+W] stays in the study period, the
        REAL year of the simulated onset differs from the observed onset
        year (an onset may cross 31 Dec otherwise), and the window does not
        overlap the observed heatwave. Draw order: target year uniform among
        the years with >= 1 admissible delta, then delta uniform among that
        year's admissible shifts. Exactly one surrogate per heatwave and
        permutation (no n_HW x n_years matrix: year rejection sampling).
        Local seasonality is preserved; only the fine timing is randomised.
    null_mode == "region_pool" (legacy, unchanged): onset doy drawn from the pooled
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

    if cfg.null_mode == "psu_window":
        assert 0 <= K < 364, "null_window_days must be < 364 (interval sampler)"
        o_month = hw["start_date"].dt.month.to_numpy()
        o_day = hw["start_date"].dt.day.to_numpy()
        # SYNCHRONISED draws: PSU sharing an ERA5 cell have the same Tmax
        # series, hence identical heatwaves. One (year, shift) is drawn per
        # DISTINCT heatwave (ERA5 cell, start, end) and copied to all its PSU,
        # so the null keeps the duplication present in the observed data
        # instead of treating the copies as independent (too narrow envelope).
        # Admissibility only depends on the dates, identical within a cluster.
        cell_of = (pd.read_parquet(cfg.paths.processed / "psu_gridmatch.parquet",
                                   columns=["psu_idx", "era5_cell"])
                     .set_index("psu_idx")["era5_cell"])
        hw_cell = hw["psu_idx"].map(cell_of).to_numpy()
        assert not pd.isna(hw_cell).any(), "ERA5 cell missing for some heatwaves"
        clus, _ = pd.factorize(pd.MultiIndex.from_arrays([hw_cell, s_obs, e_obs]))
        _, rep = np.unique(clus, return_index=True)   # first HW of each cluster
        nu = rep.size
        log.info("psu_window null: %s heatwaves = %s distinct (ERA5 cell, start, "
                 "end) clusters; one synchronised draw per cluster",
                 f"{n:,}", f"{nu:,}")
        n_adm_years = np.zeros(nu, np.int64)
        for y in years:                       # one vector pass per year, O(n)
            a = _transfer_onset(o_month[rep], o_day[rep], np.full(nu, y))
            L, U = _admissible_shifts(a, durm1[rep], s_obs[rep], e_obs[rep],
                                      hw_year[rep], W, K, study_lo, study_hi)
            n_adm_years += (y != hw_year[rep]) & (U >= L)
        if (n_adm_years == 0).any():
            bad = hw.iloc[rep[n_adm_years == 0]][["psu_idx", "start_date", "end_date"]]
            bad.to_parquet(out_dir / "axis2_null_no_admissible_year.parquet", index=False)
            raise RuntimeError(f"psu_window null: {len(bad):,} heatwave(s) have no "
                               "admissible target year — see "
                               "axis2_null_no_admissible_year.parquet.")
        if n:
            log.info("psu_window null: admissible target years per HW: min %d, "
                     "median %d, max %d", n_adm_years.min(),
                     int(np.median(n_adm_years)), n_adm_years.max())

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
    n_perm = getattr(cfg, "n_null_permutations_by_mode", {}).get(
        cfg.hw_threshold_mode, cfg.n_null_permutations)
    log.info("Axis II null: mode=%s, %d permutations x %s HW, +/-%d d, W=%d ...",
             cfg.null_mode, n_perm, f"{n:,}", K, W)
    null = {g: {k: np.full(n_perm, np.nan) for k in ("precursor", "trigger", "count_ratio",
                                                      "n_before", "n_after")}
            for g in groups}
    if cfg.null_mode == "region_pool":
        pool_lab = hw["region"].astype(str).to_numpy()
        pools = {lab: hw_doy[pool_lab == lab] for lab in np.unique(pool_lab)}
    t0 = time.time()
    for k in range(n_perm):
        if cfg.null_mode == "psu_window":
            # (1) target year (one per cluster): uniform among the other years,
            #     rejected while it has no admissible shift -> uniform among
            #     admissible years
            yi = np.empty(nu, np.int64); a_y = np.empty(nu, np.int64)
            L = np.empty(nu, np.int64); U = np.empty(nu, np.int64)
            pending = np.arange(nu)
            while pending.size:
                r_ = rep[pending]
                y_try = rng.integers(0, ny - 1, size=pending.size)
                y_try = y_try + (y_try >= obs_yi[r_])          # skip the observed year
                a_try = _transfer_onset(o_month[r_], o_day[r_], years[y_try])
                l_, u_ = _admissible_shifts(a_try, durm1[r_], s_obs[r_], e_obs[r_],
                                            hw_year[r_], W, K, study_lo, study_hi)
                ok = u_ >= l_
                acc = pending[ok]
                yi[acc], a_y[acc], L[acc], U[acc] = y_try[ok], a_try[ok], l_[ok], u_[ok]
                pending = pending[~ok]
            # (2) shift: uniform among the admissible shifts of that year;
            #     then copied to every heatwave of the cluster
            delta = rng.integers(L, U + 1)
            yi, delta = yi[clus], delta[clus]
            s_start = a_y[clus] + delta
            s_end_chk = s_start + durm1
            # one pseudo-HW per HW (same PSU array, same duration), admissible dates
            assert s_start.shape == (n,) and (s_end_chk - s_start == durm1).all()
            assert ((delta >= -K) & (delta <= K)).all()
            assert ((s_start - W >= study_lo) & (s_end_chk + W <= study_hi)).all()
            assert (years[yi] != hw_year).all() and (_year_of(s_start) != hw_year).all()
            assert ((s_end_chk + W < s_obs) | (s_start - W > e_obs)).all()
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
            null[g]["n_before"][k] = nb      # kept to decompose the ratio:
            null[g]["n_after"][k] = na       # deficit before vs excess after
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
            # decomposition of the ratio: observed counts vs their null
            # distribution (obs / null median < 1 = deficit, > 1 = excess).
            # The ratio of these two ratios approximates, but does not equal,
            # count_ratio_obs / count_ratio_null_med (median of a ratio).
            n_before_null_lo=q(nl["n_before"], .025), n_before_null_med=q(nl["n_before"], .5),
            n_before_null_hi=q(nl["n_before"], .975),
            n_after_null_lo=q(nl["n_after"], .025), n_after_null_med=q(nl["n_after"], .5),
            n_after_null_hi=q(nl["n_after"], .975),
            before_obs_over_null_med=(o["n_before_obs"] / q(nl["n_before"], .5)
                                      if q(nl["n_before"], .5) > 0 else np.nan),
            after_obs_over_null_med=(o["n_after_obs"] / q(nl["n_after"], .5)
                                     if q(nl["n_after"], .5) > 0 else np.nan),
            p_trigger_excess=pval(nl["trigger"], o["trigger_obs"]),
            p_diff_excess=pval(d_null, d_obs),
            p_count_ratio_excess=pval(nl["count_ratio"], o["count_ratio_obs"]),
            n_perm_valid_precursor=int(np.isfinite(nl["precursor"]).sum()),
            n_perm_valid_trigger=int(np.isfinite(nl["trigger"]).sum()),
            n_perm_valid_diff=int(np.isfinite(d_null).sum()),
            n_perm_valid_count_ratio=int(np.isfinite(nl["count_ratio"]).sum())))
        draws.append(pd.DataFrame({"scale": sc, "season": se, "perm": np.arange(n_perm),
                                   "precursor": nl["precursor"], "trigger": nl["trigger"],
                                   "count_ratio": nl["count_ratio"],
                                   "n_before": nl["n_before"], "n_after": nl["n_after"]}))
    out = pd.DataFrame(rows)
    out.to_parquet(out_dir / "axis2_null_region.parquet", index=False)
    pv = [c for c in out.columns if c.startswith("n_perm_valid_")]
    manifest_update("axis2_null", dict(
        null_mode=cfg.null_mode, n_permutations=n_perm, n_hw=n,
        n_distinct_hw_clusters=int(nu) if cfg.null_mode == "psu_window" else None,
        draws_synchronised_by="era5_cell, start, end" if cfg.null_mode == "psu_window" else None,
        p_values="computed and stored in axis2_null_region.parquet; not displayed",
        n_perm_valid=out[["scale", "season"] + pv].to_dict(orient="records")),
        mode=cfg.hw_threshold_mode)
    out[(out["scale"] == "Continental") & (out["season"] == "ALL")].to_parquet(
        out_dir / "axis2_null_continental.parquet", index=False)
    pd.concat(draws, ignore_index=True).to_parquet(out_dir / "axis2_null_draws.parquet", index=False)
    c = out[(out["scale"] == "Continental") & (out["season"] == "ALL")].iloc[0]
    # descriptive analysis: p-values are stored (axis2_null_region.parquet) but
    # not displayed; the comparison is reported as observed vs null envelope
    log.info("Axis II continental: count_ratio %.3f vs null median %.3f [%.3f, %.3f] | "
             "trigger %.3f vs [%.3f, %.3f] | precursor %.3f vs [%.3f, %.3f]",
             c["count_ratio_obs"], c["count_ratio_null_med"], c["count_ratio_null_lo"],
             c["count_ratio_null_hi"], c["trigger_obs"], c["trigger_null_lo"],
             c["trigger_null_hi"], c["precursor_obs"],
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


def check_axis2_consistency(cfg, in_dir: Path, out_dir: Path) -> pd.DataFrame:
    """At W = cfg.axis2_window_days the observed Axis-II quantities must be
    IDENTICAL in step06 (axis2_hw + axis2_attrib), in the observed reference
    of the null and in the W sensitivity, continental and per region:
    n_hw, n_before, n_after, precursor and trigger (ECA) rates. Writes
    axis2_consistency_check.parquet and raises on any mismatch."""
    W = cfg.axis2_window_days
    el = psu_eligibility(cfg, in_dir)
    ok_psu = el.loc[el["axis12_eligible"], "psu_idx"]
    psu = pd.read_parquet(cfg.paths.processed / "psu.parquet",
                          columns=["psu_idx", "region"])
    h = pd.read_parquet(in_dir / "axis2_hw.parquet").merge(psu, on="psu_idx", how="left")
    a = pd.read_parquet(in_dir / "axis2_attrib.parquet").merge(psu, on="psu_idx", how="left")
    h = h[h["psu_idx"].isin(ok_psu)]
    a = a[a["psu_idx"].isin(ok_psu)]
    if cfg.axis2_exclude_truncated:
        h, a = h[~h["truncated"]], a[~a["truncated"]]
    s06 = {}
    for sc in ["Continental"] + sorted(h["region"].dropna().unique()):
        hh = h if sc == "Continental" else h[h["region"] == sc]
        aa = a if sc == "Continental" else a[a["region"] == sc]
        s06[sc] = dict(n_hw=len(hh), n_before=int((aa["position"] == "before").sum()),
                       n_after=int((aa["position"] == "after").sum()),
                       precursor=hh["precursor"].mean() if len(hh) else np.nan,
                       trigger=hh["trigger"].mean() if len(hh) else np.nan)
    nl = pd.read_parquet(out_dir / "axis2_null_region.parquet")
    nl = (nl[nl["season"] == "ALL"].set_index("scale")
            .rename(columns={"n_before_obs": "n_before", "n_after_obs": "n_after",
                             "precursor_obs": "precursor", "trigger_obs": "trigger"}))
    ws = pd.read_parquet(out_dir / "axis2_window_sensitivity.parquet")
    ws = (ws[ws["window_days"] == W].set_index("scale")
            .rename(columns={"precursor_rate": "precursor", "trigger_rate": "trigger"}))
    rows = []
    for sc, ref in s06.items():
        for src_name, src in (("null_observed", nl), ("window_sensitivity", ws)):
            for c in ("n_hw", "n_before", "n_after", "precursor", "trigger"):
                v = src.loc[sc, c] if sc in src.index else np.nan
                r = ref[c]
                if c in ("precursor", "trigger"):
                    same = ((np.isnan(r) and np.isnan(v))
                            or abs(float(v) - float(r)) <= 1e-12)
                else:
                    same = sc in src.index and int(v) == int(r)
                rows.append(dict(scale=sc, source=src_name, quantity=c,
                                 step06=r, other=v, identical=bool(same)))
    rep = pd.DataFrame(rows)
    rep.to_parquet(out_dir / "axis2_consistency_check.parquet", index=False)
    manifest_update("axis2_consistency_ok", bool(rep["identical"].all()),
                    mode=cfg.hw_threshold_mode)
    if not rep["identical"].all():
        raise AssertionError("Axis II observed quantities differ between step06, "
                             "the null reference and the W sensitivity at W="
                             f"{W}:\n{rep[~rep['identical']].to_string(index=False)}")
    log.info("Axis II consistency (step06 == null reference == W sensitivity "
             "at W=%d): OK on %d scales", W, len(s06))
    return rep


def run_step07_significance(cfg, in_dir: Path | None = None,
         out_dir: Path | None = None) -> None:
    in_dir = Path(in_dir or cfg.paths.processed)
    out_dir = Path(out_dir or in_dir / "significance")
    out_dir.mkdir(parents=True, exist_ok=True)
    lmf_tables(cfg, in_dir, out_dir)
    axis2_null(cfg, in_dir, out_dir)
    axis2_window_sensitivity(cfg, in_dir, out_dir)
    check_axis2_consistency(cfg, in_dir, out_dir)

# (definitions only — run cells follow)


