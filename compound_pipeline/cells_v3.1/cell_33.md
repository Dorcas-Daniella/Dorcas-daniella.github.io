## 12. Dependence & seasonality-aware significance

step07_significance.py — Dependence and seasonality-aware significance.

This step answers the two questions a methods reviewer will ask first:

(Q1, Axis I) "Is the heat-rain co-occurrence more frequent than expected
    from the two marginal frequencies — and from their SHARED SEASONALITY?"
    Metric: Likelihood Multiplication Factor (Zscheischler & Seneviratne
    2017),  LMF = P(hot & EPE) / [P(hot) x P(EPE)], in two flavours:
      lmf_naive    : marginals pooled over the year — confounded by the
                     fact that both extremes cluster in particular seasons;
      lmf_seasonal : expected joint days computed day-of-year-wise,
                     E[joint] = sum_d  n_hot(d) * n_epe(d) / n_valid(d),
                     i.e. independence CONDITIONAL on the seasonal cycle.
    lmf_seasonal > 1 is the defensible claim of dependence beyond
    seasonality. Inputs: doy_counts.parquet from step04 (exact, no
    permutation needed), restricted to eligible PSU (epe & temp); its four
    counts share one day mask (Tmax and precip finite). 95% CIs: bootstrap
    over PSU and "cluster bootstrap CI — ERA5 cell" (PSU sharing an ERA5
    cell are resampled together; same point estimate).
    LIMITATION: the day-of-year conditioning pools all years, so
    interannual non-stationarity is not controlled by the seasonal LMF.

(Q2, Axis II) "Is the post-heatwave EPE excess more than the local rainy-season
    calendar alone produces?" STRICT null (cfg.null_mode = "psu_window", PRIMARY):
    every eligible, non-truncated observed heatwave keeps its PSU and DURATION;
    its onset month/day is transferred to another year (29 Feb -> 28 Feb in a
    non-leap year) and shifted by delta, |delta| <= K = cfg.null_window_days (15).
    Admissible (before any draw): the full +/-W window inside the study period,
    the REAL year of the simulated onset != observed onset year (crossing 31 Dec
    is allowed otherwise), no overlap of the window with the observed heatwave.
    Draw: target year uniform among the years with >= 1 admissible delta, then
    delta uniform among them; one pseudo-HW per HW and permutation (asserted).
    Draws are SYNCHRONISED within each distinct heatwave (ERA5 cell, start,
    end): PSU sharing an ERA5 cell have identical heatwaves and receive the
    same surrogate. Descriptive use: observed vs null median / envelope; the
    permutation p-values are stored, not displayed.
    A heatwave without any admissible year stops the run with a diagnostic.
    Local seasonality is preserved, only the fine timing is randomised.
    Surrogates are pushed through the SAME
    `attribute_unique` as the observed data, so the null is produced for the
    PRIMARY metric (after/before count ratio) as well as for the ECA fractions.
    Results by scale x onset season (ALL, DJF, MAM, JJA, SON). The legacy
    regional-pool null is kept as cfg.null_mode = "region_pool".
    Window sensitivity (cfg.axis2_window_sens) recomputes Axis II only, on the
    same eligible population, truncation re-evaluated for each W. At the
    primary W, step06, the null's observed reference and the W sensitivity
    must give identical n_hw, n_before, n_after and ECA rates, continental and
    per region (check_axis2_consistency, raises otherwise). Usable
    permutations are reported per group for precursor, trigger, their
    difference and the count ratio.

OUTPUTS: lmf_by_psu.parquet, lmf_by_scale.parquet, axis2_null_region.parquet,
         axis2_null_continental.parquet, axis2_null_draws.parquet,
         axis2_window_sensitivity.parquet, axis2_consistency_check.parquet
