"""Targeted synthetic tests of the corrections (C1-C17).

ARTIFICIAL DATA ONLY: passing these tests shows that the corrected functions
behave as specified on small constructed cases; it is NOT a validation on the
real ERA5 / CHIRPS / DHS data.

Run:  PIPELINE=compound_pipeline_v3.py ORIGINAL=compound_pipeline_v2_original.py \
      python -m pytest -q tests/test_corrections.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pipeline_loader import load, load_cfg

HERE = Path(__file__).resolve().parent
NEW = Path(os.environ.get("PIPELINE", HERE.parent / "compound_pipeline_v3.py"))
OLD = Path(os.environ.get("ORIGINAL", HERE.parent / "compound_pipeline_v2_original.py"))

P, PNS = load(NEW)          # corrected pipeline (functions only)
O, ONS = load(OLD)          # original v2 (functions only), for before/after checks


@pytest.fixture
def cfg(tmp_path):
    c = load_cfg(NEW, tmp_path)
    c.n_null_permutations_by_mode = {}      # tests set n_null_permutations directly
    PNS["cfg"] = c
    # manifest writes are not under test here
    PNS["manifest_update"] = lambda *a, **k: None
    return c


def D(s):
    return np.datetime64(s, "D")


def dn(s):
    return int(P.to_daynum(np.array([s], dtype="datetime64[D]"))[0])


# --------------------------------------------------------------------------
# C2 / C6 / C7 — calendar positions, leap years, annual mode unchanged
# --------------------------------------------------------------------------
def test_calendar_doy_leap_projection():
    d = pd.to_datetime(["2000-01-01", "2000-02-28", "2000-02-29", "2000-03-01",
                        "2000-12-31", "2001-02-28", "2001-03-01", "2001-12-31"])
    assert P.calendar_doy(d).tolist() == [1, 59, 60, 61, 366, 59, 61, 366]
    # every date of any year maps to a distinct position; 60 only in leap years
    for y in (1999, 2000, 2023, 2024):
        days = pd.date_range(f"{y}-01-01", f"{y}-12-31")
        pos = P.calendar_doy(days)
        assert len(np.unique(pos)) == len(days)
        assert (60 in pos) == (len(days) == 366)


def _grid(dates_values, n_psu, ref_years, index_fn):
    grid = np.full((n_psu, 366, len(ref_years)), np.nan, np.float32)
    for slot, yr in enumerate(ref_years):
        d = pd.date_range(f"{yr}-01-01", f"{yr}-12-31")
        doy = index_fn(d)
        grid[:, doy, slot] = dates_values(d, n_psu)
    return grid


def test_annual_mode_numerically_unchanged():
    rng = np.random.default_rng(0)
    ref = list(range(1985, 2015))
    vals = lambda d, n: (25 + 5 * np.sin(2 * np.pi * d.dayofyear.to_numpy() / 365)[None, :]
                         + rng.normal(0, 2, (n, len(d)))).astype(np.float32)
    old_idx = lambda d: d.dayofyear.to_numpy() - 1
    new_idx = lambda d: P.calendar_doy(d).astype(np.int64) - 1
    rng = np.random.default_rng(0); g_old = _grid(vals, 4, ref, old_idx)
    rng = np.random.default_rng(0); g_new = _grid(vals, 4, ref, new_idx)
    for corr in ("none", "loo"):
        b_old, i_old = O.annual_percentiles_with_inbase(g_old, 90.0, corr)
        b_new, i_new = P.annual_percentiles_with_inbase(g_new, 90.0, corr)
        assert np.array_equal(b_old, b_new)
        if corr != "none":
            assert np.array_equal(i_old, i_new)


def test_calendar_window_feb29_has_30_real_dates_in_non_leap_year():
    win = PNS["_window_indices"](15)
    assert all(len(w) == 31 for w in win)
    w = win[59]                                    # window centred on 29 Feb
    pos_2001 = P.calendar_doy(pd.date_range("2001-01-01", "2001-12-31")) - 1
    pos_2000 = P.calendar_doy(pd.date_range("2000-01-01", "2000-12-31")) - 1
    assert np.isin(w, pos_2001).sum() == 30 and np.isin(w, pos_2000).sum() == 31


def test_attach_tmax_threshold_uses_calendar_positions(cfg):
    base_map = pd.Series(np.arange(1, 367, dtype=float),
                         index=pd.MultiIndex.from_arrays([np.zeros(366, int),
                                                          np.arange(1, 367)]))
    blk = pd.DataFrame({"psu_idx": 0, "date": pd.to_datetime(
        ["2000-02-29", "2000-03-01", "2001-03-01", "2001-12-31"])})
    out = P._attach_tmax_threshold(blk, base_map, {}, cfg)
    assert out["doy"].tolist() == [60, 61, 61, 366]
    assert out["thr_tmax"].tolist() == [60.0, 61.0, 61.0, 366.0]


def test_attach_precip_threshold_missing_inbase_column_raises(cfg):
    pr = pd.DataFrame({"psu_idx": [0], "p95_base": [10.0], "p95_reliable": [True]})
    blk = pd.DataFrame({"psu_idx": [0, 0], "date": pd.to_datetime(["1990-01-01", "2020-01-01"])})
    P._attach_precip_threshold(blk.copy(), pr, cfg)          # "none": fine
    cfg.in_base_correction = "loo"
    with pytest.raises(KeyError):
        P._attach_precip_threshold(blk.copy(), pr, cfg)


# --------------------------------------------------------------------------
# C9 — unique attribution: tie rules, brute-force equivalence
# --------------------------------------------------------------------------
def brute_attribution(hw_psu, s, e, epe_psu, epe_day, W):
    out = []
    for p, d in zip(epe_psu, epe_day):
        ids = np.flatnonzero(hw_psu == p)
        dur = [i for i in ids if s[i] <= d <= e[i]]
        if dur:
            out.append(("during", None, 0)); continue
        aft = [i for i in ids if e[i] < d and d - e[i] <= W]
        bef = [i for i in ids if s[i] > d and s[i] - d <= W]
        ia = min(aft, key=lambda i: (-e[i], s[i], i)) if aft else None
        ib = min(bef, key=lambda i: (s[i], i)) if bef else None
        da = d - e[ia] if ia is not None else np.inf
        db = s[ib] - d if ib is not None else np.inf
        if ib is not None and db < da:
            out.append(("before", ib, -db))
        elif ia is not None:
            out.append(("after", ia, da))
        else:
            out.append(("none", None, 0))
    return out


def test_attribute_unique_matches_brute_force_with_overlaps():
    rng = np.random.default_rng(1)
    for rep in range(30):
        n_hw = 40
        hw_psu = rng.integers(0, 3, n_hw)
        s = rng.integers(0, 200, n_hw)
        # many exact ties in start and end on purpose
        s = (s // 5) * 5
        e = s + rng.choice([2, 4, 4, 9], n_hw)
        epe_psu = rng.integers(0, 3, 300)
        epe_day = rng.integers(-40, 260, 300)
        W = 30
        idx, pos, off = P.attribute_unique(hw_psu, s, e, epe_psu, epe_day, W)
        ref = brute_attribution(hw_psu, s, e, epe_psu, epe_day, W)
        names = np.array(["none", "before", "during", "after"])
        for k, (p_ref, i_ref, o_ref) in enumerate(ref):
            assert names[pos[k]] == p_ref, (rep, k)
            if p_ref in ("before", "after"):
                assert idx[k] == i_ref and off[k] == o_ref, (rep, k, idx[k], i_ref)


def test_after_tie_on_equal_end_picks_earliest_start_then_lowest_id():
    # HW0 [10,20], HW1 [5,20], HW2 [5,20]; EPE at 25 -> after, end tie
    hw_psu = np.zeros(3, int); s = np.array([10, 5, 5]); e = np.array([20, 20, 20])
    idx, pos, _ = P.attribute_unique(hw_psu, s, e, np.array([0]), np.array([25]), 30)
    assert pos[0] == 3 and idx[0] == 1
    idx_o, _, _ = O.attribute_unique(hw_psu, s, e, np.array([0]), np.array([25]), 30)
    assert idx_o[0] == 2          # documents the v2 behaviour that was corrected


def test_before_after_equal_distance_goes_to_earlier_heatwave():
    hw_psu = np.zeros(2, int); s = np.array([0, 30]); e = np.array([10, 40])
    idx, pos, off = P.attribute_unique(hw_psu, s, e, np.array([0]), np.array([20]), 30)
    assert pos[0] == 3 and idx[0] == 0 and off[0] == 10


def test_during_carrier_does_not_change_before_after_counts():
    rng = np.random.default_rng(2)
    hw_psu = np.zeros(6, int)
    s = np.array([0, 2, 50, 52, 100, 101]); e = s + np.array([10, 10, 5, 5, 3, 3])
    epe_psu = np.zeros(200, int); epe_day = rng.integers(-20, 140, 200)
    idx, pos, _ = P.attribute_unique(hw_psu, s, e, epe_psu, epe_day, 30)
    perm = rng.permutation(6)
    inv = np.argsort(perm)
    idx2, pos2, _ = P.attribute_unique(hw_psu[perm], s[perm], e[perm], epe_psu, epe_day, 30)
    assert np.array_equal(pos, pos2)
    for mask in (np.ones(6, bool), np.array([1, 0, 1, 0, 1, 0], bool)):
        nb, na = P.before_after_counts(pos, idx, mask)
        assert (nb, na) == P.before_after_counts(pos2, idx2, mask[perm])


def _axis2_inputs():
    hw = pd.DataFrame({
        "hw_id": [0, 1], "psu_idx": [0, 0],
        "start_date": pd.to_datetime(["1985-01-10", "1985-02-20"]),
        "end_date": pd.to_datetime(["1985-01-15", "1985-02-25"]),
        "duration": [6, 6], "peak_tmax": [40.0, 40.0]})
    epe = pd.DataFrame({"psu_idx": [0], "date": pd.to_datetime(["1985-01-30"]),
                        "precip": [50.0]})
    return hw, epe


def test_axis2_truncated_excluded_before_attribution(cfg):
    hw, epe = _axis2_inputs()
    lo, hi = dn("1985-01-01"), dn("2024-12-31")
    hw_out, pairs, attrib = P._axis2(hw, epe, cfg, lo, hi)
    assert hw_out["truncated"].tolist() == [True, False]
    assert len(attrib) == 1
    r = attrib.iloc[0]
    assert r["hw_id"] == 1 and r["position"] == "before" and r["offset_days"] == -21
    assert not r["truncated"]
    # v2: the EPE was captured by the truncated HW0, then lost downstream
    ONS["log"] = PNS["log"]
    _, _, attrib_o = O._axis2(hw, epe, cfg, lo, hi)
    assert attrib_o.iloc[0]["hw_id"] == 0 and attrib_o.iloc[0]["truncated"]


# --------------------------------------------------------------------------
# C11 — psu_window null: transfer, admissible shifts, draw order
# --------------------------------------------------------------------------
def test_transfer_onset_feb29():
    a = PNS["_transfer_onset"](np.array([2, 2, 3]), np.array([29, 29, 1]),
                               np.array([2001, 2004, 2001]))
    assert a.tolist() == [dn("2001-02-28"), dn("2004-02-29"), dn("2001-03-01")]


def brute_shifts(a, d, s_obs, e_obs, obs_year, W, K, lo, hi):
    ok = []
    for delta in range(-K, K + 1):
        s = a + delta
        if s - W < lo or s + d + W > hi:
            continue
        if int(PNS["_year_of"](np.array([s]))[0]) == obs_year:
            continue
        if not (s + d + W < s_obs or s - W > e_obs):
            continue
        ok.append(delta)
    return ok


@pytest.mark.parametrize("W,K,dmax", [(30, 15, 20), (60, 15, 330), (7, 3, 5)])
def test_admissible_shifts_equal_brute_force(W, K, dmax):
    rng = np.random.default_rng(3)
    lo, hi = dn("1985-01-01"), dn("2024-12-31")
    starts = np.concatenate([
        rng.integers(lo, hi - dmax, 300),
        np.array([dn(f"{y}-12-{dd}") for y in (1985, 1990, 2023) for dd in (17, 25, 31)]),
        np.array([dn(f"{y}-01-{dd:02d}") for y in (1986, 2000, 2024) for dd in (1, 5, 14)]),
        np.array([dn("2000-02-29"), dn("1996-02-29")])])
    for s_obs in starts:
        d = int(rng.integers(2, dmax + 1))
        if s_obs + d > hi:
            continue
        sd = pd.Timestamp(np.datetime64(int(s_obs), "D"))
        for y in range(1985, 2025):
            if y == sd.year:
                continue
            a = int(PNS["_transfer_onset"](np.array([sd.month]), np.array([sd.day]), np.array([y]))[0])
            L, U = PNS["_admissible_shifts"](np.array([a]), np.array([d]), np.array([s_obs]),
                                             np.array([s_obs + d]), np.array([sd.year]),
                                             W, K, lo, hi)
            ref = brute_shifts(a, d, s_obs, s_obs + d, sd.year, W, K, lo, hi)
            got = list(range(int(L[0]), int(U[0]) + 1))
            assert got == ref, (sd, y, d, got[:3], ref[:3])


def test_year_crossing_allowed_unless_it_lands_in_observed_year():
    lo, hi = dn("1985-01-01"), dn("2024-12-31")
    s_obs = dn("2000-12-28"); d = 3
    f = PNS["_admissible_shifts"]
    a95 = dn("1995-12-28")
    L, U = f(np.array([a95]), np.array([d]), np.array([s_obs]), np.array([s_obs + d]),
             np.array([2000]), 30, 15, lo, hi)
    assert (L[0], U[0]) == (-15, 15)           # may start in 1996: allowed
    a99 = dn("1999-12-28")
    L, U = f(np.array([a99]), np.array([d]), np.array([s_obs]), np.array([s_obs + d]),
             np.array([2000]), 30, 15, lo, hi)
    assert (L[0], U[0]) == (-15, 3)            # Jan 1 2000 onwards excluded


def _write_null_inputs(cfg, hws, cells=None):
    pr = cfg.paths.processed
    pr.mkdir(parents=True, exist_ok=True)
    n_psu = int(max(h[0] for h in hws)) + 1
    pd.DataFrame({"psu_idx": np.arange(n_psu), "country": "Kenya", "region": "Eastern",
                  "lat": 0.0, "lon": 37.0}).to_parquet(pr / "psu.parquet")
    pd.DataFrame({"psu_idx": np.arange(n_psu), "chirps_valid": True, "era5_valid": True,
                  "chirps_too_far": False, "chirps_all_nan": False, "era5_too_far": False,
                  "era5_cell": cells if cells is not None else 0}).to_parquet(pr / "psu_gridmatch.parquet")
    pd.DataFrame({"psu_idx": np.arange(n_psu), "p95_reliable": True,
                  "n_wet_days": 500}).to_parquet(pr / "thresholds_precip.parquet")
    hw = pd.DataFrame({"hw_id": np.arange(len(hws)), "psu_idx": [h[0] for h in hws],
                       "start_date": pd.to_datetime([h[1] for h in hws]),
                       "end_date": pd.to_datetime([h[2] for h in hws])})
    hw["duration"] = (hw.end_date - hw.start_date).dt.days + 1
    hw["peak_tmax"] = 40.0; hw["truncated"] = False
    hw["precursor"] = False; hw["trigger"] = False; hw["during"] = False
    hw.to_parquet(pr / "axis2_hw.parquet")
    pd.DataFrame({"psu_idx": [0], "date": pd.to_datetime(["1990-06-01"])}).to_parquet(pr / "epe.parquet")


def test_null_draw_order_uniform_years_then_shifts(cfg):
    # one HW at the edge: onset 1985-02-20 (W=30 -> admissible), observed year 1985
    _write_null_inputs(cfg, [(0, "1985-02-20", "1985-02-22")])
    cfg.n_null_permutations = 4000
    starts = []
    real = PNS["eca_indicators"]

    def spy(hw_psu, s, e, keys, W, span=1_000_000):
        starts.append(np.asarray(s).copy())
        return real(hw_psu, s, e, keys, W, span)
    PNS["eca_indicators"] = spy
    try:
        P.axis2_null(cfg, cfg.paths.processed, cfg.paths.processed)
    finally:
        PNS["eca_indicators"] = real
    s = np.concatenate(starts)
    assert len(s) == 4000                          # one pseudo-HW per permutation
    yrs = PNS["_year_of"](s)
    assert 1985 not in set(yrs.tolist())
    # every year 1986..2024 admissible with all 31 shifts -> year frequencies ~ uniform
    cnt = pd.Series(yrs).value_counts().reindex(range(1986, 2025), fill_value=0)
    from scipy import stats
    assert stats.chisquare(cnt.to_numpy()).pvalue > 1e-3
    shift = s - PNS["_transfer_onset"](np.full(len(s), 2), np.full(len(s), 20), yrs)
    assert shift.min() == -15 and shift.max() == 15
    assert stats.chisquare(pd.Series(shift).value_counts().sort_index().to_numpy()).pvalue > 1e-3


def test_null_draws_synchronised_within_era5_cell(cfg):
    # PSU 0 and 1 share ERA5 cell 0 and the same heatwave (identical Tmax series);
    # PSU 2 has the same dates but another cell -> independent draw
    _write_null_inputs(cfg, [(0, "2000-06-10", "2000-06-14"), (1, "2000-06-10", "2000-06-14"),
                             (2, "2000-06-10", "2000-06-14")], cells=[0, 0, 1])
    cfg.n_null_permutations = 300
    starts = []
    real = PNS["eca_indicators"]

    def spy(hw_psu, s, e, keys, W, span=1_000_000):
        starts.append(np.asarray(s).copy())
        return real(hw_psu, s, e, keys, W, span)
    PNS["eca_indicators"] = spy
    try:
        P.axis2_null(cfg, cfg.paths.processed, cfg.paths.processed)
    finally:
        PNS["eca_indicators"] = real
    S = np.vstack(starts)                            # (n_perm, 3), hw_id order
    assert (S[:, 0] == S[:, 1]).all()                # same cluster -> same surrogate
    assert (S[:, 0] != S[:, 2]).mean() > 0.9         # other cell -> independent


def test_null_keeps_before_after_counts_and_per_mode_permutations(cfg):
    _write_null_inputs(cfg, [(0, "2000-06-10", "2000-06-14"), (1, "1995-03-01", "1995-03-04")])
    pd.DataFrame({"psu_idx": [0, 0, 1], "date": pd.to_datetime(
        ["2000-06-01", "2000-06-20", "1995-03-10"])}).to_parquet(cfg.paths.processed / "epe.parquet")
    cfg.n_null_permutations_by_mode = {"calendar": 37, "annual": 11}
    cfg.hw_threshold_mode = "annual"
    P.axis2_null(cfg, cfg.paths.processed, cfg.paths.processed)
    dr = pd.read_parquet(cfg.paths.processed / "axis2_null_draws.parquet")
    nl = pd.read_parquet(cfg.paths.processed / "axis2_null_region.parquet")
    c = nl[(nl.scale == "Continental") & (nl.season == "ALL")].iloc[0]
    d = dr[(dr.scale == "Continental") & (dr.season == "ALL")]
    assert len(d) == 11                                   # per-mode count used
    assert {"n_before", "n_after"} <= set(dr.columns)
    assert c["n_before_obs"] == 1 and c["n_after_obs"] == 2
    assert np.isclose(c["n_before_null_med"], np.nanmedian(d["n_before"]))
    ok = d["n_before"] > 0
    assert np.allclose(d.loc[ok, "count_ratio"], d.loc[ok, "n_after"] / d.loc[ok, "n_before"])
    if c["n_after_null_med"] > 0:
        assert np.isclose(c["after_obs_over_null_med"], 2 / c["n_after_null_med"])


def test_null_stops_when_no_admissible_year(cfg):
    # a heatwave longer than the period allows for any other year (W=30)
    _write_null_inputs(cfg, [(0, "1985-02-01", "2024-11-01")])
    with pytest.raises(RuntimeError, match="no admissible target year"):
        P.axis2_null(cfg, cfg.paths.processed, cfg.paths.processed)
    assert (cfg.paths.processed / "axis2_null_no_admissible_year.parquet").exists()


# --------------------------------------------------------------------------
# C3 — eligibility helper
# --------------------------------------------------------------------------
def test_psu_eligibility_flags(cfg):
    pr = cfg.paths.processed; pr.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"psu_idx": [0, 1, 2, 3], "chirps_valid": [True, True, False, True],
                  "era5_valid": [True, False, True, True]}).to_parquet(pr / "psu_gridmatch.parquet")
    pd.DataFrame({"psu_idx": [0, 1, 2, 3], "p95_reliable": [True, True, False, False]}
                 ).to_parquet(pr / "thresholds_precip.parquet")
    el = P.psu_eligibility(cfg).set_index("psu_idx")
    assert el["epe_eligible"].tolist() == [True, True, False, False]
    assert el["temp_eligible"].tolist() == [True, False, True, True]
    assert el["spei_eligible"].tolist() == [True, False, False, True]
    assert el["axis12_eligible"].tolist() == [True, False, False, False]


# --------------------------------------------------------------------------
# C8 — SPEI / SPI NaN classification with the real xclim
# --------------------------------------------------------------------------
def _monthly_inputs():
    t = pd.date_range("1985-01-01", "2024-12-01", freq="MS")
    rng = np.random.default_rng(4)
    pr = pd.DataFrame(rng.gamma(2.0, 40.0, (len(t), 4)), index=t, columns=[10, 11, 12, 13])
    pr.loc[pr.index.month == 7, 11] = 0.0             # dry July every year (SPI zeros)
    pr.loc["1990-05-01", 12] = np.nan                 # one missing month
    m = (pr.index.month == 9) & (pr.index.year <= 2014) & (pr.index.year != 2000)
    pr.loc[m, 13] = np.nan                            # September: 1 calibration value only
    pet = pd.DataFrame(100.0 + rng.normal(0, 10, pr.shape), index=t, columns=pr.columns)
    return pr, pr - pet


def test_standardize_nan_categories_and_methods(cfg):
    pr, wb = _monthly_inputs()
    for kind, wide in (("spei", wb), ("spi", pr)):
        df, info = P._standardize(wide, cfg, kind)
        assert len(info["unexplained"]) == 0, info["unexplained"].head()
        nb = info["nan_by_psu"].set_index("psu_idx")
        assert (nb["n_spinup"] == 2).all()
        assert nb.loc[12, "n_missing_input"] == 3      # May, Jun, Jul 1990
        assert nb.loc[13, "n_insufficient_calibration"] > 0
        assert nb.loc[10, ["n_missing_input", "n_insufficient_calibration"]].sum() == 0
        assert nb["n_nan"].sum() == df[f"{kind}3"].isna().sum()
    _, info_s = P._standardize(wb, cfg, "spei")
    _, info_p = P._standardize(pr, cfg, "spi")
    assert info_s["method"] == "ML"                  # PWM not implemented for fisk
    assert info_p["method"] == "APP"


def test_spi_dry_month_zero_values_are_finite(cfg):
    pr, _ = _monthly_inputs()
    pr = pr[[11]].copy()
    pr.loc[(pr.index.month >= 6) & (pr.index.month <= 8), 11] = 0.0   # JJA all zero
    df, info = P._standardize(pr, cfg, "spi")
    aug = df[(df["month"] == 8) & (df["year"] > 1985)]
    assert aug["spi3"].notna().all()                 # zeros handled separately
    assert len(info["unexplained"]) == 0


def test_unexplained_nan_is_detected(cfg):
    pr, wb = _monthly_inputs()
    from xclim.indices.stats import preprocess_standardized_index
    from xclim.indices import standardized_precipitation_evapotranspiration_index as f
    da = xr.DataArray(wb.to_numpy(), dims=("time", "psu_idx"),
                      coords={"time": wb.index.values, "psu_idx": wb.columns.to_numpy()},
                      attrs={"units": "mm/month"})
    out = f(da, freq="MS", window=3, dist="fisk", method="ML",
            cal_start="1985-01-01", cal_end="2014-12-31")
    rolled, _ = preprocess_standardized_index(da, freq="MS", window=3)
    out[100, 0] = np.nan                             # inject a NaN xclim would not give
    _, unx = P._classify_si_nans(rolled, out, cfg, zero_inflated=False)
    assert len(unx) == 1 and unx.iloc[0]["psu_idx"] == 10


# --------------------------------------------------------------------------
# C13 — Axis I dating and complete PSU x decade table
# --------------------------------------------------------------------------
def test_axis1_dating_and_complete_decades(cfg, tmp_path):
    in_dir = tmp_path / "mode"; agg_dir = in_dir / "aggregations"
    agg_dir.mkdir(parents=True)
    psu = pd.DataFrame({"psu_idx": [0, 1, 2], "country": "Kenya", "region": "Eastern",
                        "era5_cell": [0, 0, 1], "scale": "Continental",
                        "axis12_eligible": [True, True, False]})
    hw = pd.DataFrame({"hw_id": [0, 1], "psu_idx": [0, 2],
                       "start_date": pd.to_datetime(["1994-12-30", "2000-05-01"])})
    hw["hw_year"] = hw.start_date.dt.year.astype("int16")
    hw["decade"] = P.assign_decade(hw["hw_year"], cfg.decade_bounds)
    pd.DataFrame({"hw_id": [0, 1], "psu_idx": [0, 2],
                  "start_date": pd.to_datetime(["1994-12-30", "2000-05-01"]),
                  "epe_date": pd.to_datetime(["1995-01-02", "2000-05-02"])}
                 ).to_parquet(in_dir / "axis1.parquet")
    PNS["_write"] = lambda df, d, name: df.to_parquet(d / f"{name}.parquet")
    P.axis1_aggregations(cfg, in_dir, agg_dir, psu, hw)
    y = pd.read_parquet(agg_dir / "axis1_by_year_continental.parquet").set_index("year")
    assert y.loc[1994, "n_hw"] == 1 and y.loc[1994, "n_hw_with_cooc"] == 1
    assert y.loc[1995, "n_cooccur_days"] == 1 and y.loc[1995, "n_hw_with_cooc"] == 0
    assert y.loc[1994, "pct_hw_with_cooc_a"] == 100.0
    assert y["n_hw"].sum() == 1                      # ineligible PSU 2 excluded
    dec = pd.read_parquet(agg_dir / "axis1_by_psu_decade.parquet")
    assert len(dec) == 2 * 4                         # all eligible PSU x decades
    d0 = dec[dec.psu_idx == 0].set_index(dec[dec.psu_idx == 0]["decade"].astype(str))
    assert d0.loc["1985-1994", "n_hw_with_cooc"] == 1 and d0.loc["1985-1994", "n_cooccur_days"] == 0
    assert d0.loc["1995-2004", "n_cooccur_days"] == 1
    assert (dec[dec.psu_idx == 1][["n_hw", "n_cooccur_days"]] == 0).all().all()


# --------------------------------------------------------------------------
# C14 — trends with missing years
# --------------------------------------------------------------------------
def test_trends_internal_gap_and_edge_nan(cfg):
    cfg.n_bootstrap = 200
    rng = np.random.default_rng(5)
    years = np.arange(1985, 2025)
    base = 0.1 * (years - 1985) + rng.normal(0, .3, len(years))
    edge = base.copy(); edge[:3] = np.nan; edge[-2:] = np.nan
    gap = base.copy(); gap[10] = np.nan
    df = pd.DataFrame({"year": years, "edge": edge, "gap": gap})
    rows = {r["metric"]: r for r in P._trend_rows(df, None, ["edge", "gap"],
                                                 np.random.default_rng(0), cfg)}
    assert rows["edge"]["inference_ok"] and np.isfinite(rows["edge"]["mk_p"])
    assert np.isfinite(rows["edge"]["ci_low_per_decade"])
    g = rows["gap"]
    assert not g["inference_ok"] and np.isnan(g["mk_p"]) and np.isnan(g["ci_low_per_decade"])
    assert np.isclose(g["slope_per_decade"], 10 * P.theil_sen_slope(years, gap))
    # BH: NaN never rejected and not counted in m
    rej = P.benjamini_hochberg(np.array([0.001, np.nan, 0.04, np.nan]), 0.05)
    assert rej.tolist() == [True, False, True, False]


# --------------------------------------------------------------------------
# C15 — figure 6 helpers
# --------------------------------------------------------------------------
def test_order_statistic_threshold():
    f = PNS["order_stat_threshold"]
    assert f([4, 1, 3, 2], .75) == 3
    assert f([1, 2, 3], 2 / 3) == 2
    assert f([5.0], .75) == 5.0
    assert np.isnan(f([], .75))
    assert f([0, 0, 0, 1], .75) == 0                 # zero threshold is acceptable
    assert f([1, np.inf, np.inf, np.inf], .75) == np.inf


def test_regime_classification_rules():
    df = pd.DataFrame({
        "cooccur_days_per_year": [1.0, 0.0, 2.0, 1.0, np.nan],
        "count_ratio_fig6": [np.inf, 0.0, 2.0, 1.0, 1.0],
        "pct_drought_psu": [50.0, 10.0, 30.0, 30.0, 30.0],
        "support_axis2": [12, 15, 3, 20, 20], "support_axis3": [30, 30, 30, 2, 30],
        "fig6_eligible": [True, True, True, True, True]})
    ok = PNS["regime_classifiable"](df, 10, 5)
    assert ok.tolist() == [True, True, False, False, False]
    thr = {"axis1": 0.0, "axis2": 1.0, "axis3": 30.0}
    lab = PNS["classify_regimes"](df, thr, ok)
    assert lab[0] == "All three"                     # inf > 1, 1 > 0, 50 > 30
    assert lab[1] == "None prominent"                # strictly above: 0 > 0 False
    assert (lab[2:] == PNS["INSUFFICIENT"]).all()


# --------------------------------------------------------------------------
# C5 — daily alignment checks
# --------------------------------------------------------------------------
def _nc(path, dates, name, lat_name, lon_name, time_name, lat_desc=False):
    lat = np.array([0.0, 0.25]); lon = np.array([10.0, 10.25])
    if lat_desc:
        lat = lat[::-1]
    v = np.ones((len(dates), 2, 2), np.float32)
    ds = xr.Dataset({k: ((time_name, lat_name, lon_name), v) for k in name},
                    coords={time_name: dates, lat_name: lat, lon_name: lon})
    ds.to_netcdf(path)


def test_open_and_align_checks(cfg):
    pr = cfg.paths.processed; pr.mkdir(parents=True, exist_ok=True)
    full = pd.date_range("1985-01-01", "2024-12-31")
    _nc(cfg.paths.era5_nc, full + pd.Timedelta(hours=12), ["tmax", "tmin"],
        "latitude", "longitude", "valid_time", lat_desc=True)
    _nc(cfg.paths.chirps_nc, pd.date_range("1984-12-01", "2025-01-31"), ["precip"],
        "latitude", "longitude", "time")
    era5, chirps, common = P._open_and_align(cfg)
    assert common.size == len(full)
    assert (era5["time"].values.astype("datetime64[D]") == common).all()
    era5.close(); chirps.close()
    # one missing day in CHIRPS -> explicit error
    _nc(cfg.paths.chirps_nc, full.delete(100), ["precip"], "latitude", "longitude", "time")
    with pytest.raises(RuntimeError, match="CHIRPS"):
        P._open_and_align(cfg)
    # duplicated date in ERA5 -> explicit error
    _nc(cfg.paths.chirps_nc, full, ["precip"], "latitude", "longitude", "time")
    dup = full.insert(50, full[50])
    _nc(cfg.paths.era5_nc, dup, ["tmax", "tmin"], "latitude", "longitude", "valid_time")
    with pytest.raises(RuntimeError, match="ERA5"):
        P._open_and_align(cfg)


# --------------------------------------------------------------------------
# C4 — manifest serialisation
# --------------------------------------------------------------------------
def test_manifest_jsonable_strict():
    j = P._jsonable({"a": np.nan, "b": np.int64(3), "c": (1, 2.5), "d": Path("/x"),
                     "e": np.bool_(True), "f": np.inf})
    s = json.dumps(j, allow_nan=False)
    assert json.loads(s) == {"a": None, "b": 3, "c": [1, 2.5], "d": "/x", "e": True, "f": None}
