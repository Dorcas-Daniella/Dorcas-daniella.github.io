"""End-to-end SMOKE RUN of the whole pipeline on ARTIFICIAL inputs
(make_synthetic_inputs.py), then post-run checks of the corrections.
Success means the code runs top to bottom and the asserted properties hold
on these artificial data — not that results on the real data are valid.

usage: python run_e2e_synthetic.py PIPELINE.py WORKDIR
(WORKDIR must contain Code_PhD/ with the three synthetic input files)
"""
import json
import os
import re
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
pipeline, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
src = pipeline.read_text()
# ---- only paths and run-size knobs are changed ------------------------------
src = re.sub(r'BASE_DIR = Path\(r".*?"\)', f'BASE_DIR = Path(r"{work}")', src, count=1)
for k, v in {"n_null_permutations": 60, "n_lmf_surrogates": 60, "n_bootstrap": 60,
             "psu_block_size": 8, "extraction_batch_size": 9}.items():
    src, n = re.subn(rf"\b{k}=\d+", f"{k}={v}", src, count=1)
    assert n == 1, k
src, n = re.subn(r'n_null_permutations_by_mode=\{[^}]*\}',
                 'n_null_permutations_by_mode={"calendar": 60, "annual": 40}', src, count=1)
assert n == 1, "n_null_permutations_by_mode"
for k in ("regime_min_support_axis2", "regime_min_support_axis3"):   # diagnostics-only path
    src, n = re.subn(rf"\b{k}=\d+", f"{k}=None", src, count=1)
    assert n == 1, k
run_file = work / "pipeline_e2e.py"
run_file.write_text(src)
g = runpy.run_path(str(run_file), run_name="__main__")

cfg, PROCESSED, MODE_DIRS = g["cfg"], g["PROCESSED"], g["MODE_DIRS"]
ok = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    ok.append(bool(cond))


man = json.loads((PROCESSED / "run_manifest.json").read_text())
check("manifest: config/seed/packages/inputs/code",
      all(k in man for k in ("config", "rng_seed", "packages", "inputs", "code")))
check("manifest: input identities (size, mtime)",
      all(man["inputs"][k]["exists"] and man["inputs"][k]["size_bytes"] > 0 for k in man["inputs"]))
check("manifest: SPEI ML / SPI APP recorded",
      man["spei_spi"]["spei"]["method_effective"] == "ML"
      and man["spei_spi"]["spi"]["method_effective"] == "APP")
check("manifest: eligibility exclusions",
      man["eligibility"]["excluded_chirps_all_nan"] >= 1
      and man["eligibility"]["excluded_era5_too_far"] >= 1
      and man["eligibility"]["excluded_wet_days_below_min"] >= 1)
nc = man["spei_spi"]["nan_categories"]["spei3"]
check("SPEI NaN categories: all explained, ineligible counted on full grid",
      nc["unexplained"] == 0 and nc["ineligible_psu_cells"] > 0 and nc["missing_input"] > 0
      and nc["insufficient_calibration"] > 0
      and nc["grid_cells"] == nc["ineligible_psu_cells"] + nc["spinup"] + nc["missing_input"]
      + nc["insufficient_calibration"] + nc["valid"])
for m in cfg.hw_threshold_modes:
    mm = man["modes"][m]
    check(f"[{m}] manifest: usable permutations per metric",
          all(r["n_perm_valid_count_ratio"] >= 0 and "n_perm_valid_diff" in r
              for r in mm["axis2_null"]["n_perm_valid"]))
    check(f"[{m}] Axis II consistency step06 == null == W-sens", mm["axis2_consistency_ok"])

el = g["psu_eligibility"](cfg)
spei = pd.read_parquet(PROCESSED / "spei3.parquet")
check("spei3: only spei_eligible PSU", set(spei.psu_idx) <= set(el.psu_idx[el.spei_eligible]))
check("is_drought nullable, <NA> exactly where index is NaN",
      str(spei["is_drought_spei3"].dtype) == "boolean"
      and (spei["is_drought_spei3"].isna() == spei["spei3"].isna()).all())

daily = pd.concat([pd.read_parquet(p) for p in (PROCESSED / "climate_daily").glob("year=*")])
for m, D in MODE_DIRS.items():
    dc = pd.read_parquet(D / "doy_counts.parquet")
    both = (daily.tmax.notna() & daily.precip.notna()).groupby(daily.psu_idx).sum()
    check(f"[{m}] doy_counts n_valid == days with Tmax & precip finite",
          (dc.groupby("psu_idx").n_valid.sum().reindex(both.index) == both).all())
    check(f"[{m}] 29 Feb kept as position 60 (10 leap years)",
          (dc[dc.doy == 60].groupby("psu_idx").n_valid.sum().max() == 10))
    epe = pd.read_parquet(D / "epe.parquet")
    check(f"[{m}] EPE only at epe_eligible PSU",
          set(epe.psu_idx) <= set(el.psu_idx[el.epe_eligible]))
    att = pd.read_parquet(D / "axis2_attrib.parquet")
    check(f"[{m}] attribution never uses truncated HW", not att.truncated.any())
    a1p = pd.read_parquet(D / "aggregations" / "axis1_by_psu.parquet")
    check(f"[{m}] Axis I population = axis12_eligible",
          set(a1p.psu_idx) == set(el.psu_idx[el.axis12_eligible]))
    a1d = pd.read_parquet(D / "aggregations" / "axis1_by_psu_decade.parquet")
    check(f"[{m}] axis1_by_psu_decade complete", len(a1d) == el.axis12_eligible.sum() * 4)
    a3p = pd.read_parquet(D / "aggregations" / "axis3_by_psu.parquet")
    check(f"[{m}] Axis III population within spei_eligible",
          set(a3p.psu_idx) <= set(el.psu_idx[el.spei_eligible]))
    files = sorted(p.name for p in (D / "aggregations").glob("*.parquet"))
    check(f"[{m}] no cell-collapse outputs", not any("cellcollapse" in f for f in files))
    tr = pd.read_parquet(D / "trends" / "all_trends_summary.parquet")
    check(f"[{m}] trends: inference_ok present, no cellcollapse scope",
          "inference_ok" in tr and not tr.scope.str.contains("cellcollapse").any())
    bad = tr[~tr.inference_ok.astype(bool)]
    check(f"[{m}] trends: inference_ok False -> MK and CI NaN",
          bad[["mk_p", "ci_low_per_decade", "ci_high_per_decade"]].isna().all().all())
    a3y = pd.read_parquet(D / "aggregations" / "axis3_by_year_country.parquet")
    check(f"[{m}] Axis III n_units_eligible present", "n_units_eligible" in a3y)
    F = g["FIG_DIRS"][m]
    check(f"[{m}] Fig 6 support diagnostics only (minimums None)",
          (F / "fig06_support_diagnostics.csv").exists()
          and not (F / "fig06_regimes.png").exists())
    sig = pd.read_parquet(D / "significance" / "lmf_by_scale.parquet")
    check(f"[{m}] LMF CI labels", "cluster bootstrap CI — ERA5 cell" in set(sig.ci_method))

# manifest completed with the full code fingerprint; null synchronised; p-values stored not shown
man = json.loads((PROCESSED / "run_manifest.json").read_text())
check("manifest: run_status completed + final config",
      man["run_status"]["state"] == "completed" and "final_config" in man["run_status"])
n_funcs_in_file = len(re.findall(r"^def ", src, flags=re.M))
check(f"manifest: all {n_funcs_in_file} top-level functions hashed at the end",
      man["code"]["n_functions_hashed"] >= n_funcs_in_file and not man["code"]["functions_not_hashable"])
for m, D in MODE_DIRS.items():
    an = man["modes"][m]["axis2_null"]
    check(f"[{m}] null: clusters recorded ({an['n_distinct_hw_clusters']:,} <= {an['n_hw']:,} HW)",
          0 < an["n_distinct_hw_clusters"] <= an["n_hw"])
    nl = pd.read_parquet(D / "significance" / "axis2_null_region.parquet")
    csv = pd.read_csv(g["FIG_DIRS"][m] / "figS_axis2_null_test_table.csv")
    check(f"[{m}] null permutations per mode ({an['n_permutations']})",
          an["n_permutations"] == {"calendar": 60, "annual": 40}[m])
    check(f"[{m}] null decomposition columns present",
          {"n_before_null_med", "n_after_null_med", "before_obs_over_null_med",
           "after_obs_over_null_med"} <= set(nl.columns))
    check(f"[{m}] p-values stored in parquet, absent from the figure table",
          "p_count_ratio_excess" in nl and not any(c.startswith("p_") for c in csv))
check("SPEI/SPI calibration moments recorded as diagnostic",
      "calibration_moments_diagnostic" in man["spei_spi"])

# annual thresholds constant per PSU (one P90 applied all year)
thr = pd.read_parquet(MODE_DIRS["annual"] / "thresholds_tmax_base.parquet")
check("annual mode: one threshold per PSU", (thr.groupby("psu_idx").thr.nunique() == 1).all())

# ---- Fig 6 with PROVISIONAL supports (figure test only) ----------------------
for m, D in MODE_DIRS.items():
    g["fig06_regimes"](D, g["FIG_DIRS"][m], cfg.regime_quantile, "", min_support=(1, 1))
    F = g["FIG_DIRS"][m]
    check(f"[{m}] Fig 6 PROVISIONAL outputs tagged",
          (F / "fig06_regimes_PROVISIONAL.png").exists()
          and (F / "fig06_diagnostics_PROVISIONAL.csv").exists())
    dg = pd.read_csv(F / "fig06_diagnostics_PROVISIONAL.csv")
    check(f"[{m}] Fig 6 thresholds finite, insufficient share reported",
          np.isfinite(dg.threshold_pooled).all() and dg.share_insufficient.notna().all())

for m in MODE_DIRS:
    check(f"[{m}] SI figure Axis III lag/index written", (g["FIG_DIRS"][m] / "figS_axis3_lag_index.png").exists())

# ---- Axis III local-baseline diagnostic ---------------------------------------
eb = pd.read_csv(PROCESSED / "figures_shared" / "axis3_expected_baseline.csv")
for m, D in MODE_DIRS.items():
    a3 = pd.read_parquet(D / "axis3.parquet")
    r = eb[(eb["mode"] == m) & (eb["index"] == "spei3") & (eb["scale"] == "Continental")].iloc[0]
    check(f"[{m}] Axis III expected baseline: observed == pooled share, expectations in (0, 100)",
          abs(r.observed_pct - 100 * a3.loc[a3.has_spei, "drought_precond"].mean()) < 1e-3
          and 0 < r.exp_month_pct < 100 and 0 < r.exp_month_decade_pct < 100)

# ---- precipitation cache fingerprint ----------------------------------------
pq = PROCESSED / "thresholds_precip.parquet"
t0 = pq.stat().st_mtime_ns
g["run_step03_thresholds"](cfg, out_dir=g["set_mode"]("calendar"))
check("precip cache reused when fingerprint unchanged", pq.stat().st_mtime_ns == t0)
cfg.min_wet_days_ref = 101
g["run_step03_thresholds"](cfg, out_dir=g["set_mode"]("calendar"))
check("precip cache recomputed when fingerprint differs", pq.stat().st_mtime_ns != t0)

print(f"\n{sum(ok)}/{len(ok)} post-run checks passed")
sys.exit(0 if all(ok) else 1)
