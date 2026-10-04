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
    # spatial matching (section 4). 20 km ~ half-diagonal of a 0.25-deg cell at
    # the equator (~19.6 km): every PSU keeps its own cell, outliers are rejected.
    max_match_km_era5=20.0, max_match_km_chirps=20.0,
    extraction_batch_size=2000,
    # heatwaves (sections 5-6)
    hw_percentile=90.0, hw_min_days=3, hw_window_half=15,
    hw_threshold_modes=("calendar", "annual"),  # BOTH are run -> PROCESSED/<mode>/
    hw_threshold_mode="calendar",   # current mode; set automatically by set_mode()
    #   calendar : HWMId-type circular window of 2*hw_window_half+1 = 31
    #              calendar positions (leap-year calendar, see calendar_doy)
    #   annual   : ONE P90 per PSU over all reference-period days, applied all year
    in_base_correction="none",      # PRIMARY "none" | "loo" | "zhang" (Zhang 2005):
    zhang_n_replicates=29,          #   loo/zhang only if a reviewer asks for them
    # EPE (sections 5,7)
    epe_percentile=95.0, wet_day_min_mm=1.0, min_wet_days_ref=100,
    # SPEI/SPI (section 8)
    spei_scale_months=3, drought_threshold=-1.0, spei_clip=3.5,
    spei_dist="fisk", spei_method="PWM",   # PWM->ML fallback is automatic & logged
    min_month_coverage=0.9,
    # Axis I
    axis1_day_tolerance=0,          # 0 strict same-day (only value reported)
    # Axis II
    axis2_window_days=30, axis2_exclude_truncated=True,
    axis2_attribution_tiebreak="earlier",
    axis2_window_sens=(7, 15, 45, 60),  # window sensitivity (Axis II only; primary W kept)
    # Axis III
    axis3_precond_lag_months=1,     # PRIMARY: antecedent month m-1
    axis3_lags=(1, 0),              # all lags computed in ONE pass; [0] must be the primary
    # significance (section 12)
    n_null_permutations=1000,       # default; QUICK TEST: set 100-200 first
    # per-mode override (calendar = primary, annual = comparison mode)
    n_null_permutations_by_mode={"calendar": 1000, "annual": 1000},
    null_mode="psu_window",         # "psu_window" (strict, PRIMARY) | "region_pool" (legacy)
    null_window_days=15,            # +/- calendar days around the observed onset (psu_window)
    null_seasons=("DJF", "MAM", "JJA", "SON"),   # onset-season decomposition of the null
    n_lmf_surrogates=1000,
    rng_seed=42,
    # trends (section 14)
    n_bootstrap=1000, block_length_years=5, fdr_alpha=0.05,
    # figures (section 17)
    regime_quantile=0.75,           # Fig 6 PRIMARY: prominent = above top quartile
    regime_quantile_sens=2/3,     # Fig 6 sensitivity: top tercile
    regime_threshold_reference="pooled",   # decade panels use the pooled full-period thresholds
    # Fig 6 minimum supports (None = support diagnostics only, no classes/figure).
    # Fix them from the support diagnostics BEFORE looking at thresholds/classes.
    regime_min_support_axis2=None,  # min n_before + n_after per PSU (and per decade)
    regime_min_support_axis3=None,  # min n_hw_spei per PSU (and per decade)
    spei_threshold_sens=(-0.5, -1.0, -1.5, -2.0),
    # processing
    psu_block_size=3000,
    # soft expectations (warn only); set None to silence
    expected_n_psu=54615, expected_n_countries=34,
)
cfg.ref_years = list(range(cfg.ref_start_year, cfg.ref_end_year + 1))
cfg.study_years = list(range(cfg.study_start_year, cfg.study_end_year + 1))

# ---- scope of the reported analysis (fail early on an unsupported config) --
assert cfg.axis1_day_tolerance == 0, "the +/-1-day tolerance is not reported"
assert cfg.axis2_attribution_tiebreak == "earlier", \
    "attribute_unique only implements the 'earlier' tie-break"

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


# ------------------------- RUN MANIFEST --------------------------------------
# PROCESSED/run_manifest.json is (re)created here at the start of every run and
# completed by the steps (eligibility exclusions, SPEI/SPI effective methods,
# NaN categories, usable permutations). The code fingerprint is recomputed at
# every update so the final manifest describes the code that actually ran.
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone

MANIFEST = PROCESSED / "run_manifest.json"
XCLIM_EXPECTED = "0.61."      # the SPEI/SPI NaN control (section 8) relies on it


def _jsonable(o):
    if isinstance(o, SimpleNamespace):
        return {k: _jsonable(v) for k, v in vars(o).items()}
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None      # strict JSON: no NaN/inf
    return o


def _file_identity(p) -> dict:
    p = Path(p)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "exists": True, "size_bytes": int(st.st_size),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}


def _package_versions() -> dict:
    import importlib
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "scipy", "xarray", "xclim", "pyarrow", "pyreadr",
                "matplotlib", "statsmodels", "cartopy", "bottleneck"):
        try:
            out[mod] = importlib.import_module(mod).__version__
        except ImportError:
            out[mod] = None
    return out


def _code_fingerprint() -> dict:
    """sha256 over the source of every function defined in this notebook
    (module __main__), plus the script file itself when run as a .py."""
    import __main__
    h = hashlib.sha256()
    n_ok, skipped = 0, []
    for name in sorted(vars(__main__)):
        obj = vars(__main__)[name]
        if inspect.isfunction(obj) and obj.__module__ == "__main__":
            try:
                src = inspect.getsource(obj)
            except (OSError, TypeError):
                skipped.append(name)
                continue
            h.update(name.encode()); h.update(src.encode()); n_ok += 1
    out = {"functions_sha256": h.hexdigest(), "n_functions_hashed": n_ok,
           "functions_not_hashable": skipped}
    # Jupyter: hash of every code cell executed in this kernel, in order
    # (covers module-level constants too; reruns are part of the record)
    try:
        cells = [c for c in get_ipython().user_ns.get("In", []) if c]   # noqa: F821
        out["executed_cells_sha256"] = hashlib.sha256(
            "\n# <cell>\n".join(cells).encode()).hexdigest()
        out["n_executed_cells"] = len(cells)
    except NameError:                       # not running under IPython
        pass
    f = getattr(__main__, "__file__", None)
    if f and Path(f).is_file():
        out["script_file"] = str(f)
        out["script_sha256"] = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    return out


def _manifest_write(m: dict) -> None:
    m["updated_utc"] = datetime.now(timezone.utc).isoformat()
    m["code"] = _code_fingerprint()
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_jsonable(m), indent=2, allow_nan=False))
    tmp.replace(MANIFEST)


def manifest_update(section: str, payload, mode: str | None = None) -> None:
    """Record `payload` under `section` (under modes/<mode>/ if mode is given)."""
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    if mode is None:
        m[section] = payload
    else:
        m.setdefault("modes", {}).setdefault(mode, {})[section] = payload
    _manifest_write(m)


_xclim_ok = str(xclim.__version__).startswith(XCLIM_EXPECTED)
if not _xclim_ok:
    warnings.warn(f"xclim {xclim.__version__} is not {XCLIM_EXPECTED}x: the SPEI/SPI "
                  "NaN control of section 8 was written against xclim 0.61 "
                  "(fit -> NaN for <= 1 valid calibration value; rolling mean "
                  "with skipna=False). Re-check it before trusting section 8.")
_manifest_write({
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "run_status": {"state": "started"},    # set to "completed" by the last cell
    "rng_seed": cfg.rng_seed,
    "config": cfg,
    "packages": _package_versions(),
    "xclim_expected": XCLIM_EXPECTED + "x", "xclim_version_ok": _xclim_ok,
    "inputs": {"era5_nc": _file_identity(ERA5_NC),
               "chirps_nc": _file_identity(CHIRPS_NC),
               "psu_rdata": _file_identity(RAW_PSU_RDATA)},
})

print("Base dir:", BASE_DIR)
print("Mode dirs:", {m: str(d) for m, d in MODE_DIRS.items()})
print("Run manifest:", MANIFEST)
