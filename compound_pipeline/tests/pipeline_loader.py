"""Load the FUNCTIONS (and a few module constants) of a pipeline .py file
without running any pipeline step: top-level imports, function definitions
and whitelisted constant assignments are executed in a fresh namespace.
Used by the synthetic tests only (artificial data, not a validation on real data)."""
import ast
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace

CONSTANTS = {"EPOCH", "_LEAP_MONTH_OFFSET", "SEASON_OF_MONTH", "DAYS_IN_YEAR",
             "GSC", "_MONTH_STARTS", "_MONTH_LENGTHS", "PRECIP_THRESHOLDS_VERSION",
             "METRICS", "SCOPES", "REGIONS_M49", "COUNTRY_TO_REGION",
             "REGIME_ORDER", "REGIME_COLORS", "INSUFFICIENT", "FIG6_COLS",
             "SUPPORT_LEVELS", "XCLIM_EXPECTED"}
SKIP_FUNCS = set()


def load(path, extra_globals=None):
    src = Path(path).read_text()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("pip install"))
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            if isinstance(node, ast.ImportFrom) and node.module and "cartopy" in node.module:
                continue
            if isinstance(node, ast.FunctionDef) and node.name in SKIP_FUNCS:
                continue
            keep.append(node)
        elif isinstance(node, ast.Assign) and all(
                isinstance(t, ast.Name) and t.id in CONSTANTS for t in node.targets):
            keep.append(node)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Subscript)
              and isinstance(node.targets[0].value, ast.Name)
              and node.targets[0].value.id == "REGIME_COLORS"):
            keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    ns = {"__name__": "pipeline_under_test", "log": logging.getLogger("test"),
          "warnings": warnings, "SimpleNamespace": SimpleNamespace, "Path": Path}
    if extra_globals:
        ns.update(extra_globals)
    exec(compile(mod, str(path), "exec"), ns)
    return SimpleNamespace(**{k: v for k, v in ns.items() if not k.startswith("__")}), ns


def load_cfg(path, processed):
    """The `cfg = SimpleNamespace(...)` of the configuration cell, with the
    paths redirected to a temporary PROCESSED directory."""
    src = Path(path).read_text()
    src = "\n".join(l for l in src.splitlines() if not l.startswith("pip install"))
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "cfg")
    ns = {"SimpleNamespace": SimpleNamespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "cfg", "exec"), ns)
    cfg = ns["cfg"]
    cfg.ref_years = list(range(cfg.ref_start_year, cfg.ref_end_year + 1))
    cfg.study_years = list(range(cfg.study_start_year, cfg.study_end_year + 1))
    processed = Path(processed)
    cfg.paths = SimpleNamespace(raw_psu_rdata=processed / "psu.RData",
                                era5_nc=processed / "era5.nc",
                                chirps_nc=processed / "chirps.nc",
                                processed=processed,
                                climate_daily=processed / "climate_daily")
    return cfg
