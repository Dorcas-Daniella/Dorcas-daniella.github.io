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
        ax.add_feature(cfeature.COASTLINE, linewidth=0.25, edgecolor="#888888", zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.15, edgecolor="#bbbbbb", zorder=1)
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
    txt = fmt.format(r["slope_per_decade"]) + f"/dec{star}"
    # no CI when inference is not valid (internal gap in the annual series)
    if (not bool(r.get("inference_ok", True)) or not np.isfinite(r["ci_low_per_decade"])
            or not np.isfinite(r["ci_high_per_decade"])):
        return txt
    return (txt + "\n[" + fmt.format(r["ci_low_per_decade"]) + ", "
            + fmt.format(r["ci_high_per_decade"]) + "]")


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
