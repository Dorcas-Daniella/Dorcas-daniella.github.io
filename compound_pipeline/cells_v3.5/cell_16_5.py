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
# Background = sign of the Theil–Sen slope and position of its 95% CI (no hypothesis test).
TREND_BG = {"Increase, 95% CI above 0": "#fdd7d1", "Increase, CI includes 0 or n/a": "#fef0e6",
            "No slope": "#f5f5f5", "Decrease, CI includes 0 or n/a": "#dae8f5",
            "Decrease, 95% CI below 0": "#b8d4ea", "Not assessable": "#ffffff"}
COUNTRY_REGION = PSU_GEO.groupby("country")["region"].first().to_dict()


def trend_category(tr, axis, country, metric):
    r = slope_row(tr, axis, "country", country, metric)
    if r is None or not np.isfinite(r["slope_per_decade"]):
        return "Not assessable"
    s, lo, hi = r["slope_per_decade"], r["ci_low_per_decade"], r["ci_high_per_decade"]
    ci_ok = bool(r.get("inference_ok", True)) and np.isfinite(lo) and np.isfinite(hi)
    if s > 0:
        return "Increase, 95% CI above 0" if ci_ok and lo > 0 else "Increase, CI includes 0 or n/a"
    if s < 0:
        return "Decrease, 95% CI below 0" if ci_ok and hi < 0 else "Decrease, CI includes 0 or n/a"
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
    handles = [mpatches.Patch(fc="none", ec="none", label=r"$\bf{Background = Theil–Sen\ trend\ (95\%\ CI)}$")]
    handles += [mpatches.Patch(fc=TREND_BG[k], ec="#aaa", lw=.5, label=k) for k in TREND_BG]
    handles += [mpatches.Patch(fc="none", ec="none", label=r"$\bf{Regions}$")]
    handles += [mpatches.Patch(fc=REGION_COLORS[r], label=r) for r in REGIONS]
    axl = fig.add_axes([0.02, 0.05, 0.14, 0.35]); axl.axis("off")
    axl.legend(handles=handles, loc="center", fontsize=7, frameon=True, edgecolor="#ccc")
    savefig(fig, FIG, filename)


def figS_geo_paired(D, FIG, axis, col_a, col_b, lab_a, lab_b, title, filename):
    t = agg(D, f"{axis}_by_year_country")
    n_units = PSU_GEO.groupby("country").size()               # all PSU of the country
    if "n_units_eligible" in t.columns:     # Axis II: eligible PSU; Axis III: PSU with SPEI
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


