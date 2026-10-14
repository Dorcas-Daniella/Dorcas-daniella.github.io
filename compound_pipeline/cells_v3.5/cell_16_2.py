# ---- 16.2 Figure 2: legacy rendering (old fig02 v7) + Extended Data Fig. 3 (Axis-III definitions) ----
# Needs 16.0 and 16.0b. Annual series = PROCESSED/<mode>/aggregations/axis*_by_year_
# {continental,region}.parquet; slope and 95% CI printed on each panel = step-14
# trends (trends/all_trends_summary.parquet: Theil–Sen per decade, moving-block
# bootstrap CI), i.e. the numbers of the trend tables. The Theil–Sen line itself is
# drawn from the plotted series, as before (same slope, checked below).
LEG_FIG2_METRICS = {
    "axis1": "cooccur_days_per_unit",   # co-occurrence days per eligible PSU per year (EPE year)
    "axis2": "count_ratio_a",           # Σ after / Σ before EPE, unique attribution (PRIMARY)
    "axis3": "pct_drought_b",           # mean over PSU of the % HW with antecedent drought (PRIMARY);
}                                       # "pct_drought_a" = pooled % over all HW, as the old script computed it


def leg2_compute_annual(D):
    """{axis: {scale: array over YEARS}} read from the pipeline aggregations."""
    out = {}
    for axis, metric in LEG_FIG2_METRICS.items():
        print(f"  Loading {axis} ({metric})...")
        out[axis] = series_by_scale(D, axis, metric)
    # the old script masked years with <= 5 EPE before HW; step 14 keeps every year
    nb = series_by_scale(D, "axis2", "n_before")
    low = {sc: int(np.sum(v <= 5)) for sc, v in nb.items() if np.sum(v <= 5)}
    if low:
        print(f"  note: years with <= 5 EPE before HW (masked by the old script, kept as in step 14): {low}")
    return out


def leg2_compute_lowess(years, values, frac=0.3):
    mask = ~np.isnan(values)
    if mask.sum() < 5:
        return np.full_like(values, np.nan, dtype=float)
    if HAS_SM:
        smoothed = sm.nonparametric.lowess(values[mask], years[mask], frac=frac)
        return np.interp(years, smoothed[:, 0], smoothed[:, 1])
    return pd.Series(values).rolling(5, center=True, min_periods=3).mean().values


def leg2_compute_trend_with_ci(years, values):
    """Theil-Sen line of the plotted series (the CI shown comes from step 14)."""
    mask = ~np.isnan(values)
    if mask.sum() < 5:
        return None, None, None, None, np.full_like(values, np.nan, dtype=float)
    slope, intercept, lo_slope, hi_slope = stats.theilslopes(values[mask], years[mask])
    trend_line = slope * years + intercept
    return slope, intercept, lo_slope, hi_slope, trend_line


def leg2_compute_row_ylim(series_dict, col_order):
    all_values = []
    for region in col_order:
        values = series_dict.get(region)
        if values is not None:
            mask = ~np.isnan(values)
            if mask.any():
                all_values.extend(values[mask])
    if not all_values:
        return 0, 1
    p99 = np.percentile(all_values, 99)
    return 0, p99 * 1.10


def leg2_create_figure(axe1, axe2, axe3, tr):
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    gs = GridSpec(3, 5, figure=fig, hspace=0.45, wspace=0.35)
    col_order = ['Continental', 'Western', 'Eastern', 'Central', 'Southern']
    axes_config = [
        {'axis': 'axis1', 'series': axe1, 'ylabel': 'Events / PSU / year',
         'row_title': 'HW–EPE compound events', 'row_label': 'a', 'fmt': '{:+.3f}'},
        {'axis': 'axis2', 'series': axe2, 'ylabel': 'After / Before ratio',
         'row_title': 'EPE asymmetry around HW', 'row_label': 'b', 'fmt': '{:+.2f}'},
        {'axis': 'axis3', 'series': axe3, 'ylabel': '% of HW',
         'row_title': 'HW with antecedent drought', 'row_label': 'c', 'fmt': '{:+.2f}'},
    ]
    print("\n" + "=" * 90)
    print("THEIL-SEN SLOPES PER DECADE, 95% CI (step 14: moving-block bootstrap; MK Hamed–Rao p)")
    print("=" * 90)

    for row, rcfg in enumerate(axes_config):
        if rcfg['series'] is None:
            continue
        metric = LEG_FIG2_METRICS[rcfg['axis']]
        ymin, ymax = leg2_compute_row_ylim(rcfg['series'], col_order)
        print(f"\n  {rcfg['row_title']}  [{metric}]:")
        print(f"  {'Region':>15s} | {'Slope/decade':>14s} | {'95% CI':>22s} | MK p")
        print(f"  {'-' * 72}")

        for col_idx, region in enumerate(col_order):
            ax = fig.add_subplot(gs[row, col_idx])
            color = LEG_REGION_COLORS_CONT[region]
            values = rcfg['series'].get(region)

            if values is not None:
                mask = ~np.isnan(values)
                ax.scatter(YEARS[mask], values[mask], s=6, color=color,
                           alpha=0.5, edgecolors='none', zorder=3)
                smooth_ = leg2_compute_lowess(YEARS, values)
                ax.plot(YEARS, smooth_, color=color, linewidth=1.5, zorder=4)

                slope, intercept, _, _, trend_line = leg2_compute_trend_with_ci(YEARS, values)
                if slope is not None:
                    ax.plot(YEARS, trend_line, color=color, linewidth=0.7,
                            linestyle='-', alpha=0.55, zorder=2)

                r = slope_row(tr, rcfg['axis'], 'continental' if region == 'Continental' else 'region',
                              region, metric)
                if r is None or not np.isfinite(r['slope_per_decade']):
                    print(f"  {region:>15s} | no step-14 trend for {metric}")
                else:
                    slope_dec = r['slope_per_decade']
                    lo_dec, hi_dec = r['ci_low_per_decade'], r['ci_high_per_decade']
                    if slope is not None and not np.isclose(slope * 10, slope_dec, rtol=1e-6, atol=1e-9):
                        print(f"  ⚠ {region}: Theil–Sen slope of the plotted series ({slope * 10:+.4f}) "
                              f"differs from step 14 ({slope_dec:+.4f})")
                    ci_ok = np.isfinite(lo_dec) and np.isfinite(hi_dec)   # no CI if internal gap
                    f = rcfg['fmt']
                    txt = f.format(slope_dec) + '/decade'
                    if ci_ok:
                        txt += '\n[' + f.format(lo_dec) + ', ' + f.format(hi_dec) + ']'
                    ax.text(0.97, 0.93, txt,
                            transform=ax.transAxes, fontsize=5.5,
                            ha='right', va='top', color='#555555',
                            fontweight='medium')
                    ci_txt = f"[{lo_dec:+.3f}, {hi_dec:+.3f}]" if ci_ok else "n/a (gap)"
                    print(f"  {region:>15s} | {slope_dec:>+12.3f}   | {ci_txt:>22s} | {r['mk_p']:.3g}")

            ax.set_ylim(ymin, ymax)                         # shared y-axis per row
            ax.set_xlim(YEARS[0] - 1, YEARS[-1] + 1)
            ax.tick_params(axis='both', which='major', labelsize=5.5)
            ax.yaxis.grid(True, linewidth=0.2, color='#dddddd', zorder=0)
            if row == 0:                                    # column headers
                ax.text(0.5, 1.18, region, transform=ax.transAxes,
                        fontsize=8, fontweight='bold', ha='center',
                        va='bottom', color=color)
            if col_idx == 0:                                # row title + ylabel
                ax.set_ylabel(rcfg['ylabel'], fontsize=6.5)
                ax.set_title(rcfg['row_title'], fontsize=7.5,
                             fontweight='bold', pad=4, color='#333333')
                ax.text(-0.32, 1.08, rcfg['row_label'],
                        transform=ax.transAxes, fontsize=14,
                        fontweight='bold', va='bottom', ha='left')
            else:
                ax.set_title('', pad=4)
    return fig


def leg2_main(D, FIG, mode):
    print("=" * 60)
    print(f"FIGURE 2 v7 — Harmonized Y-axes + CI output   [hw_threshold_mode = {mode}]")
    print("=" * 60)
    leg_require(D, [f"aggregations/{a}_by_year_{s}.parquet" for a in LEG_FIG2_METRICS
                    for s in ("continental", "region")] + ["trends/all_trends_summary.parquet"])
    tr = trends(D)
    print("\nReading annual series...")
    s = leg2_compute_annual(D)
    print("\nGenerating figure...")
    fig = leg2_create_figure(s["axis1"], s["axis2"], s["axis3"], tr)
    leg_save(fig, FIG, "fig02_temporal_trends")


# ---- Extended Data Fig. 3: drought before heatwaves under three definitions -------
# No hypothesis test; a = annual series + LOWESS per scale, b = Theil–Sen trend per
# decade with its 95% moving-block-bootstrap CI (step 14).
AXIS3_VARIANTS = [("pct_drought_b", "SPEI-3, month before onset (primary)", "#1f2d3d", "-"),
                  ("pct_drought_lag0_b", "SPEI-3, onset month", "#8e6fc1", "--"),
                  ("pct_drought_spi_b", "SPI-3, month before onset", "#e08214", "-")]


def figS_axis3_lag_index(D, FIG):
    tr = trends(D)
    markers = ["o", "s", "^"]
    fig = plt.figure(figsize=(15, 8.2), facecolor="white")
    gs = GridSpec(2, len(SCALES), figure=fig, height_ratios=[1, 0.85], hspace=0.42, wspace=0.10,
                  top=0.88, bottom=0.08, left=0.06, right=0.99)
    ser = {metric: series_by_scale(D, "axis3", metric) for metric, *_ in AXIS3_VARIANTS}
    allv = np.concatenate([v[np.isfinite(v)] for d in ser.values() for v in d.values()])
    ymax = np.percentile(allv, 99.5) * 1.08 if allv.size else 100
    axs = []
    for j, sc in enumerate(SCALES):
        ax = fig.add_subplot(gs[0, j], sharey=axs[0] if axs else None); axs.append(ax)
        for metric, lab, col, ls in AXIS3_VARIANTS:
            y = ser[metric][sc]; m = np.isfinite(y)
            ax.scatter(YEARS[m], y[m], s=7, color=col, alpha=.22, edgecolors="none", zorder=2)
            ax.plot(YEARS, smooth(y), color=col, lw=2.2, ls=ls, zorder=4)
        ax.set_title(sc, fontsize=11, fontweight="bold", color="#222")
        ax.set_xlim(YEARS[0] - 1, YEARS[-1] + 1); ax.set_ylim(0, ymax)
        ax.set_xticks([1990, 2005, 2020]); ax.tick_params(labelsize=9)
        ax.yaxis.grid(True, lw=.3, color="#e3e3e3", zorder=0)
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        if j == 0:
            ax.set_ylabel("Heatwaves preceded by drought (%)", fontsize=10)
        else:
            ax.tick_params(labelleft=False)
    panel_label(axs[0], "a", x=-0.22, y=1.06, size=14)

    # b: trend per decade with 95% CI, three definitions side by side for each scale
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(SCALES)); off = np.linspace(-0.22, 0.22, len(AXIS3_VARIANTS))
    for k, (metric, lab, col, ls) in enumerate(AXIS3_VARIANTS):
        for xi, sc in zip(x, SCALES):
            r = slope_row(tr, "axis3", "continental" if sc == "Continental" else "region", sc, metric)
            if r is None or not np.isfinite(r["slope_per_decade"]):
                continue
            s_ = r["slope_per_decade"]; lo, hi = r["ci_low_per_decade"], r["ci_high_per_decade"]
            has_ci = bool(r.get("inference_ok", True)) and np.isfinite(lo) and np.isfinite(hi)
            ax.errorbar(xi + off[k], s_, yerr=[[s_ - lo], [hi - s_]] if has_ci else None,
                        fmt=markers[k], ms=7, color=col, mfc=col if k != 1 else "white",
                        mec=col, mew=1.6, elinewidth=1.8, capsize=4, zorder=3)
    ax.axhline(0, color="#777", lw=.8, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(SCALES, fontsize=10)
    ax.set_xlim(-0.6, len(SCALES) - 0.4); ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("Trend (percentage points per decade)", fontsize=10)
    ax.yaxis.grid(True, lw=.3, color="#e3e3e3", zorder=0)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    panel_label(ax, "b", x=-0.045, y=1.03, size=14)

    handles = [plt.Line2D([0], [0], color=col, lw=2.2, ls=ls, marker=markers[k], ms=7,
                          mfc=col if k != 1 else "white", mec=col, mew=1.6, label=lab)
               for k, (metric, lab, col, ls) in enumerate(AXIS3_VARIANTS)]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=10.5,
               frameon=False, bbox_to_anchor=(0.5, 0.995), handlelength=3)
    savefig(fig, FIG, "figS_axis3_lag_index")


for _mode, _D in MODE_DIRS.items():
    with leg_style(LEG_RC_FIG2):
        leg2_main(_D, FIG_DIRS[_mode], _mode)
    figS_axis3_lag_index(_D, FIG_DIRS[_mode])          # Extended Data Fig. 3
