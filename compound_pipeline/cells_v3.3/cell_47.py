# ---- 16.2 Figure 2: temporal trends (annual series + Theil–Sen from step09) ----
FIG2_ROWS = [
    ("axis1", "cooccur_days_per_unit", "Co-occurrence days / PSU / year", "HW–EPE compound events", "{:+.3f}"),
    ("axis2", "count_ratio_a", "EPE after / before HW", "EPE asymmetry (count ratio, PRIMARY)", "{:+.2f}"),
    ("axis2", "rate_diff_b", "Trigger − precursor fraction", "EPE asymmetry (ECA fractions, secondary)", "{:+.3f}"),
    ("axis3", "pct_drought_b", "% of HW", "HW with antecedent drought (lag 1)", "{:+.2f}"),
]


def fig02_trends(D, FIG):
    tr = trends(D)
    fig = plt.figure(figsize=(14, 10), facecolor="white")
    gs = GridSpec(len(FIG2_ROWS), 5, figure=fig, hspace=0.5, wspace=0.35)
    for i, (axis, metric, ylab, title, fmt) in enumerate(FIG2_ROWS):
        ser = series_by_scale(D, axis, metric)
        allv = np.concatenate([v[np.isfinite(v)] for v in ser.values()])
        ymax = np.percentile(allv, 99) * 1.1 if allv.size else 1
        ymin = min(0, np.nanmin(allv) * 1.1) if allv.size else 0
        for j, sc in enumerate(SCALES):
            ax = fig.add_subplot(gs[i, j]); col = REGION_COLORS[sc]; y = ser[sc]; m = np.isfinite(y)
            ax.scatter(YEARS[m], y[m], s=6, color=col, alpha=.5, edgecolors="none", zorder=3)
            ax.plot(YEARS, smooth(y), color=col, lw=1.5, zorder=4)
            ts = theilsen_line(y)
            if ts is not None:
                ax.plot(YEARS, ts[0] * YEARS + ts[1], color=col, lw=.7, alpha=.55, zorder=2)
            txt = slope_text(tr, axis, "continental" if sc == "Continental" else "region", sc, metric, fmt)
            ax.text(.97, .93, txt, transform=ax.transAxes, fontsize=5.5, ha="right", va="top", color="#555")
            ax.set_ylim(ymin, ymax); ax.set_xlim(YEARS[0] - 1, YEARS[-1] + 1)
            ax.yaxis.grid(True, lw=.2, color="#ddd", zorder=0)
            if i == 0:
                ax.text(.5, 1.18, sc, transform=ax.transAxes, fontsize=8, fontweight="bold",
                        ha="center", va="bottom", color=col)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=6.5)
                ax.set_title(title, fontsize=7.5, fontweight="bold", pad=4, loc="left")
                panel_label(ax, "abcd"[i], x=-0.32, y=1.08, size=13)
    fig.text(.01, -.01, "Slopes: Theil–Sen per decade, 95% moving-block-bootstrap CI (step 14; no CI "
             "when the annual series has internal gaps); * = Mann–Kendall (Hamed–Rao) p < 0.05.",
             fontsize=6, color="#888", style="italic")
    savefig(fig, FIG, "fig02_temporal_trends")


# Axis III variants: primary SPEI-3 at lag 1, SPEI-3 at lag 0 (onset month; includes
# the heatwave's own PET) and SPI-3 at lag 1 (precipitation only).
AXIS3_VARIANTS = [("pct_drought_b", "SPEI-3, lag 1 (primary)", "#2c3e50", "-"),
                  ("pct_drought_lag0_b", "SPEI-3, lag 0", "#8e7cc3", "--"),
                  ("pct_drought_spi_b", "SPI-3, lag 1", "#1b9e77", "-")]


def figS_axis3_lag_index(D, FIG):
    """SI figure: annual % of HW with drought for the three Axis-III variants,
    per scale, with the Theil–Sen slope per decade of each (step-14 trends)."""
    tr = trends(D)
    fig, axs = plt.subplots(1, len(SCALES), figsize=(16, 3.8), facecolor="white", sharey=True)
    for j, sc in enumerate(SCALES):
        ax = axs[j]; lines = []
        for k, (metric, lab, col, ls) in enumerate(AXIS3_VARIANTS):
            y = series_by_scale(D, "axis3", metric)[sc]; m = np.isfinite(y)
            ax.scatter(YEARS[m], y[m], s=4, color=col, alpha=.35, edgecolors="none", zorder=3)
            ax.plot(YEARS, smooth(y), color=col, lw=1.4, ls=ls, zorder=4, label=lab)
            txt = slope_text(tr, "axis3", "continental" if sc == "Continental" else "region", sc,
                             metric, "{:+.2f}").replace("\n", " ")
            lines.append((txt or "n/a", col))
        for k, (txt, col) in enumerate(lines):
            ax.text(.03, .97 - k * .075, txt, transform=ax.transAxes, fontsize=5.5, color=col,
                    ha="left", va="top")
        ax.set_title(sc, fontsize=9, fontweight="bold", color=REGION_COLORS[sc])
        ax.set_xlim(YEARS[0] - 1, YEARS[-1] + 1); ax.yaxis.grid(True, lw=.2, color="#ddd", zorder=0)
        if j == 0:
            ax.set_ylabel(f"% of HW with antecedent drought (index < {cfg.drought_threshold})", fontsize=7)
    axs[-1].legend(fontsize=6, loc="lower right", frameon=False)
    fig.text(.01, -.03, "Per-PSU mean of the % of HW with drought (pct_*_b). Slopes: Theil–Sen per decade, "
             "95% moving-block-bootstrap CI; * = Mann–Kendall (Hamed–Rao) p < 0.05. Lag 0 uses the onset month, "
             "whose SPEI includes the heatwave's own PET; SPI is precipitation-only.",
             fontsize=6, color="#888", style="italic")
    savefig(fig, FIG, "figS_axis3_lag_index")


for _mode, _D in MODE_DIRS.items():
    fig02_trends(_D, FIG_DIRS[_mode])
    figS_axis3_lag_index(_D, FIG_DIRS[_mode])
