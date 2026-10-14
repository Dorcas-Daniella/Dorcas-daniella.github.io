# ---- Extended Data Fig. 3: drought before heatwaves under three definitions ----
# New cell, to paste at the END of compound_pipeline_v3.ipynb and run after the whole notebook.
# Replaces figS_axis3_lag_index (no hypothesis test, larger text):
#   a  annual % of heatwaves preceded by drought (points) and LOWESS curves, per scale
#   b  Theil–Sen trend per decade with its 95% moving-block-bootstrap CI
# Output: <mode>/figures/figS_axis3_lag_index.png/.pdf

# Axis III variants: primary SPEI-3 at lag 1, SPEI-3 at lag 0 (onset month; includes
# the heatwave's own PET) and SPI-3 at lag 1 (precipitation only).
AXIS3_VARIANTS = [("pct_drought_b", "SPEI-3, month before onset (primary)", "#1f2d3d", "-"),
                  ("pct_drought_lag0_b", "SPEI-3, onset month", "#8e6fc1", "--"),
                  ("pct_drought_spi_b", "SPI-3, month before onset", "#e08214", "-")]


def figS_axis3_lag_index(D, FIG):
    """Extended Data figure: % of HW preceded by drought for the three Axis-III
    definitions. a, annual series (points) and LOWESS curves per scale;
    b, Theil–Sen slope per decade with its 95% moving-block-bootstrap CI.
    No hypothesis test is shown; details go in the figure legend."""
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
    figS_axis3_lag_index(_D, FIG_DIRS[_mode])
