# ---- Old-style Figures 4, 5 and 6 (layout of the original notebook), new files --------
# New cell, to paste at the END of compound_pipeline_v3.ipynb.
# Run it after cells 2, 4, 6, 45 and 49 (helpers, configuration and Fig 6 rules).
# Same data and rules as the pipeline figures, so the numbers match the text:
#   Fig 4: unique attribution of each EPE to the nearest heatwave (axis2_attrib.parquet,
#          aggregations/axis2_by_country and axis2_by_region_decade);
#   Fig 5: SPEI-3 of the month before heatwave onset (axis3.parquet, drought_precond;
#          aggregations/axis3_by_region_decade);
#   Fig 6: regimes of cell 49 (q75 of the classifiable PSU, minimum supports of cell 4).
# Output: fig4_old_style, fig5_old_style, fig6_old_style (.png and .pdf) in FIG_DIRS[mode].
import cartopy.crs as ccrs
import cartopy.feature as cfeature

try:
    from adjustText import adjust_text
    HAS_ADJUST = True
except ImportError:
    HAS_ADJUST = False
    print("Tip: pip install adjustText for better label placement")

OLD_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.3, "ytick.major.width": 0.3,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.grid": False, "legend.frameon": False,
}
OLD_RC_FIG6 = {
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6, "axes.linewidth": 0.4, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "figure.facecolor": "white", "axes.facecolor": "white",
}
OLD_REGION_COLORS = {"Western": "#D95F02", "Eastern": "#1B9E77",
                     "Central": "#7570B3", "Southern": "#E7298A"}
OLD_REGIONS = ["Western", "Eastern", "Central", "Southern"]
OLD_SHORT_NAMES = {"Burkina Faso": "Burkina F.", "Central African Republic": "CAR",
                   "Congo Democratic Republic": "DR Congo", "Cote d'Ivoire": "Côte d'Iv.",
                   "Sierra Leone": "Sierra L.", "South Africa": "S. Africa"}


def _old_save(fig, FIG, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", format=ext, dpi=300 if ext == "png" else None,
                    facecolor="white")
    plt.close(fig)
    print(f"  ok {FIG / name}.png/.pdf")


def _region_decade_matrix(rd, col, regions):
    M = np.full((len(regions), len(DECADES)), np.nan)
    for i, reg in enumerate(regions):
        for j, dec in enumerate(DECADES):
            r = rd[(rd["region"] == reg) & (rd["decade"] == dec)]
            if len(r):
                M[i, j] = r[col].iloc[0]
    return M


# ============================================================================
# FIGURE 4 — EPE temporal asymmetry around heatwaves (Axis II)
# ============================================================================
def fig4_old_style(D, FIG):
    W = cfg.axis2_window_days
    cs = agg(D, "axis2_by_country")
    rd = agg(D, "axis2_by_region_decade"); rd["decade"] = rd["decade"].astype(str)
    att = pd.read_parquet(D / "axis2_attrib.parquet")
    el = psu_eligibility(cfg, D)
    att = att[att["psu_idx"].isin(el.loc[el["axis12_eligible"], "psu_idx"])]
    att = att[~att["truncated"] & att["position"].isin(["before", "after"])].copy()
    att["decade"] = assign_decade(pd.to_datetime(att["start_date"]).dt.year,
                                  cfg.decade_bounds).astype(str)

    with plt.rc_context(OLD_RC):
        fig = plt.figure(figsize=(14, 5), facecolor="white")
        gs = GridSpec(1, 3, figure=fig, wspace=0.35)

        # (a) after vs before per country — all labeled
        ax_a = fig.add_subplot(gs[0, 0])
        max_val = max(cs["before_per_psu"].max(), cs["after_per_psu"].max()) * 1.1
        ax_a.plot([0, max_val], [0, max_val], "k--", linewidth=0.5, alpha=0.4,
                  zorder=1, label="1:1 line (symmetry)")
        for region in OLD_REGIONS:
            sub = cs[cs["region"] == region]
            ax_a.scatter(sub["before_per_psu"], sub["after_per_psu"],
                         c=OLD_REGION_COLORS[region], s=30, alpha=0.8,
                         edgecolors="white", linewidth=0.3, zorder=3, label=f"{region} Africa")
        texts = []
        for _, row in cs.iterrows():
            texts.append(ax_a.text(row["before_per_psu"], row["after_per_psu"],
                                   OLD_SHORT_NAMES.get(row["country"], row["country"]),
                                   fontsize=4, color=OLD_REGION_COLORS.get(row["region"], "#555555"),
                                   alpha=0.85, fontweight="medium"))
        if HAS_ADJUST and texts:
            adjust_text(texts, ax=ax_a,
                        arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.3),
                        force_text=(0.5, 0.5), force_points=(0.3, 0.3),
                        expand_text=(1.2, 1.4), expand_points=(1.3, 1.3),
                        only_move={"points": "y", "text": "xy"})
        ax_a.set_xlabel("EPE before HW (count / PSU)", fontsize=7.5)
        ax_a.set_ylabel("EPE after HW (count / PSU)", fontsize=7.5)
        ax_a.set_title(" EPE occurrence before vs. after heatwaves", fontsize=9,
                       fontweight="bold", pad=8)
        ax_a.legend(fontsize=5.5, loc="upper left", markerscale=0.7)
        ax_a.set_xlim(0, max_val); ax_a.set_ylim(0, max_val)
        ax_a.text(-0.08, 1.08, "a", transform=ax_a.transAxes, fontsize=13, fontweight="bold", va="top")

        # (b) distribution of EPE timing, by decade (day 0 = inside the heatwave: no value)
        ax_b = fig.add_subplot(gs[0, 1])
        bins = np.arange(-W - 0.5, W + 1.5, 1)
        for dec in DECADES:
            sub = att.loc[att["decade"] == dec, "offset_days"]
            if len(sub) > 0:
                counts, edges = np.histogram(sub, bins=bins, density=True)
                centers = (edges[:-1] + edges[1:]) / 2
                counts[centers == 0] = np.nan
                ax_b.plot(centers, counts, color=DECADE_COLORS[dec], linewidth=1.2,
                          alpha=0.8, label=dec.replace("-", "–"))
        ax_b.axvline(x=0, color="#888888", linewidth=0.5, linestyle="-", zorder=1)
        ax_b.set_xlabel("Days from nearest heatwave boundary", fontsize=7.5)
        ax_b.set_ylabel("Density", fontsize=7.5)
        ax_b.set_title("Distribution of EPE timing relative to HW", fontsize=9,
                       fontweight="bold", pad=8)
        ax_b.legend(fontsize=6, loc="upper left")
        ax_b.set_xlim(-W - 2, W + 2)
        ymin, ymax = ax_b.get_ylim()
        ax_b.text(-W / 2, ymax * 0.05, "Before HW start", fontsize=6, ha="center",
                  color="#0072B2", alpha=0.7, fontstyle="italic")
        ax_b.text(W / 2, ymax * 0.05, "After HW end", fontsize=6, ha="center",
                  color="#D73027", alpha=0.7, fontstyle="italic")
        ax_b.text(-0.08, 1.08, "b", transform=ax_b.transAxes, fontsize=13, fontweight="bold", va="top")

        # (c) after/before ratio by region and decade
        ax_c = fig.add_subplot(gs[0, 2])
        regions_order = OLD_REGIONS + ["Continental"]
        matrix = _region_decade_matrix(rd, "count_ratio", regions_order)
        im = ax_c.imshow(matrix, cmap="OrRd", aspect="auto", vmin=1, vmax=6)
        ax_c.set_xticks(range(len(DECADES)))
        ax_c.set_xticklabels([d.replace("-", "–") for d in DECADES], fontsize=6.5,
                             rotation=30, ha="right")
        ax_c.set_yticks(range(len(regions_order))); ax_c.set_yticklabels(regions_order, fontsize=7)
        for i in range(len(regions_order)):
            for j in range(len(DECADES)):
                val = matrix[i, j]
                if np.isfinite(val):
                    ax_c.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                              fontweight="bold", color="white" if val > 4 else "#333333")
        ax_c.set_title("After/before ratio by region and decade", fontsize=9,
                       fontweight="bold", pad=8)
        ax_c.text(-0.2, 1.08, "c", transform=ax_c.transAxes, fontsize=13, fontweight="bold", va="top")
        cbar = plt.colorbar(im, ax=ax_c, shrink=0.7, pad=0.04, extend="both")
        cbar.set_label("Ratio", fontsize=7); cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.3)
        _old_save(fig, FIG, "fig4_old_style")


# ============================================================================
# FIGURE 5 — Drought amplification of heatwaves (Axis III, month before onset)
# ============================================================================
def _old_boxplots(ax, data_no, data_dr, pos, width):
    for d, dx, fc, mc in ((data_no, -width / 2, "#b3cde3", "#2166ac"),
                          (data_dr, width / 2, "#e6550d", "#b2182b")):
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        bp = ax.boxplot([d], positions=[pos + dx], widths=width * 0.7, patch_artist=True,
                        showfliers=False, zorder=3, medianprops=dict(color="#333333", linewidth=1),
                        whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5))
        bp["boxes"][0].set_facecolor(fc); bp["boxes"][0].set_edgecolor("#333333")
        bp["boxes"][0].set_linewidth(0.5); bp["boxes"][0].set_alpha(0.8)
        ax.scatter(pos + dx, np.mean(d), marker="D", s=14, color=mc,
                   edgecolor="white", linewidth=0.4, zorder=5)


def fig5_old_style(D, FIG):
    a3 = pd.read_parquet(D / "axis3.parquet",
                         columns=["psu_idx", "duration", "peak_tmax", "has_spei", "drought_precond"])
    a3 = a3.merge(PSU_GEO[["psu_idx", "region"]], on="psu_idx", how="left")
    el = psu_eligibility(cfg, D)
    a3 = a3[a3["psu_idx"].isin(el.loc[el["spei_eligible"], "psu_idx"]) & a3["has_spei"]]
    dr = a3["drought_precond"].astype(bool)
    rd = agg(D, "axis3_by_region_decade"); rd["decade"] = rd["decade"].astype(str)

    width = 0.32
    legend_patches = [
        mpatches.Patch(facecolor="#b3cde3", edgecolor="#333333", linewidth=0.5, label="No antecedent drought"),
        mpatches.Patch(facecolor="#e6550d", edgecolor="#333333", linewidth=0.5,
                       label="Antecedent drought (SPEI-3 < −1)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#666666", markersize=4, label="Mean"),
    ]
    with plt.rc_context(OLD_RC):
        fig = plt.figure(figsize=(14, 5), facecolor="white")
        gs = GridSpec(1, 3, figure=fig, wspace=0.35)
        for k, (col, ylab, title) in enumerate((("duration", "HW duration (days)", "Heatwave duration"),
                                               ("peak_tmax", "Maximum temperature (°C)",
                                                "Heatwave intensity"))):
            ax = fig.add_subplot(gs[0, k])
            for i, region in enumerate(OLD_REGIONS):
                m = (a3["region"] == region).to_numpy()
                _old_boxplots(ax, a3.loc[m & ~dr.to_numpy(), col].to_numpy(float),
                              a3.loc[m & dr.to_numpy(), col].to_numpy(float), i, width)
            ax.set_xticks(np.arange(len(OLD_REGIONS))); ax.set_xticklabels(OLD_REGIONS, fontsize=7.5)
            ax.set_ylabel(ylab, fontsize=8)
            ax.set_title(title, fontsize=9, fontweight="bold", pad=10)
            ax.yaxis.grid(True, linewidth=0.15, color="#e0e0e0", zorder=0)
            ax.legend(handles=legend_patches, fontsize=6.5, loc="upper right")
            ax.text(-0.08, 1.02, "ab"[k], transform=ax.transAxes, fontsize=14,
                    fontweight="bold", va="bottom")

        ax_c = fig.add_subplot(gs[0, 2])
        regions_ext = OLD_REGIONS + ["Continental"]
        matrix = _region_decade_matrix(rd, "pct_drought_a", regions_ext)
        im = ax_c.imshow(matrix, cmap="YlOrBr", aspect=1.2, vmin=0, vmax=50)
        ax_c.set_xticks(range(len(DECADES)))
        ax_c.set_xticklabels([d.replace("-", "–") for d in DECADES], fontsize=7,
                             rotation=30, ha="right")
        ax_c.set_yticks(range(len(regions_ext))); ax_c.set_yticklabels(regions_ext, fontsize=7.5)
        for i in range(len(regions_ext)):
            for j in range(len(DECADES)):
                val = matrix[i, j]
                if np.isfinite(val):
                    ax_c.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.5,
                              fontweight="bold", color="white" if val > 35 else "#333333")
        ax_c.set_title("HW preceded by drought (%)", fontsize=9,
                       fontweight="bold", pad=10)
        ax_c.text(-0.25, 1.02, "c", transform=ax_c.transAxes, fontsize=14, fontweight="bold", va="bottom")
        cbar = plt.colorbar(im, ax=ax_c, shrink=0.8, pad=0.03, aspect=30, extend="max")
        cbar.set_label("Share of HW (%)", fontsize=7); cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.2)
        _old_save(fig, FIG, "fig5_old_style")


# ============================================================================
# FIGURE 6 — Compound-event regimes (classification of cell 49)
# ============================================================================
def fig6_old_style(D, FIG):
    df = regime_inputs(D, "full")
    min2, min3 = cfg.regime_min_support_axis2, cfg.regime_min_support_axis3
    ok = regime_classifiable(df, min2, min3)
    thr = {a: order_stat_threshold(df.loc[ok, c], cfg.regime_quantile) for a, c in FIG6_COLS.items()}
    df["regime"] = classify_regimes(df, thr, ok).to_numpy()
    counts = df.loc[ok, "regime"].value_counts()
    n_cls, n_insuf = int(ok.sum()), int((~ok).sum())
    print(f"  Fig 6 ({D.name}): {n_cls:,} classifiable PSU, {n_insuf:,} insufficient data "
          f"(min support {min2}/{min3}, q = {cfg.regime_quantile})")
    for cat in REGIME_ORDER:
        print(f"    {cat:40s}: {counts.get(cat, 0):6,} ({100 * counts.get(cat, 0) / max(n_cls, 1):.1f}%)")

    fig_width = 180 / 25.4
    with plt.rc_context(OLD_RC_FIG6):
        fig = plt.figure(figsize=(fig_width, fig_width * 0.55), facecolor="white")

        # (a) bars, % of classifiable PSU
        ax1 = fig.add_axes([0.05, 0.12, 0.36, 0.82])
        y_pos = np.arange(len(REGIME_ORDER))
        vals = [counts.get(c, 0) for c in REGIME_ORDER]
        bars = ax1.barh(y_pos, vals, color=[REGIME_COLORS[c] for c in REGIME_ORDER],
                        edgecolor="white", linewidth=0.3, height=0.7)
        ax1.set_yticks(y_pos); ax1.set_yticklabels(REGIME_ORDER, fontsize=5.5)
        ax1.set_xlabel(f"Number of PSU (n = {n_cls:,} classifiable; {INSUFFICIENT.lower()}: "
                       f"{n_insuf:,}, {100 * n_insuf / len(df):.1f}%)", fontsize=6)
        ax1.invert_yaxis()
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
        ax1.tick_params(axis="both", labelsize=5.5, pad=2)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val > 0:
                ax1.text(val + n_cls * 0.01, i, f"{100 * val / n_cls:.1f}%",
                         va="center", fontsize=5, color="#555")
        ax1.set_xlim(0, max(vals) * 1.18)
        ax1.text(-0.08, 1.05, "a", transform=ax1.transAxes, fontsize=8, fontweight="bold",
                 va="bottom", ha="right")

        # (b) map
        ax2 = fig.add_axes([0.48, 0.08, 0.50, 0.88], projection=ccrs.PlateCarree())
        ax2.set_extent([-19, 52, -36, 26], crs=ccrs.PlateCarree())
        ax2.add_feature(cfeature.BORDERS, linewidth=0.2, color="#999999")
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#666666")
        ax2.add_feature(cfeature.OCEAN, facecolor="#f5f5f5", zorder=0)
        ax2.add_feature(cfeature.LAND, facecolor="white", zorder=0)
        plot_order = [INSUFFICIENT] + list(reversed(REGIME_ORDER))
        for cat in plot_order:
            sub = df[df["regime"] == cat]
            if len(sub) > 0:
                ax2.scatter(sub["lon"], sub["lat"], c=REGIME_COLORS[cat], s=0.3, alpha=0.5,
                            transform=ccrs.PlateCarree(), rasterized=True,
                            zorder=2 if cat not in ("None prominent", INSUFFICIENT) else 1)
        ax2.text(-0.08, 1.05, "b", transform=ax2.transAxes, fontsize=8, fontweight="bold",
                 va="bottom", ha="right")
        handles = [mpatches.Patch(facecolor=REGIME_COLORS[c], edgecolor="none", label=c)
                   for c in REGIME_ORDER + [INSUFFICIENT]
                   if (counts.get(c, 0) if c != INSUFFICIENT else n_insuf) > 0]
        ax2.legend(handles=handles, fontsize=4.5, loc="lower left", frameon=True, framealpha=0.95,
                   edgecolor="#cccccc", handlelength=1.0, handleheight=0.8,
                   borderpad=0.4, labelspacing=0.3)
        _old_save(fig, FIG, "fig6_old_style")


for _mode, _D in MODE_DIRS.items():
    print(f"=== {_mode} ===")
    fig4_old_style(_D, FIG_DIRS[_mode])
    fig5_old_style(_D, FIG_DIRS[_mode])
    fig6_old_style(_D, FIG_DIRS[_mode])
