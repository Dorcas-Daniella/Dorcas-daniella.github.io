# ---- Figures without hypothesis-test markers (no p-values, no "*", no FDR) ----
# New cell, to paste at the END of compound_pipeline_v3.ipynb and run after the whole notebook
# (or at least after cells 2, 4 and the figure section 16). It replaces the helpers that still
# showed Mann–Kendall stars / FDR categories and regenerates only the affected figures:
#   Fig 2               fig02_temporal_trends          slopes + 95% CI, no "*"
#   Extended Data Fig 3 figS_axis3_lag_index           slopes + 95% CI, no "*"
#   Supplementary Fig 1 figS02_geo_axis1 (+ figS03/04) background = sign of slope and 95% CI
#   tableS01 (figure + CSV)                            trends with 95% CI instead of "*"
# The MK p-values are still computed in the trends step but are no longer shown anywhere.

def slope_text(tr, axis, scope, scale, metric, fmt="{:+.2f}"):
    r = slope_row(tr, axis, scope, scale, metric)
    if r is None or not np.isfinite(r["slope_per_decade"]):
        return ""
    txt = fmt.format(r["slope_per_decade"]) + "/dec"
    # no CI when inference is not valid (internal gap in the annual series)
    if (not bool(r.get("inference_ok", True)) or not np.isfinite(r["ci_low_per_decade"])
            or not np.isfinite(r["ci_high_per_decade"])):
        return txt
    return (txt + "\n[" + fmt.format(r["ci_low_per_decade"]) + ", "
            + fmt.format(r["ci_high_per_decade"]) + "]")


# Background = sign of the Theil–Sen slope and position of its 95% CI (no hypothesis test).
TREND_BG = {"Increase, 95% CI above 0": "#fdd7d1", "Increase, CI includes 0 or n/a": "#fef0e6",
            "No slope": "#f5f5f5", "Decrease, CI includes 0 or n/a": "#dae8f5",
            "Decrease, 95% CI below 0": "#b8d4ea", "Not assessable": "#ffffff"}

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
    fig.text(.01, -.01, "Slopes: Theil–Sen per decade, 95% moving-block-bootstrap CI (no CI "
             "when the annual series has internal gaps).",
             fontsize=6, color="#888", style="italic")
    savefig(fig, FIG, "fig02_temporal_trends")


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
             "95% moving-block-bootstrap CI. Lag 0 uses the onset month, "
             "whose SPEI includes the heatwave's own PET; SPI is precipitation-only.",
             fontsize=6, color="#888", style="italic")
    savefig(fig, FIG, "figS_axis3_lag_index")


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


def tableS01(D, FIG):
    tr = trends(D)
    p1 = agg(D, "axis1_by_psu").groupby("country").agg(N_PSU=("psu_idx", "size"), CCE_days_per_PSU=("n_cooccur_days", "mean"))
    p2 = agg(D, "axis2_by_country").set_index("country")[["n_before", "n_after", "count_ratio"]]
    a3 = agg(D, "axis3_by_year_country").groupby("country")[["n_hw", "n_hw_spei", "n_hw_drought"]].sum()
    a3["Pct_HW_drought"] = 100 * a3["n_hw_drought"] / a3["n_hw_spei"].replace(0, np.nan)
    tab = (p1.join(p2).join(a3)).reset_index()
    # same unit as the Axis-I trend (days / PSU / yr); the 40-yr total is kept in the CSV
    tab["CCE_days_per_PSU_yr"] = tab["CCE_days_per_PSU"] / len(YEARS)
    tab["Region"] = tab["country"].map(COUNTRY_REGION)
    for axis, metric, name in (("axis1", "cooccur_days_per_unit", "CCE_trend_per_decade"),
                               ("axis2", "count_ratio_a", "Ratio_trend_per_decade"),
                               ("axis3", "pct_drought_b", "Drought_trend_per_decade")):
        rows = tr[(tr["axis"] == axis) & (tr["scope"] == "country") & (tr["metric"] == metric)].set_index("scale")
        tab[name] = tab["country"].map(rows["slope_per_decade"])
        ok = rows["inference_ok"].astype(bool)
        tab[name + "_ci_lo"] = tab["country"].map(rows["ci_low_per_decade"].where(ok))
        tab[name + "_ci_hi"] = tab["country"].map(rows["ci_high_per_decade"].where(ok))
    tab = tab.rename(columns={"country": "Country", "n_before": "N_EPE_before", "n_after": "N_EPE_after", "count_ratio": "EPE_ratio",
                              "n_hw": "N_HW", "n_hw_spei": "N_HW_with_SPEI", "n_hw_drought": "N_HW_drought"})
    tab["_o"] = tab["Region"].map({r: i for i, r in enumerate(REGIONS)})
    tab = tab.sort_values(["_o", "Country"]).drop(columns="_o")
    FIG.mkdir(parents=True, exist_ok=True); tab.round(4).to_csv(FIG / "tableS01_country_statistics.csv", index=False)
    show = tab[["Country", "Region", "N_PSU", "CCE_days_per_PSU_yr", "CCE_trend_per_decade", "EPE_ratio",
                "Ratio_trend_per_decade", "Pct_HW_drought", "Drought_trend_per_decade"]].copy()
    for c in show.columns[3:]:
        show[c] = show[c].map(lambda v: f"{v:.3g}" if pd.notna(v) else "NA")
    for c in ("CCE_trend_per_decade", "Ratio_trend_per_decade", "Drought_trend_per_decade"):
        show[c] = [v if pd.isna(a) or pd.isna(b) else f"{v} [{a:.3g}, {b:.3g}]"
                   for v, a, b in zip(show[c], tab[c + "_ci_lo"], tab[c + "_ci_hi"])]
    fig, ax = plt.subplots(figsize=(14, 11), facecolor="white"); ax.axis("off")
    tb = ax.table(cellText=show.to_numpy().tolist(), colLabels=list(show.columns), cellLoc="center", loc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(6); tb.scale(1, 1.3)
    for j in range(len(show.columns)):
        tb[0, j].set_facecolor("#4a4a4a"); tb[0, j].set_text_props(color="white", fontweight="bold")
    fig.suptitle("Table S1 — Country-level compound climate event statistics", fontsize=10, fontweight="bold")
    fig.text(0.5, 0.06,
             f"CCE_days_per_PSU_yr: HW–EPE co-occurrence days per PSU per year ({YEARS[0]}–{YEARS[-1]} mean; same unit as its trend). "
             "EPE_ratio: Σ EPE after / Σ EPE before HW (unique attribution, W = ±30 d). Pct_HW_drought: pooled share of HW with "
             "SPEI-3 < −1 at m−1;\nits trend uses the per-PSU mean (pct_drought_b). Trends: Theil–Sen per decade [95% moving-block-bootstrap CI; "
             "none when the annual series has gaps]. A slope of 0 can result from a series with "
             "mostly zero years (Theil–Sen median of pairwise slopes).", ha="center", va="bottom", fontsize=6, color="#444")
    savefig(fig, FIG, "tableS01_country_statistics")
    return tab


for _mode, _D in MODE_DIRS.items():
    _F = FIG_DIRS[_mode]
    fig02_trends(_D, _F)
    figS_axis3_lag_index(_D, _F)
    figS_geo_single(_D, _F, "axis1", "cooccur_days_per_unit",
                    "Axis I: HW–EPE co-occurrence days per PSU per year, by country", "figS02_geo_axis1")
    figS_geo_single(_D, _F, "axis2", "count_ratio_a",
                    "Axis II: EPE after/before count ratio, by country", "figS03_geo_axis2",
                    ylim=(0, 16), clip=15)
    figS_geo_single(_D, _F, "axis3", "pct_drought_b",
                    "Axis III: % HW with antecedent drought, by country", "figS04_geo_axis3", ylim=(0, 100))
    tableS01(_D, _F)
