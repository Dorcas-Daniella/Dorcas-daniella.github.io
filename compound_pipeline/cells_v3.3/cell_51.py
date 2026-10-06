# ---- 16.6 Figures S5 (seasonality), S6 (comparative evolution), S8 (latitude), S9 (decades), S10 (month strat.), Table S1
def figS05_seasonality(D, FIG):
    m1 = agg(D, "axis1_by_month_region"); m2 = agg(D, "axis2_by_onset_month_region")
    m3 = agg(D, "axis3_by_onset_month_region")
    fig = plt.figure(figsize=(14, 10), facecolor="white"); gs = GridSpec(2, 2, figure=fig, hspace=.35, wspace=.3)
    spec = [(m1, "month", "cooccur_days_per_unit_year", "Co-occurrence days / PSU / year", "HW–EPE compound events (month of EPE)", None),
            (m2, "onset_month", "count_ratio", "EPE after / before HW", "EPE asymmetry (month of HW onset)", 1.0),
            (m3, "onset_month", "pct_drought_a", "% of HW with antecedent drought", "HW ∩ antecedent drought (month of onset)", None),
            (m3, "onset_month", "hw_per_unit_year", "HW / PSU / year", "All heatwaves (baseline seasonality)", None)]
    for k, (t, mcol, vcol, ylab, title, hline) in enumerate(spec):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        for reg in REGIONS:
            s = t[t["region"] == reg].set_index(mcol)[vcol].reindex(range(1, 13))
            v = s.to_numpy(float)
            if vcol == "count_ratio":
                v = np.clip(v, None, 15)
            ax.plot(range(1, 13), v, color=REGION_COLORS[reg], lw=2, marker="o", ms=4, label=reg)
        if hline is not None:
            ax.axhline(hline, color="#888", lw=.5, ls="--")
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_LABELS); ax.set_xlabel("Month")
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=7); ax.yaxis.grid(True, lw=.15, color="#e0e0e0", zorder=0); panel_label(ax, "abcd"[k], x=-0.12)
    savefig(fig, FIG, "figS05_seasonality")


def figS06_comparative(D, FIG):
    a2c, a2r = agg(D, "axis2_by_year_continental"), agg(D, "axis2_by_year_region")
    a3c, a3r = agg(D, "axis3_by_year_continental"), agg(D, "axis3_by_year_region")
    n_all = {"Continental": len(PSU_GEO), **PSU_GEO.groupby("region").size().to_dict()}

    def pick(c, r, sc):
        t = c if sc == "Continental" else r[r["region"] == sc]
        return t.set_index("year").reindex(YEARS)
    fig = plt.figure(figsize=(15, 7), facecolor="white"); gs = GridSpec(2, 5, figure=fig, hspace=.4, wspace=.3)
    rows = [("a", a2c, a2r, "n_before", "n_after", "n_units_eligible", "#4575b4", "#d73027", "EPE before HW", "EPE after HW", "EPE / PSU / year"),
            ("b", a3c, a3r, "n_hw_nodrought", "n_hw_drought", "n_units_eligible", "#4575b4", "#e6550d", "HW, no antecedent drought", "HW, antecedent drought", "HW / PSU / year")]
    for i, (lab, c, r, ca, cb, cn, cola, colb, la, lb, ylab) in enumerate(rows):
        data = {}
        for sc in SCALES:
            t = pick(c, r, sc)
            n = t[cn].max() if cn else n_all[sc]
            data[sc] = (t[ca].fillna(0).to_numpy(float) / n, t[cb].fillna(0).to_numpy(float) / n)
        ymax = max(np.nanmax(v) for pair in data.values() for v in pair) * 1.15
        for j, sc in enumerate(SCALES):
            ax = fig.add_subplot(gs[i, j]); a, b = data[sc]
            ax.scatter(YEARS, a, s=5, color=cola, alpha=.35, edgecolors="none"); ax.scatter(YEARS, b, s=5, color=colb, alpha=.35, edgecolors="none")
            sa, sb = smooth(a), smooth(b)
            ax.plot(YEARS, sa, color=cola, lw=1.8, label=la); ax.plot(YEARS, sb, color=colb, lw=1.8, label=lb)
            ax.fill_between(YEARS, sa, sb, alpha=.08, color=colb)
            ax.set_xlim(YEARS[0] - 1, YEARS[-1] + 1); ax.set_ylim(0, ymax)
            ax.set_title(sc, fontsize=9, fontweight="bold", color=REGION_COLORS[sc]); ax.set_xticks([1990, 2000, 2010, 2020])
            ax.yaxis.grid(True, lw=.12, color="#e0e0e0", zorder=0)
            if j == 0:
                ax.set_ylabel(ylab); ax.legend(fontsize=5.5, loc="upper left"); panel_label(ax, lab, x=-0.3, size=13)
    savefig(fig, FIG, "figS06_comparative_evolution")


def figS08_latitude(D, FIG):
    df = psu_metric_table(D); df["lat_bin"] = (df["lat"] // 5) * 5 + 2.5
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), facecolor="white")
    spec = [("axis1", "HW–EPE co-occurrence days / PSU / yr"), ("axis2", "EPE after/before ratio"), ("axis3", "% HW with antecedent drought")]
    for i, (axis, ylab) in enumerate(spec):
        ax = axes.flat[i]; col = AXIS_PSU_COLS[axis]
        for reg in REGIONS:
            s = df[df["region"] == reg]
            ax.scatter(s["lat"], s[col], s=.3, alpha=.12, c=REGION_COLORS[reg], rasterized=True, label=reg)
        g = df.groupby("lat_bin")[col]
        med, q25, q75 = g.median(), g.quantile(.25), g.quantile(.75)
        ax.fill_between(med.index, q25.to_numpy(), q75.to_numpy(), alpha=.15, color="k")
        ax.plot(med.index, med.to_numpy(), "k-", lw=1.5, label="Median (5° bins)")
        ax.set_ylabel(ylab); ax.set_xlabel("Latitude (°N)"); panel_label(ax, "abc"[i])
        if axis == "axis2":
            ax.set_ylim(0, min(np.nanquantile(df[col], .99), 20)); ax.axhline(1, color="grey", ls="--", lw=.5)
        if i == 0:
            ax.legend(fontsize=5.5, markerscale=5)
    ax = axes.flat[3]
    for axis, color in zip(("axis1", "axis2", "axis3"), ("#e67e22", "#3498db", "#e74c3c")):
        med = df.groupby("lat_bin")[AXIS_PSU_COLS[axis]].median(); v = med.to_numpy(float)
        vn = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)) if np.nanmax(v) > np.nanmin(v) else np.zeros_like(v)
        ax.plot(med.index, vn, "o-", color=color, ms=3, lw=1.3, label=AXIS_NAMES[axis])
    ax.set_xlabel("Latitude (°N)"); ax.set_ylabel("Normalised median (0–1)"); ax.legend(fontsize=6); panel_label(ax, "d")
    for a in axes.flat:
        a.axvspan(-5, 5, alpha=.04, color="#FFA500", zorder=0)
    fig.tight_layout(); savefig(fig, FIG, "figS08_latitude_gradient")


def figS09_decadal(D, FIG):
    t1, t2, t3 = agg(D, "axis1_by_region_decade"), agg(D, "axis2_by_region_decade"), agg(D, "axis3_by_region_decade")
    for t in (t1, t2, t3):
        t["decade"] = t["decade"].astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), facecolor="white")
    spec = [(t1, "cooccur_days_per_unit_year", "Co-occurrence days / PSU / yr", "Axis I"),
            (t2, "count_ratio", "EPE after/before ratio", "Axis II"), (t3, "pct_drought_a", "% HW with antecedent drought", "Axis III")]
    x = np.arange(len(DECADES)); w = .2
    for ax, (t, col, ylab, title) in zip(axes, spec):
        for i, reg in enumerate(REGIONS):
            v = [t[(t["region"] == reg) & (t["decade"] == d)][col].squeeze() if len(t[(t["region"] == reg) & (t["decade"] == d)]) else np.nan for d in DECADES]
            v = np.array(v, float)
            ax.bar(x + i * w, v, w, color=REGION_COLORS[reg], label=reg, alpha=.85, edgecolor="white", lw=.3)
            for j in range(1, len(v)):
                if np.isfinite(v[j - 1]) and v[j - 1] > 0 and np.isfinite(v[j]):
                    pc = (v[j] - v[j - 1]) / v[j - 1] * 100
                    if abs(pc) > 15:
                        ax.annotate(f"{pc:+.0f}%", xy=(x[j] + i * w, v[j]), fontsize=4.5, ha="center", va="bottom",
                                    color="#006600" if pc > 0 else "#CC0000", rotation=45)
        ax.set_xticks(x + 1.5 * w); ax.set_xticklabels([d.replace("-", "\n") for d in DECADES], fontsize=6)
        ax.set_ylabel(ylab); ax.set_title(title, fontweight="bold"); panel_label(ax, "abc"[list(axes).index(ax)], x=-0.15)
    axes[0].legend(fontsize=6, ncol=2); fig.tight_layout(); savefig(fig, FIG, "figS09_decadal_change")


def figS10_monthly(D, FIG, min_n=100):
    t = agg(D, "axis2_by_onset_month_region")
    R = np.full((len(SCALES), 12), np.nan); N = np.zeros((len(SCALES), 12))
    for i, sc in enumerate(SCALES):
        for j in range(12):
            r = t[(t["region"] == sc) & (t["onset_month"] == j + 1)]
            if len(r):
                R[i, j] = r["count_ratio"].iloc[0]; N[i, j] = r["n_before"].iloc[0] + r["n_after"].iloc[0]
    # a cell is shown only with >= min_n attributed EPE AND a defined ratio
    reliable = (N >= min_n) & np.isfinite(R)
    R_show = np.where(reliable, R, np.nan)                     # masked cells -> blank
    fig = plt.figure(figsize=(15, 6.5), facecolor="white")
    gs = GridSpec(2, 2, figure=fig, width_ratios=[3, 1.3], height_ratios=[1, .05], hspace=.3, wspace=.3)
    ax = fig.add_subplot(gs[0, 0])
    cmap = mcolors.LinearSegmentedColormap.from_list("r", ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"])
    cmap.set_bad("#ffffff")
    im = ax.imshow(np.log2(np.clip(R_show, .1, 64)), cmap=cmap, norm=mcolors.TwoSlopeNorm(vmin=np.log2(.1), vcenter=0, vmax=np.log2(16)), aspect="auto")
    for i in range(len(SCALES)):
        for j in range(12):
            if not reliable[i, j] or not np.isfinite(R[i, j]):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#888")
            else:
                ax.text(j, i, f"{R[i, j]:.1f}" if R[i, j] < 10 else f"{R[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        fontweight="bold" if R[i, j] >= 1.5 else "normal", color="white" if abs(np.log2(max(R[i, j], .1))) > 2.5 else "k")
    ax.set_xticks(range(12)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(len(SCALES))); ax.set_yticklabels(SCALES); ax.set_xlabel("Month of heatwave onset")
    ax.set_title("After/before EPE count ratio by month of HW onset", fontsize=11, fontweight="bold"); panel_label(ax, "a", x=-0.06)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[1, 0]), orientation="horizontal", extend="both")
    tv = [.125, .25, .5, 1, 2, 4, 8, 16]; cb.set_ticks(np.log2(tv)); cb.set_ticklabels([str(v) for v in tv]); cb.set_label("Ratio (log2 scale)")
    ax = fig.add_subplot(gs[0, 1]); frac = []; med = []
    for i in range(len(SCALES)):
        rv = R[i, reliable[i]]; frac.append(100 * np.sum(rv > 1) / len(rv) if len(rv) else np.nan); med.append(np.nanmedian(rv) if len(rv) else np.nan)
    ax.barh(range(len(SCALES)), frac, color=[REGION_COLORS[s] for s in SCALES], height=.7)
    for i, (f, m) in enumerate(zip(frac, med)):
        ax.text((f if np.isfinite(f) else 0) + 1.5, i,
                f"{f:.0f}% (median {m:.1f})" if np.isfinite(f) else "no reliable month",
                va="center", fontsize=8)
    ax.set_yticks(range(len(SCALES))); ax.set_yticklabels(SCALES); ax.invert_yaxis(); ax.set_xlim(0, 130)
    ax.set_xlabel("Months with ratio > 1 (%)"); ax.axvline(50, color="gray", ls=":", lw=.8)
    ax.set_title("Fraction of months with post-HW EPE excess", fontsize=11, fontweight="bold"); panel_label(ax, "b", x=-0.15)
    fig.text(.02, .0, f"— : fewer than {min_n} attributed EPE or undefined ratio (no EPE before HW); excluded from panel b. Unique attribution, W = ±{cfg.axis2_window_days} d, non-truncated HW.", fontsize=7, color="#666", style="italic")
    savefig(fig, FIG, "figS10_monthly_stratification")


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
        tab[name] = tab["country"].map(rows["slope_per_decade"]); tab[name + "_FDRsig"] = tab["country"].map(rows["mk_significant_fdr"])
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
        sig = tab[c + "_FDRsig"].fillna(False).astype(bool).to_numpy()
        show[c] = [v + "*" if f else v for v, f in zip(show[c], sig)]
    fig, ax = plt.subplots(figsize=(14, 11), facecolor="white"); ax.axis("off")
    tb = ax.table(cellText=show.to_numpy().tolist(), colLabels=list(show.columns), cellLoc="center", loc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(6); tb.scale(1, 1.3)
    for j in range(len(show.columns)):
        tb[0, j].set_facecolor("#4a4a4a"); tb[0, j].set_text_props(color="white", fontweight="bold")
    fig.suptitle("Table S1 — Country-level compound climate event statistics", fontsize=10, fontweight="bold")
    fig.text(0.5, 0.06,
             f"CCE_days_per_PSU_yr: HW–EPE co-occurrence days per PSU per year ({YEARS[0]}–{YEARS[-1]} mean; same unit as its trend). "
             "EPE_ratio: Σ EPE after / Σ EPE before HW (unique attribution, W = ±30 d). Pct_HW_drought: pooled share of HW with "
             "SPEI-3 < −1 at m−1;\nits trend uses the per-PSU mean (pct_drought_b). Trends: Theil–Sen per decade; * = Mann–Kendall "
             "(Hamed–Rao) significant after Benjamini–Hochberg FDR across countries. A slope of 0 can result from a series with "
             "mostly zero years (Theil–Sen median of pairwise slopes).", ha="center", va="bottom", fontsize=6, color="#444")
    savefig(fig, FIG, "tableS01_country_statistics")
    return tab


for _mode, _D in MODE_DIRS.items():
    _F = FIG_DIRS[_mode]
    figS05_seasonality(_D, _F); figS06_comparative(_D, _F); figS08_latitude(_D, _F)
    figS09_decadal(_D, _F); figS10_monthly(_D, _F); tableS01(_D, _F)
