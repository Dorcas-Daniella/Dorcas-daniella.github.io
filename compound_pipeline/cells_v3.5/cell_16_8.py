# ---- 16.8 calendar vs annual threshold comparison + shared drought baseline ----
def fig_mode_comparison(FIG):
    """Headline metrics of each threshold mode side by side (needs both runs)."""
    HEAD = [("axis1", "cooccur_days_per_unit", "Co-occurrence days / PSU / yr"),
            ("axis2", "count_ratio_a", "EPE after / before HW"),
            ("axis2", "rate_diff_b", "Trigger − precursor"),
            ("axis3", "pct_drought_b", "% HW antecedent drought")]
    modes = list(MODE_DIRS); cols = {"calendar": "#2c3e50", "annual": "#e67e22"}
    fig, axes = plt.subplots(2, len(HEAD), figsize=(4 * len(HEAD), 7), facecolor="white")
    rows = []
    for j, (axis, metric, lab) in enumerate(HEAD):
        for m in modes:
            D = MODE_DIRS[m]; tr = trends(D)
            ser = series_by_scale(D, axis, metric); x = np.arange(len(SCALES)); w = .8 / len(modes)
            mean_v = [np.nanmean(ser[sc]) for sc in SCALES]
            axes[0, j].bar(x + modes.index(m) * w, mean_v, w, color=cols.get(m, "#999"), label=m, alpha=.85)
            sl = [slope_row(tr, axis, "continental" if sc == "Continental" else "region", sc, metric) for sc in SCALES]
            s = np.array([r["slope_per_decade"] if r is not None else np.nan for r in sl])
            lo = np.array([r["ci_low_per_decade"] if r is not None else np.nan for r in sl])
            hi = np.array([r["ci_high_per_decade"] if r is not None else np.nan for r in sl])
            axes[1, j].errorbar(x + modes.index(m) * w, s, yerr=[s - lo, hi - s], fmt="o", color=cols.get(m, "#999"), capsize=3, label=m)
            for sc, mv, r in zip(SCALES, mean_v, sl):
                rows.append(dict(mode=m, axis=axis, metric=metric, scale=sc, mean_over_years=mv,
                                 slope_per_decade=r["slope_per_decade"] if r is not None else np.nan,
                                 ci_lo=r["ci_low_per_decade"] if r is not None else np.nan,
                                 ci_hi=r["ci_high_per_decade"] if r is not None else np.nan))
        for i in range(2):
            axes[i, j].set_xticks(x + (len(modes) - 1) * w / 2); axes[i, j].set_xticklabels(SCALES, rotation=30, ha="right")
        axes[0, j].set_title(f"{AXIS_NAMES[axis]}\n{lab}", fontsize=8, fontweight="bold"); axes[0, j].set_ylabel("Mean of annual values")
        axes[1, j].set_ylabel("Theil–Sen slope / decade (95% CI)"); axes[1, j].axhline(0, color="grey", lw=.5)
    # HW counts by mode
    txt = ("HW detected on all PSU — " + " | ".join(
        f"{m}: {len(pd.read_parquet(MODE_DIRS[m] / 'heatwaves.parquet', columns=['psu_idx'])):,}" for m in modes)
        + " (top row: mean of annual values, not the pooled ratio)")
    axes[0, 0].legend(fontsize=7); fig.suptitle("Calendar vs annual Tmax threshold — headline metrics\n" + txt, fontsize=10, fontweight="bold")
    fig.tight_layout(); savefig(fig, FIG, "figS_threshold_mode_comparison")
    pd.DataFrame(rows).round(4).to_csv(FIG / "figS_threshold_mode_comparison_table.csv", index=False)


def fig_drought_baseline(FIG):
    spei = pd.read_parquet(PROCESSED / "spei3.parquet"); col = f"spei{cfg.spei_scale_months}"
    # frequency over the AVAILABLE months only (NaN months are not 'no drought')
    d = spei.groupby("psu_idx")[col].apply(lambda v: 100 * (v.dropna() < cfg.drought_threshold).mean()).rename("pct_months_drought").reset_index()
    d = d.merge(PSU_GEO, on="psu_idx", how="left")
    fig = plt.figure(figsize=(8, 7.3), facecolor="white"); gs = GridSpec(2, 1, figure=fig, height_ratios=[1, .04], hspace=.08)
    ax = map_ax(fig, gs[0], lw=0.8); sc = map_scatter(ax, d, "pct_months_drought", cmap="YlOrBr", norm=mcolors.Normalize(0, 40), s=1.2)
    thr = f"{cfg.drought_threshold:g}".replace("-", "−")
    ax.set_title(f"Baseline drought frequency: % of months with SPEI-3 < {thr}, {YEARS[0]}–{YEARS[-1]}", fontsize=10, fontweight="bold")
    cax = fig.add_subplot(gs[1]); shrink_cax(cax)
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="max"); style_colorbar(cb, "% of months")
    region_legend(fig, [cax])
    savefig(fig, FIG, "figS_drought_baseline")


def axis3_expected_baseline(FIG):
    """Axis III against its LOCAL baseline. For every HW with an antecedent
    index value, the expected probability of drought is the drought frequency
    (index < cfg.drought_threshold) of the SAME PSU and calendar month, over all
    years (`exp_month`) or within the same decade (`exp_month_decade`, which also
    removes the background trend). Observed share / expected share > 1 = HW are
    preceded by drought more often than that place, season (and decade) would
    imply. Done for SPEI (PET-dependent) and SPI (precipitation only)."""
    sname, pname = f"spei{cfg.spei_scale_months}", f"spi{cfg.spei_scale_months}"
    sp = pd.read_parquet(PROCESSED / "spei3.parquet", columns=["psu_idx", "year", "month", sname, pname])
    sp["dec"] = assign_decade(sp["year"], cfg.decade_bounds).astype(str)
    base = {}
    for idx in (sname, pname):
        v = sp.dropna(subset=[idx]).assign(d=lambda t: t[idx] < cfg.drought_threshold)
        base[idx] = (v.groupby(["psu_idx", "month"])["d"].mean(),
                     v.groupby(["psu_idx", "month", "dec"])["d"].mean(),
                     float(v["d"].mean()))
    rows = []
    for m, D in MODE_DIRS.items():
        a3 = pd.read_parquet(D / "axis3.parquet", columns=["psu_idx", "year", "month", "has_spei", "has_spi",
                                                         "drought_precond", "drought_precond_spi"])
        a3["dec"] = assign_decade(a3["year"], cfg.decade_bounds).astype(str)
        a3 = a3.merge(PSU_GEO[["psu_idx", "region"]], on="psu_idx", how="left")
        for idx, has, obs in ((sname, "has_spei", "drought_precond"), (pname, "has_spi", "drought_precond_spi")):
            t = a3[a3[has]].copy()
            clim, cdec, allm = base[idx]
            t["e1"] = clim.reindex(pd.MultiIndex.from_frame(t[["psu_idx", "month"]])).to_numpy(float)
            t["e2"] = cdec.reindex(pd.MultiIndex.from_frame(t[["psu_idx", "month", "dec"]])).to_numpy(float)
            for sc in ["Continental"] + REGIONS:
                g = t if sc == "Continental" else t[t["region"] == sc]
                if not len(g):
                    continue
                o, e1, e2 = 100 * g[obs].mean(), 100 * np.nanmean(g["e1"]), 100 * np.nanmean(g["e2"])
                rows.append(dict(mode=m, index=idx, scale=sc, n_hw=len(g), pct_all_months=100 * allm,
                                 observed_pct=o, exp_month_pct=e1, exp_month_decade_pct=e2,
                                 ratio_month=o / e1 if e1 > 0 else np.nan,
                                 ratio_month_decade=o / e2 if e2 > 0 else np.nan))
    out = pd.DataFrame(rows)
    FIG.mkdir(parents=True, exist_ok=True); out.round(3).to_csv(FIG / "axis3_expected_baseline.csv", index=False)
    for r in out[out["scale"] == "Continental"].itertuples():
        log.info("Axis III %s %s: observed %.1f%% vs expected %.1f%% (PSU-month) / %.1f%% (PSU-month-decade) "
                 "-> x%.2f", r.mode, r.index, r.observed_pct, r.exp_month_pct, r.exp_month_decade_pct,
                 r.ratio_month_decade)
    return out


def figS_axis3_expected_baseline(eb, FIG):
    """Extended Data figure (primary threshold mode only; the CSV keeps both modes):
    (a) observed % of HW preceded by drought vs. its local expectation (drought
    frequency of the same PSU, calendar month and decade), SPEI-3 and SPI-3;
    (b) observed / expected ratio (bars: same PSU, month and decade; dots: same
    PSU and month)."""
    sname, pname = f"spei{cfg.spei_scale_months}", f"spi{cfg.spei_scale_months}"
    modes = [m for m in MODE_DIRS if m in set(eb["mode"])]
    prim = "annual" if "annual" in modes else modes[0]       # primary mode of the paper
    e = eb[eb["mode"] == prim]
    idx_style = [(sname, "SPEI-3", "#b35806"), (pname, "SPI-3", "#1b9e77")]
    x = np.arange(len(SCALES))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), facecolor="white",
                            gridspec_kw=dict(wspace=.22, left=.06, right=.99, top=.80, bottom=.10))

    ax = axs[0]; w = .2
    for k, (idx, lab, color) in enumerate(idx_style):
        t = e[e["index"] == idx].set_index("scale").reindex(SCALES)
        ax.bar(x - .3 + (2 * k) * w, t["observed_pct"], w * .95, color=color, label=f"{lab}, observed")
        ax.bar(x - .3 + (2 * k + 1) * w, t["exp_month_decade_pct"], w * .95, color=color, alpha=.35,
               label=f"{lab}, expected")
    ax.set_xticks(x); ax.set_xticklabels(SCALES, fontsize=10)
    ax.set_ylabel("Heatwaves preceded by drought (%)", fontsize=10)
    ax.legend(fontsize=9, ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))
    panel_label(ax, "a", x=-0.08, y=1.17, size=14)

    ax = axs[1]; w = .32
    for k, (idx, lab, color) in enumerate(idx_style):
        t = e[e["index"] == idx].set_index("scale").reindex(SCALES)
        xx = x + (k - .5) * w
        ax.bar(xx, t["ratio_month_decade"], w * .9, color=color, label=lab)
        ax.scatter(xx, t["ratio_month"], s=22, color="k", zorder=4,
                   label="Expected from cluster and month only" if k == 0 else None)
    ax.axhline(1, color="#666", lw=.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(SCALES, fontsize=10); ax.set_ylim(0, None)
    ax.set_ylabel("Observed / expected", fontsize=10)
    ax.legend(fontsize=9, ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))
    panel_label(ax, "b", x=-0.08, y=1.17, size=14)
    for a in axs:
        a.tick_params(axis="y", labelsize=9); a.yaxis.grid(True, lw=.3, color="#e3e3e3", zorder=0)
        a.set_axisbelow(True)
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
    log.info("figS_axis3_expected_baseline: %s mode", prim)
    savefig(fig, FIG, "figS_axis3_expected_baseline")


_ROOT_FIG = PROCESSED / "figures_shared"
fig_drought_baseline(_ROOT_FIG)
_eb = axis3_expected_baseline(_ROOT_FIG)
figS_axis3_expected_baseline(_eb, _ROOT_FIG)
if len(MODE_DIRS) > 1:
    fig_mode_comparison(_ROOT_FIG)

# ---- run completed: final manifest update ------------------------------------
# Recomputes the code fingerprint now that EVERY function (sections 13-16
# included) is defined, and stores the final configuration. A manifest still
# showing run_status = "started" belongs to an interrupted run.
manifest_update("run_status", dict(state="completed",
                                   completed_utc=datetime.now(timezone.utc).isoformat(),
                                   final_config=cfg))
print("Run completed; manifest:", MANIFEST)

