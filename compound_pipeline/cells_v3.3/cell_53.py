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
                                 ci_hi=r["ci_high_per_decade"] if r is not None else np.nan,
                                 mk_p=r["mk_p"] if r is not None else np.nan))
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
    fig = plt.figure(figsize=(8, 8), facecolor="white"); gs = GridSpec(2, 1, figure=fig, height_ratios=[1, .04], hspace=.05)
    ax = map_ax(fig, gs[0]); sc = map_scatter(ax, d, "pct_months_drought", cmap="YlOrBr", norm=mcolors.Normalize(0, 40), s=1.2)
    ax.set_title(f"Baseline drought frequency: % of months with SPEI-3 < {cfg.drought_threshold}, {YEARS[0]}–{YEARS[-1]}", fontsize=10, fontweight="bold")
    cb = fig.colorbar(sc, cax=fig.add_subplot(gs[1]), orientation="horizontal", extend="max"); cb.set_label("% of months")
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
    """SI figure from axis3_expected_baseline: (a) primary mode, observed % of HW
    preceded by drought vs. its local expectation (same PSU and calendar month,
    within the same decade), SPEI and SPI; (b) observed / expected ratio for the
    two expectations, both indices, both threshold modes."""
    sname, pname = f"spei{cfg.spei_scale_months}", f"spi{cfg.spei_scale_months}"
    modes = [m for m in MODE_DIRS if m in set(eb["mode"])]
    prim = "calendar" if "calendar" in modes else modes[0]
    x = np.arange(len(SCALES))
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), facecolor="white", gridspec_kw=dict(wspace=.22))
    ax = axs[0]
    spec = [(sname, "observed_pct", "SPEI-3 observed", "#b35806", 1.0),
            (sname, "exp_month_decade_pct", "SPEI-3 expected", "#b35806", .35),
            (pname, "observed_pct", "SPI-3 observed", "#1b9e77", 1.0),
            (pname, "exp_month_decade_pct", "SPI-3 expected", "#1b9e77", .35)]
    w = .2
    for k, (idx, col, lab, color, alpha) in enumerate(spec):
        v = eb[(eb["mode"] == prim) & (eb["index"] == idx)].set_index("scale").reindex(SCALES)[col]
        ax.bar(x - .3 + k * w, v, w * .95, color=color, alpha=alpha, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(SCALES); ax.set_ylabel(f"% of HW preceded by drought (index < {cfg.drought_threshold:g})")
    ax.set_title(f"Observed vs. locally expected share ({prim} mode)\n"
                 "expected = drought frequency of the same PSU, calendar month and decade",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=6.5, ncol=2, frameon=False); panel_label(ax, "a", x=-0.1, y=1.08)
    ax = axs[1]
    combos = [(m, idx) for m in modes for idx in (sname, pname)]
    w = .8 / len(combos)
    colors = {sname: "#b35806", pname: "#1b9e77"}
    for k, (m, idx) in enumerate(combos):
        t = eb[(eb["mode"] == m) & (eb["index"] == idx)].set_index("scale").reindex(SCALES)
        xx = x - .4 + w * (k + .5)
        ax.bar(xx, t["ratio_month_decade"], w * .9, color=colors[idx], alpha=1.0 if m == prim else .45,
               hatch=None if m == prim else "//", edgecolor="white", lw=.3,
               label=f"{idx.upper()[:-1]}-3, {m}")
        ax.scatter(xx, t["ratio_month"], s=10, color="k", zorder=4,
                   label="expected from PSU-month only" if k == 0 else None)
    ax.axhline(1, color="#888", lw=.6, ls="--"); ax.set_xticks(x); ax.set_xticklabels(SCALES)
    ax.set_ylabel("Observed / expected"); ax.set_ylim(0, None)
    ax.set_title("Observed / expected (> 1 = more often than locally expected)\n"
                 "bars: same PSU, month and decade · dots: same PSU and month",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.08)); panel_label(ax, "b", x=-0.1, y=1.08)
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
