# ---- 16.7 Null-model figure and sensitivity figure --------------------------------
def fig_null(D, FIG):
    nl = sig(D, "axis2_null_region"); dr = sig(D, "axis2_null_draws")
    seasons = ["ALL"] + list(cfg.null_seasons)
    mode_txt = ("same PSU, other year, onset ±%d d" % cfg.null_window_days if cfg.null_mode == "psu_window"
                else "regional onset-doy pool")
    fig = plt.figure(figsize=(14, 13.5), facecolor="white")
    gs = GridSpec(3, 2, figure=fig, hspace=.45, wspace=.3, height_ratios=[1, 1, .9])
    # (a) PRIMARY: count ratio obs vs null, by scale (ALL seasons)
    ax = fig.add_subplot(gs[0, 0]); t = nl[nl["season"] == "ALL"].set_index("scale").reindex(SCALES)
    x = np.arange(len(SCALES))
    ax.bar(x, t["count_ratio_obs"], .5, color=[REGION_COLORS[s] for s in SCALES], alpha=.85, label="observed")
    ax.errorbar(x, t["count_ratio_null_med"], yerr=[t["count_ratio_null_med"] - t["count_ratio_null_lo"],
                t["count_ratio_null_hi"] - t["count_ratio_null_med"]], fmt="s", color="k", ms=4, capsize=4, lw=1,
                label="null median & 95% envelope")
    for i, (o_, m_) in enumerate(zip(t["count_ratio_obs"], t["count_ratio_null_med"])):   # effect size, no p-value
        ax.text(x[i], max(o_, t["count_ratio_null_hi"].iloc[i]) * 1.04,
                f"×{o_ / m_:.2f} null median" if np.isfinite(o_ / m_) else "", ha="center", fontsize=6)
    ax.axhline(1, color="#888", lw=.5, ls="--"); ax.set_xticks(x); ax.set_xticklabels(SCALES)
    ax.set_ylabel("EPE after / before HW (unique attribution)")
    ax.set_title("PRIMARY metric vs. strict null (%s)" % mode_txt, fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2); panel_label(ax, "a")
    # (b) secondary: ECA fractions
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x - .18, t["precursor_obs"], .36, color="#91bfdb", label="precursor (EPE before HW)")
    ax.bar(x + .18, t["trigger_obs"], .36, color="#d73027", label="trigger (EPE after HW)")
    ax.errorbar(x - .18, (t["precursor_null_lo"] + t["precursor_null_hi"]) / 2, yerr=(t["precursor_null_hi"] - t["precursor_null_lo"]) / 2,
                fmt="none", ecolor="k", capsize=3, lw=1)
    ax.errorbar(x + .18, (t["trigger_null_lo"] + t["trigger_null_hi"]) / 2, yerr=(t["trigger_null_hi"] - t["trigger_null_lo"]) / 2,
                fmt="none", ecolor="k", capsize=3, lw=1, label="null 95% envelope")
    ax.set_xticks(x); ax.set_xticklabels(SCALES); ax.set_ylabel(f"Share of HW with ≥1 EPE within {cfg.axis2_window_days} d")
    ax.set_title("Secondary metric: ECA fractions vs. null", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3); panel_label(ax, "b")
    # (c) seasonal decomposition (onset season) — count ratio
    ax = fig.add_subplot(gs[1, 0]); w = .8 / len(seasons)
    for k, se in enumerate(seasons):
        ts = nl[nl["season"] == se].set_index("scale").reindex(SCALES)
        xx = x - .4 + w * (k + .5)
        ax.bar(xx, ts["count_ratio_obs"], w * .9, color=plt.cm.viridis(k / max(1, len(seasons) - 1)), alpha=.85, label=se)
        ax.errorbar(xx, ts["count_ratio_null_med"], yerr=[(ts["count_ratio_null_med"] - ts["count_ratio_null_lo"]).clip(lower=0),
                    (ts["count_ratio_null_hi"] - ts["count_ratio_null_med"]).clip(lower=0)], fmt="_", color="k", ms=5, capsize=2, lw=.8)
    ax.axhline(1, color="#888", lw=.5, ls="--"); ax.set_xticks(x); ax.set_xticklabels(SCALES)
    ax.set_ylabel("EPE after / before HW"); ax.set_title("By season of HW onset (bars = observed, black = null envelope)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, ncol=5); panel_label(ax, "c")
    # (d) null distribution, continental
    ax = fig.add_subplot(gs[1, 1]); d = dr[(dr["scale"] == "Continental") & (dr["season"] == "ALL")]["count_ratio"].dropna()
    ax.hist(d, bins=40, color="#999", alpha=.8, label=f"null ({len(d)} permutations)")
    obs = t.loc["Continental", "count_ratio_obs"]; ax.axvline(obs, color="#d73027", lw=2, label=f"observed = {obs:.2f}")
    ax.set_xlabel("EPE after / before HW"); ax.set_ylabel("Count"); ax.set_title("Null distribution, Continental", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6); panel_label(ax, "d")
    # (e) decomposition: observed EPE before / after HW relative to the null median
    #     (< 1 = deficit, > 1 = excess vs. the same season in other years); the black
    #     bars are the null 95% envelope expressed relative to its own median
    ax = fig.add_subplot(gs[2, :])
    need = ["n_before_obs", "n_before_null_med", "n_before_null_lo", "n_before_null_hi",
            "n_after_obs", "n_after_null_med", "n_after_null_lo", "n_after_null_hi"]
    if all(c in t.columns for c in need):
        for side, off, col, lab in (("before", -.18, "#4575b4", "EPE before HW"),
                                    ("after", .18, "#d73027", "EPE after HW")):
            med = t[f"n_{side}_null_med"].to_numpy(float)
            r = t[f"n_{side}_obs"].to_numpy(float) / med
            lo = t[f"n_{side}_null_lo"].to_numpy(float) / med
            hi = t[f"n_{side}_null_hi"].to_numpy(float) / med
            ax.bar(x + off, r, .34, color=col, alpha=.85, label=f"{lab}: observed / null median")
            ax.errorbar(x + off, np.ones_like(r), yerr=[1 - lo, hi - 1], fmt="none", ecolor="k",
                        capsize=3, lw=1)
            for xi, v in zip(x + off, r):
                if np.isfinite(v):         # value inside the bar, clear of the envelope at 1
                    ax.text(xi, .04, f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                            color="white", fontweight="bold")
        ax.errorbar([], [], yerr=[], fmt="none", ecolor="k", capsize=3, lw=1, label="null 95% envelope")
        ax.axhline(1, color="#888", lw=.6, ls="--"); ax.set_xticks(x); ax.set_xticklabels(SCALES)
        ax.set_ylabel("Observed / null median")
        ax.set_ylim(0, max(1.25, np.nanmax(t[["n_after_obs"]].to_numpy(float) / t[["n_after_null_med"]].to_numpy(float)) * 1.15))
        ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(.5, -.1), ncol=3)
    else:
        ax.text(.5, .5, "decomposition not available (null run before v3.2)", ha="center", va="center",
                transform=ax.transAxes); ax.set_axis_off()
    ax.set_title("Decomposition of the asymmetry: EPE before vs. after HW relative to the null "
                 "(< 1 = deficit, > 1 = excess)", fontsize=9, fontweight="bold")
    panel_label(ax, "e", x=-0.05)
    savefig(fig, FIG, "figS_axis2_null_test")
    nl.drop(columns=[c for c in nl.columns if c.startswith("p_")]).round(4).to_csv(
        FIG / "figS_axis2_null_test_table.csv", index=False)   # p-values kept in the parquet only


def fig_sensitivity(D, FIG):
    ws = sig(D, "axis2_window_sensitivity"); a3 = pd.read_parquet(D / "axis3.parquet")
    _el = psu_eligibility(cfg, D)
    a3 = a3[a3["psu_idx"].isin(_el.loc[_el["spei_eligible"], "psu_idx"])]       # Axis III population
    a3 = a3.merge(PSU_GEO[["psu_idx", "region"]], on="psu_idx", how="left"); a3["year"] = pd.to_datetime(a3["start_date"]).dt.year
    sname = f"spei{cfg.spei_scale_months}"; pname = f"spi{cfg.spei_scale_months}"
    Ws = sorted(ws["window_days"].unique()); thr = list(cfg.spei_threshold_sens)
    fig = plt.figure(figsize=(16, 9), facecolor="white"); gs = GridSpec(2, 6, figure=fig, hspace=.4, wspace=.4)
    # (a) W sensitivity by region — count ratio
    ax = fig.add_subplot(gs[0, 0:3]); x = np.arange(len(Ws)); w = .8 / len(REGIONS)
    for i, reg in enumerate(REGIONS):
        v = ws[ws["scale"] == reg].set_index("window_days")["count_ratio"].reindex(Ws)
        ax.bar(x + i * w, v.to_numpy(), w, color=REGION_COLORS[reg], label=reg, alpha=.85, edgecolor="white", lw=.3)
    ax.set_xticks(x + (len(REGIONS) - 1) * w / 2); ax.set_xticklabels([f"±{W}" for W in Ws]); ax.axhline(1, color="grey", ls="--", lw=.5)
    ax.set_xlabel("Window W (days)"); ax.set_ylabel("EPE after / before HW"); ax.legend(fontsize=7, ncol=4)
    ax.set_title(f"Axis II count ratio by region and window (primary W = ±{cfg.axis2_window_days})", fontsize=9, fontweight="bold"); panel_label(ax, "a", x=-0.07)
    # (b) continental W sensitivity: count ratio + trigger-precursor
    ax = fig.add_subplot(gs[0, 3:6]); c = ws[ws["scale"] == "Continental"].set_index("window_days").reindex(Ws)
    ax.plot(Ws, c["count_ratio"], "o-", color="#2c3e50", lw=1.8, ms=6, label="count ratio (primary)")
    for W, r in zip(Ws, c["count_ratio"]):
        ax.annotate(f"{r:.2f}", (W, r), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7, color="#2c3e50")
    ax2 = ax.twinx(); ax2.plot(Ws, c["rate_diff"], "s--", color="#d73027", lw=1.2, ms=5, label="trigger − precursor (secondary)")
    ax2.set_ylabel("Trigger − precursor fraction", color="#d73027"); ax2.spines["right"].set_visible(True)
    ax.axvline(cfg.axis2_window_days, color="#e74c3c", ls=":", lw=.8); ax.axhline(1, color="grey", ls="--", lw=.5)
    ax.set_xticks(Ws); ax.set_xticklabels([f"±{W}" for W in Ws]); ax.set_xlabel("Window W (days)"); ax.set_ylabel("EPE after / before HW")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right", frameon=False)
    ax.set_title("Continental Axis II metrics by window size", fontsize=9, fontweight="bold"); panel_label(ax, "b", x=-0.07)
    # (c) SPEI threshold: % HW preconditioned and duration ratio (primary lag)
    s = a3[a3["has_spei"]]
    ax = fig.add_subplot(gs[1, 0:2]); cols = ["#4292c6", "#f4a582", "#ef6548", "#67000d"][:len(thr)]
    pct = [100 * (s[sname] < t).mean() for t in thr]
    dur = [s.loc[s[sname] < t, "duration"].mean() / s.loc[~(s[sname] < t), "duration"].mean() for t in thr]
    ax.bar(range(len(thr)), pct, color=cols, edgecolor="white")
    for i, v in enumerate(pct):
        ax.text(i, v + .8, f"{v:.1f}%", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(thr))); ax.set_xticklabels([f"< {t}" for t in thr]); ax.set_xlabel("SPEI-3 threshold (antecedent month)")
    ax.set_ylabel("% HW with antecedent drought"); ax.set_title("Drought threshold sensitivity", fontsize=9, fontweight="bold"); panel_label(ax, "c", x=-0.1)
    ax = fig.add_subplot(gs[1, 2:4]); ax.bar(range(len(thr)), dur, color=cols, edgecolor="white"); ax.axhline(1, color="grey", ls="--", lw=.5)
    for i, v in enumerate(dur):
        ax.text(i, v + .01, f"{v:.2f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(thr))); ax.set_xticklabels([f"< {t}" for t in thr]); ax.set_xlabel("SPEI-3 threshold")
    ax.set_ylabel("Duration ratio (drought / no drought)"); ax.set_title("HW duration amplification", fontsize=9, fontweight="bold"); panel_label(ax, "d", x=-0.1)
    # (e) lag 1 vs lag 0, SPEI vs SPI by region (% HW preconditioned)
    ax = fig.add_subplot(gs[1, 4:6]); variants = [("drought_precond", "has_spei", "SPEI-3, lag 1 (primary)", "#2c3e50")]
    for L in list(cfg.axis3_lags)[1:]:
        variants.append((f"drought_precond_lag{L}", f"has_spei_lag{L}", f"SPEI-3, lag {L}", "#7570B3"))
    variants.append(("drought_precond_spi", "has_spi", "SPI-3, lag 1", "#1B9E77"))
    w = .8 / len(variants); x = np.arange(len(SCALES))
    for k, (col, has, lab, color) in enumerate(variants):
        v = [100 * a3.loc[a3[has] & ((a3["region"] == sc) if sc != "Continental" else True), col].mean() for sc in SCALES]
        ax.bar(x + k * w, v, w, color=color, label=lab, alpha=.85, edgecolor="white", lw=.3)
    ax.set_xticks(x + (len(variants) - 1) * w / 2); ax.set_xticklabels(SCALES); ax.set_ylabel("% HW with antecedent drought")
    ax.set_title("Axis III: lag and index sensitivity", fontsize=9, fontweight="bold"); ax.legend(fontsize=6.5); panel_label(ax, "e", x=-0.1)
    fig.suptitle("Sensitivity analyses — Axis II window, Axis III threshold / lag / index", fontsize=11, fontweight="bold", y=.995)
    savefig(fig, FIG, "figS_sensitivity")
    # threshold x trend table (Theil-Sen of the annual continental %), for the text
    rows = []
    for t in thr:
        yr = s.groupby("year")[sname].apply(lambda v: 100 * (v < t).mean()).reindex(YEARS)
        ts = theilsen_line(yr.to_numpy(float))
        rows.append(dict(threshold=t, pct_hw=100 * (s[sname] < t).mean(),
                         duration_ratio=s.loc[s[sname] < t, "duration"].mean() / s.loc[~(s[sname] < t), "duration"].mean(),
                         trend_pct_per_decade=ts[0] * 10 if ts else np.nan,
                         trend_ci_lo=ts[2] * 10 if ts else np.nan, trend_ci_hi=ts[3] * 10 if ts else np.nan,
                         trend_ci_method="scipy.stats.theilslopes 95% CI (not the block bootstrap of step 14)"))
    pd.DataFrame(rows).round(4).to_csv(FIG / "figS_sensitivity_spei_threshold_table.csv", index=False)
    ws.round(4).to_csv(FIG / "figS_sensitivity_window_table.csv", index=False)


AXIS2_BUFFERS = (0, 1, 2, 3, 5)


def axis2_boundary_buffer(D, FIG):
    """Axis II sensitivity to EPE close to the heatwave boundaries (rain that
    terminates a heatwave falls on day +1): the after/before ratio recomputed
    without the attributed EPE within g days of onset/end (|offset| <= g), same
    eligible population and non-truncated heatwaves as the primary metric.
    g = 0 reproduces the primary ratio. The null is not recomputed here."""
    el = psu_eligibility(cfg, D)
    a = pd.read_parquet(D / "axis2_attrib.parquet", columns=["psu_idx", "position", "offset_days", "truncated"])
    a = a[a["psu_idx"].isin(el.loc[el["axis12_eligible"], "psu_idx"]) & ~a["truncated"]
          & a["position"].isin(["before", "after"])]
    off = a["offset_days"].abs()
    rows = []
    for gap in AXIS2_BUFFERS:
        s = a[off > gap]
        nb, na = int((s["position"] == "before").sum()), int((s["position"] == "after").sum())
        rows.append(dict(buffer_days=gap, n_before=nb, n_after=na, count_ratio=na / nb if nb else np.nan))
    out = pd.DataFrame(rows)
    nA, nB = (a["position"] == "after").sum(), (a["position"] == "before").sum()
    out["pct_after_on_day_plus1"] = 100 * ((a["position"] == "after") & (a["offset_days"] == 1)).sum() / max(nA, 1)
    out["pct_before_on_day_minus1"] = 100 * ((a["position"] == "before") & (a["offset_days"] == -1)).sum() / max(nB, 1)
    FIG.mkdir(parents=True, exist_ok=True); out.round(4).to_csv(FIG / "figS_axis2_boundary_buffer_table.csv", index=False)
    log.info("Axis II boundary buffer (%s): %s", D.name,
             ", ".join(f"±{r.buffer_days} d -> {r.count_ratio:.3f}" for r in out.itertuples()))
    return out


for _mode, _D in MODE_DIRS.items():
    fig_null(_D, FIG_DIRS[_mode]); fig_sensitivity(_D, FIG_DIRS[_mode])
    axis2_boundary_buffer(_D, FIG_DIRS[_mode])
