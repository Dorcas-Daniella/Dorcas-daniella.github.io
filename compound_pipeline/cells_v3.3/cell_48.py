# ---- 16.3 Figures 4 & 5: Axis II asymmetry deep dive, Axis III amplification --
SHORT_NAMES = {"Burkina Faso": "Burkina F.", "Central African Republic": "CAR",
               "Congo Democratic Republic": "DR Congo", "Cote d'Ivoire": "Côte d'Iv.",
               "Sierra Leone": "Sierra L.", "South Africa": "S. Africa"}


def fig04_asymmetry(D, FIG):
    W = cfg.axis2_window_days
    bc = agg(D, "axis2_by_country")
    att = pd.read_parquet(D / "axis2_attrib.parquet")
    _el = psu_eligibility(cfg, D)
    att = att[att["psu_idx"].isin(_el.loc[_el["axis12_eligible"], "psu_idx"])]   # same population as 4a/4c
    att = att[~att["truncated"] & att["position"].isin(["before", "after"])].copy()
    att["decade"] = assign_decade(pd.to_datetime(att["start_date"]).dt.year, cfg.decade_bounds).astype(str)
    rd = agg(D, "axis2_by_region_decade"); rd["decade"] = rd["decade"].astype(str)

    fig = plt.figure(figsize=(14, 5), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    # (a) after vs before per PSU, per country
    ax = fig.add_subplot(gs[0, 0])
    mx = max(bc["before_per_psu"].max(), bc["after_per_psu"].max()) * 1.1
    ax.plot([0, mx], [0, mx], "k--", lw=.5, alpha=.4, label="1:1 (symmetry)")
    for reg in REGIONS:
        s = bc[bc["region"] == reg]
        ax.scatter(s["before_per_psu"], s["after_per_psu"], c=REGION_COLORS[reg], s=30,
                   alpha=.85, edgecolors="white", lw=.3, zorder=3, label=f"{reg} Africa")
    for _, r in bc.iterrows():
        ax.text(r["before_per_psu"], r["after_per_psu"], SHORT_NAMES.get(r["country"], r["country"]),
                fontsize=4.5, color=REGION_COLORS.get(r["region"], "#555"), alpha=.9)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("EPE before HW (count / PSU)"); ax.set_ylabel("EPE after HW (count / PSU)")
    ax.set_title("EPE before vs. after heatwaves (unique attribution)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=5.5, loc="upper left"); panel_label(ax, "a")
    # (b) timing distribution by decade
    ax = fig.add_subplot(gs[0, 1])
    bins = np.arange(-W - .5, W + 1.5, 1)
    for dec in DECADES:
        s = att.loc[att["decade"] == dec, "offset_days"]
        if len(s):
            cnt, edges = np.histogram(s, bins=bins, density=True)
            ctr = (edges[:-1] + edges[1:]) / 2
            cnt[ctr == 0] = np.nan        # day 0 = inside the HW (Axis I), not a possible offset: break the line
            ax.plot(ctr, cnt, color=DECADE_COLORS[dec], lw=1.2, label=dec.replace("-", "–"))
    ax.axvline(0, color="#888", lw=.5)
    ax.set_xlabel("Days from nearest heatwave boundary (− before onset, + after end)"); ax.set_ylabel("Density")
    ax.set_title("EPE timing relative to HW, by decade", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6); panel_label(ax, "b")
    # (c) count ratio region x decade
    ax = fig.add_subplot(gs[0, 2])
    M = np.full((len(SCALES), len(DECADES)), np.nan)
    for i, sc in enumerate(SCALES):
        for j, dec in enumerate(DECADES):
            r = rd[(rd["region"] == sc) & (rd["decade"] == dec)]
            if len(r):
                M[i, j] = r["count_ratio"].iloc[0]
    im = ax.imshow(M, cmap="OrRd", aspect="auto", vmin=1, vmax=max(2, np.nanpercentile(M, 95)))
    ax.set_xticks(range(len(DECADES))); ax.set_xticklabels([d.replace("-", "–") for d in DECADES], rotation=30, ha="right")
    ax.set_yticks(range(len(SCALES))); ax.set_yticklabels(SCALES)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7, fontweight="bold",
                        color="white" if M[i, j] > 0.75 * np.nanmax(M) else "#333")
    ax.set_title("After/before count ratio by region and decade", fontsize=9, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, shrink=.7, pad=.04, extend="both"); cb.set_label("Ratio", fontsize=7)
    panel_label(ax, "c", x=-0.12)
    savefig(fig, FIG, "fig04_axis2_asymmetry")


def _paired_box(ax, d0, d1, pos, width=.32):
    for d, dx, fc, mc in ((d0, -width / 2, "#b3cde3", "#2166ac"), (d1, width / 2, "#e6550d", "#b2182b")):
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        bp = ax.boxplot([d], positions=[pos + dx], widths=width * .7, patch_artist=True,
                        showfliers=False, zorder=3, medianprops=dict(color="#333", lw=1),
                        whiskerprops=dict(lw=.5), capprops=dict(lw=.5))
        bp["boxes"][0].set(facecolor=fc, edgecolor="#333", lw=.5, alpha=.85)
        ax.scatter(pos + dx, d.mean(), marker="D", s=14, color=mc, edgecolor="white", lw=.4, zorder=5)


def fig05_drought(D, FIG):
    a3 = pd.read_parquet(D / "axis3.parquet").merge(PSU_GEO[["psu_idx", "region"]], on="psu_idx", how="left")
    _el = psu_eligibility(cfg, D)
    a3 = a3[a3["psu_idx"].isin(_el.loc[_el["spei_eligible"], "psu_idx"]) & a3["has_spei"]]
    rd = agg(D, "axis3_by_region_decade"); rd["decade"] = rd["decade"].astype(str)
    fig = plt.figure(figsize=(14, 5), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    leg = [mpatches.Patch(fc="#b3cde3", ec="#333", lw=.5, label="No antecedent drought"),
           mpatches.Patch(fc="#e6550d", ec="#333", lw=.5, label="Antecedent drought (SPEI-3 < −1)"),
           plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#666", markersize=4, label="Mean")]
    for k, (col, ylab, title) in enumerate((("duration", "HW duration (days)", "Heatwave duration"),
                                           ("peak_tmax", "Peak Tmax (°C)", "Heatwave intensity"))):
        ax = fig.add_subplot(gs[0, k])
        for i, reg in enumerate(REGIONS):
            s = a3[a3["region"] == reg]
            _paired_box(ax, s.loc[~s["drought_precond"], col].to_numpy(float),
                        s.loc[s["drought_precond"], col].to_numpy(float), i)
        ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels(REGIONS)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=9, fontweight="bold")
        ax.yaxis.grid(True, lw=.15, color="#e0e0e0", zorder=0)
        ax.legend(handles=leg, fontsize=6, loc="upper right"); panel_label(ax, "ab"[k])
    ax = fig.add_subplot(gs[0, 2])
    M = np.full((len(SCALES), len(DECADES)), np.nan)
    for i, sc in enumerate(SCALES):
        for j, dec in enumerate(DECADES):
            r = rd[(rd["region"] == sc) & (rd["decade"] == dec)]
            if len(r):
                M[i, j] = r["pct_drought_a"].iloc[0]
    im = ax.imshow(M, cmap="YlOrBr", aspect="auto", vmin=0, vmax=50)
    ax.set_xticks(range(len(DECADES))); ax.set_xticklabels([d.replace("-", "–") for d in DECADES], rotation=30, ha="right")
    ax.set_yticks(range(len(SCALES))); ax.set_yticklabels(SCALES)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7.5, fontweight="bold",
                        color="white" if M[i, j] > 35 else "#333")
    ax.set_title("% HW with antecedent drought, by region and decade", fontsize=9, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, shrink=.8, pad=.03, extend="max"); cb.set_label("% of HW", fontsize=7)
    panel_label(ax, "c", x=-0.15)
    savefig(fig, FIG, "fig05_drought_amplification")


for _mode, _D in MODE_DIRS.items():
    fig04_asymmetry(_D, FIG_DIRS[_mode])
    fig05_drought(_D, FIG_DIRS[_mode])
