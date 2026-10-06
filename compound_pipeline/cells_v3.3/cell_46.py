# ---- 16.1 Figures 1 & 3: spatial distribution (pooled) and by decade ---------
MAP_CFG = {
    "axis1": dict(cmap="YlOrRd", norm=None, extend="max"),
    "axis2": dict(cmap="RdBu_r", norm=mcolors.TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=6), extend="max"),
    "axis3": dict(cmap="YlOrBr", norm=mcolors.Normalize(vmin=0, vmax=80), extend="max"),
}


def _norm_for(axis, values):
    n = MAP_CFG[axis]["norm"]
    if n is None:
        v = values[np.isfinite(values) & (values > 0)]
        n = mcolors.Normalize(vmin=0, vmax=np.percentile(v, 95) if v.size else 1)
    return n


UNDEF_COLOR = "#b0b0b0"


def plot_undefined_ratio(ax, df, legend=True):
    """Axis II maps: PSU with heatwaves but no EPE attributed BEFORE them have
    an undefined after/before ratio. They are drawn in grey (instead of being
    dropped, which would look like 'no PSU'); returns their count."""
    u = df[df["n_before"].eq(0) & df["count_ratio"].isna()]
    if len(u):
        map_scatter(ax, u, c=UNDEF_COLOR, s=0.6, zorder=3)
    if legend:
        ax.legend(handles=[plt.Line2D([0], [0], marker="o", ls="", color=UNDEF_COLOR, ms=3,
                                      label="no EPE before HW (ratio undefined)")],
                  loc="lower left", fontsize=6, frameon=False, handletextpad=0.2)
    return len(u)


def fig01_spatial(D, FIG):
    df = psu_metric_table(D)
    fig = plt.figure(figsize=(14, 5.2), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 0.05], wspace=0.05, hspace=0.05)
    for j, (axis, lab) in enumerate(zip(("axis1", "axis2", "axis3"), "abc")):
        col = AXIS_PSU_COLS[axis]
        ax = map_ax(fig, gs[0, j])
        if axis == "axis2":
            nu = plot_undefined_ratio(ax, df)
            n2 = int(df["n_before"].notna().sum())
            log.info("fig01 axis2: %d of %d PSU with no EPE before HW (ratio undefined, grey); "
                     "of which %d with EPE after HW", nu, n2,
                     int((df["n_before"].eq(0) & df["n_after"].gt(0)).sum()))
        sc = map_scatter(ax, df, col, cmap=MAP_CFG[axis]["cmap"], norm=_norm_for(axis, df[col].to_numpy()))
        ax.set_title(AXIS_NAMES[axis], fontsize=9, fontweight="bold", pad=6)
        panel_label(ax, lab, x=0.0, y=1.02)
        cb = fig.colorbar(sc, cax=fig.add_subplot(gs[1, j]), orientation="horizontal",
                          extend=MAP_CFG[axis]["extend"])
        cb.set_label(AXIS_PSU_LABELS[axis], fontsize=7)
        cb.ax.tick_params(labelsize=6)
    savefig(fig, FIG, "fig01_spatial_distribution")


def fig03_decades(D, FIG):
    tabs = {"axis1": agg(D, "axis1_by_psu_decade"), "axis2": agg(D, "axis2_by_psu_decade"),
            "axis3": agg(D, "axis3_by_psu_decade")}
    fig = plt.figure(figsize=(14, 12.5), facecolor="white")
    gs = GridSpec(6, 4, figure=fig, wspace=0.03, hspace=0.08,
                  height_ratios=[1, 0.05, 1, 0.05, 1, 0.05])
    for i, (axis, lab) in enumerate(zip(("axis1", "axis2", "axis3"), "abc")):
        col = AXIS_PSU_COLS[axis]
        t = tabs[axis].copy(); t["decade"] = t["decade"].astype(str)
        t = t.merge(PSU_GEO, on="psu_idx", how="left")
        norm = _norm_for(axis, t[col].to_numpy())
        sc = None
        for j, dec in enumerate(DECADES):
            ax = map_ax(fig, gs[2 * i, j])
            sub = t[t["decade"] == dec]
            if axis == "axis2":
                nu = plot_undefined_ratio(ax, sub[sub["n_hw"] > 0], legend=(j == 0))
                log.info("fig03 axis2 %s: %d PSU with HW but no EPE before HW (grey)", dec, nu)
            if len(sub.dropna(subset=[col])):
                sc = map_scatter(ax, sub, col, cmap=MAP_CFG[axis]["cmap"], norm=norm)
            if i == 0:
                ax.set_title(dec.replace("-", "–"), fontsize=9, fontweight="bold", pad=6)
            if j == 0:
                ax.text(-0.04, 0.5, AXIS_NAMES[axis], transform=ax.transAxes, rotation=90,
                        fontsize=7.5, fontweight="bold", va="center", ha="right")
                panel_label(ax, lab, x=-0.08, y=1.02)
        if sc is not None:
            pos = gs[2 * i + 1, :].get_position(fig)
            cax = fig.add_axes([pos.x0 + pos.width * .15, pos.y0 + pos.height * .3,
                                pos.width * .7, pos.height * .4])
            cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend=MAP_CFG[axis]["extend"])
            cb.set_label(AXIS_PSU_LABELS[axis] + ", by decade", fontsize=7)
            cb.ax.tick_params(labelsize=6)
    savefig(fig, FIG, "fig03_spatiotemporal_decades")


for _mode, _D in MODE_DIRS.items():
    fig01_spatial(_D, FIG_DIRS[_mode])
    fig03_decades(_D, FIG_DIRS[_mode])
