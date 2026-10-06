# ---- 16.4 Figure 6: regimes (single relative rule for the three axes) + first vs last decade
REGIME_ORDER = ["All three", "Post-HW precip. + Drought-heat", "Heat-precip. + Post-HW precip.",
                "Heat-precip. + Drought-heat", "Post-HW precip. only", "Drought-heat only",
                "Heat-precip. only", "None prominent"]
REGIME_COLORS = {"All three": "#2c3e50", "Post-HW precip. + Drought-heat": "#8e44ad",
                 "Heat-precip. + Post-HW precip.": "#e67e22", "Heat-precip. + Drought-heat": "#e74c3c",
                 "Post-HW precip. only": "#3498db", "Drought-heat only": "#1abc9c",
                 "Heat-precip. only": "#f39c12", "None prominent": "#bdc3c7"}


def regime_label(x1, x2, x3):
    if x1 and x2 and x3: return "All three"
    if x2 and x3:        return "Post-HW precip. + Drought-heat"
    if x1 and x2:        return "Heat-precip. + Post-HW precip."
    if x1 and x3:        return "Heat-precip. + Drought-heat"
    if x2:               return "Post-HW precip. only"
    if x3:               return "Drought-heat only"
    if x1:               return "Heat-precip. only"
    return "None prominent"


INSUFFICIENT = "Insufficient data"
REGIME_COLORS[INSUFFICIENT] = "#7f7f7f"     # distinct mid grey (not 'None prominent')
FIG6_COLS = {"axis1": "cooccur_days_per_year", "axis2": "count_ratio_fig6",
             "axis3": "pct_drought_psu"}
SUPPORT_LEVELS = (5, 10, 20)


def regime_inputs(D, period):
    """Per-PSU inputs of Fig 6 for one period ("full" or a decade label):
    eligibility, the three metrics and the supports (Axis II: n_before +
    n_after; Axis III: n_hw_spei; Axis I: none). The Axis-II ratio is REBUILT
    for this figure only: nb = na = 0 -> NaN (insufficient); nb = 0 < na ->
    +inf; nb > 0 -> na / nb (0 when na = 0). Tables keep NaN when nb = 0."""
    el = psu_eligibility(cfg, D)
    if period == "full":
        a1 = agg(D, "axis1_by_psu")[["psu_idx", "cooccur_days_per_year"]]
        a2 = agg(D, "axis2_by_psu")[["psu_idx", "n_before", "n_after"]]
        a3 = agg(D, "axis3_by_psu")[["psu_idx", "pct_drought_psu", "n_hw_spei"]]
    else:
        def dec(name, cols):
            t = agg(D, name); t["decade"] = t["decade"].astype(str)
            return t.loc[t["decade"] == period, ["psu_idx"] + cols]
        a1 = dec("axis1_by_psu_decade", ["cooccur_days_per_year"])
        a2 = dec("axis2_by_psu_decade", ["n_before", "n_after"])
        a3 = dec("axis3_by_psu_decade", ["pct_drought_psu", "n_hw_spei"])
    df = (PSU_GEO.merge(el, on="psu_idx", how="left")
          .merge(a1, on="psu_idx", how="left").merge(a2, on="psu_idx", how="left")
          .merge(a3, on="psu_idx", how="left"))
    nb = df["n_before"].to_numpy(float); na = df["n_after"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["count_ratio_fig6"] = np.where(nb > 0, na / np.where(nb > 0, nb, 1.0),
                                          np.where(na > 0, np.inf, np.nan))
    df["support_axis2"] = nb + na
    df["support_axis3"] = df["n_hw_spei"].to_numpy(float)
    df["fig6_eligible"] = (df["axis12_eligible"].fillna(False)
                           & df["spei_eligible"].fillna(False)).astype(bool)
    return df


def regime_classifiable(df, min2, min3):
    """Classifiable on the three axes: eligible, no missing metric, supports
    >= the configured minimums. Everything else = 'Insufficient data'."""
    missing = df[list(FIG6_COLS.values())].isna().any(axis=1).to_numpy()
    ok2 = (df["support_axis2"] >= min2).to_numpy()            # NaN -> False
    ok3 = (df["support_axis3"] >= min3).to_numpy()
    return df["fig6_eligible"].to_numpy() & ~missing & ok2 & ok3


def order_stat_threshold(values, q):
    """x_(ceil(q n)) of the n sorted values (1-indexed); NaN if n = 0.
    (A 1e-9 guard keeps e.g. (2/3)*n from rounding up past an integer.)"""
    v = np.sort(np.asarray(values, float)); n = v.size
    if n == 0:
        return np.nan
    k = min(max(int(np.ceil(q * n - 1e-9)), 1), n)
    return float(v[k - 1])


def classify_regimes(df, thresholds, classifiable):
    """ONE relative rule for the three axes: prominent <=> value STRICTLY above
    the pooled full-period threshold of that axis (also applied to decades).
    Non-classifiable PSU -> 'Insufficient data'."""
    flags = {a: (df[c].to_numpy(float) > thresholds[a]) for a, c in FIG6_COLS.items()}
    lab = [regime_label(*t) for t in zip(flags["axis1"], flags["axis2"], flags["axis3"])]
    lab = np.where(classifiable, np.array(lab, dtype=object), INSUFFICIENT)
    return pd.Series(lab, index=df.index)


def regime_support_diagnostics(inputs):
    """Support diagnostics only (no thresholds, no classes): what is needed to
    fix cfg.regime_min_support_axis2/3 before looking at the regimes."""
    rows = []
    for period, df in inputs.items():
        el = df["fig6_eligible"].to_numpy()
        for axis, sup in (("axis2", "support_axis2"), ("axis3", "support_axis3")):
            s = df.loc[el, sup]
            r = dict(period=period, axis=axis, n_psu=len(df), n_eligible=int(el.sum()),
                     n_support_nan=int(s.isna().sum()), n_support_zero=int((s == 0).sum()),
                     support_p25=s.quantile(.25), support_median=s.median(),
                     support_p75=s.quantile(.75))
            for k in SUPPORT_LEVELS:
                r[f"n_support_ge_{k}"] = int((s >= k).sum())
                r[f"share_support_ge_{k}"] = float((s >= k).mean()) if len(s) else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def regime_full_diagnostics(inputs, thresholds, min2, min3):
    rows = []
    for period, df in inputs.items():
        ok = regime_classifiable(df, min2, min3)
        el = df["fig6_eligible"].to_numpy()
        for axis, col in FIG6_COLS.items():
            v = df[col].to_numpy(float); t = thresholds[axis]
            vc = v[ok]
            r = dict(period=period, axis=axis, threshold_pooled=t, n_psu=len(df),
                     n_eligible=int(el.sum()), n_classifiable=int(ok.sum()),
                     share_insufficient=float(1 - ok.mean()) if len(df) else np.nan,
                     n_nan_eligible=int(np.isnan(v[el]).sum()),
                     n_zero=int((vc == 0).sum()), n_inf=int(np.isinf(vc).sum()),
                     n_ties_at_threshold=int((vc == t).sum()),
                     n_above=int((vc > t).sum()),
                     share_above=float((vc > t).mean()) if vc.size else np.nan)
            if axis != "axis1":
                s = df.loc[el, f"support_{axis}"]
                for k in SUPPORT_LEVELS:
                    r[f"n_support_ge_{k}"] = int((s >= k).sum())
            rows.append(r)
    return pd.DataFrame(rows)


def _regime_map(ax, df, title):
    for r in [INSUFFICIENT] + REGIME_ORDER[::-1]:
        s = df[df["regime"] == r]
        if len(s):
            map_scatter(ax, s, c=REGIME_COLORS[r], s=0.8)
    ax.set_title(title, fontsize=9, fontweight="bold")


def fig06_regimes(D, FIG, q, tag, min_support=None):
    """Fig 6. With cfg.regime_min_support_axis2/3 = None: support diagnostics
    ONLY (fix the minimums from them before looking at thresholds/classes).
    `min_support=(min2, min3)` is a PROVISIONAL override to test the figure:
    every output is then tagged '_PROVISIONAL'."""
    assert cfg.regime_threshold_reference == "pooled", \
        "Fig 6 uses pooled full-period thresholds for every decade"
    FIG.mkdir(parents=True, exist_ok=True)
    periods = ["full"] + DECADES
    inputs = {p: regime_inputs(D, p) for p in periods}
    regime_support_diagnostics(inputs).round(4).to_csv(
        FIG / "fig06_support_diagnostics.csv", index=False)
    if min_support is not None:
        min2, min3 = min_support
        tag = f"{tag}_PROVISIONAL"
        log.warning("Fig 6: PROVISIONAL minimum supports %s (figure test only).", min_support)
    else:
        min2, min3 = cfg.regime_min_support_axis2, cfg.regime_min_support_axis3
    if min2 is None or min3 is None:
        log.warning("Fig 6: minimum supports not set -> support diagnostics only "
                    "(%s). Set cfg.regime_min_support_axis2/3, then rerun.",
                    FIG / "fig06_support_diagnostics.csv")
        return None

    full = inputs["full"]
    ok_full = regime_classifiable(full, min2, min3)
    thr = {a: order_stat_threshold(full.loc[ok_full, c], q) for a, c in FIG6_COLS.items()}
    regime_full_diagnostics(inputs, thr, min2, min3).round(6).to_csv(
        FIG / f"fig06_diagnostics{tag}.csv", index=False)
    pd.Series(thr).rename("threshold").to_csv(FIG / f"fig06_regime_thresholds{tag}.csv")
    bad = {a: t for a, t in thr.items() if not np.isfinite(t)}
    if bad:
        raise RuntimeError(f"Fig 6: non-finite pooled threshold(s) {bad} (q={q}); "
                           f"diagnostics written to fig06_diagnostics{tag}.csv")

    by = {}
    for p, df in inputs.items():
        ok = regime_classifiable(df, min2, min3)
        df = df.copy()
        df["regime"] = classify_regimes(df, thr, ok)
        df["classifiable"] = ok
        by[p] = df
    pooled = by["full"]
    shares = pd.DataFrame({p: by[p].loc[by[p]["classifiable"], "regime"]
                           .value_counts(normalize=True).reindex(REGIME_ORDER).fillna(0) * 100
                           for p in DECADES})
    counts = pd.DataFrame({p: by[p]["regime"].value_counts()
                           .reindex(REGIME_ORDER + [INSUFFICIENT]).fillna(0).astype(int)
                           for p in periods})
    n_cls = {p: int(by[p]["classifiable"].sum()) for p in periods}
    pct_insuf = {p: 100 * (1 - by[p]["classifiable"].mean()) for p in periods}
    pct_inel = {p: 100 * (1 - by[p]["fig6_eligible"].mean()) for p in periods}
    f_, l_ = by[FIRST_DEC].set_index("psu_idx"), by[LAST_DEC].set_index("psu_idx")
    both = f_.index[f_["classifiable"]].intersection(l_.index[l_["classifiable"]])
    trans = pd.crosstab(f_.loc[both, "regime"], l_.loc[both, "regime"]).reindex(
        index=REGIME_ORDER, columns=REGIME_ORDER).fillna(0).astype(int)
    # persistence: same regime in both decades, vs. expected if the two decades were
    # independent with the same marginal shares (sum_i r_i * c_i)
    _T = trans.to_numpy(float); _N = max(1.0, _T.sum())
    same_pct = 100 * np.trace(_T) / _N
    chance_pct = 100 * float((_T.sum(1) / _N) @ (_T.sum(0) / _N))
    log.info("Fig 6%s: %d PSU classifiable in %s and %s; same regime %.1f%% (%.1f%% expected "
             "by chance)", tag, len(both), FIRST_DEC, LAST_DEC, same_pct, chance_pct)

    fig = plt.figure(figsize=(15, 12.5), facecolor="white")
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], hspace=0.34, wspace=0.3,
                  top=0.86, bottom=0.08, left=0.13, right=0.90)
    # (a) counts and shares, pooled (shares among classifiable PSU)
    ax = fig.add_subplot(gs[0, 0])
    cnt = counts["full"].reindex(REGIME_ORDER)
    y = np.arange(len(REGIME_ORDER))[::-1]
    ax.barh(y, cnt.to_numpy(), color=[REGIME_COLORS[r] for r in REGIME_ORDER], height=.65)
    for yi, c in zip(y, cnt.to_numpy()):
        ax.text(c + max(1, cnt.max()) * .01, yi, f"{100 * c / max(1, n_cls['full']):.1f}%",
                va="center", fontsize=7, color="#555")
    ax.set_yticks(y); ax.set_yticklabels(REGIME_ORDER, fontsize=7); ax.set_xlabel("Number of PSU")
    ax.set_xlim(0, max(1, cnt.max()) * 1.2)
    ax.set_title(f"Regimes, {YEARS[0]}–{YEARS[-1]}\n(n = {n_cls['full']:,} classifiable; "
                 f"{pct_insuf['full']:.1f}% insufficient data,\nincl. {pct_inel['full']:.1f}% "
                 f"ineligible PSU)", fontsize=9, fontweight="bold")
    panel_label(ax, "a", x=-0.55)
    # (b) pooled map
    ax = map_ax(fig, gs[0, 1]); _regime_map(ax, pooled, f"Regimes, {YEARS[0]}–{YEARS[-1]}"); panel_label(ax, "b", x=0)
    ax.legend(handles=[mpatches.Patch(color=REGIME_COLORS[r], label=r) for r in REGIME_ORDER + [INSUFFICIENT]],
              loc="lower left", fontsize=6, frameon=True, framealpha=.9)
    # (c) shares by decade among classifiable PSU (stacked), n and insufficient share on top
    ax = fig.add_subplot(gs[0, 2])
    bottom = np.zeros(len(DECADES))
    for r in REGIME_ORDER:
        v = shares.loc[r].to_numpy()
        ax.bar(range(len(DECADES)), v, bottom=bottom, color=REGIME_COLORS[r], width=.7, label=r)
        bottom += v
    for j, d in enumerate(DECADES):
        ax.text(j, 101, f"n={n_cls[d]:,}\ninsuff. {pct_insuf[d]:.1f}%", ha="center",
                va="bottom", fontsize=6, color="#555")
    ax.set_xticks(range(len(DECADES))); ax.set_xticklabels([d.replace("-", "–") for d in DECADES])
    ax.set_ylabel("% of classifiable PSU"); ax.set_ylim(0, 112)
    ax.set_title("Regime shares by decade (classifiable PSU)", fontsize=9, fontweight="bold"); panel_label(ax, "c")
    # (d, e) first vs last decade maps
    for k, d in enumerate((FIRST_DEC, LAST_DEC)):
        ax = map_ax(fig, gs[1, k]); _regime_map(ax, by[d], d.replace("-", "–")); panel_label(ax, "de"[k], x=0)
    # (f) transition matrix first -> last, PSU classifiable in both decades
    ax = fig.add_subplot(gs[1, 2])
    P = trans.to_numpy(float); P = 100 * P / max(1, P.sum())
    im = ax.imshow(P, cmap="Blues", vmin=0, vmax=max(1, P.max()))
    ax.set_xticks(range(len(REGIME_ORDER))); ax.set_xticklabels(REGIME_ORDER, rotation=60, ha="right", fontsize=6)
    ax.set_yticks(range(len(REGIME_ORDER))); ax.set_yticklabels(REGIME_ORDER, fontsize=6)
    ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
    ax.set_xlabel(f"Regime in {LAST_DEC.replace('-', '–')}"); ax.set_ylabel(f"Regime in {FIRST_DEC.replace('-', '–')}")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] >= 0.5:
                ax.text(j, i, f"{P[i, j]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if P[i, j] > .6 * P.max() else "#333")
    for i in range(P.shape[0]):                       # outline the diagonal (same regime)
        ax.add_patch(mpatches.Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="#333", lw=.6))
    ax.set_title(f"Transitions {FIRST_DEC.replace('-', '–')} → {LAST_DEC.replace('-', '–')}, "
                 f"% of n = {len(both):,} PSU\nclassifiable in both decades; same regime "
                 f"{same_pct:.1f}% (chance {chance_pct:.1f}%)", fontsize=9, fontweight="bold")
    panel_label(ax, "f", x=-0.08, y=1.12)
    prov = "  [PROVISIONAL minimum supports — test only]" if min_support is not None else ""
    fig.suptitle(f"Compound-event regimes — prominent = strictly above the order statistic "
                 f"$x_{{(\\lceil {q:.2f}\\,n \\rceil)}}$ of the classifiable PSU, {YEARS[0]}–{YEARS[-1]}\n"
                 f"pooled thresholds applied to every decade · minimum support: Axis II = {min2} EPE, "
                 f"Axis III = {min3} HW{prov}", fontsize=10, fontweight="bold", y=.975)
    savefig(fig, FIG, f"fig06_regimes{tag}")
    shares.round(2).to_csv(FIG / f"fig06_regime_shares_by_decade{tag}.csv")
    counts.T.assign(n_classifiable=pd.Series(n_cls),
                    pct_insufficient=pd.Series(pct_insuf)).round(2).to_csv(
        FIG / f"fig06_regime_counts_by_period{tag}.csv")
    trans.to_csv(FIG / f"fig06_regime_transitions{tag}.csv")
    pd.DataFrame([dict(n_both=len(both), same_regime_pct=same_pct, chance_pct=chance_pct,
                       pct_insufficient_full=pct_insuf["full"], pct_ineligible_full=pct_inel["full"])]
                 ).round(2).to_csv(FIG / f"fig06_transition_summary{tag}.csv", index=False)
    return shares


for _mode, _D in MODE_DIRS.items():
    fig06_regimes(_D, FIG_DIRS[_mode], cfg.regime_quantile, "")                        # main (q75)
    fig06_regimes(_D, FIG_DIRS[_mode], cfg.regime_quantile_sens, "_sens_tercile")      # sensitivity
