# ---- Supplementary Tables 1-4 (annual mode) -> one Excel file + one CSV per table ----
# New cell, to paste at the END of compound_pipeline_v3.ipynb; run it after cells 2 and 4.
# Reads only files already written by the pipeline (no recomputation).
#   Table 1  LMF by region                 <- annual/significance/lmf_by_scale.parquet
#   Table 2  Country statistics + trends   <- annual/figures/tableS01_country_statistics.csv
#                                             + annual/trends/all_trends_summary.parquet
#   Table 3  Sensitivity values (a, b, c)  <- annual/figures/figS_axis2_boundary_buffer_table.csv,
#                                             figS_sensitivity_spei_threshold_table.csv,
#                                             figS_sensitivity_window_table.csv
#   Table 4  DHS surveys used              <- processed_3/psu.parquet
# Output: processed_3/supplementary/Supplementary_Tables.xlsx (one sheet per table) and CSVs.
from pathlib import Path
import pandas as pd

NICE_COUNTRY = {"Congo Democratic Republic": "Democratic Republic of the Congo",
                "Cote d'Ivoire": "Côte d'Ivoire"}
REGION_ORDER = ["Western", "Central", "Eastern", "Southern"]


def _ci(est, lo, hi, nd):
    if pd.isna(est):
        return ""
    if pd.isna(lo) or pd.isna(hi):
        return f"{est:.{nd}f}"
    return f"{est:.{nd}f} ({lo:.{nd}f} to {hi:.{nd}f})"


def supp_table1_lmf(sig_dir):
    # CI from the cluster bootstrap over ERA5 cells (the one reported in the text);
    # the PSU-level rows only provide the number of PSUs.
    l = pd.read_parquet(Path(sig_dir) / "lmf_by_scale.parquet")
    l = l[l["scale"].isin(["Continental"] + REGION_ORDER)]
    cell = l[l["unit"] == "era5_cell"].set_index("scale")
    n_psu = l[l["unit"] == "psu_idx"].set_index("scale")["n_units"]
    order = [s for s in ["Continental"] + REGION_ORDER if s in cell.index]
    cell = cell.loc[order]
    return pd.DataFrame({
        "Scale": order,
        "Number of PSUs": n_psu.reindex(order).astype("Int64").to_numpy(),
        "Number of ERA5 cells": cell["n_units"].astype(int).to_numpy(),
        "Likelihood multiplication factor (95% CI)": [
            _ci(e, a, b, 2) for e, a, b in zip(cell["lmf_seasonal"], cell["lmf_seasonal_lo"], cell["lmf_seasonal_hi"])],
    })


def supp_table2_countries(fig_dir, trends_path):
    c = pd.read_csv(Path(fig_dir) / "tableS01_country_statistics.csv")
    tr = pd.read_parquet(trends_path)
    tr = tr[tr["scope"] == "country"]

    def trend(metric, nd):
        s = tr[tr["metric"] == metric].set_index("scale")
        return {k: _ci(r.slope_per_decade, r.ci_low_per_decade, r.ci_high_per_decade, nd)
                for k, r in s.iterrows()}
    t1, t2, t3 = trend("cooccur_days_per_unit", 4), trend("count_ratio_a", 2), trend("pct_drought_b", 2)
    c["_r"] = c["Region"].map({r: i for i, r in enumerate(REGION_ORDER)})
    c = c.sort_values(["_r", "Country"])
    return pd.DataFrame({
        "Region": c["Region"],
        "Country": c["Country"].replace(NICE_COUNTRY),
        "Number of PSUs": c["N_PSU"].astype(int),
        "Co-occurrence days per PSU per year": c["CCE_days_per_PSU_yr"].round(4),
        "EPEs before heatwaves": c["N_EPE_before"].astype(int),
        "EPEs after heatwaves": c["N_EPE_after"].astype(int),
        "After-to-before ratio": c["EPE_ratio"].round(2),
        "Heatwaves with SPEI-3": c["N_HW_with_SPEI"].astype(int),
        "Heatwaves preceded by drought (%)": c["Pct_HW_drought"].round(1),
        "Trend, co-occurrence days per PSU per year, per decade (95% CI)": c["Country"].map(t1),
        "Trend, after-to-before ratio, per decade (95% CI)": c["Country"].map(t2),
        "Trend, heatwaves preceded by drought, percentage points per decade (95% CI)": c["Country"].map(t3),
    })


def supp_table3_sensitivity(fig_dir):
    fig_dir = Path(fig_dir)
    b = pd.read_csv(fig_dir / "figS_axis2_boundary_buffer_table.csv")
    a = pd.DataFrame({
        "EPEs excluded within (days of the heatwave boundaries)": b["buffer_days"].astype(int),
        "EPEs before heatwaves": b["n_before"].astype(int),
        "EPEs after heatwaves": b["n_after"].astype(int),
        "After-to-before ratio": b["count_ratio"].round(2),
    })
    s = pd.read_csv(fig_dir / "figS_sensitivity_spei_threshold_table.csv")
    bb = pd.DataFrame({
        "Drought threshold (SPEI-3 below)": s["threshold"],
        "Heatwaves preceded by drought (%)": s["pct_hw"].round(1),
        "Ratio of mean durations, drought / no drought": s["duration_ratio"].round(3),
        "Trend of the share, percentage points per decade (95% CI)": [
            _ci(e, lo, hi, 2) for e, lo, hi in zip(s["trend_pct_per_decade"], s["trend_ci_lo"], s["trend_ci_hi"])],
    })
    w = pd.read_csv(fig_dir / "figS_sensitivity_window_table.csv")
    w["_s"] = w["scale"].map({x: i for i, x in enumerate(["Continental"] + REGION_ORDER)})
    w = w.sort_values(["window_days", "_s"])
    cc = pd.DataFrame({
        "Window (± days)": w["window_days"].astype(int),
        "Scale": w["scale"],
        "EPEs before heatwaves": w["n_before"].astype(int),
        "EPEs after heatwaves": w["n_after"].astype(int),
        "After-to-before ratio": w["count_ratio"].round(2),
    })
    return a, bb, cc


def supp_table4_dhs(psu_path):
    p = pd.read_parquet(psu_path, columns=["country", "region", "survey_id", "survey_year"])
    g = (p.groupby(["region", "country", "survey_id", "survey_year"], dropna=False)
           .size().rename("Number of PSUs retained").reset_index())
    g["_r"] = g["region"].map({r: i for i, r in enumerate(REGION_ORDER)})
    g = g.sort_values(["_r", "country", "survey_year"]).drop(columns="_r")
    return g.rename(columns={"region": "Region", "country": "Country", "survey_id": "DHS survey",
                             "survey_year": "Survey year"}).replace({"Country": NICE_COUNTRY})


def make_supplementary_tables(D, FIGD, psu_path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sheets = {"Table 1": supp_table1_lmf(Path(D) / "significance"),
              "Table 2": supp_table2_countries(FIGD, Path(D) / "trends" / "all_trends_summary.parquet")}
    t3a, t3b, t3c = supp_table3_sensitivity(FIGD)
    sheets.update({"Table 3a": t3a, "Table 3b": t3b, "Table 3c": t3c})
    if Path(psu_path).exists():
        sheets["Table 4"] = supp_table4_dhs(psu_path)
    else:
        print(f"psu.parquet not found ({psu_path}): Table 4 skipped")
    for name, df in sheets.items():
        df.to_csv(out_dir / f"Supplementary_{name.replace(' ', '_')}.csv", index=False, encoding="utf-8-sig")
    xl = out_dir / "Supplementary_Tables.xlsx"
    with pd.ExcelWriter(xl) as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            for j, col in enumerate(df.columns):          # readable column widths
                width = max(len(str(col)), *(len(str(v)) for v in df[col].head(200))) + 2
                ws.column_dimensions[ws.cell(1, j + 1).column_letter].width = min(width, 45)
    print("Supplementary tables ->", xl)
    return sheets


if "MODE_DIRS" in globals():
    make_supplementary_tables(MODE_DIRS["annual"], FIG_DIRS["annual"], PROCESSED / "psu.parquet",
                              PROCESSED / "supplementary")
