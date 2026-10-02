# ---- checks: section 8 -------------------------------------------------------
spei = pd.read_parquet(PROCESSED / "spei3.parquet")
_el = psu_eligibility(cfg)
assert not spei.psu_idx.isin(set(_el.loc[~_el.spei_eligible, "psu_idx"])).any()
assert str(spei[f"is_drought_spei{cfg.spei_scale_months}"].dtype) == "boolean"
print("SPEI/SPI methods & NaN categories:",
      json.dumps(json.loads(MANIFEST.read_text())["spei_spi"], indent=1))
ref = spei[(spei.year >= cfg.ref_start_year) & (spei.year <= cfg.ref_end_year)]
for col in (f"spei{cfg.spei_scale_months}", f"spi{cfg.spei_scale_months}"):
    v = ref[col].dropna()
    print(f"{col}: ref mean {v.mean():+.3f} (exp ~0) | std {v.std():.3f} "
          f"(exp ~1) | NaN whole period {spei[col].isna().mean():.2%}")
    # diagnostic, not a stop criterion: a correct SPI with a large zero mass
    # does not have mean 0 / std 1 (see run_step05_spei); errors and
    # unexplained NaN already stop the run in step05.
    if abs(v.mean()) >= 0.15 or abs(v.std() - 1) >= 0.2:
        warnings.warn(f"{col}: pooled mean/std outside 0 +/- 0.15 / 1 +/- 0.2 — "
                      "diagnostic only; inspect the fits (zero mass for SPI).")
dr = spei.groupby("year")[f"is_drought_spei{cfg.spei_scale_months}"].mean()
print("drought-month fraction by year (head/tail):")
print(pd.concat([dr.head(3), dr.tail(3)]).round(3))
