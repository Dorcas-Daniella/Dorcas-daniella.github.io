# Methods (draft v1)

*Values in [square brackets] must be completed or checked by the authors. All counts and parameters refer to the final run (`run_manifest.json`, `run_status = completed`). Reference list at the end to be verified against the original sources.*

---

## 1. Study population and climate data

**Survey clusters.** We used the GPS coordinates of the primary sampling units (PSU, survey clusters) of the Demographic and Health Surveys (DHS) conducted in 34 sub-Saharan African countries [list of surveys and years: Table Sx]. Records with missing coordinates, coordinates at (0°, 0°) or outside 35°S–25°N and 18°W–52°E were removed. PSU sharing exactly the same coordinates across survey rounds were retained once (earliest survey). The final sample comprises **54,615 PSU**. Countries were grouped into the four United Nations M49 sub-regions: Western, Eastern, Middle (hereafter Central) and Southern Africa (Table S1). DHS coordinates are randomly displaced (up to 2 km in urban and 5 km in rural clusters, 10 km for 1% of rural clusters) [ref. DHS GPS guidelines]; this displacement is small compared with the ERA5 grid (0.25°) but may shift the matched CHIRPS cell (0.05°). Each PSU is treated as a sampled locality; no demographic weighting is applied.

**Climate data.** Daily maximum and minimum 2-m air temperature (Tmax, Tmin) were taken from the ERA5 reanalysis (0.25°) [ref. Hersbach et al. 2020] and daily precipitation from CHIRPS v2.0 (0.05°) [ref. Funk et al. 2015], for 1 January 1985 – 31 December 2024. Each PSU was assigned the nearest grid cell of each product; a match was rejected when the cell centre lay more than 20 km away (about the half-diagonal of a 0.25° cell at the equator) or when the cell contained no valid data. ERA5 daily values follow UTC days, whereas CHIRPS days follow the reporting convention of its gauge inputs; a possible ±1-day offset between the two products could not be quantified and is acknowledged as a limitation.

**Eligibility.** All 54,615 PSU had a valid ERA5 cell. A valid CHIRPS cell was available for 53,802 PSU. The extreme-precipitation threshold (Section 2) additionally required at least 100 wet days in the reference period, which retained 53,782 PSU. Analyses involving precipitation extremes (Axes I and II, the likelihood multiplication factor and the null model) use these 53,782 PSU; analyses of drought (Axis III) use the 53,802 PSU with both ERA5 and CHIRPS data, restricted to heatwaves for which the drought index is available. The same population enters numerators, denominators, tables and figures of each axis. The 54,615 PSU fall into 10,713 distinct ERA5 cells (5.1 PSU per cell on average); because PSU in the same ERA5 cell share identical temperature series, this pseudo-replication is accounted for in all uncertainty estimates (Sections 3–4).

## 2. Hazard definitions

**Reference period.** All percentile thresholds and the calibration of the drought indices use 1985–2014. Thresholds estimated in this period are applied to all years without in-base correction; exceedance frequencies inside the reference period are therefore slightly biased low relative to 2015–2024 [ref. Zhang et al. 2005].

**Heatwaves (HW).** A heatwave is a run of at least three consecutive days with Tmax strictly above the 90th percentile threshold. In the primary (*calendar*) mode, the threshold is computed for each calendar day from the reference-period Tmax values within a 31-day window centred on that day [ref. Russo et al. 2014], on a fixed 366-position calendar (29 February occupies its own position and contributes data only in leap years). The threshold therefore follows the seasonal cycle, and heatwaves are anomalies relative to the time of year. As a sensitivity analysis (*annual* mode), a single 90th percentile of all reference-period days is used for each PSU, so that heatwaves concentrate in the hot season. For each heatwave we recorded the onset and end dates, the duration and the peak Tmax. Over all PSU, 9,608,094 heatwaves were detected in calendar mode and 10,743,161 in annual mode.

**Extreme precipitation events (EPE).** An EPE day is a day with precipitation strictly greater than the PSU-specific 95th percentile of wet-day (≥ 1 mm) precipitation in the reference period. In total, 12,787,839 EPE days were identified.

**Drought.** Meteorological drought was characterised with the 3-month Standardized Precipitation–Evapotranspiration Index (SPEI-3) [ref. Vicente-Serrano et al. 2010], computed from the monthly climatic water balance (precipitation minus potential evapotranspiration, PET). PET was estimated with the Hargreaves–Samani equation [ref. Hargreaves & Samani 1985] from Tmax, Tmin and extraterrestrial radiation [ref. Allen et al. 1998], the only formulation compatible with the available variables. A month was used only if at least 90% of its days had valid data. The log-logistic distribution was fitted by maximum likelihood for each calendar month on 1985–2014 with xclim [version 0.61.1; ref. Bourgault et al. 2023]. Values for 2015–2024 are therefore extrapolations outside the calibration period. The 3-month Standardized Precipitation Index (SPI-3) [ref. McKee et al. 1993], fitted with a gamma distribution and a separate probability for zero totals, was computed in the same way as a precipitation-only alternative. Drought is defined as an index value below −1. The first index value is March 1985. Every missing value was assigned to a documented cause (ineligible PSU, spin-up months, missing input, insufficient calibration data); unexplained missing values stopped the computation.

## 3. Three axes of compound hazards

### Axis I — Same-day co-occurrence of heat and extreme precipitation

A co-occurrence day is a day that is both a heatwave day and an EPE day at the same PSU. We report the number of co-occurrence days per PSU per year (averaged over all eligible PSU, zeros included) and the percentage of heatwaves containing at least one co-occurrence day.

To test whether heat and extreme rainfall co-occur more or less often than expected from their individual frequencies, we computed the likelihood multiplication factor (LMF) [ref. Zscheischler & Seneviratne 2017]:

LMF = N_joint / E[N_joint],

where N_joint is the observed number of co-occurrence days. In the *naive* version, E[N_joint] = N_hot · N_EPE / N_valid, with all days pooled. Because both extremes have marked seasonal cycles, our primary *seasonal* version conditions the expectation on the day of the year d:

E_seasonal[N_joint] = Σ_d N_hot(d) · N_EPE(d) / N_valid(d).

All four counts are computed on the same set of days (both Tmax and precipitation available). LMF < 1 indicates that the two extremes co-occur less often than expected under independence. The LMF is pooled over PSU (ratio of summed observed to summed expected counts), and its 95% confidence interval is obtained from 1,000 bootstrap resamples of ERA5 cells (cluster bootstrap), so that PSU sharing a temperature series are resampled together; a PSU-level bootstrap is reported for comparison. The day-of-year conditioning pools all years and therefore does not account for interannual non-stationarity.

### Axis II — Timing of extreme precipitation around heatwaves

For each heatwave we considered a window of W = 30 days before onset and W = 30 days after the end. Heatwaves whose window extended beyond the study period were excluded (87,380 heatwaves in calendar mode among eligible PSU), leaving 9,361,759 heatwaves (10,501,242 in annual mode).

*Primary metric.* Each EPE located within the window of at least one heatwave of the same PSU was attributed to a single heatwave, the nearest one (distance to onset for EPE before it, to its end for EPE after it; ties assigned to the earlier heatwave). EPE falling during a heatwave belong to Axis I. The primary metric is the after/before count ratio, R = N_after / N_before, computed as a ratio of sums over all heatwaves of a group (continent, region, country, decade). R > 1 means that extreme rainfall is more frequent after heatwaves than before them.

*Secondary metrics.* Following event coincidence analysis [ref. Donges et al. 2016], we also computed, for each heatwave, a precursor indicator (at least one EPE in the 30 days before onset) and a trigger indicator (at least one EPE in the 30 days after the end). Their PSU-averaged difference (trigger − precursor) is reported as a secondary measure. These indicators are evaluated for each heatwave separately, so an EPE may inform several heatwaves; the primary count ratio avoids this by unique attribution.

*Null model.* Because rainfall and heat both follow strong seasonal cycles, an after/before asymmetry can arise simply from the timing of heatwaves relative to the rainy season. We therefore compared R with a seasonally constrained null distribution. In each of 1,000 permutations, every heatwave kept its PSU and its duration, and its onset (month and day) was moved to another year, randomly shifted by up to ±15 days (29 February mapped to 28 February in non-leap years). A displacement was admissible only if the full ±30-day window stayed within the study period, the new onset fell in a different calendar year from the observed one, and the window did not overlap the observed heatwave. The target year was drawn uniformly among the years with at least one admissible shift, then the shift uniformly among the admissible values. PSU in the same ERA5 cell have identical heatwaves; one draw was therefore made per distinct heatwave (ERA5 cell, onset, end), giving 1,830,045 independent units in calendar mode (2,071,704 in annual mode), and copied to all PSU of that cell. Surrogate heatwaves were processed with exactly the same attribution procedure as the observed ones. The analysis is descriptive: we report the observed ratio, the median and 2.5–97.5 percentile envelope of the null, and their ratio. The null also stores the numbers of EPE before and after the surrogate heatwaves, which allows the asymmetry to be decomposed into a deficit of EPE before heatwaves (N_before,obs / median N_before,null < 1) and/or an excess after them (N_after,obs / median N_after,null > 1). Results are given for the continent, each region and each season of heatwave onset (DJF, MAM, JJA, SON). One-sided permutation p-values were computed but are not used for inference.

*Sensitivity analyses.* (i) Window length W = 7, 15, 45 and 60 days, with truncation re-evaluated for each W. (ii) Exclusion of EPE within 1, 2, 3 or 5 days of the heatwave boundaries, to assess the contribution of rainfall that terminates a heatwave (the null was not recomputed for these variants). At W = 30 days, the observed numbers of heatwaves, EPE before and after, and the event-coincidence rates were verified to be identical across the main analysis, the null model and the window sensitivity, for the continent and each region.

### Axis III — Heatwaves preceded by drought

For each heatwave we extracted SPEI-3 for the calendar month preceding the onset month (lag 1, primary). Using the preceding month avoids the circularity of the onset month, whose PET is inflated by the heatwave itself. The primary metric is the percentage of heatwaves preceded by drought (SPEI-3 < −1), among heatwaves for which the index is available (99.7% of heatwaves at the 53,802 PSU in calendar mode; heatwaves with onset in January–March 1985 have no antecedent value).

Sensitivity analyses used SPEI-3 in the onset month (lag 0), SPI-3 at lag 1 (precipitation only), and drought thresholds of −0.5, −1.5 and −2. We also compared the duration and peak Tmax of heatwaves with and without antecedent drought.

Because the drought frequency differs between places and seasons and, for SPEI, increases over time, the observed percentage was compared with its local expectation. For each heatwave, the expected probability of drought is the frequency of drought in the same PSU and the same calendar month over all years (place and season), or within the same decade (place, season and background trend). The ratio of observed to expected percentages, averaged over heatwaves, measures how much more often heatwaves are preceded by drought than the local conditions alone would imply. It was computed for SPEI-3 and SPI-3.

## 4. Aggregation, temporal trends and regimes

**Aggregation.** Metrics were first computed per PSU and then aggregated by country, region, the continent, year and decade (1985–1994, 1995–2004, 2005–2014, 2015–2024). Two types of summaries are distinguished: event-pooled values (ratio of sums for the count ratio; share of all heatwaves for percentages) and unweighted means of per-PSU values. Unless stated otherwise, levels are reported as event-pooled values. Temporal trends are computed on annual series: the yearly ratio of sums for the after/before count ratio, the yearly mean over PSU (zeros included) for co-occurrence days, and the yearly mean of per-PSU values for the event-coincidence difference and the drought percentages. Heatwaves and Axis II quantities are dated by the onset year of the heatwave, co-occurrence days by their own date.

**Trends.** Linear trends of the annual series (1985–2024) were estimated with the Theil–Sen slope [ref. Sen 1968], expressed per decade. Monotonic trends were tested with the Mann–Kendall test, using a conservative Hamed–Rao variance correction for autocorrelation [ref. Hamed & Rao 1998] (only significant autocorrelation lags enter the correction, which is applied only when it increases the variance). 95% confidence intervals of the slope were obtained from 1,000 moving-block bootstrap resamples with 5-year blocks [ref. Künsch 1989]. Country-level tests were corrected for multiple testing with the Benjamini–Hochberg false discovery rate (α = 0.05) [ref. Benjamini & Hochberg 1995], separately for each axis and metric. Series with an internal gap keep their slope, but no test or interval is computed. When the Mann–Kendall test and the bootstrap interval disagree, the trend is described as weak.

**Compound-hazard regimes.** To summarise the three axes jointly, each PSU was classified according to which axes are prominent at that location. The three metrics are co-occurrence days per year (Axis I), the after/before ratio (Axis II; set to +∞ when no EPE precedes but at least one follows the heatwaves) and the percentage of heatwaves preceded by drought (Axis III). An axis is prominent when its value is strictly above the 75th percentile (order statistic x₍⌈0.75n⌉₎) of the PSU classifiable on all three axes over 1985–2024 (top tercile as sensitivity). This yields eight regimes, from "none prominent" to "all three". A PSU was classifiable only if at least 10 EPE were attributed to its heatwaves (Axis II) and at least 10 of its heatwaves had an antecedent SPEI-3 value (Axis III). These minimum supports were fixed from the distribution of supports alone, before thresholds or classes were examined. Over 1985–2024 they retain 99.2% (Axis II) and 100% (Axis III) of eligible PSU in calendar mode. Decadal maps apply the same 1985–2024 thresholds to each decade; their changes therefore combine changes in hazard frequency and, for Axis II, the larger sampling variability of decadal ratios. Transitions between 1985–1994 and 2015–2024 are computed on PSU classifiable in both decades, and the proportion of PSU keeping the same regime is compared with the proportion expected by chance given the regime frequencies of each decade.

## 5. Threshold-mode sensitivity

The full analysis (all axes, null model, trends and regimes) was repeated with the annual heatwave threshold. The calendar mode is the primary analysis because it defines heatwaves as anomalies relative to the time of year, which makes the three axes less dependent on the seasonal cycle; the annual mode is reported in the Supplementary Information.

## 6. Software and reproducibility

All computations were performed in Python [version] with NumPy, pandas, xarray, SciPy and xclim (0.61.1). Random draws use a fixed seed (42). A run manifest records the configuration, package versions, the identity (size, modification time) of each input file, a fingerprint of the code, the eligibility exclusions, the effective fitting methods of SPEI and SPI, the number of usable permutations, and the outcome of the internal consistency checks; it is marked as completed only at the end of the run. The code is available at [repository / DOI].

---

## References to verify

- Allen, R. G. et al. (1998). *Crop evapotranspiration*. FAO Irrigation and Drainage Paper 56.
- Benjamini, Y. & Hochberg, Y. (1995). J. R. Stat. Soc. B 57, 289–300.
- Bourgault, P. et al. (2023). xclim: xarray-based climate data analytics. J. Open Source Softw. 8, 5415.
- Donges, J. F. et al. (2016). Event coincidence analysis for quantifying statistical interrelationships between event time series. Eur. Phys. J. Spec. Top. 225, 471–487.
- Funk, C. et al. (2015). The climate hazards infrared precipitation with stations. Sci. Data 2, 150066.
- Hamed, K. H. & Rao, A. R. (1998). A modified Mann–Kendall trend test for autocorrelated data. J. Hydrol. 204, 182–196.
- Hargreaves, G. H. & Samani, Z. A. (1985). Reference crop evapotranspiration from temperature. Appl. Eng. Agric. 1, 96–99.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol. Soc. 146, 1999–2049.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. Ann. Stat. 17, 1217–1241.
- McKee, T. B., Doesken, N. J. & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. 8th Conf. Applied Climatology.
- Russo, S. et al. (2014). Magnitude of extreme heat waves in present climate and their projection in a warming world. J. Geophys. Res. Atmos. 119, 12500–12512.
- Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. J. Am. Stat. Assoc. 63, 1379–1389.
- Vicente-Serrano, S. M., Beguería, S. & López-Moreno, J. I. (2010). A multiscalar drought index sensitive to global warming: the SPEI. J. Clim. 23, 1696–1718.
- Zhang, X. et al. (2005). Avoiding inhomogeneity in percentile-based indices of temperature extremes. J. Clim. 18, 1641–1651.
- Zscheischler, J. & Seneviratne, S. I. (2017). Dependence of drivers affects risks associated with compound events. Sci. Adv. 3, e1700263.
- [DHS GPS displacement documentation]
