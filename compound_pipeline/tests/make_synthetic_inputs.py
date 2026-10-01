"""Build ARTIFICIAL inputs (RData + ERA5 + CHIRPS NetCDF) for an end-to-end
smoke run of the pipeline. Nothing here resembles the real data beyond the
file layout (variable / dimension names, units, latitude orientation).

Built-in traps:
  * PSU on an all-NaN CHIRPS cell        -> chirps_all_nan (EPE & SPEI ineligible)
  * PSU 7 deg away from any ERA5 column  -> era5_too_far (temp & SPEI ineligible)
  * very dry PSU (< 100 wet days)        -> p95_reliable False, SPEI eligible
  * 20 missing precip days in Mar 1990   -> SPEI/SPI 'missing input'
  * Septembers 1985-2014 missing but one -> 'insufficient calibration'
  * a few missing Tmax days              -> excluded from the LMF counts
usage: python make_synthetic_inputs.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr
import xarray as xr

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2024)

lat_vals = [-28.0, -20.0, -10.0, 0.0, 8.0, 15.0]
lon_vals = [-10.0, 0.0, 10.0, 20.0, 30.0, 40.0]
countries = {(-28.0,): "Lesotho", (-20.0,): "Namibia", (-10.0,): "Angola",
             (0.0,): "Kenya", (8.0,): "Nigeria", (15.0,): "Ethiopia"}
rows = []
for la in lat_vals:
    for lo in lon_vals:
        c = countries[(la,)]
        if c == "Nigeria" and lo >= 20: c = "Ghana"
        if c == "Kenya" and lo <= 0: c = "Cameroon"
        rows.append((c, la + rng.uniform(-.05, .05), lo + rng.uniform(-.05, .05)))
rows.append(("Kenya", 0.02, 47.0))            # off the ERA5 grid (7 deg from lon 40)
rows.append(("Kenya", 0.03, 47.02))           # duplicate locality nearby (kept)
rows.append(("Ethiopia", 15.01, 40.01))       # second PSU on the very dry cell
rows.append(("Ghana", 8.01, 30.01))
psu = pd.DataFrame(rows, columns=["CountryName", "LATNUM", "LONGNUM"])
psu["SurveyId"] = [f"{c[:2].upper()}2015DHS" for c in psu.CountryName]
psu["SurveyYear"] = "2015"
psu["PSU"] = np.arange(1, len(psu) + 1, dtype=float)
psu = psu[["CountryName", "SurveyId", "SurveyYear", "PSU", "LATNUM", "LONGNUM"]]
pyreadr.write_rdata(str(out / "PSU_geocoordinates.RData"), psu, df_name="psu")

days = pd.date_range("1985-01-01", "2024-12-31")
T = len(days)
doy = days.dayofyear.to_numpy()
yrs = (days.year.to_numpy() - 1985) / 40.0

# ---- ERA5: tmax/tmin in Kelvin, latitude DESCENDING, valid_time ------------
e_lat = np.array(lat_vals[::-1]); e_lon = np.array(lon_vals)
tmax = np.empty((T, len(e_lat), len(e_lon)), np.float32)
for i, la in enumerate(e_lat):
    for j, lo in enumerate(e_lon):
        noise = np.zeros(T); eps = rng.normal(0, 1.6, T)
        for t in range(1, T):
            noise[t] = 0.75 * noise[t - 1] + eps[t]
        season = 4 * np.sin(2 * np.pi * (doy - 30 - 10 * i) / 365.25)
        tmax[:, i, j] = 273.15 + 30 + season + 1.5 * yrs + noise
tmax[3000:3004, 1, 2] = np.nan                  # a few missing Tmax days
era5 = xr.Dataset({"tmax": (("valid_time", "latitude", "longitude"), tmax),
                   "tmin": (("valid_time", "latitude", "longitude"), tmax - 11.0)},
                  coords={"valid_time": days + pd.Timedelta(hours=0), "latitude": e_lat,
                          "longitude": e_lon, "number": 0})
era5.to_netcdf(out / "ERA5_SubSahara_Tmax_Tmin_Daily_1985_2024.nc")

# ---- CHIRPS: precip mm/day, latitude ascending, longer time axis -----------
c_days = pd.date_range("1984-11-01", "2025-01-31")
cdoy = c_days.dayofyear.to_numpy()
c_lat = np.array(lat_vals); c_lon = np.array(lon_vals + [47.0])
pr = np.empty((len(c_days), len(c_lat), len(c_lon)), np.float32)
for i, la in enumerate(c_lat):
    for j, lo in enumerate(c_lon):
        p_wet = np.clip(0.25 + 0.2 * np.sin(2 * np.pi * (cdoy - 60 * (i % 3)) / 365.25), 0.03, 1)
        wet = rng.random(len(c_days)) < p_wet
        pr[:, i, j] = np.where(wet, rng.gamma(0.8, 9.0, len(c_days)), 0.0)
pr[:, 0, 0] = np.nan                            # ocean / no-data cell (lat -28, lon -10)
dry = rng.random(len(c_days)) < 0.004           # very dry cell (lat 15, lon 40)
pr[:, 5, 5] = np.where(dry, rng.gamma(0.8, 9.0, len(c_days)), 0.0)
m90 = (c_days >= "1990-03-05") & (c_days <= "1990-03-24")
pr[m90, 4, 2] = np.nan                          # lat 8, lon 10: 20 missing days
sep = (c_days.month == 9) & (c_days.year <= 2014) & (c_days.year != 2000)
pr[sep, 3, 4] = np.nan                          # lat 0, lon 30: Septembers missing
chirps = xr.Dataset({"precip": (("time", "latitude", "longitude"), pr)},
                    coords={"time": c_days, "latitude": c_lat, "longitude": c_lon})
chirps.to_netcdf(out / "chirps_v2_1981_2024.nc")
print("synthetic inputs written to", out, "| PSU rows:", len(psu))
