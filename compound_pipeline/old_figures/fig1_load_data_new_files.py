# =============================================================================
# DATA — new pipeline files (processed_3), same columns as the old figure code
# =============================================================================
from pathlib import Path

PROCESSED_DIR = Path(BASE_DIR) / "Code_PhD" / "processed_3"   # dossier de sortie du pipeline v3
MODE = "calendar"                                                # "calendar" (principal) ou "annual"
AGG_DIR = PROCESSED_DIR / MODE / "aggregations"


def load_data():
    print(f"Loading data ({MODE})...")
    psu = pd.read_parquet(PROCESSED_DIR / "psu.parquet",
                          columns=["psu_idx", "lat", "lon", "country", "region"])
    a1 = pd.read_parquet(AGG_DIR / "axis1_by_psu.parquet")[["psu_idx", "n_cooccur_days"]]
    a2 = pd.read_parquet(AGG_DIR / "axis2_by_psu.parquet")[["psu_idx", "n_before", "n_after"]]
    a3 = pd.read_parquet(AGG_DIR / "axis3_by_psu.parquet")[["psu_idx", "pct_drought_psu"]]
    df = psu.merge(a1, on="psu_idx", how="left").merge(a2, on="psu_idx", how="left") \
            .merge(a3, on="psu_idx", how="left")

    # mêmes colonnes que l'ancien code
    df["axe1_count"] = df["n_cooccur_days"]                      # nb de jours HW+EPE, 1985–2024
    df["axe2_ratio"] = np.where(df["n_before"] > 0, df["n_after"] / df["n_before"], np.nan)
    df["axe2_ratio"] = df["axe2_ratio"].clip(upper=8)
    df["axe3_pct"] = df["pct_drought_psu"]                       # % HW précédées d'une sécheresse

    print(f"  {len(df):,} PSU · {df['country'].nunique()} countries")
    print(f"  (a) Compound events — mean: {df['axe1_count'].mean():.1f}, "
          f"median: {df['axe1_count'].median():.0f}, max: {df['axe1_count'].max():.0f}")
    print(f"  (b) After/before ratio — mean: {df['axe2_ratio'].mean():.2f}")
    print(f"  (c) % HW+drought — mean: {df['axe3_pct'].mean():.1f}%")
    return df
