# Plan des figures et tableaux (validé)

Mode principal : **calendar**. Les fichiers viennent de `processed_3/calendar/figures/`, sauf mention contraire.

## Figures principales

| Article | Fichier | Contenu |
|---|---|---|
| Fig. 1 | `fig01_spatial_distribution` | Cartes des 3 axes, 1985–2024 (PSU sans EPE avant les HW en gris) |
| Fig. 2 | `fig02_temporal_trends` | Séries annuelles et tendances, continent + 4 régions |
| Fig. 3 | `fig03_spatiotemporal_decades` | Cartes des 3 axes par décennie |
| Fig. 4 | `fig04_axis2_asymmetry` | Asymétrie de l'Axe II : pays, calendrier des EPE autour des HW, région × décennie |
| Fig. 5 | `fig05_drought_amplification` | Axe III : durée, intensité, région × décennie |
| Fig. 6 | `fig06_regimes` | Régimes (seuil au 75e percentile), décennies, transitions |

## Figures supplémentaires

| SI | Fichier | Contenu |
|---|---|---|
| Fig. S1 | `figS_axis2_null_test` | Null de l'Axe II : ratio observé et null par région, fractions ECA, par saison, distribution, **e : décomposition avant / après** |
| Fig. S2 | `figS06_comparative_evolution` | Nombres absolus d'EPE avant et après les HW ; HW avec et sans sécheresse |
| Fig. S3 | `figS05_seasonality` | Saisonnalité des 3 axes et des HW |
| Fig. S4 | `figS08_latitude_gradient` | Gradient en latitude |
| Fig. S5 | `figS_sensitivity` | Fenêtre W, seuil de sécheresse, durée, lag et indice |
| Fig. S6 | `figS_axis3_lag_index` | SPEI au lag 1, SPEI au lag 0 et SPI au lag 1 dans le temps |
| Fig. S7 | `figures_shared/figS_drought_baseline` | Fréquence de sécheresse de référence (SPEI-3 < −1) |
| Fig. S8 | `figures_shared/figS_axis3_expected_baseline` | **Axe III comparé au niveau attendu localement** (SPEI/SPI, régions, les deux modes) |
| Fig. S9 | `fig06_regimes_sens_tercile` | Régimes au seuil du tercile |
| Fig. S10 | `figures_shared/figS_threshold_mode_comparison` | Comparaison calendar / annual |
| Fig. S11 | `annual/figures/fig01_spatial_distribution` | Mode annual : cartes |
| Fig. S12 | `annual/figures/figS_axis2_null_test` | Mode annual : null (fortement saisonnier) |
| Fig. S13 | `annual/figures/fig06_regimes` | Mode annual : régimes |

Non utilisées : `figS09_decadal_change` (doublon), `figS10_monthly_stratification` (doublon de la Fig. S3 b), `figS02/S03/S04_geo_*` et `figS03b/S04b_*` (doublons du Tableau S1 et de la Fig. 4a ; pas encore relues), les autres figures du mode annual (résumées par la Fig. S10).

## Tableaux supplémentaires

| SI | Source |
|---|---|
| Tableau S1 — Statistiques par pays | `tableS01_country_statistics.csv` |
| Tableau S2 — Enquêtes DHS | à compléter |
| Tableau S3 — Éligibilité et nombres d'événements | manifeste, diagnostic de la section 15 |
| Tableau S4 — Null par région et saison, avec décomposition (sans p-values) | `figS_axis2_null_test_table.csv` |
| Tableau S5 — Sensibilité à W et zone tampon | `axis2_window_sensitivity`, `figS_axis2_boundary_buffer_table.csv` |
| Tableau S6 — Axe III observé et attendu localement | `figures_shared/axis3_expected_baseline.csv` |
| Tableau S7 — LMF par région | `significance/lmf_by_scale` |
| Tableau S8 — Toutes les tendances | `trends/all_trends_summary` |
| Tableau S9 — Diagnostics de la Fig. 6 | `fig06_*.csv` |
