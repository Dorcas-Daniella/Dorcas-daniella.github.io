# Pipeline « compound climate events » — v2 → v3 : rapport des corrections

## Livrables

| Fichier | Contenu |
|---|---|
| `compound_pipeline_v3.py` | Pipeline corrigé complet (format jupytext *percent* : `# %%` = cellule ; exécutable comme script) |
| `compound_pipeline_v3.ipynb` | Le même, en notebook, **sans sorties** (54 cellules) |
| `compound_pipeline_v2_original.py` | Source de la v2 transmise, convertie à l'identique (référence du diff) |
| `compound_pipeline_v2_to_v3.diff` | Différences unifiées v2 → v3 |
| `tests/test_corrections.py` | 27 tests ciblés sur données artificielles (v3 et, pour comparaison, v2) |
| `tests/make_synthetic_inputs.py`, `tests/run_e2e_synthetic.py` | Run complet de bout en bout sur données artificielles + 44 contrôles post-run |

Non versionnés volontairement : le `.ipynb` d'origine (ses sorties contiennent notamment des coordonnées de PSU DHS) et le fichier `PSU_geocoordinates.RData`.

**Correspondance des cellules.** Les cellules 10 (patch CHIRPS) et 22 (`pip install -U xclim`) ont été supprimées. Une cellule *n* de la v2 devient *n* pour *n* ≤ 9, *n* − 1 pour 11 ≤ *n* ≤ 21, et *n* − 2 pour *n* ≥ 23.

## Tableau de suivi

Les statuts distinguent trois niveaux de preuve :
- **« Vérifiée (synthétique) »** : un test exécuté ici, sur données artificielles, a réussi ;
- **« Exécutée (synthétique) »** : le code a tourné sans erreur dans le run complet artificiel, sans test dédié du résultat ;
- **« À valider au run »** : dépend des données réelles.

Aucune correction n'est validée sur les données climatiques réelles.

| # | Correction | Emplacement modifié | Vérification | Statut |
|---|---|---|---|---|
| 1 | Périmètre : `in_base_correction="none"` | cfg (c4) | Déjà `"none"` dans la v2 | Déjà conforme |
| 1 | LOO / Zhang réservés au reviewer | commentaires cfg, c15, c18 ; branches conservées | — | Appliquée (documentation) |
| 1 | Tolérance ±1 j non rapportée | `assert cfg.axis1_day_tolerance == 0` (c4) ; textes c11, c18, c27 | Exécutée dans le run complet | Appliquée |
| 1 | Cell-collapse retiré | 6 appels `by_year(..., collapse=True)`, scopes `*_cellcollapse`, textes c0, c7, c8 (log), c11, c35, c39, c42 | Contrôle e2e : aucun fichier ni scope `cellcollapse` | Vérifiée (synthétique) |
| 2 | Suppression `axis3_indices`, `null_doy_pool` | cfg | Recherche : aucun autre usage | Appliquée |
| 2 | `assert axis2_attribution_tiebreak == "earlier"` ; minimums fig. 6 = `None` | cfg | Exécutée | Appliquée |
| 2 | `calendar_doy()` (29 fév = 60, 1er mars = 61) | cœur (c6) | `test_calendar_doy_leap_projection` | Vérifiée (synthétique) |
| 2 | « ETCCDI » → fenêtre circulaire de 31 positions de type HWMId | cfg, c15, docstrings | — | Appliquée |
| 3 | Helper unique `psu_eligibility()` (+ `axis12_eligible` = epe & temp) | cœur (c6) ; utilisé dans step04-check, step05, LMF, null, sens. W, step08, figures 4b/5/6/sens., c21, c26, c32, c45 | `test_psu_eligibility_flags` ; contrôles e2e de population (Axes I, II, III, EPE, SPEI) | Vérifiée (synthétique) |
| 4 | Manifeste `run_manifest.json` : config, graine, versions, empreinte du code, identité NetCDF/RData | fin de c4 | e2e : présence des clés et identités ; `test_manifest_jsonable_strict` | Vérifiée (synthétique) |
| 4 | Complété au fil du run : éligibilité, jours exclus du LMF, méthodes SPEI/SPI, catégories de NaN, permutations utilisables, contrôle de cohérence de l'Axe II | c16, step04, step05, null, `check_axis2_consistency` | e2e | Vérifiée (synthétique) |
| 4 | Avertissement si xclim ≠ 0.61.x | c4 | Non déclenché ici (0.61.1 installé) | Appliquée, non testée |
| 5 | Suppression de la cellule 10 | — | La v2 plante en c10 sur un `processed_3` vide (`FileNotFoundError`, observé) ; la v3 passe | Vérifiée (synthétique) |
| 5 | `_open_and_align` : unicité, ordre, pas journalier, couverture exacte 1985–2024, sélection ordonnée, contiguïté | c12 | `test_open_and_align_checks` (jour manquant, doublon, horodatage 12 h, latitudes décroissantes) | Vérifiée (synthétique) |
| 6 | Grille Tmax indexée par `calendar_doy − 1`, 31 positions conservées | `_tmax_thresholds` | `test_calendar_window_feb29_has_30_real_dates...` | Vérifiée (synthétique) |
| 6 | Mode annual numériquement identique | — | Fonction (`test_annual_mode_numerically_unchanged`) et bout en bout : seuils, HW et EPE **identiques** v2/v3 en mode annual | Vérifiée (synthétique) |
| 6 | Empreinte JSON du cache précipitation | `_precip_cache_fingerprint`, `run_step03` | e2e : cache réutilisé si l'empreinte est inchangée, recalculé si `min_wet_days_ref` change | Vérifiée (synthétique) |
| 6 | Avertissement zhang → LOO (EPE) ; replis et moyenne des seuils ≠ moyenne des taux documentés | `_precip_thresholds`, docstrings, c15 | Non exécuté (branche reviewer) | Appliquée, non testée |
| 7 | `calendar_doy` dans `_attach_tmax_threshold` ; 29 fév conservé dans `doy_counts` | step04 | `test_attach_tmax_threshold...` ; e2e : position 60 = 10 années bissextiles | Vérifiée (synthétique) |
| 7 | Erreur si `p95_yXXXX` manque (correction in-base active) | `_attach_precip_threshold` | `test_attach_precip_threshold_missing_inbase_column_raises` | Vérifiée (synthétique) |
| 7 | Masque commun Tmax & précip pour `n_valid`, `n_hwday`, `n_epe`, `n_joint` ; jours exclus journalisés | step04 | e2e : `n_valid` = jours où les deux sont finis ; v2/v3 : HW/EPE identiques, seuls les comptes masqués changent | Vérifiée (synthétique) |
| 8 | Suppression de la cellule `pip install` | — | — | Appliquée |
| 8 | SPEI/SPI limités aux PSU `spei_eligible` | `run_step05_spei` | e2e | Vérifiée (synthétique) |
| 8 | Suppression de l'`except Exception` ; repli PWM → ML conservé ; méthode effective lue dans xclim, constante entre blocs | `_standardize`, `run_step05_spei` | `test_standardize_nan_categories_and_methods` (SPEI = ML, SPI = APP) ; e2e sur 5 blocs | Vérifiée (synthétique) |
| 8 | Classification des NaN (inéligible / amorçage / entrée manquante / calibration ≤ 1 valeur, zéros exclus pour le SPI) ; arrêt avec diagnostic sinon | `_classify_si_nans` | Tests : catégories attendues, zéros du SPI finis, NaN injecté détecté ; e2e : 0 non expliqué, bilan exact sur la grille complète | Vérifiée (synthétique) |
| 8 | `is_drought_*` booléen nullable ; fractions sur les mois disponibles | step05, c26, carte de sécheresse | e2e : `<NA>` exactement là où l'indice est NaN | Vérifiée (synthétique) |
| 9 | Exclusion des HW tronquées **avant** l'attribution ; remappage vers les `hw_id` ; drapeau conservé ; contrôle legacy sur les paires non tronquées | `_axis2` | `test_axis2_truncated_excluded_before_attribution` (la v2 perdait l'EPE) ; e2e : vectorisé == legacy | Vérifiée (synthétique) |
| 9 | Égalités : after (fin égale → début le plus ancien → id) ; before (début égal → id) | `attribute_unique` | Comparaison à une implémentation brute sur 30 × 300 cas avec égalités ; le test documente aussi le comportement corrigé de la v2 | Vérifiée (synthétique) |
| 9 | Le choix du porteur « during » n'affecte pas avant/après | docstring | `test_during_carrier_does_not_change_before_after_counts` | Vérifiée (synthétique) |
| 10 | LMF : population éligible + masque commun ; IC « cluster bootstrap CI — ERA5 cell » ; retrait de « the CI to quote » | `lmf_tables`, c35, c37 | e2e : libellés, population | Vérifiée (synthétique) |
| 11 | Null `psu_window` : transfert mois/jour (29 fév → 28), bornes entières, 3 exclusions, tirage année puis décalage, franchissement du 31 déc., assertions, arrêt si aucune année admissible, sans matrice n × 40 ; `region_pool` inchangé | `_transfer_onset`, `_year_of`, `_admissible_shifts`, `axis2_null` | Bornes = recherche brute (3 jeux W/K/durée, dont W = 60 et durées ≤ 330 j) ; franchissement d'année ; uniformité année et décalage (χ², 4 000 tirages) ; arrêt avec diagnostic | Vérifiée (synthétique) |
| 12 | Sensibilité W : même éligibilité, troncature par W ; identité step06 = référence du null = sensibilité W (continental + régions) | `_load_axis2_for_null`, `check_axis2_consistency` | e2e : identique sur 5 échelles, dans les deux modes | Vérifiée (synthétique) |
| 12 | `n_perm_valid` séparé : précurseur, déclencheur, différence, ratio | `axis2_null` | e2e (manifeste) | Vérifiée (synthétique) |
| 13 | Helper dans `_scaffold`, `n_units_eligible`, populations des Axes I, II et III | `_scaffold`, `axis{1,2,3}_aggregations` | e2e (populations) | Vérifiée (synthétique) |
| 13 | `n_hw_with_cooc` daté par le début de la HW (année, décennie) ; `n_cooccur_days` par l'EPE ; table PSU × décennie complète | `axis1_aggregations` | `test_axis1_dating_and_complete_decades` (HW 30/12/1994, EPE 02/01/1995) ; e2e | Vérifiée (synthétique) |
| 14 | Pente de Sen sur les années disponibles ; trou interne → MK et IC NaN, `inference_ok = False` ; NaN de bord tolérés ; scopes cell-collapse retirés | `_trend_rows`, `SCOPES` | `test_trends_internal_gap_and_edge_nan` ; e2e | Vérifiée (synthétique) |
| 14 | FDR sur les seules p-valeurs calculables | — | `benjamini_hochberg` ignorait déjà les NaN (test) | Déjà conforme |
| 15 | Figure 6 : données insuffisantes (inéligible / métrique manquante / support), ratio reconstruit (+∞ / 0), statistique d'ordre x₍⌈qn⌉₎ poolée appliquée aux décennies, « strictement supérieur », arrêt si seuil NaN/+∞, gris distinct, parts et transitions sur les PSU classables, effectifs, diagnostics | c51 (cellule réécrite) | `test_order_statistic_threshold`, `test_regime_classification_rules` ; e2e : diagnostics de support seuls quand minimums = None ; figure avec minimums **PROVISOIRES (1, 1)** marquée `_PROVISIONAL` | Vérifiée (synthétique) — minimums à fixer |
| 16 | Fig. 4b éligibilité ; S03b/S04b et S06 b : PSU disposant du SPEI ; S10 : ratios indéfinis exclus, cellules masquées ; carte de sécheresse sur mois disponibles ; `slope_text` sans IC si `inference_ok = False` | c47–c55 | e2e : toutes les figures produites sans erreur | Exécutée (synthétique) ; rendu visuel non inspecté |
| 17 | Textes et Methods (c0, 7, 11, 15, 18, 24, 27, 31, 35, 39, 42, 46) | markdown + docstrings | Relecture du diff | Appliquée ; valeurs du run final à reporter |

## Choix d'interprétation (à confirmer si besoin)

1. **C11 — non-recouvrement.** La fenêtre simulée complète `[s − W, s + d + W]` ne doit pas recouvrir la HW observée `[s_obs, e_obs]`, c'est-à-dire la même fenêtre que pour la contrainte de période.
   - Preuve utilisée par l'échantillonneur : pour K < 364, chaque exclusion retire une queue de [−K, K], donc l'ensemble admissible reste un intervalle. Une assertion bloque le run si ce n'était pas le cas.
   - Tirage de l'année par rejet : la loi est exactement uniforme sur les années admissibles.
2. **C15 — avec minimums `None`.** Seuls les diagnostics de support (`fig06_support_diagnostics.csv`) sont produits ; ni seuils, ni classes, ni figure. Les diagnostics complets sont écrits une fois les minimums fixés.
   - Pour tester la figure : `fig06_regimes(..., min_support=(m2, m3))`. Toutes les sorties sont alors suffixées `_PROVISIONAL` et la figure porte la mention.
   - « Inéligible » = non (`epe & temp`) ou non `spei_eligible`, puisque la PSU doit être classable sur les 3 axes.
3. **C16 — « PSU disposant du SPEI sur la période ».** PSU `spei_eligible` ayant au moins une valeur SPEI-3 finie sur 1985–2024. C'est une constante par groupe (`n_units_eligible` des tables `axis3_by_year_*`), comme pour l'Axe II.
4. **C10 — libellé.** Traduit en anglais comme le reste du notebook : « cluster bootstrap CI — ERA5 cell ». La ligne d'IC « bootstrap CI — PSU » est conservée et libellée, sans être présentée comme préférable.
5. **C3.** J'ajoute la colonne dérivée `axis12_eligible = epe_eligible & temp_eligible` pour éviter de répéter la conjonction. Les sorties de step06 (`axis1.parquet`, `axis2_*.parquet`, `axis3.parquet`) restent exhaustives ; l'éligibilité est appliquée par tous leurs consommateurs.
6. **C1 — cell-collapse.** Les appels, sorties, scopes et textes sont retirés. Le paramètre `collapse=False` des fonctions internes `by_year` est conservé mais n'est plus appelé (diff minimal). Dites-le si vous voulez aussi supprimer ce code mort.
7. **C7.** `n_joint_tol1` est conservée dans `doy_counts` comme colonne diagnostique, avec le même masque. Elle n'est utilisée nulle part puisque la tolérance vaut 0, ce qui est vérifié par assertion.

## Limites et points restant à vérifier sur vos données

- **Rien n'a été exécuté sur les données ERA5/CHIRPS réelles.** Les tests utilisent des données artificielles (40 PSU). Seule l'étape step01 a tourné sur votre RData réel : 54 615 PSU, 34 pays, répartition régionale identique à vos anciennes sorties.
- Le contrôle des NaN SPEI/SPI reproduit le comportement d'xclim **0.61.1** (lu dans son code source). Avec une autre version, un NaN légitime pourrait être classé « non expliqué » et arrêter le run : c'est voulu, et l'avertissement de C4 le signale.
- Mode calendar : les seuils changent (sur les données artificielles, jusqu'à 0,27 °C et 16 HW de plus sur 7 982). L'ampleur réelle reste à mesurer.
- Le nouveau null exige une année admissible pour chaque HW. Une HW de plusieurs centaines de jours ou un W très grand arrêterait le run (diagnostic `axis2_null_no_admissible_year.parquet`).
- Coût : le null ajoute un passage de 40 années × n HW avant les permutations, puis un rejet d'années par permutation. C'est le même ordre de grandeur qu'avant, mais pas mesuré à 9,5 M de HW.
- Les avertissements pandas `FutureWarning` (downcasting `fillna`, `observed=False`) proviennent de lignes de la v2 non modifiées ; ils ont été laissés en l'état.
- `_code_fingerprint` hache le source des fonctions définies dans le noyau. Dans Jupyter, cela suppose que `inspect.getsource` fonctionne (cas standard d'ipykernel) ; les fonctions non hachables sont listées dans le manifeste.
- Les figures ont été produites, mais leur rendu visuel n'a pas été inspecté (pas de cartopy ici : fond de carte simplifié).

## Validation locale (votre exécution)

```bash
# 0. environnement : vérifier xclim 0.61.x
python -c "import xclim; print(xclim.__version__)"
# 1. tests synthétiques (quelques secondes ; aucune donnée réelle)
cd compound_pipeline/tests && python -m pytest -q test_corrections.py          # attendu : 27 passed
# 2. run complet artificiel (~3 min)
python make_synthetic_inputs.py /tmp/cce_synth/Code_PhD
python run_e2e_synthetic.py ../compound_pipeline_v3.py /tmp/cce_synth           # attendu : 44/44 post-run checks passed
```

**Run de contrôle sur données réelles** (étape 3 de votre plan) :
- éditer `BASE_DIR` en cellule 4 ;
- vider `processed_3`, redémarrer le noyau, exécuter le notebook v3 de haut en bas avec `n_null_permutations = 100–200`.

Critères de passage :
- aucune exception ; en particulier aucun `unexplained NaN` (step05), aucune `AssertionError` de cohérence de l'Axe II et aucune HW sans année admissible ;
- `run_manifest.json` contient `eligibility`, `spei_spi` (SPEI `method_effective = "ML"`, SPI `"APP"`, `nan_categories.*.unexplained = 0`), et pour chaque mode `lmf_excluded_days`, `axis2_null.n_perm_valid` et `axis2_consistency_ok = true` ;
- `processed_3/<mode>/significance/axis2_consistency_check.parquet` a la colonne `identical` entièrement vraie ;
- `processed_3/<mode>/figures/fig06_support_diagnostics.csv` existe, sans `fig06_regimes.png`.

Ensuite :
- inscrire `regime_min_support_axis2/3` d'après les seuls diagnostics de support, avant de consulter seuils et classes ;
- relancer depuis un état propre avec les nombres complets de permutations et de bootstraps ;
- reporter dans les Methods de la cellule 0 les valeurs entre crochets, lues dans le manifeste du run final.

## Ajouts après le premier run réel (v3.1)

| Point | Modification | Vérification | Statut |
|---|---|---|---|
| Null synchronisé | `axis2_null` : un seul tirage (année, décalage) par HW distincte (cellule ERA5, début, fin), recopié sur toutes les PSU de la cellule. Ces PSU ont des séries Tmax identiques, donc des HW identiques. Avant, les copies étaient traitées comme indépendantes et l'enveloppe était trop étroite. La médiane du null est inchangée en espérance. | `test_null_draws_synchronised_within_era5_cell` ; e2e : grappes enregistrées dans le manifeste | Vérifiée (synthétique) |
| p-values non affichées | Toujours calculées et conservées dans `axis2_null_region.parquet`. Retirées du log, de la console, de la figure `figS_axis2_null_test` (le panneau a montre désormais le rapport observé / médiane du null) et du CSV joint à la figure. | e2e | Vérifiée (synthétique) |
| 3.1 Assertion de normalité SPEI/SPI | Remplacée par un avertissement. Les moments de la période de calibration sont enregistrés comme diagnostic (`spei_spi.calibration_moments_diagnostic`). Les arrêts sur erreur xclim et sur NaN inexpliqué sont inchangés. | xclim 0.61.1 : un SPI correct avec 50 % de zéros a une moyenne de +0,42 et un écart-type de 0,60, et faisait échouer l'assertion ; avec 20 % de zéros, elle passe | Appliquée |
| 3.2 Manifeste complet | `run_status` passe de started à completed (dernière cellule), avec la configuration finale ; l'empreinte est recalculée après la dernière étape (119/119 fonctions). Sous Jupyter, les cellules exécutées sont aussi hachées (`executed_cells_sha256`). | e2e (mode script) ; le hachage des cellules Jupyter n'est pas testable ici | Appliquée |
| 3.3 Dénominateur de l'Axe III (S04b, S06b) | Inchangé : cohorte fixe des PSU disposant du SPEI. C'est un choix, à expliciter. Hors amorçage, la grille SPEI est quasi complète : 0,42 % de NaN au run précédent, contre 0,417 % pour le seul amorçage. Le seul effet réel concerne 1985 : les débuts de janvier à mars n'y sont pas classables au lag 1, donc les comptes par PSU de 1985 sont sous-estimés. Les pourcentages (`pct_drought_*`, tendances) ne sont pas concernés. | Raisonnement sur les sorties du run précédent | Non modifiée |

## Ajouts v3.2

| Point | Modification | Vérification | Statut |
|---|---|---|---|
| Permutations par mode | `cfg.n_null_permutations_by_mode = {"calendar": 1000, "annual": 1000}` (calendar = mode principal, annual = mode de comparaison ; 1000 dans les deux modes, choix retenu pour l’exécution finale). Repli sur `n_null_permutations` si le paramètre est absent. Nombre effectif enregistré dans le manifeste. | Test unitaire (11 permutations en annual) ; e2e (60 / 40) | Vérifiée (synthétique) |
| Décomposition du ratio | Le null conserve `n_before` et `n_after` à chaque permutation (`axis2_null_draws.parquet`). La table `axis2_null_region.parquet` ajoute leurs enveloppes (`n_before_null_lo/med/hi`, `n_after_null_lo/med/hi`) et `before_obs_over_null_med`, `after_obs_over_null_med` : < 1 = déficit, > 1 = excès par rapport au calendrier saisonnier. Le produit de ces deux rapports approche `count_ratio_obs / count_ratio_null_med` sans l'égaler, la médiane d'un ratio différant du ratio des médianes. | Test unitaire (cohérence draws / ratio / médianes) ; e2e | Vérifiée (synthétique) |

## Ajouts v3.3 (affichage seulement ; aucun résultat modifié)

| Point | Modification | Vérification | Statut |
|---|---|---|---|
| Journal en double | `log.propagate = False` (cellule 2) : chaque message ne s'affiche plus qu'une fois, même si une bibliothèque a configuré le logger racine. | Tests unitaires 29/29 | Appliquée |
| Cartes Axe II (Fig 1b, Fig 3b) | Les PSU ayant des HW mais aucun EPE attribué avant (`n_before = 0`, ratio indéfini) étaient retirées des cartes et ressemblaient à des zones sans PSU. Elles sont maintenant tracées en gris ; la classe grise figure dans la légende, sous forme d'une case « undefined » placée devant la barre de couleur. Leur nombre est écrit dans le log, par carte et par décennie. | Test sur un petit tableau (2 PSU indéfinies tracées, PSU non éligible exclue) ; e2e 55/55 ; rendu de la légende contrôlé visuellement | Vérifiée (synthétique) |
| Barre de couleur Axe II | `extend="max"` au lieu de `"both"` : un ratio ne peut pas être négatif. | e2e | Appliquée |
| Libellé Fig 3 | « (per decade) » → « , by decade » : valeur de la décennie, et non pente par décennie. | e2e | Appliquée |
| Avertissements cartopy | `edgecolor=` au lieu de `color=` pour les côtes et frontières : supprime les avertissements « facecolor will have no effect ». | e2e : aucun avertissement de ce type | Appliquée |
