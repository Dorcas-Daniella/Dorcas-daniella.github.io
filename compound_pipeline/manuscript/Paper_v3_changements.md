# Paper v3 (mode annuel) : ce qui change par rapport à Paper_Dorcas_v2

Le texte, la structure, les figures et l'interprétation d'ensemble sont conservés. L'introduction est inchangée. Les changements viennent de deux choix de méthode et des corrections du code.

## 1. Ce qui ne change pas (résultats confirmés)

| Résultat | v2 | v3 |
|---|---|---|
| Vagues de chaleur détectées | 10,7 millions | 10,7 millions |
| PSU sans co-occurrence | 55,5 % | 54,8 % |
| Maximum par PSU | 63 | 63 |
| Taux continental (par PSU et par an) | 0,041 | 0,042 |
| Madagascar / Rwanda (tendance par décennie) | +0,087 / +0,043 | +0,090 / +0,043 |
| Ratio après/avant, tendance | −0,36 par décennie | −0,37 par décennie |
| Pays où l'asymétrie est la plus forte | Niger, Guinée, Burkina Faso, Mali | les mêmes |
| Allongement des vagues de chaleur après sécheresse le plus marqué | Afrique australe et de l'Ouest | les mêmes |
| Hausse de la part avec sécheresse, ordre des régions | australe > Ouest > centrale > Est | le même |

## 2. Les deux choix de méthode modifiés

1. **Axe II : chaque EPE est compté une seule fois** (attribué à la vague de chaleur la plus proche). Dans la v2, un EPE pouvait compter pour plusieurs vagues de chaleur. Ratio : 2,39 → 3,11. L'ancienne mesure « par vague de chaleur » est donnée en complément.
2. **Axe III : SPEI du mois précédant la vague de chaleur** (au lieu du mois où elle commence). Le mois du début contient la chaleur de la vague elle-même, ce qui crée en partie sa propre sécheresse. Le lag 0 est donné en supplément.

## 3. Ajouts

- Test contre la saisonnalité (test null, 1 000 permutations) pour l'Axe II, avec la décomposition avant / après.
- Facteur de co-occurrence comparé au hasard (LMF) pour l'Axe I.
- SPI (pluie seule) et niveau de sécheresse attendu localement pour l'Axe III.
- Tests de sensibilité : fenêtre, zone tampon, seuil de sécheresse, seuil calendaire.
- Littérature africaine dans la Discussion (Fredy, Alasow, Ayugi, Adeyeri, Liu ; Tian déjà cité).

## 4. Phrases modifiées

| Où | v2 | v3 | Pourquoi |
|---|---|---|---|
| Résultats, Axe I | 1,6 fois par PSU ; 55,5 % | 1,7 jour par PSU ; 54,8 % ; + phrase LMF | Nouveaux chiffres ; ajout du LMF |
| Résultats, Axe I | « despite high heatwave frequency » | retiré | Avec un seuil en percentile, la fréquence des vagues de chaleur est la même partout |
| Résultats, Axe III | 30,5 % « coincided with drought » | 26,6 % « preceded by drought » | Mois précédent |
| Résultats, baisse du ratio | « mechanically reduces their ratio » | les EPE avant les vagues de chaleur augmentent plus vite (×2,6) que ceux après (×1,9) | C'est ce que montrent les données |
| Résultats, tendances sécheresse | +4,70 « % per decade » | +6,50 « percentage points per decade » ; + phrase SPI | Lag 1 ; unité correcte |
| Résultats, Axe II | 3,09 / 1,29 millions, ratio 2,39 ; « sensitive to prior heat accumulation » | 1,70 / 0,55 million, ratio 3,11 ; + test null (attendu 1,84 ; ×1,69 ; déficit avant 0,63) ; « heatwaves concentrate at the end of the dry season » | Comptage unique ; saisonnalité testée |
| Résultats, pays | 33 sur 34, Gabon 0,99 | 34 sur 34, Gabon 1,01 | Nouveaux chiffres |
| Résultats, Fig. 4b | « 5 to 15 days after » | pic au premier jour après la fin de la vague de chaleur ; après > avant sur la plus grande partie de la fenêtre ; pic plus marqué en 2015-2024 | La figure change avec le comptage unique |
| Résultats, durée et température | +1 à 1,5 jour (+17 à 31 %) ; +1,9 °C | +0,55 jour (+9 %) ; australe +0,97 jour (+19 %), +1,7 °C ; l'écart de température augmente dans le temps | Lag 1 |
| Résultats, parts par décennie | 23,6 → 37,6 % ; 63,5 % | 15,5 → 34,9 % ; 59,8 % ; + comparaison au niveau attendu localement (×1,10) | Lag 1 |
| Résultats, couplage | « suggests a strengthening land-atmosphere coupling » | retiré | La hausse disparaît avec le SPI |
| Résultats, régimes | « most locations exposed to multiple prominent hazard types » ; 20,6 % pour chaleur–pluie + pluie après la vague de chaleur | « at most one prominent hazard type » ; 51 514 PSU classables (94 %) ; 39,5 % aucun type (43,8 % attendus par hasard) ; chaleur–pluie + pluie après la vague de chaleur devient la plus rare (1,4 % contre 4,2 % attendus), les trois ensemble 0,1 % ; par décennie : aucun type 64,6 → 24,6 %, sécheresse–chaleur seule 10,2 → 28,1 % | Même seuil (quart supérieur) pour les trois axes au lieu de seuils mixtes (> 0, > 1,5, médiane) ; ratio avec +0,5 ; catégorie « Insufficient data » |
| Discussion, 3 résultats clés | « drought systematically prolongs heatwaves… intensifying » | « heatwaves preceded by drought last longer and are hotter, and a rapidly expanding share… » ; 2e résultat complété (saisonnalité, déficit avant) | Cohérence avec les résultats |
| Discussion, sécheresse | « has strengthened over time » ; « approaching levels reported in Mediterranean Europe » | retirés ; ajout de la littérature africaine et de la limite SPI | Non démontré ; pas de référence pour l'Europe méditerranéenne |
| Discussion, Clausius–Clapeyron | « physically consistent with the release of this moisture » | asymétrie en partie saisonnière, déficit avant, conditions anticycloniques ; Liu et al. 2026 | Contredit par le test null |
| Discussion, conclusion | « gain physical coherence, evolving… to distinct regional-scale regimes » | « net shift from locations without any prominent hazard towards regimes dominated by drought-heat coupling and heat-precipitation co-occurrence » | Les régimes changent localement (22 % inchangés, 20 % attendus par hasard) |
| Limites | — | + saisonnalité du seuil annuel ; + SPEI après 2014 ; + décalage d'un jour ERA5 / CHIRPS | Transparence |
| Méthodes | Paragraphe « Sensitivity analyses » regroupant tous les tests ; CHIRPS 0,25° ; SPEI « generalized logistic » ; Axe II avec doublons ; mois du début ; pas de test ; classification des pays fort / modéré / faible | CHIRPS 0,05° ; log-logistique (xclim) ; attribution unique ; test null ; mois précédent ; IC par bootstrap en blocs + Mann-Kendall ; règles de la Fig. 6 | Nouveau code |

## 5. À faire de votre côté

1. Refaire les figures avec vos anciens codes, en lisant les nouveaux fichiers. Fig. 4b : mettre le jour 0 à NaN (sinon la courbe tombe à 0). Fig. 4c : échelle de couleurs jusqu'à 6 au moins (ou logarithmique) et texte blanc sur les cases foncées. Fig. 5c : corriger le symbole « % » du titre. La Fig. 5 contient un tableau `STATS` écrit à la main avec les anciens chiffres : il faut le recalculer.
2. Extended Data Fig. S13 : ajouter la version avec un minimum de 10 EPE (`fig06_regimes_sens_min10`) en panneau b, à côté de la version tercile.
3. Supplementary Table S2 (enquêtes DHS) et lien du dépôt à compléter.
4. La figure S6 (classification des pays) venait de l'ancien code. Il faut soit la refaire à partir des nouvelles tendances, soit la retirer, puisque la méthode correspondante n'est plus décrite.
