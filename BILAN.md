# Bilan du projet : détection de ranges et achat de cassure (actions françaises, position longue)

État au 30/10/2026. Données Yahoo Finance journalières, du 01/04/2020 au 30/09/2026.

## 1. Conclusion
- **La détection des ranges fonctionne.** Sur la référence annotée (5 actions, 24 ranges), 17 ranges sont détectés la veille de leur cassure. La résistance calculée se situe à +0,3 % en médiane de celle tracée à la main, y compris sur les deux actions de contrôle (Sodexo, Solutions 30).
- **Acheter la cassure de ces ranges n'apporte pas d'avantage mesurable.** Sur 171 actions du SRD (environ 1 500 cassures par le haut), aucune des 48 variantes testées ne remplit les critères fixés à l'avance. Les entrées au hasard font même légèrement mieux, de +0,02 à +0,05 R.
- **Décision :** ne pas engager de capital sur ce signal en l'état. Le pipeline de détection reste réutilisable.

## 2. Pipeline retenu (étapes 1 à 4)
| Étape | Contenu |
|---|---|
| 1 | Cours ajustés et bruts (ouverture, haut, bas, clôture) et volume. Univers : référence + SRD (.PA, au moins 250 séances). Une variation de plus de 30 % en une séance est marquée comme saut. |
| 2 | Fenêtres glissantes de 22 séances : pente du log du cours, volatilité (ATR / cours moyen), ratio de consolidation. Fenêtres contenant un saut marquées. |
| 3 | K-means (K = 3, log + échelle robuste, n_init = 50) par action, sans les fenêtres contenant un saut. Le cluster du milieu n'est un range que si `indice_tendance` ≤ 0,5 (déplacement net en unités de volatilité, soit environ la médiane du hasard). |
| 4 | Épisodes : interruptions de 3 séances ou moins absorbées, validation à 10 séances, maintien tant que le prix reste dans ses bandes. Bandes par quantiles (support 5 à 10 % des bas, résistance 90 à 95 % des hauts), calculées depuis le début de la fenêtre de détection. Cassure confirmée par 2 clôtures au-delà de la bande + 0,25 ATR(22). Niveaux aussi exprimés en cours bruts. |

Aucune information future n'est utilisée dans les étapes 1 à 4 : chaque niveau est calculé avec les seules séances passées.

## 3. Validation de la détection (étape 4, `reference.csv`)
| Mesure | Résultat |
|---|---|
| Ranges vus en épisode validé la veille de la cassure | 17 / 24 |
| Cassures détectées dans le bon sens, à ±3 séances | 15 / 24 |
| Écart médian entre résistance calculée et résistance annotée (cours bruts) | +0,3 % |
| Part des jours de tendance classés en épisode | 16 % (3 tendances sur 17 à plus de 25 %) |

Limites connues :
- les ranges de moins de 20 séances environ ne sont pas détectés ;
- les ranges des périodes très volatiles (2020) sont classés « autre » par K-means.

## 4. Tests de performance (étapes 5 à 7)
Protocole commun :
- entrée à l'ouverture de la séance qui suit la cassure confirmée ;
- stops S1 (bas de la résistance − 0,25 ATR), S3 (milieu du range) ou S4 (entrée − 2 ATR) ;
- frais de 0,2 % par aller-retour, une position à la fois par action ;
- résultats mesurés en R (multiple du risque initial) ;
- comparaison à des entrées au hasard, avec la même distance de stop et la même sortie.

Critères fixés avant chaque test :
- R moyen positif sur les sous-échantillons ;
- meilleur que le hasard ;
- borne basse de l'intervalle de confiance à 95 % au-dessus de 0.

| Test | Variantes | Variantes valides | R moyen typique | Par rapport au hasard |
|---|---|---|---|---|
| Étape 5 : cassure, filtre de volume (max des volumes J1/J2 ≥ 1,5 × médiane sur 22 séances), sortie fixe à 5 ou 10 séances | 12 | 0 | −0,02 | légèrement en dessous |
| Étape 5, piste A : sortie par stop suiveur (chandelier 3 ATR, canal 10 séances, au plus 60 séances) | 12 | 0 | −0,02 | en dessous (20 cas sur 24) |
| Étape 6, piste C : entrée sur retour vers la résistance (ordre limité valable 10 séances, moitié A) | 12 | 0 | −0,03 | en dessous |
| Étape 7, piste B : filtre de marché (indice au-dessus de sa moyenne mobile à 200 séances, moitié A) | 12 | 0 | −0,10 | en dessous |

Observations :
- **Volume :** pas de relation régulière entre l'intensité du volume et le résultat.
- **Retard d'entrée :** en médiane, l'entrée se fait 1,1 ATR au-dessus de la résistance. Explication plausible, non démontrée : l'achat après deux clôtures de hausse expose au retournement de court terme documenté sur les actions individuelles (Jegadeesh 1990 ; Lehmann 1990).
- **Année :** 2022 est la plus défavorable et 2024 la plus favorable, mais le filtre de marché ne transforme pas cette différence en avantage.
- **Étape 7 :** le résultat reporté a été obtenu avec un indice de remplacement équipondéré (moitié A), Yahoo étant inaccessible depuis l'environnement d'analyse. Il reste à confirmer avec ^FCHI en lançant `etape7.py`, qui télécharge l'indice et le met en cache dans `data_test/cac40.csv`.

## 5. Précautions méthodologiques
- **Moitié B réservée :** une action sur deux dans l'ordre alphabétique n'a pas été utilisée pour les étapes 6 et 7. Elle reste disponible pour confirmer une seule fois une règle future, fixée à l'avance.
- **Nombre de variantes :** 48 variantes ont été testées en tout. Sur ces mêmes données, toute nouvelle variante a une probabilité croissante de réussir par hasard (Bailey et al., 2014).
- **Biais du survivant :** la liste du SRD est celle d'aujourd'hui, sans les sociétés radiées depuis 2020. Ce biais avantage en principe les résultats, ce qui renforce le constat négatif.
- **Volume :** le volume Yahoo des titres .PA ne couvre qu'Euronext Paris ; il est utilisé en relatif uniquement.
- **Cours ajustés :** ils dépendent de la date de téléchargement (un dividende futur modifie tout l'historique). Les niveaux à exécuter sur le marché sont à lire en cours bruts. Le jour d'un détachement de dividende, un stop en cours brut est à abaisser du montant du dividende.

## 6. Pistes si le projet reprend
1. Utiliser la détection de range comme **filtre de contexte** pour un autre signal, plutôt que comme signal d'entrée.
2. Restreindre l'univers aux **grandes capitalisations** (CAC 40), où les cassures peuvent se comporter autrement que sur les petites valeurs du SRD.
3. Toute nouvelle règle : la fixer et l'écrire avant le test, la tester sur la moitié A, puis la confirmer une seule fois sur la moitié B.

## Sources
- Jegadeesh (1990), « Evidence of Predictable Behavior of Security Returns », *Journal of Finance*.
- Lehmann (1990), « Fads, Martingales, and Market Efficiency », *Quarterly Journal of Economics*.
- Gervais, Kaniel & Mingelgrin (2001), « The High-Volume Return Premium », *Journal of Finance*.
- Moskowitz, Ooi & Pedersen (2012), « Time Series Momentum », *Journal of Financial Economics*.
- Faber (2007), « A Quantitative Approach to Tactical Asset Allocation », *Journal of Wealth Management*.
- Bailey, Borwein, López de Prado & Zhu (2014), « The Probability of Backtest Overfitting ».
