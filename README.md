# Projet test: système d'aide à la prise de position sur actions françaises (position longue) sur les breakouts confirmés de range
- L'idée est de tester sur plusieurs actions françaises la détection de pattern de range sur la période avril 2020 à septembre 2026: Veolia, Vivendi et Airbus pour la phase d'entraînement, sodexo et solution30 pour la phase de test
- La détection est réalisée par clusterisation sur des fenêtres glissantes de 2 mois de cotations (22 séances boursières)
- 3 clusters K-means sont créés selon 3 features calculés par fenêtre glissante: la pente de la tendance, la volatilité, le ratio de consolidation des prix
- ne sont ensuite retenus que les range significatifs (durée de l'épisode minimale à déterminer), les épisodes de range entrecoupés de courte interruption (à déterminer nombre de séances séances ou moins) sont fusionnés
- des supports et des résistances sont tracés sur ces épisodes de range
- à compléter au fur et à mesure de l'avancement de la réflexion avec l'IA

## étape 1: chargement des données
- récupérer les actions Veolia, Vivendi, Airbus, Sodexo et Solution30 (symboles Yahoo: VIE.PA, SW.PA, VIV.PA, AIR.PA et S30.PA) via Yahoo finance et l'utilisation de la librarie yfinance dans Python
- récupérer le cours de clôture, le cours bas, le cours haut journaliers, pour ces trois cours récupérer les valeurs non ajustées et ajustées, le volume
- récupérer les variables du 01/04/2020 au 30/09/2026
- création d'une nouvelle variable intitulée saut, qui vaut vrai quand la clôture varie de plus de 30 % par rapport à la veille, faux sino
- vérification et stockage des variables dans un fichier etape1.csv dans le sous dossier data_test, séparateur ";"

## étape 2: calcul des features nécessaires à la clusterisation
- à partir des données de l'étape 1, créer des variables supplémentaires
- variable 1: un identifiant qui représente une fenêtre glissante de 22 séances boursières, pas de la fenêtre: elle avance d'une séance boursière à la fois
- variable 2: un premier feature, le coefficient de la tendance de la regression linéaire du logarithme du cours de l'action sur la fenêtre de 22 séances, une valeur par fenêtre glissante
- variable 3: un second feature, la volatilité du cours de l'action sur la fenêtre de 22 séances. Option retenue pour le calcul de la volatilité: ATR normalisé par le cours
  - true range journalier: max(haut - bas, |haut - clôture de la veille|, |bas - clôture de la veille|)
  - ATR: moyenne des true range sur la fenêtre de 22 séances
  - volatilité = ATR / cours de clôture moyen de la fenêtre (même dénominateur que le ratio de consolidation ci dessous)
- variable 4: un troisième feature, le ratio de consolidation du cours de l'action sur la fenêtre de 22 séances, une valeur par identifiant donc. Pour calculer ce ratio, applique la formule suivante: (cours max - cours min)/cours moyen 
- variable 5: intitulée fenetre_avec_saut, qui vaut vrai quand la fenêtre de 22 séances contient au moins un saut
- vérification et stockage des variables créées dans un fichier etape2.csv dans le sous dossier data_test, séparateur ";"

## étape 3: clusterisation pour mettre en évidence les patterns de range
- la mise à l’échelle et K-means sont ajustés sans les fenêtres contenant un saut. Ces fenêtres reçoivent le régime exclu, sans numéro de cluster. Elles apparaissent en orange sur le graphique
- mise à l'échelle des 3 features: passage au log puis mise à l'échelle robuste, c'est à dire: Log, puis (x − médiane) / écart interquartile. Le log ne s'applique pas au feature pente qui peut être négatif
- recherche de clusters via la méthode Kmeans, avec K=3, n_init=50, random_state=0
- nommage automatique des clusters par pente moyenne de leurs fenêtres (la plus faible = autre, celle du milieu = range, la plus forte = autre)
- rattachement de chaque fenêtre au jour date_fin, pour ne pas utiliser d’information future
- ajout d'une variable filtre intitulée indice_tendance pour ne récupérer que les range significatifs. On ne garde que les fenêtres « range » où |pente| × 21 / (volatilité × √22) reste inférieur ou égal à 0,5
- ajout d'un graphique:
  - chaque jour,uniquement pour le régime range, le segment de cours prend la couleur du régime de la fenêtre qui se termine ce jour-là
  - couleur : bleu pour range, gris pour autre
  - le graphique, intitulé etape3, au format png, est stocké dans le dossier data_test
- vérification et stockage des variables créées dans un fichier fichier etape3.csv dans le sous dossier data_test, séparateur ";"

## étape 4: épisodes de range, bandes de support et de résistance, cassures
- n'utiliser que le passé connu: aucune information future n'est utilisée; traitement action par action
- interruptions (régime autre ou exclu) de 3 séances ou moins absorbées si le range reprend, sinon épisode clos à sa dernière séance range
- épisode validé à partir de sa 10e séance
- une fois l’épisode validé, une séance qui n’est plus range ne compte pas comme interruption si sa clôture reste entre le bas de la bande de support et le haut de la bande de résistance de la veille, sauf séance exclue
- bandes (méthode des quantiles): support = quantiles 5 à 10 % des cours bas, résistance = quantiles 90 à 95 % des cours hauts, calculées depuis le début de la fenêtre de détection (22 séances) qui a ouvert l'épisode, au plus les 44 dernières séances; bandes figées pendant une interruption
- fin d'épisode par cassure confirmée: 2 clôtures consécutives au-delà de la bande d'au moins 0,25 × ATR(22); nouvel épisode possible seulement après une séance non range
- bandes calculées en cours ajustés puis converties en cours bruts (facteur cloture_ajustee / cloture_brute)
- évaluation: comparaison à data_test/reference.csv (range vu la veille de la cassure, cassure détectée à ±3 séances, écart de résistance en cours bruts, part de jours de tendance en épisode) et suite des cassures par le haut (rendement à 5 et 10 séances, fausse cassure = retour sous la bande de résistance en 10 séances)
- sorties: etape4.csv, etape4_episodes.csv, etape4_evaluation.csv, etape4_cassures.csv, etape4.png dans data_test, séparateur ";"