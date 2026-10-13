# Projet test: système d'aide à la prise de position sur actions françaises (position longue uniquement) 
- L'idée est de tester sur l'action française Veolia SA la détection de pattern de range sur la période avril 2020 à septembre 2026.
- La détection est réalisée par clusterisation sur des fenêtres glissantes de 2 mois de cotations (44 séances boursières)
- 3 clusters K-means sont créés selon 3 features calculés par fenêtre glissante: la pente de la tendance, la volatilité, le ratio de consolidation des prix

## étape 1:
- récupérer l'action Veolia SA (symbole Yahoo: VIE.PA) via Yahoo finance et l'utilisation de la library yfinance
- récupérer le cours de clôture, le cours bas, le cours haut journaliers ajustés
- récupérer les 3 cours depuis le 01/04/2020 jusqu'au 30/09/2026
- vérification: stockage des cours dans un fichier csv dans le sous dossier data_test, séparateur ";"

## étape 2:
- à partir de ces cours: créer des variables dans un dataframe
- variable 1: un identifiant qui représente une fenêtre glissante de 44 séances boursières, pas de la fenêtre: elle avance d'une séance boursière à la fois
- variable 2: un premier feature, le coefficient de la tendance de la regression linéaire du logarithme du cours de l'action sur la fenêtre de 44 séances, une valeur par identifiant donc
- variable 3: un second feature, la volatilité du cours de l'action sur la fenêtre de 44 séances, une valeur par identifiant donc. Option retenue pour le calcul de la volatilité: ATR normalisé par le cours
  - true range journalier: max(haut - bas, |haut - clôture de la veille|, |bas - clôture de la veille|)
  - ATR: moyenne des true range sur la fenêtre de 44 séances
  - volatilité = ATR / cours de clôture moyen de la fenêtre (même dénominateur que le ratio de consolidation)
- variable 4: un troisième feature, le ratio de consolidation du cours de l'action sur la fenêtre de 44 séances, une valeur par identifiant donc. Pour calculer ce ratio, applique la formule suivante: (cours max - cours min)/cours moyen 
- vérification: stockage du dataframe dans un fichier csv dans le sous dossier data_test, séparateur ";"

## étape 3:
- mise à l'échelle des 3 features: log puis échelle robuste, c'est à dire: Log, puis (x − médiane) / écart interquartile)
- recherche de clusters via la méthode Kmeans, avec K=3, n_init=50, random_state=0
- nommage automatique des clusters selon la pente de leur centre (la plus faible = baissière, celle du milieu = range, la plus forte = haussière)
- rattachement de chaque fenêtre au jour date_fin, pour ne pas utiliser d’information future
- vérification: dans un fichier csv: cours journalier ajusté et de id_fenetre, du numéro de cluster et de son nom. Dans le sous dossier data_test, séparateur ";" 