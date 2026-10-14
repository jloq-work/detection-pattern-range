# Projet test: système d'aide à la prise de position sur actions françaises (position longue uniquement) et seulement sur les patterns de range
- L'idée est de tester sur l'action française Veolia SA la détection de pattern de range sur la période avril 2020 à septembre 2026.
- La détection est réalisée par clusterisation sur des fenêtres glissantes de 2 mois de cotations (44 séances boursières)
- 3 clusters K-means sont créés selon 3 features calculés par fenêtre glissante: la pente de la tendance, la volatilité, le ratio de consolidation des prix
- ne sont ensuite retenus que les range significatifs (durée de l'épisode minimale de 15 séances), les épisodes de range entrecoupés de courte interruption (5 séances ou moins) sont fusionnés
- des supports et des résistances sont tracés sur ces épisodes de range

## étape 1: chargement des données
- récupérer l'action Veolia SA (symbole Yahoo: VIE.PA) via Yahoo finance et l'utilisation de la library yfinance
- récupérer le cours de clôture, le cours bas, le cours haut journaliers ajustés, le volume
- récupérer les variables du 01/04/2020 au 30/09/2026
- vérification et stockage des variables dans un fichier csv dans le sous dossier data_test, séparateur ";"

## étape 2: calcul des features nécessaires à la clusterisation
- à partir des données de l'étape 1, créer des variables supplémentaires
- variable 1: un identifiant qui représente une fenêtre glissante de 44 séances boursières, pas de la fenêtre: elle avance d'une séance boursière à la fois
- variable 2: un premier feature, le coefficient de la tendance de la regression linéaire du logarithme du cours de l'action sur la fenêtre de 44 séances, une valeur par identifiant donc
- variable 3: un second feature, la volatilité du cours de l'action sur la fenêtre de 44 séances, une valeur par identifiant donc. Option retenue pour le calcul de la volatilité: ATR normalisé par le cours
  - true range journalier: max(haut - bas, |haut - clôture de la veille|, |bas - clôture de la veille|)
  - ATR: moyenne des true range sur la fenêtre de 44 séances
  - volatilité = ATR / cours de clôture moyen de la fenêtre (même dénominateur que le ratio de consolidation ci dessous)
- variable 4: un troisième feature, le ratio de consolidation du cours de l'action sur la fenêtre de 44 séances, une valeur par identifiant donc. Pour calculer ce ratio, applique la formule suivante: (cours max - cours min)/cours moyen 
- vérification et stockage des variables créées dans le même fichier csv que celui créé à l'étape 1

## étape 3: clusterisation et filtre des patterns de range
- mise à l'échelle des 3 features: log puis échelle robuste, c'est à dire: Log, puis (x − médiane) / écart interquartile. Le log ne s'applique pas au feature pente qui peut être négatif
- recherche de clusters via la méthode Kmeans, avec K=3, n_init=50, random_state=0
- nommage automatique des clusters par pente moyenne de leurs fenêtres (la plus faible = autre, celle du milieu = range, la plus forte = autre)
- rattachement de chaque fenêtre au jour date_fin, pour ne pas utiliser d’information future
- ajout d'un graphique:
  - chaque jour,uniquement pour le régime range, le segment de cours prend la couleur du régime de la fenêtre qui se termine ce jour-là
  - couleur : bleu pour range
  - le graphique, au format png, est stocké dans le dossier data_test
- vérification et stockage des variables créées dans le même fichier csv que celui créé à l'étape 1

## étape 4: récupération des range pertinents et traçage des supports et résistances
- fusionner les épisodes de range séparés par des interruptions (cluster=autre) d'une durée de 5 séances boursières ou moins
- créer une nouvelle variable d'identification des range: épisode de range d'une durée minimales de 15 séances boursières
- les bandes de niveaux des supports et des résistances sont calculées seulement en utilisant les séances boursières connues au moment du calcul, autrement dit les séances passées, aucune information future n'est utilisée
- utiliser et comparer 2 méthodes pour déterminer les bandes des supports et des résistances
  - méthode 1: utilisation des quantiles, bande de support=quantiles 5 à 10% des cours bas et bande de résistance=quantile 90 à 95% des cours hauts
  - méthode 2: point de retournements regroupés, repérer les creux et sommets locaux (un point plus bas ou plus haut que ses 3 voisins de chaque côté), regrouper ceux qui sont à moins de 0,5*ATR l'un de l'autre, et retenir les niveaux touchés au moins 2 fois
- plan de comparaison des 2 méthodes:
  - produire un tableau des épisodes de range retenus par méthode avec les informations suivantes: dates, durée, support et résistance selon les méthodes, nombre de contacts
  - mesure du taux de respect pour chaque méthode: quand le cours entre dans la zone de support, part des cas où il cloture au dessus 5 et 10 séances plus tard, contre la part des cas où il casse sous la zone
  - contrôle visuel:un graphique sur toute la période avec les zones en bandes semi transparentes