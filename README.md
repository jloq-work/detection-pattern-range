# Projet test: système d'aide à la prise de position sur actions françaises (position longue) sur les breakouts confirmés de range
- L'idée est de tester sur plusieurs actions française: Veolia, Sodexo, Vivendi et Airbus la détection de pattern de range sur la période avril 2020 à septembre 2026
- La détection est réalisée par clusterisation sur des fenêtres glissantes de 2 mois de cotations (22 séances boursières)
- 3 clusters K-means sont créés selon 3 features calculés par fenêtre glissante: la pente de la tendance, la volatilité, le ratio de consolidation des prix
- ne sont ensuite retenus que les range significatifs (durée de l'épisode minimale à déterminer), les épisodes de range entrecoupés de courte interruption (à déterminer nombre de séances séances ou moins) sont fusionnés
- des supports et des résistances sont tracés sur ces épisodes de range
- à compléter au fur et à mesure de l'avancement de la réflexion avec l'IA

## étape 1: chargement des données
- récupérer les actions Veolia, SES, Vivendi et Airbus (symboles Yahoo: VIE.PA, SW.PA, VIV.PA et AIR.PA) via Yahoo finance et l'utilisation de la librarie yfinance dans Python
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

## étape 4: récupération des range pertinents et traçage des supports et résistances
- n'utiliser que le passé connu: aucune information future n'est utilisée
- fusionner les épisodes de range séparés par des interruptions (cluster=autre) d'une durée de 5 séances boursières ou moins: si le range reprend dans les 5 séances, l’épisode continue. Sinon, il est clos rétroactivement au début de l’interruption
- créer une nouvelle variable d'identification des range: épisode de range d'une durée minimale de 15 séances boursières, range validé seulement à partir de sa 15e séance
- période de calcul des bandes de niveaux des supports et des résistances: partir du début de l’épisode, avec au moins 15 séances et au plus les 44 dernières
- utiliser et comparer 2 méthodes pour déterminer les bandes des supports et des résistances
  - méthode 1: utilisation des quantiles, bande de support=quantiles 5 à 10% des cours bas et bande de résistance=quantile 90 à 95% des cours hauts
  - méthode 2: point de retournements regroupés, repérer les creux et sommets locaux (un point plus bas ou plus haut que ses 3 voisins de chaque côté), regrouper ceux qui sont à moins de 0,5*ATR l'un de l'autre, et retenir les niveaux touchés au moins 2 fois. Un creux ou un sommet n’est connu que 3 séances après coup. ATR calculé sur les 44 dernières séances. Règle pour choisir le support et la résistance parmi les niveaux trouvés: le niveau confirmé le plus proche sous le cours pour le support, et le plus proche au-dessus pour la résistance
  - plan de comparaison des 2 méthodes:
    - produire un tableau des épisodes de range retenus par méthode avec les informations suivantes: dates, durée, support et résistance selon les méthodes, nombre de contacts
    - mesure du taux de respect pour chaque méthode: quand le cours entre dans la zone de support, part des cas où il cloture au dessus 5 et 10 séances plus tard, contre la part des cas où il casse sous la zone
    - contrôle visuel:un graphique sur toute la période avec les zones en bandes semi transparentes