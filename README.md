## Projet test: système d'aide à la prise de position sur actions françaises (position longue uniquement) 
L'idée est de tester sur l'action française Veolia SA la détection de pattern de range sur la période avril 2020 à septembre 2026.
La détection est réalisée par clusterisation sur des fenêtres glissantes de 2 mois de cotations.
3 clusters K-means sont créés avec 3 features: la pente de la tendance, la volatilité normalisée, le ratio de consolidation des prix

#etape 1:
-récupérer l'action Veolia SA (symbole Yahoo: VIE.PA) via Yahoo finance et l'utilisation de la library yfinance
-récupérer le cours de clôture journalier ajusté
-récupérer le cours depuis le 01/04/2020 jusqu'au 30/09/2026
-à partir de ce cours: créer des variables dans un dataframe
-variable 1: un identifiant qui représente une fenêtre glissante de 44 séances boursières
-variable 2: un premier feature, le coefficient de la tendance de la regression linéaire du cours de l'action sur la fenêtre de 44 séances, une valeur par identifiant donc
-variable 3: un second feature
-filter les actions pour lesquelles le montant en euros du volume moyen échangé est supérieur ou égal à 10 000 000 d'euros (moyenne calculé sur 6 mois de cotation)
-détecter les patterns de range: 2 conditions à respecter sur au moins 21 jours d'ouverture de la bourse (période>=21 jours)
	-condition 1: le prix de l'action doit osciller entre -3;+3% par rapport au prix moyen de l'action sur la période
	AND
	-condition 2: le volume d'échange de titres de l'action doit osciller entre -5;+5% par rapport au volume moyen sur la période
-récupérer les données dans un fichier csv: séparateur ";"