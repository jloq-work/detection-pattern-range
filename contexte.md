 
##étape 1:
-récupérer 10 actions côtées au Service de Réglèment différé via Yahoo finance et l'utilisation de la library yfinance
-je veux le cours de clôture ajusté
-les symboles des 10 actions: AF.PA, CS.PA, BN.PA, GET.PA, MC.PA, ORA.PA, RNO.PA, HO.PA, GLE.PA, VIE.PA 
-filter les actions pour lesquelles le montant en euros du volume moyen échangé est supérieur ou égal à 10 000 000 d'euros (moyenne calculé sur 6 mois de cotation)
-détecter les patterns de range: 2 conditions à respecter sur au moins 21 jours d'ouverture de la bourse (période>=21 jours)
	-condition 1: le prix de l'action doit osciller entre -3;+3% par rapport au prix moyen de l'action sur la période
	AND
	-condition 2: le volume d'échange de titres de l'action doit osciller entre -5;+5% par rapport au volume moyen sur la période
-récupérer les données dans un fichier csv: séparateur ";"