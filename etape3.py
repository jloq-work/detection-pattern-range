"""Étape 3 : clustering K-means (K=3) des fenêtres glissantes de 44 séances de Veolia SA.

Choix retenus (cf. diagnostic) :
- transformation log de la volatilité et du ratio de consolidation (distributions asymétriques à droite),
  pente laissée telle quelle (symétrique, signée) ;
- mise à l'échelle robuste (médiane / écart interquartile) des 3 features, moins sensible aux extrêmes 2020 et 2022 ;
- K-means : K=3, init k-means++, n_init=50, random_state fixé pour la reproductibilité ;
- nommage des clusters par la pente moyenne : la plus faible = baissière, intermédiaire = range, la plus forte = haussière ;
- chaque fenêtre est rattachée à sa dernière séance (date_fin) : aucune information future n'est utilisée.
Limite connue et acceptée : le cluster « range » regroupe aussi les tendances lentes en période de faible volatilité.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

N_CLUSTERS = 3
N_INIT = 50
RANDOM_STATE = 0
FEATURES = ["pente", "volatilite_atr", "ratio_consolidation"]
FEATURES_LOG = ["volatilite_atr", "ratio_consolidation"]
NOMS_REGIMES = ["baissiere", "range", "haussiere"]  # par pente moyenne croissante

DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_COURS = DOSSIER_DATA / "veolia_cours.csv"
FICHIER_FEATURES = DOSSIER_DATA / "veolia_features.csv"
FICHIER_CLUSTERS = DOSSIER_DATA / "veolia_clusters.csv"


def preparer_features(features: pd.DataFrame) -> np.ndarray:
    x = features[FEATURES].copy()
    x[FEATURES_LOG] = np.log(x[FEATURES_LOG])
    return RobustScaler().fit_transform(x)


def clusteriser(features: pd.DataFrame) -> pd.DataFrame:
    x = preparer_features(features)
    kmeans = KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=N_INIT, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(x)

    # les numéros de K-means sont arbitraires : on les renumérote 0, 1, 2 par pente moyenne croissante
    ordre = features.assign(c=labels).groupby("c")["pente"].mean().sort_values().index
    renumerotation = {ancien: nouveau for nouveau, ancien in enumerate(ordre)}
    resultat = features.copy()
    resultat["cluster"] = pd.Series(labels).map(renumerotation).to_numpy()
    resultat["regime"] = resultat["cluster"].map(dict(enumerate(NOMS_REGIMES)))

    print(f"Silhouette : {silhouette_score(x, labels):.3f} (surestimée : fenêtres recouvrantes)")
    return resultat


def resume(resultat: pd.DataFrame) -> None:
    centres = resultat.groupby(["cluster", "regime"])[FEATURES].mean()
    centres.insert(0, "nb_fenetres", resultat.groupby(["cluster", "regime"]).size())
    print("\nCentres en unités d'origine :")
    print(centres.round(4).to_string())

    blocs = (resultat["cluster"].diff() != 0).cumsum()
    print(f"\nDurée moyenne d'un régime : {resultat.groupby(blocs).size().mean():.1f} séances")
    print("\nRépartition par année (part des fenêtres) :")
    print(pd.crosstab(resultat["date_fin"].dt.year, resultat["regime"], normalize="index").round(2).to_string())


def main() -> None:
    cours = pd.read_csv(FICHIER_COURS, sep=";", parse_dates=["date"], index_col="date")
    features = pd.read_csv(FICHIER_FEATURES, sep=";", parse_dates=["date_debut", "date_fin"])

    resultat = clusteriser(features)
    resume(resultat)

    sortie = cours.join(
        resultat.set_index("date_fin")[["id_fenetre", "cluster", "regime"]], how="left"
    )
    sortie["id_fenetre"] = sortie["id_fenetre"].astype("Int64")
    sortie["cluster"] = sortie["cluster"].astype("Int64")
    sortie.to_csv(FICHIER_CLUSTERS, sep=";", date_format="%Y-%m-%d")
    print(f"\n{len(sortie)} séances enregistrées dans {FICHIER_CLUSTERS} "
          f"({sortie['cluster'].isna().sum()} premières séances sans cluster)")


if __name__ == "__main__":
    main()
