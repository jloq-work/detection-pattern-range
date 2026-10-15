"""Étape 3 : clustering K-means (K=3) des fenêtres glissantes de 44 séances pour mettre en évidence les patterns de range.

Choix retenus (cf. README) :
- transformation log de la volatilité et du ratio de consolidation (distributions asymétriques à droite),
  pente laissée telle quelle (peut être négative) ;
- mise à l'échelle robuste (médiane / écart interquartile) des 3 features ;
- K-means : K=3, init k-means++, n_init=50, random_state=0 ;
- clusters numérotés 0, 1, 2 par pente moyenne croissante ; le cluster du milieu (1) est nommé « range »,
  les deux autres « autre » ;
- chaque fenêtre est rattachée à sa dernière séance (date_fin) : aucune information future n'est utilisée.
Limite connue et acceptée : le cluster « range » regroupe aussi les tendances lentes en période de faible volatilité.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

N_CLUSTERS = 3
N_INIT = 50
RANDOM_STATE = 0
FEATURES = ["pente", "volatilite_atr", "ratio_consolidation"]
FEATURES_LOG = ["volatilite_atr", "ratio_consolidation"]
NOMS_REGIMES = ["autre", "range", "autre"]  # par pente moyenne croissante

DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_COURS = DOSSIER_DATA / "etape1.csv"
FICHIER_FEATURES = DOSSIER_DATA / "etape2.csv"
FICHIER_SORTIE = DOSSIER_DATA / "etape3.csv"
FICHIER_GRAPHIQUE = DOSSIER_DATA / "etape3.png"

COULEUR_RANGE = "#2a78d6"
COULEUR_AUTRE = "#c3c2b7"
ENCRE_PRINCIPALE = "#0b0b0b"
ENCRE_DISCRETE = "#898781"
GRILLE = "#e1e0d9"
AXE = "#c3c2b7"


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

    blocs = (resultat["regime"] != resultat["regime"].shift()).cumsum()
    durees = resultat.groupby(blocs).agg(regime=("regime", "first"), duree=("regime", "size"))
    print(f"\nÉpisodes de range bruts (avant fusion et filtre de l'étape 4) : "
          f"{(durees['regime'] == 'range').sum()}, durée moyenne "
          f"{durees.loc[durees['regime'] == 'range', 'duree'].mean():.1f} séances")
    print("\nRépartition par année (part des fenêtres) :")
    print(pd.crosstab(resultat["date_fin"].dt.year, resultat["regime"], normalize="index").round(2).to_string())


def tracer_graphique(sortie: pd.DataFrame) -> None:
    """Cours de clôture ajusté : segments en bleu lorsque la fenêtre se terminant à la séance d'arrivée est un range."""
    dates = mdates.date2num(sortie.index.to_pydatetime())
    cloture = sortie["cloture_ajustee"].to_numpy()
    points = np.column_stack([dates, cloture])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    est_range = (sortie["regime"].iloc[1:] == "range").to_numpy()
    couleurs = np.where(est_range, COULEUR_RANGE, COULEUR_AUTRE)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=120)
    ax.add_collection(LineCollection(segments, colors=couleurs, linewidths=2, capstyle="round", joinstyle="round"))

    legende = [Line2D([], [], color=COULEUR_RANGE, linewidth=2.5, label="Range"),
               Line2D([], [], color=COULEUR_AUTRE, linewidth=2.5, label="Autre (ou sans cluster : 44 premières séances)")]
    ax.legend(handles=legende, loc="upper left", frameon=False, fontsize=10, labelcolor=ENCRE_PRINCIPALE)

    ax.set_title("Veolia SA (VIE.PA) : cours de clôture ajusté, régime range K-means en bleu "
                 "(fenêtre de 44 séances rattachée à sa dernière séance)",
                 loc="left", fontsize=12, color=ENCRE_PRINCIPALE, pad=12)
    ax.set_ylabel("€", color=ENCRE_DISCRETE, rotation=0, labelpad=12)
    ax.set_xlim(dates[0], dates[-1] + 30)
    ax.set_ylim(cloture.min() * 0.95, cloture.max() * 1.05)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=GRILLE, linewidth=1)
    ax.grid(axis="x", which="major", color=GRILLE, linewidth=1)
    ax.tick_params(colors=ENCRE_DISCRETE, length=0)
    ax.tick_params(axis="x", which="minor", length=3, color=AXE)
    for cote in ["top", "right", "left"]:
        ax.spines[cote].set_visible(False)
    ax.spines["bottom"].set_color(AXE)

    fig.tight_layout()
    fig.savefig(FICHIER_GRAPHIQUE)
    plt.close(fig)
    print(f"Graphique enregistré dans {FICHIER_GRAPHIQUE}")


def main() -> None:
    cours = pd.read_csv(FICHIER_COURS, sep=";", parse_dates=["date"], index_col="date")
    features = pd.read_csv(FICHIER_FEATURES, sep=";", parse_dates=["date_debut", "date_fin"])

    resultat = clusteriser(features)
    resume(resultat)

    sortie = cours.join(resultat.set_index("date_fin")[["id_fenetre", "cluster", "regime"]], how="left")
    sortie["id_fenetre"] = sortie["id_fenetre"].astype("Int64")
    sortie["cluster"] = sortie["cluster"].astype("Int64")
    sortie.to_csv(FICHIER_SORTIE, sep=";", date_format="%Y-%m-%d")
    print(f"\n{len(sortie)} séances enregistrées dans {FICHIER_SORTIE} "
          f"({sortie['cluster'].isna().sum()} premières séances sans cluster)")
    tracer_graphique(sortie)


if __name__ == "__main__":
    main()
