"""Étape 3 : clustering K-means (K=3) des fenêtres glissantes de 44 séances pour mettre en évidence les patterns de range.

Un modèle (mise à l'échelle + K-means) est ajusté séparément pour chaque action : la définition du range
est relative au comportement propre de chaque titre (sa volatilité habituelle notamment).

Choix retenus (cf. README) :
- transformation log de la volatilité et du ratio de consolidation (distributions asymétriques à droite),
  pente laissée telle quelle (peut être négative) ;
- mise à l'échelle robuste (médiane / écart interquartile) des 3 features ;
- K-means : K=3, init k-means++, n_init=50, random_state=0 ;
- clusters numérotés 0, 1, 2 par pente moyenne croissante ; le cluster du milieu (1) est nommé « range »,
  les deux autres « autre » ;
- chaque fenêtre est rattachée à sa dernière séance (date_fin) : aucune information future n'est utilisée ;
- les fenêtres contenant un saut (> 30 % en une séance, cf. étapes 1 et 2) sont retirées de l'ajustement
  (mise à l'échelle et K-means) et reçoivent le régime « exclu » (pas de numéro de cluster) ;
- filtre anti-tendance après K-means : indice_tendance = |pente| × 43 / (volatilite_atr × √44), c'est-à-dire le
  déplacement net de la droite de tendance sur la fenêtre, rapporté à la variation attendue du seul hasard ;
  une fenêtre n'est « range » que si elle est dans le cluster du milieu ET si indice_tendance <= 0,5
  (environ la médiane obtenue sur des séries sans tendance). Sinon elle devient « autre » ; son numéro de
  cluster K-means est conservé pour garder la trace du filtre.
Limite connue : une tendance lente sur plusieurs mois mais faible sur chaque fenêtre de 44 séances passe le filtre.
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
CLUSTER_RANGE = 1
SEUIL_INDICE_TENDANCE = 0.5

DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_COURS = DOSSIER_DATA / "etape1.csv"
FICHIER_FEATURES = DOSSIER_DATA / "etape2.csv"
FICHIER_SORTIE = DOSSIER_DATA / "etape3.csv"
FICHIER_GRAPHIQUE = DOSSIER_DATA / "etape3.png"

COULEUR_RANGE = "#2a78d6"
COULEUR_AUTRE = "#c3c2b7"
COULEUR_EXCLU = "#eb6834"
ENCRE_PRINCIPALE = "#0b0b0b"
ENCRE_DISCRETE = "#898781"
GRILLE = "#e1e0d9"
AXE = "#c3c2b7"


def preparer_features(features: pd.DataFrame, echelle: RobustScaler | None = None) -> tuple[np.ndarray, RobustScaler]:
    x = features[FEATURES].copy()
    x[FEATURES_LOG] = np.log(x[FEATURES_LOG])
    if echelle is None:
        echelle = RobustScaler().fit(x)
    return echelle.transform(x), echelle


def indice_tendance(features: pd.DataFrame) -> pd.Series:
    """Déplacement net de la droite de régression sur la fenêtre, en unités de volatilité attendue sur la fenêtre."""
    n = 44
    return features["pente"].abs() * (n - 1) / (features["volatilite_atr"] * np.sqrt(n))


def clusteriser(features: pd.DataFrame) -> pd.DataFrame:
    """Clustering des fenêtres d'une seule action, ajusté sur les seules fenêtres sans saut."""
    sans_saut = ~features["fenetre_avec_saut"].astype(bool)
    x, _ = preparer_features(features[sans_saut])
    kmeans = KMeans(n_clusters=N_CLUSTERS, init="k-means++", n_init=N_INIT, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(x)

    # les numéros de K-means sont arbitraires : on les renumérote 0, 1, 2 par pente moyenne croissante
    ordre = features[sans_saut].assign(c=labels).groupby("c")["pente"].mean().sort_values().index
    renumerotation = {ancien: nouveau for nouveau, ancien in enumerate(ordre)}
    resultat = features.copy()
    resultat["cluster"] = pd.Series(pd.NA, index=resultat.index, dtype="Int64")
    resultat.loc[sans_saut, "cluster"] = pd.Series(labels).map(renumerotation).to_numpy()
    resultat["regime"] = resultat["cluster"].map(dict(enumerate(NOMS_REGIMES))).astype(object)
    resultat.loc[~sans_saut, "regime"] = "exclu"

    # filtre anti-tendance : le cluster du milieu n'est « range » que si la fenêtre est assez plate
    resultat["indice_tendance"] = indice_tendance(resultat)
    filtree = (resultat["cluster"] == CLUSTER_RANGE).fillna(False) & (resultat["indice_tendance"] > SEUIL_INDICE_TENDANCE)
    resultat.loc[filtree, "regime"] = "autre"

    print(f"  Fenêtres exclues (contiennent un saut) : {(~sans_saut).sum()}")
    print(f"  Fenêtres du cluster range retirées par le filtre anti-tendance : {filtree.sum()} "
          f"sur {(resultat['cluster'] == CLUSTER_RANGE).sum()}")
    print(f"  Silhouette : {silhouette_score(x, labels):.3f} (surestimée : fenêtres recouvrantes)")
    return resultat


def resume(resultat: pd.DataFrame) -> None:
    ajustees = resultat[resultat["regime"] != "exclu"]
    centres = ajustees.groupby(["cluster", "regime"])[FEATURES + ["indice_tendance"]].mean()
    centres.insert(0, "nb_fenetres", ajustees.groupby(["cluster", "regime"]).size())
    print("  Centres en unités d'origine :")
    print(centres.round(4).to_string())

    blocs = (resultat["regime"] != resultat["regime"].shift()).cumsum()
    durees = resultat.groupby(blocs).agg(regime=("regime", "first"), duree=("regime", "size"))
    print(f"  Épisodes de range bruts (avant fusion et filtre de l'étape 4) : "
          f"{(durees['regime'] == 'range').sum()}, durée moyenne "
          f"{durees.loc[durees['regime'] == 'range', 'duree'].mean():.1f} séances")
    print("  Répartition par année (part des fenêtres) :")
    print(pd.crosstab(resultat["date_fin"].dt.year, resultat["regime"], normalize="index").round(2).to_string())


def tracer_panneau(ax: plt.Axes, sortie: pd.DataFrame, symbole: str) -> None:
    """Cours de clôture ajusté : segments en bleu lorsque la fenêtre se terminant à la séance d'arrivée est un range."""
    dates = mdates.date2num(sortie.index.to_pydatetime())
    cloture = sortie["cloture_ajustee"].to_numpy()
    points = np.column_stack([dates, cloture])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    regime = sortie["regime"].iloc[1:].to_numpy()
    est_range = regime == "range"
    couleurs = np.select([est_range, regime == "exclu"], [COULEUR_RANGE, COULEUR_EXCLU], COULEUR_AUTRE)
    ax.add_collection(LineCollection(segments, colors=couleurs, linewidths=1.6, capstyle="round", joinstyle="round"))

    ax.set_title(f"{symbole} : {est_range.mean():.0%} des séances en range", loc="left", fontsize=11,
                 color=ENCRE_PRINCIPALE)
    ax.set_ylabel("€", color=ENCRE_DISCRETE, rotation=0, labelpad=12)
    ax.set_xlim(dates[0], dates[-1] + 30)
    ax.set_ylim(cloture.min() * 0.95, cloture.max() * 1.05)
    ax.grid(axis="y", color=GRILLE, linewidth=1)
    ax.grid(axis="x", which="major", color=GRILLE, linewidth=1)
    ax.tick_params(colors=ENCRE_DISCRETE, length=0)
    for cote in ["top", "right", "left"]:
        ax.spines[cote].set_visible(False)
    ax.spines["bottom"].set_color(AXE)


def tracer_graphique(sortie: pd.DataFrame) -> None:
    symboles = list(dict.fromkeys(sortie["symbole"]))
    fig, axes = plt.subplots(len(symboles), 1, figsize=(13, 3.6 * len(symboles)), dpi=120, sharex=True, squeeze=False)
    for ax, symbole in zip(axes[:, 0], symboles):
        tracer_panneau(ax, sortie[sortie["symbole"] == symbole].set_index("date"), symbole)
    axes[-1, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    legende = [Line2D([], [], color=COULEUR_RANGE, linewidth=2.5, label="Range"),
               Line2D([], [], color=COULEUR_AUTRE, linewidth=2.5, label="Autre (ou sans cluster : 44 premières séances)")]
    if (sortie["regime"] == "exclu").any():
        legende.append(Line2D([], [], color=COULEUR_EXCLU, linewidth=2.5,
                              label="Exclu : fenêtre contenant un saut de plus de 30 % en une séance"))
    axes[0, 0].legend(handles=legende, loc="upper left", frameon=False, fontsize=10, labelcolor=ENCRE_PRINCIPALE)
    fig.suptitle("Cours de clôture ajustés, régime range K-means en bleu "
                 f"(un modèle par action, filtre indice_tendance <= {SEUIL_INDICE_TENDANCE} ; fenêtre de 44 séances rattachée à sa dernière séance)",
                 x=0.01, ha="left", fontsize=12, color=ENCRE_PRINCIPALE)
    fig.tight_layout()
    fig.savefig(FICHIER_GRAPHIQUE)
    plt.close(fig)
    print(f"Graphique enregistré dans {FICHIER_GRAPHIQUE}")


def main() -> None:
    cours = pd.read_csv(FICHIER_COURS, sep=";", parse_dates=["date"])
    features = pd.read_csv(FICHIER_FEATURES, sep=";", parse_dates=["date_debut", "date_fin"])

    resultats = []
    for symbole, f in features.groupby("symbole", sort=False):
        print(f"\n=== {symbole} ===")
        resultat = clusteriser(f.reset_index(drop=True))
        resume(resultat)
        resultats.append(resultat)
    resultat = pd.concat(resultats, ignore_index=True).rename(columns={"date_fin": "date"})

    sortie = cours.merge(resultat[["symbole", "date", "id_fenetre", "cluster", "indice_tendance", "regime"]],
                         on=["symbole", "date"], how="left")
    sortie["id_fenetre"] = sortie["id_fenetre"].astype("Int64")
    sortie["cluster"] = sortie["cluster"].astype("Int64")
    sortie.to_csv(FICHIER_SORTIE, sep=";", index=False, date_format="%Y-%m-%d")
    print(f"\n{len(sortie)} lignes enregistrées dans {FICHIER_SORTIE}")
    tracer_graphique(sortie)


if __name__ == "__main__":
    main()
