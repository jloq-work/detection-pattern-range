"""Étape 4 : épisodes de range retenus, bandes de supports et résistances (2 méthodes) et comparaison.

Règles (cf. README) — tout est calculé séance par séance avec le seul passé connu :
- épisode : démarre à la première séance « range » ; une interruption (« autre ») de 5 séances ou moins est
  absorbée si le range reprend ; au-delà, l'épisode est clos rétroactivement à sa dernière séance « range » ;
- validation : un épisode est validé (et reçoit un id_range) à partir de sa 15e séance, comptée depuis son début ;
- fenêtre de calcul des bandes : depuis le début de l'épisode, au plus les 44 dernières séances (au moins 15) ;
- méthode 1 (quantiles) : support = [q5 %, q10 %] des cours bas, résistance = [q90 %, q95 %] des cours hauts ;
- méthode 2 (points de retournement) : creux / sommets plus bas / plus hauts que leurs 3 voisins de chaque côté,
  connus 3 séances après coup ; regroupés si l'écart avec le premier point du groupe est <= 0,5 × ATR(44) ;
  niveaux retenus si touchés au moins 2 fois ; support = niveau le plus proche sous la clôture,
  résistance = niveau le plus proche au-dessus. La bande d'un niveau va du plus bas au plus haut de ses points.

Choix d'implémentation à valider :
- statut « pause » pendant une interruption d'un épisode validé : les bandes de la dernière séance « valide »
  sont conservées, aucune nouvelle bande n'est calculée et aucun événement n'est compté ;
- creux et sommets sont mis en commun pour la méthode 2 (un ancien sommet sous le cours peut servir de support) ;
- taux de respect : une entrée dans la zone de support est une séance dont le plus bas touche le haut de la bande
  calculée la veille alors que la clôture de la veille était au-dessus ; à 5 et 10 séances, la clôture est
  « au-dessus » (> haut de bande), « cassure » (< bas de bande) ou « dans la zone ». Cette mesure d'évaluation
  regarde par construction les séances suivantes ; les bandes, elles, n'utilisent que le passé.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

MAX_INTERRUPTION = 5
DUREE_MIN = 15
FENETRE_MAX = 44
Q_SUPPORT = (0.05, 0.10)
Q_RESISTANCE = (0.90, 0.95)
VOISINS = 3
TOLERANCE_ATR = 0.5
CONTACTS_MIN = 2
HORIZONS = (5, 10)

DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_ENTREE = DOSSIER_DATA / "etape3.csv"
FICHIER_SORTIE = DOSSIER_DATA / "etape4.csv"
FICHIER_EPISODES = DOSSIER_DATA / "etape4_episodes.csv"
FICHIER_RESPECT = DOSSIER_DATA / "etape4_respect.csv"
FICHIER_GRAPHIQUE = DOSSIER_DATA / "etape4.png"

COULEUR_SUPPORT = "#2a78d6"
COULEUR_RESISTANCE = "#eb6834"
COULEUR_COURS = "#52514e"
FOND_EPISODE = "#efeee9"
ENCRE_PRINCIPALE = "#0b0b0b"
ENCRE_DISCRETE = "#898781"
GRILLE = "#e1e0d9"
AXE = "#c3c2b7"

COLONNES_BANDES = ["sup_bas", "sup_haut", "sup_contacts", "res_bas", "res_haut", "res_contacts"]


# ---------------------------------------------------------------- épisodes
def construire_episodes(regime: pd.Series) -> tuple[pd.DataFrame, list[dict]]:
    """Machine à états séance par séance : statut, id_range et début d'épisode connus à chaque date."""
    n = len(regime)
    est_range = (regime == "range").to_numpy()
    statut = np.full(n, "", dtype=object)
    id_range = np.full(n, np.nan)
    debut = np.full(n, -1)
    episodes, courant, nb_valides = [], None, 0

    def clore(ep: dict) -> None:
        if ep["id_range"] is not None:
            episodes.append(ep)

    for i in range(n):
        if est_range[i]:
            if courant is None:
                courant = {"debut": i, "dernier_range": i, "interruption": 0, "id_range": None, "validation": None}
            else:
                courant["interruption"] = 0
                courant["dernier_range"] = i
        elif courant is not None:
            courant["interruption"] += 1
            if courant["interruption"] > MAX_INTERRUPTION:
                clore(courant)
                courant = None
        if courant is None:
            continue

        debut[i] = courant["debut"]
        if courant["interruption"] == 0 and courant["id_range"] is None and i - courant["debut"] + 1 >= DUREE_MIN:
            nb_valides += 1
            courant["id_range"], courant["validation"] = nb_valides, i
        if courant["interruption"] > 0:
            statut[i] = "pause"
        else:
            statut[i] = "valide" if courant["id_range"] is not None else "construction"
        if courant["id_range"] is not None:
            id_range[i] = courant["id_range"]

    if courant is not None:
        clore(courant)
    etat = pd.DataFrame({"statut_range": statut, "id_range": id_range, "debut_pos": debut}, index=regime.index)
    return etat, episodes


# ---------------------------------------------------------------- bandes
def atr(cours: pd.DataFrame, fenetre: int) -> pd.Series:
    veille = cours["cloture_ajustee"].shift()
    tr = pd.concat([cours["haut_ajuste"] - cours["bas_ajuste"], (cours["haut_ajuste"] - veille).abs(),
                    (cours["bas_ajuste"] - veille).abs()], axis=1).max(axis=1, skipna=False)
    return tr.rolling(fenetre).mean()


def points_de_retournement(serie: np.ndarray, creux: bool) -> np.ndarray:
    """Booléen : point strictement plus bas (creux) ou plus haut (sommet) que ses VOISINS voisins de chaque côté."""
    n = len(serie)
    res = np.zeros(n, dtype=bool)
    for j in range(VOISINS, n - VOISINS):
        voisins = np.r_[serie[j - VOISINS:j], serie[j + 1:j + 1 + VOISINS]]
        res[j] = serie[j] < voisins.min() if creux else serie[j] > voisins.max()
    return res


def bandes_quantiles(bas: np.ndarray, haut: np.ndarray) -> dict:
    return {"sup_bas": np.quantile(bas, Q_SUPPORT[0]), "sup_haut": np.quantile(bas, Q_SUPPORT[1]),
            "sup_contacts": np.nan,
            "res_bas": np.quantile(haut, Q_RESISTANCE[0]), "res_haut": np.quantile(haut, Q_RESISTANCE[1]),
            "res_contacts": np.nan}


def bandes_retournements(niveaux: np.ndarray, cloture: float, tolerance: float) -> dict:
    vide = dict.fromkeys(COLONNES_BANDES, np.nan)
    groupes, groupe = [], []
    for v in np.sort(niveaux):
        if groupe and v - groupe[0] > tolerance:
            groupes.append(groupe)
            groupe = []
        groupe.append(v)
    if groupe:
        groupes.append(groupe)
    groupes = [g for g in groupes if len(g) >= CONTACTS_MIN]

    dessous = [g for g in groupes if np.mean(g) < cloture]
    dessus = [g for g in groupes if np.mean(g) > cloture]
    if dessous:
        g = max(dessous, key=np.mean)
        vide.update(sup_bas=min(g), sup_haut=max(g), sup_contacts=len(g))
    if dessus:
        g = min(dessus, key=np.mean)
        vide.update(res_bas=min(g), res_haut=max(g), res_contacts=len(g))
    return vide


def calculer_bandes(cours: pd.DataFrame, etat: pd.DataFrame) -> pd.DataFrame:
    bas, haut = cours["bas_ajuste"].to_numpy(), cours["haut_ajuste"].to_numpy()
    cloture = cours["cloture_ajustee"].to_numpy()
    atr44 = atr(cours, FENETRE_MAX).to_numpy()
    est_creux, est_sommet = points_de_retournement(bas, True), points_de_retournement(haut, False)
    statut, debut = etat["statut_range"].to_numpy(), etat["debut_pos"].to_numpy()

    m1 = np.full((len(cours), len(COLONNES_BANDES)), np.nan)
    m2 = np.full_like(m1, np.nan)
    for i in range(len(cours)):
        if statut[i] == "pause" and not np.isnan(etat["id_range"].iat[i]):
            m1[i], m2[i] = m1[i - 1], m2[i - 1]  # bandes figées pendant l'interruption
            continue
        if statut[i] != "valide":
            continue
        debut_fenetre = max(debut[i], i - FENETRE_MAX + 1)
        m1[i] = list(bandes_quantiles(bas[debut_fenetre:i + 1], haut[debut_fenetre:i + 1]).values())

        # points confirmés : dans la fenêtre et connus (j + VOISINS <= i)
        j = np.arange(debut_fenetre, i - VOISINS + 1)
        niveaux = np.r_[bas[j][est_creux[j]], haut[j][est_sommet[j]]]
        m2[i] = list(bandes_retournements(niveaux, cloture[i], TOLERANCE_ATR * atr44[i]).values())

    bandes = pd.concat([pd.DataFrame(m1, index=cours.index, columns=[f"m1_{c}" for c in COLONNES_BANDES]),
                        pd.DataFrame(m2, index=cours.index, columns=[f"m2_{c}" for c in COLONNES_BANDES])], axis=1)
    bandes.insert(0, "atr44", atr44)
    return bandes.drop(columns=["m1_sup_contacts", "m1_res_contacts"])


# ---------------------------------------------------------------- évaluation
def evenements(cours: pd.DataFrame, etat: pd.DataFrame, bandes: pd.DataFrame, methode: str) -> pd.DataFrame:
    """Entrées dans la zone de support (bandes de la veille) et issue à 5 et 10 séances."""
    cloture, bas, haut = cours["cloture_ajustee"], cours["bas_ajuste"], cours["haut_ajuste"]
    sup_bas, sup_haut = bandes[f"{methode}_sup_bas"].shift(), bandes[f"{methode}_sup_haut"].shift()
    res_bas = bandes[f"{methode}_res_bas"].shift()
    valide_veille = etat["statut_range"].shift() == "valide"

    entree_sup = valide_veille & (cloture.shift() > sup_haut) & (bas <= sup_haut)
    entree_res = valide_veille & (cloture.shift() < res_bas) & (haut >= res_bas)

    ev = pd.DataFrame({"id_range": etat["id_range"].shift(), "sup_bas": sup_bas, "sup_haut": sup_haut})[entree_sup]
    for h in HORIZONS:
        futur = cloture.shift(-h)[entree_sup]
        ev[f"issue_{h}"] = np.select([futur.isna(), futur > ev["sup_haut"], futur < ev["sup_bas"]],
                                     ["indisponible", "au_dessus", "cassure"], "dans_zone")
    ev.attrs["entrees_resistance"] = etat["id_range"].shift()[entree_res]
    return ev


def taux_respect(ev_par_methode: dict) -> pd.DataFrame:
    lignes = []
    for methode, ev in ev_par_methode.items():
        for h in HORIZONS:
            issues = ev[f"issue_{h}"][ev[f"issue_{h}"] != "indisponible"]
            n = len(issues)
            lignes.append({"methode": methode, "horizon": h, "nb_entrees": n,
                           "part_au_dessus": (issues == "au_dessus").mean() if n else np.nan,
                           "part_dans_zone": (issues == "dans_zone").mean() if n else np.nan,
                           "part_cassure": (issues == "cassure").mean() if n else np.nan})
    return pd.DataFrame(lignes)


def tableau_episodes(cours: pd.DataFrame, episodes: list[dict], bandes: pd.DataFrame, etat: pd.DataFrame,
                     ev_par_methode: dict) -> pd.DataFrame:
    dates = cours.index
    lignes = []
    for ep in episodes:
        k = ep["id_range"]
        jours_valides = (etat["id_range"] == k) & (etat["statut_range"] == "valide")
        ligne = {"id_range": k, "date_debut": dates[ep["debut"]].date(), "date_validation": dates[ep["validation"]].date(),
                 "date_fin": dates[ep["dernier_range"]].date(), "duree_seances": ep["dernier_range"] - ep["debut"] + 1,
                 "seances_valides": int(jours_valides.sum())}
        for m, ev in ev_par_methode.items():
            b = bandes[jours_valides]
            ligne[f"{m}_support_median"] = ((b[f"{m}_sup_bas"] + b[f"{m}_sup_haut"]) / 2).median()
            ligne[f"{m}_resistance_median"] = ((b[f"{m}_res_bas"] + b[f"{m}_res_haut"]) / 2).median()
            ligne[f"{m}_contacts_support"] = int((ev["id_range"] == k).sum())
            ligne[f"{m}_contacts_resistance"] = int((ev.attrs["entrees_resistance"] == k).sum())
            if m == "m2":
                ligne["m2_part_jours_sans_support"] = b["m2_sup_bas"].isna().mean()
                ligne["m2_part_jours_sans_resistance"] = b["m2_res_bas"].isna().mean()
        lignes.append(ligne)
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------- graphique
def tracer_graphique(cours: pd.DataFrame, bandes: pd.DataFrame, etat: pd.DataFrame) -> None:
    titres = {"m1": "Méthode 1 : quantiles (support q5–q10 % des bas, résistance q90–q95 % des hauts)",
              "m2": "Méthode 2 : points de retournement regroupés (≥ 2 contacts, tolérance 0,5 × ATR44)"}
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=120, sharex=True)
    for ax, m in zip(axes, ["m1", "m2"]):
        # fond grisé : séances où un épisode de range validé est en cours (statut valide ou pause)
        en_cours = etat["id_range"].notna()
        ax.fill_between(cours.index, 0, 1, where=en_cours, transform=ax.get_xaxis_transform(),
                        color=FOND_EPISODE, linewidth=0, step="post")
        ax.plot(cours.index, cours["cloture_ajustee"], color=COULEUR_COURS, linewidth=1.1)
        for cote, couleur in [("sup", COULEUR_SUPPORT), ("res", COULEUR_RESISTANCE)]:
            b, h = bandes[f"{m}_{cote}_bas"], bandes[f"{m}_{cote}_haut"]
            ax.fill_between(cours.index, b, h, where=b.notna(), color=couleur, alpha=0.55, linewidth=0, step="post")
        ax.set_title(titres[m], loc="left", fontsize=11, color=ENCRE_PRINCIPALE)
        ax.set_ylabel("€", color=ENCRE_DISCRETE, rotation=0, labelpad=12)
        ax.grid(axis="y", color=GRILLE, linewidth=1)
        ax.tick_params(colors=ENCRE_DISCRETE, length=0)
        for c in ["top", "right", "left"]:
            ax.spines[c].set_visible(False)
        ax.spines["bottom"].set_color(AXE)
    axes[0].legend(handles=[Line2D([], [], color=COULEUR_COURS, linewidth=1.5, label="Clôture ajustée"),
                            Patch(color=COULEUR_SUPPORT, alpha=0.55, label="Bande de support"),
                            Patch(color=COULEUR_RESISTANCE, alpha=0.55, label="Bande de résistance"),
                            Patch(color=FOND_EPISODE, label="Épisode de range validé en cours")],
                   loc="upper left", frameon=False, fontsize=10)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("Veolia SA (VIE.PA) : supports et résistances des épisodes de range validés (calcul sans information future)",
                 x=0.01, ha="left", fontsize=13, color=ENCRE_PRINCIPALE)
    fig.tight_layout()
    fig.savefig(FICHIER_GRAPHIQUE)
    plt.close(fig)
    print(f"Graphique enregistré dans {FICHIER_GRAPHIQUE}")


# ---------------------------------------------------------------- programme
def main() -> None:
    cours = pd.read_csv(FICHIER_ENTREE, sep=";", parse_dates=["date"], index_col="date")
    etat, episodes = construire_episodes(cours["regime"])
    bandes = calculer_bandes(cours, etat)
    ev = {m: evenements(cours, etat, bandes, m) for m in ["m1", "m2"]}

    sortie = pd.concat([cours, etat.drop(columns="debut_pos"), bandes], axis=1)
    sortie["id_range"] = sortie["id_range"].astype("Int64")
    for c in ["m2_sup_contacts", "m2_res_contacts"]:
        sortie[c] = sortie[c].astype("Int64")
    sortie.to_csv(FICHIER_SORTIE, sep=";", date_format="%Y-%m-%d")
    print(f"{len(sortie)} séances enregistrées dans {FICHIER_SORTIE}")

    tableau = tableau_episodes(cours, episodes, bandes, etat, ev)
    tableau.to_csv(FICHIER_EPISODES, sep=";", index=False, float_format="%.2f")
    print(f"\n{len(tableau)} épisodes de range validés (enregistrés dans {FICHIER_EPISODES}) :")
    print(tableau.round(2).to_string(index=False))

    respect = taux_respect(ev)
    respect.to_csv(FICHIER_RESPECT, sep=";", index=False, float_format="%.3f")
    print(f"\nTaux de respect des supports (enregistré dans {FICHIER_RESPECT}) :")
    print(respect.round(2).to_string(index=False))

    tracer_graphique(cours, bandes, etat)


if __name__ == "__main__":
    main()
