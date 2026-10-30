"""Étape 4 : épisodes de range retenus, bandes de support et de résistance (quantiles), cassures et évaluation.

Toutes les règles sont appliquées séance par séance, action par action, avec le seul passé connu :
- épisode : démarre à une séance « range » (étape 3) ; une interruption (« autre » ou « exclu ») de
  MAX_INTERRUPTION séances ou moins est absorbée si le range reprend ; au-delà, l'épisode est clos
  rétroactivement à sa dernière séance « range » (motif « fin_regime ») ;
- validation : l'épisode est validé à partir de sa DUREE_MIN-ième séance, comptée depuis son début ;
- maintien par confinement du prix (MAINTIEN_PAR_CONFINEMENT) : une fois l'épisode validé, une séance qui n'est
  plus « range » à l'étape 3 ne compte pas comme interruption si sa clôture reste comprise entre le bas de la bande
  de support et le haut de la bande de résistance de la veille (sauf séance « exclu ») ; l'étape 3 ouvre
  l'épisode, le confinement du prix le maintient. Retenu car il améliore la détection sur les actions de contrôle
  (Sodexo, Solutions 30) sans augmenter la part de jours de tendance en épisode ;
- bandes (méthode des quantiles), calculées depuis le début de la fenêtre de détection (22 séances) qui a
  ouvert l'épisode — fenêtre plate grâce au filtre anti-tendance de l'étape 3 —, au plus les FENETRE_BANDES dernières : support = [q5 %, q10 %] des cours bas, résistance = [q90 %, q95 %] des cours hauts ;
  pendant une interruption, les bandes de la dernière séance « range » sont conservées (statut « pause ») ;
- fin par sortie de prix : quand CLOTURES_CONFIRMATION (2) clôtures consécutives dépassent le haut de la bande
  de résistance d'au moins MARGE_CASSURE_ATR (0,25) × ATR(22) (« cassure_haut »), ou passent sous le bas de la
  bande de support de la même marge (« cassure_bas »), l'épisode se termine à la 2e clôture (séance de cassure
  confirmée). Les bandes sont figées pendant la confirmation. Un nouvel épisode ne peut démarrer qu'après au moins
  une séance non « range » (sinon la même fenêtre de détection relancerait aussitôt un épisode) ;
- les bandes sont calculées sur les cours ajustés (cohérents avec les étapes 2 et 3) puis converties en cours
  bruts avec le facteur d'ajustement du jour (cloture_ajustee / cloture_brute) : ce sont les niveaux réellement
  visibles sur un graphique de cours, à utiliser pour les comparaisons et, plus tard, pour l'entrée et le stop.

Évaluation (regarde par construction les séances suivantes ; les bandes n'utilisent que le passé) :
- comparaison à data_test/reference.csv (ranges : date_fin = séance de cassure ; lignes « incertain » exclues) ;
- suite des cassures par le haut : rendement à 5 et 10 séances, fausse cassure (retour sous le bas de la bande
  de résistance dans les 10 séances).
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MAX_INTERRUPTION = 3
DUREE_MIN = 10
FENETRE_BANDES = 44
Q_SUPPORT = (0.05, 0.10)
Q_RESISTANCE = (0.90, 0.95)
RETRODATER_BANDES = True   # bandes calculées dès le début de la fenêtre de détection qui a ouvert l'épisode
TAILLE_FENETRE_DETECTION = 22  # doit être identique à etape2.py / etape3.py
MARGE_CASSURE_ATR = 0.25   # la clôture doit dépasser la bande d'au moins MARGE × ATR(22)
CLOTURES_CONFIRMATION = 2  # nombre de clôtures consécutives au-delà de la bande pour acter la cassure
MAINTIEN_PAR_CONFINEMENT = True   # épisode validé maintenu tant que la clôture reste dans ses bandes, même si le régime n'est plus « range »
TOLERANCE_CASSURE = 3  # séances d'écart admises entre la cassure détectée et celle de la référence
HORIZONS = (5, 10)

DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_ENTREE = DOSSIER_DATA / "etape3.csv"
FICHIER_REFERENCE = DOSSIER_DATA / "reference.csv"
FICHIER_SORTIE = DOSSIER_DATA / "etape4.csv"
FICHIER_EPISODES = DOSSIER_DATA / "etape4_episodes.csv"
FICHIER_EVALUATION = DOSSIER_DATA / "etape4_evaluation.csv"
FICHIER_CASSURES = DOSSIER_DATA / "etape4_cassures.csv"
FICHIER_GRAPHIQUE = DOSSIER_DATA / "etape4.png"

COULEUR_SUPPORT = "#2a78d6"
COULEUR_RESISTANCE = "#eb6834"
COULEUR_COURS = "#52514e"
FOND_EPISODE = "#efeee9"
COULEUR_REFERENCE = "#0b0b0b"
ENCRE_PRINCIPALE = "#0b0b0b"
ENCRE_DISCRETE = "#898781"
GRILLE = "#e1e0d9"
AXE = "#c3c2b7"

BANDES = ["sup_bas", "sup_haut", "res_bas", "res_haut"]


# ---------------------------------------------------------------- épisodes et bandes
def bandes_quantiles(bas: np.ndarray, haut: np.ndarray) -> list[float]:
    return [np.quantile(bas, Q_SUPPORT[0]), np.quantile(bas, Q_SUPPORT[1]),
            np.quantile(haut, Q_RESISTANCE[0]), np.quantile(haut, Q_RESISTANCE[1])]


def traiter_action(c: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Machine à états séance par séance pour une action (c trié par date, index 0..n-1)."""
    n = len(c)
    est_range = (c["regime"] == "range").to_numpy()
    est_exclu = (c["regime"] == "exclu").to_numpy()
    bas, haut, cloture = c["bas_ajuste"].to_numpy(), c["haut_ajuste"].to_numpy(), c["cloture_ajustee"].to_numpy()
    veille = c["cloture_ajustee"].shift()
    tr = pd.concat([c["haut_ajuste"] - c["bas_ajuste"], (c["haut_ajuste"] - veille).abs(),
                    (c["bas_ajuste"] - veille).abs()], axis=1).max(axis=1)
    atr = tr.rolling(TAILLE_FENETRE_DETECTION, min_periods=1).mean().to_numpy()

    statut = np.full(n, "", dtype=object)
    evenement = np.full(n, "", dtype=object)
    id_range = np.full(n, np.nan)
    bandes = np.full((n, len(BANDES)), np.nan)
    episodes, ep, arme, nb_valides = [], None, True, 0

    def confine(ep: dict, i: int) -> bool:
        """Épisode validé, séance non exclue et clôture comprise dans les bandes de la veille."""
        if ep["id_range"] is None or est_exclu[i] or np.isnan(bandes[i - 1, 0]):
            return False
        return bandes[i - 1, 0] <= cloture[i] <= bandes[i - 1, 3]

    def clore(fin: int, motif: str) -> None:
        if ep["id_range"] is not None:
            ep.update(fin=fin, motif_fin=motif)
            episodes.append(ep)

    for i in range(n):
        # 1. sortie de prix : comparée aux bandes connues la veille (figées pendant une confirmation en cours)
        en_attente = False
        if ep is not None and ep["id_range"] is not None and not np.isnan(bandes[i - 1, 0]):
            sup_bas, _, _, res_haut = bandes[i - 1]
            marge = MARGE_CASSURE_ATR * atr[i]
            sens = "cassure_haut" if cloture[i] > res_haut + marge else "cassure_bas" if cloture[i] < sup_bas - marge else None
            ep["sens"], ep["compte"] = sens, (ep.get("compte", 0) + 1 if sens and sens == ep.get("sens") else int(bool(sens)))
            if sens and ep["compte"] >= CLOTURES_CONFIRMATION:
                statut[i], evenement[i], id_range[i], bandes[i] = sens, sens, ep["id_range"], bandes[i - 1]
                clore(i, sens)
                ep, arme = None, False
                continue
            en_attente = bool(sens)

        # 2. évolution de l'épisode selon le régime du jour
        if ep is None:
            if not est_range[i]:
                arme = True
                continue
            if not arme:
                continue
            ep = {"debut": i, "dernier_range": i, "interruption": 0, "id_range": None, "validation": None,
                  "debut_bandes": max(0, i - TAILLE_FENETRE_DETECTION + 1) if RETRODATER_BANDES else i}
        elif est_range[i] or (MAINTIEN_PAR_CONFINEMENT and confine(ep, i)):
            ep["interruption"], ep["dernier_range"] = 0, i
        else:
            ep["interruption"] += 1
            if ep["interruption"] > MAX_INTERRUPTION:
                clore(ep["dernier_range"], "fin_regime")
                ep, arme = None, True
                continue

        if ep["interruption"] == 0 and ep["id_range"] is None and i - ep["debut"] + 1 >= DUREE_MIN:
            nb_valides += 1
            ep["id_range"], ep["validation"] = nb_valides, i

        if ep["interruption"] > 0 or (en_attente and ep["id_range"] is not None):
            statut[i] = "pause" if ep["interruption"] > 0 else "valide"
            if ep["id_range"] is not None:
                bandes[i] = bandes[i - 1]
        elif ep["id_range"] is None:
            statut[i] = "construction"
        else:
            statut[i] = "valide"
            debut_fenetre = max(ep["debut_bandes"], i - FENETRE_BANDES + 1)
            bandes[i] = bandes_quantiles(bas[debut_fenetre:i + 1], haut[debut_fenetre:i + 1])
        if ep["id_range"] is not None:
            id_range[i] = ep["id_range"]

    if ep is not None:
        clore(ep["dernier_range"], "fin_donnees")

    sortie = pd.DataFrame(bandes, columns=BANDES, index=c.index)
    facteur = (c["cloture_ajustee"] / c["cloture_brute"]).to_numpy()
    for b in BANDES:
        sortie[f"{b}_brut"] = sortie[b] / facteur
    sortie.insert(0, "evenement", evenement)
    sortie.insert(0, "id_range", id_range)
    sortie.insert(0, "statut_range", statut)
    return sortie, episodes


def tableau_episodes(c: pd.DataFrame, episodes: list[dict], sortie: pd.DataFrame) -> pd.DataFrame:
    lignes = []
    for ep in episodes:
        veille = ep["fin"] - 1 if ep["motif_fin"].startswith("cassure") else ep["fin"]
        b = sortie.loc[veille]
        lignes.append({"symbole": c["symbole"].iat[0], "id_range": ep["id_range"],
                       "date_debut": c["date"].iat[ep["debut"]].date(),
                       "date_validation": c["date"].iat[ep["validation"]].date(),
                       "date_fin": c["date"].iat[ep["fin"]].date(), "motif_fin": ep["motif_fin"],
                       "duree_seances": ep["fin"] - ep["debut"] + 1,
                       "support_brut": (b["sup_bas_brut"] + b["sup_haut_brut"]) / 2,
                       "resistance_brut": (b["res_bas_brut"] + b["res_haut_brut"]) / 2,
                       "res_haut_brut": b["res_haut_brut"]})
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------- évaluation
def evaluer_reference(donnees: pd.DataFrame) -> pd.DataFrame:
    if not FICHIER_REFERENCE.exists():
        print("Pas de fichier de référence : évaluation ignorée")
        return pd.DataFrame()
    ref = pd.read_csv(FICHIER_REFERENCE, sep=";", decimal=",", parse_dates=["date_debut", "date_fin"])
    incertain = (ref["type"] == "incertain") | ref["commentaire"].fillna("").str.contains("incertain")
    lignes = []
    for _, r in ref[~incertain].iterrows():
        g = donnees[donnees["symbole"] == r["symbole"]].reset_index(drop=True)
        ligne = {"symbole": r["symbole"], "type": r["type"], "date_debut": r["date_debut"].date(),
                 "date_fin": r["date_fin"].date(), "commentaire": r["commentaire"]}
        if g.empty:
            ligne["statut_evaluation"] = "action absente des données"
            lignes.append(ligne)
            continue
        periode = g[g["date"].between(r["date_debut"], r["date_fin"])]
        en_episode = periode["statut_range"].isin(["valide", "pause"])
        ligne["part_jours_en_episode_valide"] = en_episode.mean()
        if r["type"] == "range":
            avant = g.index[g["date"] < r["date_fin"]]
            iv = avant[-1]
            ligne["vu_la_veille_de_la_cassure"] = bool(g.at[iv, "statut_range"] in ("valide", "pause"))
            sens = "cassure_bas" if "bas" in str(r["commentaire"]) else "cassure_haut"
            ic = iv + 1
            proches = g.loc[max(ic - TOLERANCE_CASSURE, 0):ic + TOLERANCE_CASSURE]
            ligne["cassure_detectee"] = bool((proches["evenement"] == sens).any())
            if ligne["vu_la_veille_de_la_cassure"] and pd.notna(r.get("resistance")):
                ligne["resistance_reference"] = r["resistance"]
                ligne["res_haut_brut_veille"] = g.at[iv, "res_haut_brut"]
                ligne["ecart_resistance_pct"] = 100 * (g.at[iv, "res_haut_brut"] / r["resistance"] - 1)
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def suite_des_cassures(donnees: pd.DataFrame) -> pd.DataFrame:
    lignes = []
    for symbole, g in donnees.groupby("symbole", sort=False):
        g = g.reset_index(drop=True)
        for i in g.index[g["evenement"] == "cassure_haut"]:
            ligne = {"symbole": symbole, "date": g.at[i, "date"].date(), "id_range": g.at[i, "id_range"],
                     "cloture_brute": g.at[i, "cloture_brute"], "res_haut_brut": g.at[i, "res_haut_brut"]}
            for h in HORIZONS:
                ligne[f"rendement_{h}"] = (g.at[i + h, "cloture_ajustee"] / g.at[i, "cloture_ajustee"] - 1
                                           if i + h < len(g) else np.nan)
            suite = g.loc[i + 1:i + max(HORIZONS), "cloture_ajustee"]
            ligne["fausse_cassure_10"] = float((suite < g.at[i, "res_bas"]).any()) if len(suite) == max(HORIZONS) else np.nan
            lignes.append(ligne)
    return pd.DataFrame(lignes)


def afficher_resume(evaluation: pd.DataFrame, cassures: pd.DataFrame) -> None:
    if not evaluation.empty and "part_jours_en_episode_valide" in evaluation:
        e = evaluation[evaluation["part_jours_en_episode_valide"].notna()]
        r, t = e[e["type"] == "range"], e[e["type"] == "tendance"]
        print("\nComparaison à la référence :")
        print(f"  ranges vus en épisode validé la veille de la cassure : {int(r['vu_la_veille_de_la_cassure'].astype(bool).sum())} / {len(r)}")
        print(f"  cassures détectées dans le bon sens à ±{TOLERANCE_CASSURE} séances : {int(r['cassure_detectee'].astype(bool).sum())} / {len(r)}")
        if "ecart_resistance_pct" in r and r["ecart_resistance_pct"].notna().any():
            print(f"  écart médian résistance (haut de bande, brut) / référence : "
                  f"{r['ecart_resistance_pct'].median():+.1f} % (sur {r['ecart_resistance_pct'].notna().sum()} ranges)")
        print(f"  part des jours de tendance en épisode validé : {t['part_jours_en_episode_valide'].mean():.0%} ; "
              f"tendances à plus de 25 % : {(t['part_jours_en_episode_valide'] > 0.25).sum()} / {len(t)}")
        absentes = (evaluation.loc[evaluation["statut_evaluation"].notna(), "symbole"].unique()
                    if "statut_evaluation" in evaluation else [])
        if len(absentes):
            print(f"  lignes ignorées, action absente des données : {', '.join(absentes)}")
    if not cassures.empty:
        print(f"\nCassures par le haut : {len(cassures)} ; rendement médian à 5 séances "
              f"{cassures['rendement_5'].median():+.1%}, à 10 séances {cassures['rendement_10'].median():+.1%} ; "
              f"fausses cassures (retour sous la bande de résistance en 10 séances) : "
              f"{cassures['fausse_cassure_10'].astype(float).mean():.0%}")


# ---------------------------------------------------------------- graphique
def tracer_graphique(donnees: pd.DataFrame) -> None:
    ref = (pd.read_csv(FICHIER_REFERENCE, sep=";", decimal=",", parse_dates=["date_debut", "date_fin"])
           if FICHIER_REFERENCE.exists() else pd.DataFrame(columns=["symbole", "type"]))
    symboles = list(dict.fromkeys(donnees["symbole"]))
    fig, axes = plt.subplots(len(symboles), 1, figsize=(14, 3.8 * len(symboles)), dpi=120, sharex=True, squeeze=False)
    for ax, symbole in zip(axes[:, 0], symboles):
        g = donnees[donnees["symbole"] == symbole]
        en_episode = g["statut_range"].isin(["valide", "pause"])
        ax.fill_between(g["date"], 0, 1, where=en_episode, transform=ax.get_xaxis_transform(),
                        color=FOND_EPISODE, linewidth=0, step="post")
        ax.plot(g["date"], g["cloture_brute"], color=COULEUR_COURS, linewidth=1.0)
        for cote, couleur in [("sup", COULEUR_SUPPORT), ("res", COULEUR_RESISTANCE)]:
            b, h = g[f"{cote}_bas_brut"], g[f"{cote}_haut_brut"]
            ax.fill_between(g["date"], b, h, where=b.notna(), color=couleur, alpha=0.55, linewidth=0, step="post")
        for evt, marqueur in [("cassure_haut", "^"), ("cassure_bas", "v")]:
            e = g[g["evenement"] == evt]
            ax.scatter(e["date"], e["cloture_brute"], marker=marqueur, s=36, color=ENCRE_PRINCIPALE, zorder=3)
        for _, r in ref[(ref["symbole"] == symbole) & (ref["type"] == "range")].iterrows():
            ax.plot([r["date_debut"], r["date_fin"]], [0.02, 0.02], transform=ax.get_xaxis_transform(),
                    color=COULEUR_REFERENCE, linewidth=3, solid_capstyle="butt")
        ax.set_title(f"{symbole} : {int((g['evenement'] == 'cassure_haut').sum())} cassures par le haut, "
                     f"{int((g['evenement'] == 'cassure_bas').sum())} par le bas", loc="left", fontsize=11,
                     color=ENCRE_PRINCIPALE)
        ax.set_ylabel("€ (brut)", color=ENCRE_DISCRETE, labelpad=8)
        ax.grid(axis="y", color=GRILLE, linewidth=1)
        ax.tick_params(colors=ENCRE_DISCRETE, length=0)
        for cote in ["top", "right", "left"]:
            ax.spines[cote].set_visible(False)
        ax.spines["bottom"].set_color(AXE)
    axes[-1, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 0].legend(handles=[
        Line2D([], [], color=COULEUR_COURS, linewidth=1.5, label="Clôture (cours brut)"),
        Patch(color=COULEUR_SUPPORT, alpha=0.55, label="Bande de support"),
        Patch(color=COULEUR_RESISTANCE, alpha=0.55, label="Bande de résistance"),
        Patch(color=FOND_EPISODE, label="Épisode de range validé"),
        Line2D([], [], marker="^", linestyle="", color=ENCRE_PRINCIPALE, label="Cassure par le haut (▼ par le bas)"),
        Line2D([], [], color=COULEUR_REFERENCE, linewidth=3, label="Range de la référence annotée")],
        loc="upper left", frameon=False, fontsize=9, ncol=2)
    fig.suptitle("Épisodes de range, bandes de support et de résistance (quantiles) et cassures — calcul sans "
                 "information future, niveaux affichés en cours bruts", x=0.01, ha="left", fontsize=12,
                 color=ENCRE_PRINCIPALE)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(FICHIER_GRAPHIQUE)
    plt.close(fig)
    print(f"Graphique enregistré dans {FICHIER_GRAPHIQUE}")


# ---------------------------------------------------------------- programme
def main() -> None:
    donnees = pd.read_csv(FICHIER_ENTREE, sep=";", parse_dates=["date"])
    morceaux, tableaux = [], []
    for symbole, c in donnees.groupby("symbole", sort=False):
        c = c.sort_values("date").reset_index(drop=True)
        sortie, episodes = traiter_action(c)
        morceaux.append(pd.concat([c, sortie], axis=1))
        tableaux.append(tableau_episodes(c, episodes, sortie))
    donnees = pd.concat(morceaux, ignore_index=True)
    donnees["id_range"] = donnees["id_range"].astype("Int64")
    donnees.to_csv(FICHIER_SORTIE, sep=";", index=False, date_format="%Y-%m-%d")
    print(f"{len(donnees)} lignes enregistrées dans {FICHIER_SORTIE}")

    episodes = pd.concat(tableaux, ignore_index=True)
    episodes.to_csv(FICHIER_EPISODES, sep=";", index=False, float_format="%.2f")
    print(f"\n{len(episodes)} épisodes validés (enregistrés dans {FICHIER_EPISODES}) :")
    print(episodes.groupby("symbole").agg(episodes=("id_range", "size"), duree_mediane=("duree_seances", "median"))
          .join(pd.crosstab(episodes["symbole"], episodes["motif_fin"])).to_string())

    evaluation = evaluer_reference(donnees)
    if not evaluation.empty:
        evaluation.to_csv(FICHIER_EVALUATION, sep=";", index=False, float_format="%.3f")
    cassures = suite_des_cassures(donnees)
    cassures.to_csv(FICHIER_CASSURES, sep=";", index=False, float_format="%.4f")
    afficher_resume(evaluation, cassures)
    tracer_graphique(donnees)


if __name__ == "__main__":
    main()
