"""Étape 5 : test de performance des positions prises sur cassure confirmée d'un range (position longue uniquement).

Signal (cf. étape 4) : cassure par le haut confirmée = 2 clôtures consécutives (J1, J2) au-dessus du haut de la
bande de résistance d'au moins 0,25 × ATR(22).
Filtre de volume : max(volume J1, volume J2) >= SEUIL_VOLUME × médiane des volumes des 22 séances précédant J1.

Position :
- entrée à l'ouverture de J3 (première séance après la cassure confirmée) ;
- stop calculé avec les bandes figées à la clôture de J2 :
    S1 = bas de la bande de résistance - 0,25 × ATR(22)     (retour franc dans le range)
    S3 = milieu du range = (haut du support + bas de la résistance) / 2
    S4 = prix d'entrée - 2 × ATR(22)                        (stop de volatilité)
- stop touché si le plus bas de la séance l'atteint : sortie au stop, ou à l'ouverture si le cours ouvre dessous ;
- sorties testées (SORTIES) :
    fixe_5 / fixe_10 : sortie à la clôture de la 5e / 10e séance après la séance d'entrée ;
    T1_chandelier : stop suiveur = plus haut atteint depuis l'entrée - 3 × ATR(22) ;
    T2_canal10    : stop suiveur = plus bas des 10 dernières séances ;
  les stops suiveurs ne font que monter (jamais sous le stop initial), sont recalculés à chaque clôture et
  s'appliquent dès la séance suivante ; durée maximale DUREE_MAX_SUIVEUR séances (sortie à la clôture) ;
- frais : FRAIS_ALLER_RETOUR du prix d'entrée ; une seule position à la fois par action et par variante.
Les rendements sont calculés en cours ajustés (dividendes inclus) ; entrée et stop sont aussi donnés en cours bruts,
niveaux réellement visibles sur le marché (le jour d'un détachement de dividende, un stop en cours brut est à abaisser
du montant du dividende).

Points de comparaison :
- les mêmes cassures sans filtre de volume ;
- des entrées tirées au hasard dans la même action (hors périodes exclues), avec un stop à la même distance en ATR
  que la position réelle (N_TIRAGES tirages par position) : le signal doit faire mieux que le hasard.
Robustesse : résultats séparés pour deux moitiés de l'univers (actions A et B, en alternant dans l'ordre alphabétique)
et pour deux périodes (2020-2023 et 2024-2026).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEUIL_VOLUME = 1.5
FENETRE_VOLUME = 22
CLOTURES_CONFIRMATION = 2  # doit être identique à etape4.py
TAILLE_ATR = 22
STOPS = {"S1": "bas de résistance - 0,25 ATR", "S3": "milieu du range", "S4": "entrée - 2 ATR"}
DUREE_MAX_SUIVEUR = 60
SORTIES = {"fixe_5": ("fixe", 5), "fixe_10": ("fixe", 10),
           "T1_chandelier": ("chandelier", DUREE_MAX_SUIVEUR), "T2_canal10": ("canal", DUREE_MAX_SUIVEUR)}
MULTIPLE_CHANDELIER = 3.0
CANAL = 10
FRAIS_ALLER_RETOUR = 0.002
N_TIRAGES = 20
GRAINE = 0
ANNEE_COUPURE = 2024

DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_ENTREE = DOSSIER_DATA / "etape4.csv"
FICHIER_POSITIONS = DOSSIER_DATA / "etape5_positions.csv"
FICHIER_SYNTHESE = DOSSIER_DATA / "etape5_synthese.csv"
FICHIER_GRAPHIQUE = DOSSIER_DATA / "etape5.png"

COULEUR_SIGNAL = "#2a78d6"
COULEUR_HASARD = "#c3c2b7"
ENCRE_PRINCIPALE = "#0b0b0b"
ENCRE_DISCRETE = "#898781"
GRILLE = "#e1e0d9"


# ---------------------------------------------------------------- simulation d'une position
def simuler(o, h, b, c, atr, e: int, stop: float, sortie: str) -> tuple[float, int, bool]:
    """Position ouverte à l'ouverture de la séance e ; renvoie (prix de sortie, séance de sortie, sortie sur stop ?)."""
    mode, duree = SORTIES[sortie]
    plus_haut = -np.inf
    for d in range(e, e + duree + 1):
        if d > e and o[d] <= stop:
            return o[d], d, True
        if b[d] <= stop:
            return stop, d, True
        if mode == "chandelier":  # mise à jour à la clôture, appliquée à partir de la séance suivante
            plus_haut = max(plus_haut, h[d])
            stop = max(stop, plus_haut - MULTIPLE_CHANDELIER * atr[d])
        elif mode == "canal":
            stop = max(stop, b[max(0, d - CANAL + 1):d + 1].min())
    return c[e + duree], e + duree, False


def niveaux_stop(ligne_j2: pd.Series, entree: float, atr: float) -> dict:
    return {"S1": ligne_j2["res_bas"] - 0.25 * atr,
            "S3": (ligne_j2["sup_haut"] + ligne_j2["res_bas"]) / 2,
            "S4": entree - 2 * atr}


def traiter_action(g: pd.DataFrame, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    g = g.sort_values("date").reset_index(drop=True)
    n = len(g)
    o, h = g["ouverture_ajustee"].to_numpy(), g["haut_ajuste"].to_numpy()
    b, c = g["bas_ajuste"].to_numpy(), g["cloture_ajustee"].to_numpy()
    v = g["volume"].to_numpy(dtype=float)
    veille = g["cloture_ajustee"].shift()
    tr = pd.concat([g["haut_ajuste"] - g["bas_ajuste"], (g["haut_ajuste"] - veille).abs(),
                    (g["bas_ajuste"] - veille).abs()], axis=1).max(axis=1)
    atr = tr.rolling(TAILLE_ATR).mean().to_numpy()
    facteur = (g["cloture_ajustee"] / g["cloture_brute"]).to_numpy()
    eligibles = np.flatnonzero((g["regime"].fillna("") != "exclu").to_numpy() & ~np.isnan(atr) & ~np.isnan(o))

    positions, hasard = [], []
    occupe_jusqua = {}  # (filtre, stop, sortie) -> dernière séance occupée
    for j2 in np.flatnonzero((g["evenement"] == "cassure_haut").to_numpy()):
        j1, e = j2 - (CLOTURES_CONFIRMATION - 1), j2 + 1
        if e >= n or j1 < FENETRE_VOLUME or np.isnan(atr[j2]) or np.isnan(o[e]):
            continue
        mediane = np.median(v[j1 - FENETRE_VOLUME:j1])
        ratio_volume = v[j1:j2 + 1].max() / mediane if mediane > 0 else np.nan
        volume_ok = bool(ratio_volume >= SEUIL_VOLUME)
        entree = o[e]
        for nom, stop in niveaux_stop(g.loc[j2], entree, atr[j2]).items():
            for horizon in SORTIES:
                if e + SORTIES[horizon][1] >= n:
                    continue
                for filtre in ("sans_filtre", "volume"):
                    if filtre == "volume" and not volume_ok:
                        continue
                    cle = (filtre, nom, horizon)
                    base = {"symbole": g.at[j2, "symbole"], "date_cassure": g.at[j2, "date"].date(),
                            "date_entree": g.at[e, "date"].date(), "filtre": filtre, "stop": nom, "horizon": horizon,
                            "ratio_volume": ratio_volume, "ecart_ouverture_atr": (entree - g.at[j2, "res_haut"]) / atr[j2]}
                    if e <= occupe_jusqua.get(cle, -1):
                        positions.append({**base, "statut": "ignorée : position déjà ouverte"})
                        continue
                    if entree <= stop:
                        positions.append({**base, "statut": "ignorée : ouverture sous le stop"})
                        continue
                    sortie, jour_sortie, stoppee = simuler(o, h, b, c, atr, e, stop, horizon)
                    occupe_jusqua[cle] = jour_sortie
                    risque = entree - stop
                    positions.append({**base, "statut": "prise", "entree": entree, "stop_niveau": stop,
                                      "entree_brut": entree / facteur[e], "stop_brut": stop / facteur[e],
                                      "risque_pct": risque / entree, "sortie": sortie,
                                      "date_sortie": g.at[jour_sortie, "date"].date(), "stoppee": stoppee,
                                      "stop_initial_touche": bool(stoppee and sortie <= stop),
                                      "duree_seances": jour_sortie - e + 1,
                                      "rendement": sortie / entree - 1 - FRAIS_ALLER_RETOUR,
                                      "R": (sortie - entree) / risque - FRAIS_ALLER_RETOUR * entree / risque})
                    # entrées au hasard dans la même action, stop à la même distance en ATR
                    distance_atr = risque / atr[j2]
                    candidats = eligibles[(eligibles >= 1) & (eligibles + SORTIES[horizon][1] < n)]
                    for x in rng.choice(candidats, size=min(N_TIRAGES, len(candidats)), replace=False):
                        st = o[x] - distance_atr * atr[x - 1]
                        if o[x] <= st:
                            continue
                        s, _, _ = simuler(o, h, b, c, atr, x, st, horizon)
                        hasard.append({"filtre": filtre, "stop": nom, "horizon": horizon,
                                       "R": (s - o[x]) / (o[x] - st) - FRAIS_ALLER_RETOUR * o[x] / (o[x] - st),
                                       "rendement": s / o[x] - 1 - FRAIS_ALLER_RETOUR})
    return positions, hasard


# ---------------------------------------------------------------- synthèse
def synthese(positions: pd.DataFrame, hasard: pd.DataFrame) -> pd.DataFrame:
    prises = positions[positions["statut"] == "prise"]
    lignes = []
    for (filtre, stop, horizon), p in prises.groupby(["filtre", "stop", "horizon"]):
        hz = hasard[(hasard["filtre"] == filtre) & (hasard["stop"] == stop) & (hasard["horizon"] == horizon)]
        demi_ic = 1.96 * p["R"].std(ddof=1) / np.sqrt(len(p)) if len(p) > 1 else np.nan
        ligne = {"filtre": filtre, "stop": stop, "horizon": horizon, "positions": len(p),
                 "part_stoppees": p["stoppee"].mean(), "part_gagnantes": (p["rendement"] > 0).mean(),
                 "rendement_moyen": p["rendement"].mean(), "rendement_median": p["rendement"].median(),
                 "risque_median_pct": p["risque_pct"].median(),
                 "R_moyen": p["R"].mean(), "R_moyen_ic95_bas": p["R"].mean() - demi_ic,
                 "R_moyen_ic95_haut": p["R"].mean() + demi_ic, "R_median": p["R"].median(),
                 "R_moyen_hasard": hz["R"].mean(), "pire_R": p["R"].min(), "meilleur_R": p["R"].max(),
                 "part_R_sup_2": (p["R"] > 2).mean(), "part_stop_initial": p["stop_initial_touche"].mean(),
                 "duree_moyenne": p["duree_seances"].mean()}
        for nom, masque in [("A", p["moitie"] == "A"), ("B", p["moitie"] == "B"),
                            (f"avant_{ANNEE_COUPURE}", p["annee"] < ANNEE_COUPURE),
                            (f"depuis_{ANNEE_COUPURE}", p["annee"] >= ANNEE_COUPURE)]:
            ligne[f"R_moyen_{nom}"] = p.loc[masque, "R"].mean()
            ligne[f"n_{nom}"] = int(masque.sum())
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def tracer(synth: pd.DataFrame) -> None:
    synth = synth.sort_values(["filtre", "horizon", "stop"]).reset_index(drop=True)
    etiquettes = [f"{'volume' if r.filtre == 'volume' else 'sans filtre'}\n{r.stop}, {r.horizon}" for r in synth.itertuples()]
    x = np.arange(len(synth))
    fig, ax = plt.subplots(figsize=(18, 6), dpi=120)
    ax.bar(x - 0.2, synth["R_moyen"], width=0.38, color=COULEUR_SIGNAL, label="Cassures (R moyen, IC 95 %)")
    ax.errorbar(x - 0.2, synth["R_moyen"], yerr=[synth["R_moyen"] - synth["R_moyen_ic95_bas"],
                                                 synth["R_moyen_ic95_haut"] - synth["R_moyen"]],
                fmt="none", ecolor=ENCRE_PRINCIPALE, capsize=3, linewidth=1)
    ax.bar(x + 0.2, synth["R_moyen_hasard"], width=0.38, color=COULEUR_HASARD, label="Entrées au hasard (R moyen)")
    for xi, n in zip(x, synth["positions"]):
        ax.annotate(f"n={n}", (xi - 0.2, 0), xytext=(0, -14), textcoords="offset points", ha="center",
                    fontsize=8, color=ENCRE_DISCRETE)
    ax.axhline(0, color=ENCRE_DISCRETE, linewidth=1)
    ax.set_xticks(x, etiquettes, fontsize=8)
    ax.set_ylabel("R moyen par position (après frais)", color=ENCRE_DISCRETE)
    ax.grid(axis="y", color=GRILLE, linewidth=1)
    ax.tick_params(colors=ENCRE_DISCRETE, length=0)
    for cote in ["top", "right", "left"]:
        ax.spines[cote].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Positions sur cassure confirmée de range : gain moyen en multiples du risque initial (R), "
                 "comparé à des entrées au hasard", loc="left", fontsize=12, color=ENCRE_PRINCIPALE)
    fig.tight_layout()
    fig.savefig(FICHIER_GRAPHIQUE)
    plt.close(fig)
    print(f"Graphique enregistré dans {FICHIER_GRAPHIQUE}")


def main() -> None:
    donnees = pd.read_csv(FICHIER_ENTREE, sep=";", parse_dates=["date"])
    manquantes = {"ouverture_ajustee", "ouverture_brute"} - set(donnees.columns)
    if manquantes:
        raise RuntimeError(f"Colonnes absentes de {FICHIER_ENTREE.name} : {manquantes} — relancer les étapes 1 à 4")
    rng = np.random.default_rng(GRAINE)
    symboles = sorted(donnees["symbole"].unique())
    moitie = {s: "AB"[k % 2] for k, s in enumerate(symboles)}

    toutes, hasard = [], []
    for symbole in symboles:
        p, hz = traiter_action(donnees[donnees["symbole"] == symbole], rng)
        toutes += p
        hasard += hz
    positions = pd.DataFrame(toutes)
    if positions.empty:
        print("Aucune cassure exploitable.")
        return
    positions["moitie"] = positions["symbole"].map(moitie)
    positions["annee"] = pd.to_datetime(positions["date_entree"]).dt.year
    positions.to_csv(FICHIER_POSITIONS, sep=";", index=False, float_format="%.4f")
    hasard = pd.DataFrame(hasard)

    synth = synthese(positions, hasard)
    synth.to_csv(FICHIER_SYNTHESE, sep=";", index=False, float_format="%.4f")
    cassures = positions.drop_duplicates(["symbole", "date_cassure"])
    print(f"{len(symboles)} actions, {len(cassures)} cassures par le haut exploitables, dont "
          f"{int((cassures['ratio_volume'] >= SEUIL_VOLUME).sum())} avec le filtre de volume "
          f"(seuil {SEUIL_VOLUME} × médiane des {FENETRE_VOLUME} séances)")
    print(f"Écart médian entre l'ouverture de J3 et le haut de la résistance : "
          f"{cassures['ecart_ouverture_atr'].median():+.2f} ATR")
    colonnes = ["filtre", "stop", "horizon", "positions", "part_stop_initial", "part_gagnantes", "duree_moyenne",
                "part_R_sup_2", "rendement_moyen",
                "risque_median_pct", "R_moyen", "R_moyen_ic95_bas", "R_moyen_ic95_haut", "R_moyen_hasard",
                "R_moyen_A", "R_moyen_B", f"R_moyen_avant_{ANNEE_COUPURE}", f"R_moyen_depuis_{ANNEE_COUPURE}"]
    print(f"\nSynthèse (enregistrée dans {FICHIER_SYNTHESE}) :")
    print(synth[colonnes].round(3).to_string(index=False))
    print(f"\nDétail des positions dans {FICHIER_POSITIONS}")
    tracer(synth)


if __name__ == "__main__":
    main()
