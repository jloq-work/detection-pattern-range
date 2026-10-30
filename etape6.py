"""Étape 6 (piste C) : entrée sur retour vers la résistance après une cassure confirmée — moitié A de l'univers seulement.

Règle fixée avant tout calcul :
- signal : cassure par le haut confirmée (étape 4), avec ou sans le filtre de volume de l'étape 5 ;
- ordre d'achat à cours limité au haut de la bande de résistance (figée à J2), valable de J3 à J3 + 9 (10 séances) :
  exécuté à l'ouverture si elle est au niveau de l'ordre ou en dessous, sinon au niveau de l'ordre si le plus bas
  l'atteint ; annulé si une clôture passe sous le stop avant exécution (configuration invalidée) ;
- stops initiaux S1 / S3 / S4 de l'étape 5 (S4 calculé depuis le prix d'achat) ; le jour de l'achat, si le plus bas
  atteint aussi le stop, la position est considérée stoppée (hypothèse défavorable : ordre des prix inconnu) ;
- sorties : fixe_10 et T1_chandelier (cf. etape5.py) ; frais 0,2 % aller-retour ; une position à la fois par action.
Comparaisons : entrées au hasard (même action, même distance de stop en ATR, même sortie) et entrée à la cassure
(ouverture de J3, règles de l'étape 5), calculées sur la même moitié A.
La moitié B de l'univers (une action sur deux dans l'ordre alphabétique) est réservée : elle n'est ni chargée ni
calculée ici ; elle ne servira qu'une fois, pour confirmer une règle déjà choisie.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import etape5 as e5

VALIDITE_ORDRE = 10
SORTIES_TESTEES = ["fixe_10", "T1_chandelier"]
STOPS_TESTES = ["S1", "S3", "S4"]
MOITIE = "A"

DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_ENTREE = DOSSIER_DATA / "etape4.csv"
FICHIER_POSITIONS = DOSSIER_DATA / "etape6_positions.csv"
FICHIER_SYNTHESE = DOSSIER_DATA / "etape6_synthese.csv"


def symboles_moitie(symboles, moitie: str) -> list[str]:
    tries = sorted(symboles)
    return [s for k, s in enumerate(tries) if "AB"[k % 2] == moitie]


def traiter_action(g: pd.DataFrame, rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    g = g.sort_values("date").reset_index(drop=True)
    n = len(g)
    o, h = g["ouverture_ajustee"].to_numpy(), g["haut_ajuste"].to_numpy()
    b, c = g["bas_ajuste"].to_numpy(), g["cloture_ajustee"].to_numpy()
    v = g["volume"].to_numpy(dtype=float)
    veille = g["cloture_ajustee"].shift()
    tr = pd.concat([g["haut_ajuste"] - g["bas_ajuste"], (g["haut_ajuste"] - veille).abs(),
                    (g["bas_ajuste"] - veille).abs()], axis=1).max(axis=1)
    atr = tr.rolling(e5.TAILLE_ATR).mean().to_numpy()
    eligibles = np.flatnonzero((g["regime"].fillna("") != "exclu").to_numpy() & ~np.isnan(atr) & ~np.isnan(o))

    positions, hasard, occupe = [], [], {}
    for j2 in np.flatnonzero((g["evenement"] == "cassure_haut").to_numpy()):
        j1, j3 = j2 - (e5.CLOTURES_CONFIRMATION - 1), j2 + 1
        if j3 + VALIDITE_ORDRE + e5.DUREE_MAX_SUIVEUR >= n or j1 < e5.FENETRE_VOLUME or np.isnan(atr[j2]):
            continue
        mediane = np.median(v[j1 - e5.FENETRE_VOLUME:j1])
        volume_ok = bool(mediane > 0 and v[j1:j2 + 1].max() / mediane >= e5.SEUIL_VOLUME)
        limite = g.at[j2, "res_haut"]
        for filtre in ("sans_filtre", "volume"):
            if filtre == "volume" and not volume_ok:
                continue
            for nom in STOPS_TESTES:
                for sortie in SORTIES_TESTEES:
                    cle = (filtre, nom, sortie)
                    base = {"symbole": g.at[j2, "symbole"], "date_cassure": g.at[j2, "date"].date(),
                            "filtre": filtre, "stop": nom, "sortie": sortie}
                    if j3 <= occupe.get(cle, -1):
                        positions.append({**base, "statut": "ignorée : position déjà ouverte"})
                        continue
                    # recherche de l'exécution de l'ordre à cours limité
                    execution, prix, statut = None, np.nan, "non exécutée"
                    for d in range(j3, j3 + VALIDITE_ORDRE):
                        prix_prevu = o[d] if o[d] <= limite else limite
                        stop = (e5.niveaux_stop(g.loc[j2], prix_prevu, atr[j2])[nom])
                        if o[d] <= limite or b[d] <= limite:
                            execution, prix = d, prix_prevu
                            break
                        if c[d] < stop:
                            statut = "annulée : clôture sous le stop"
                            break
                    if execution is None:
                        positions.append({**base, "statut": statut})
                        continue
                    stop = e5.niveaux_stop(g.loc[j2], prix, atr[j2])[nom]
                    if prix <= stop:
                        positions.append({**base, "statut": "annulée : exécution sous le stop"})
                        continue
                    risque = prix - stop
                    if b[execution] <= stop:  # hypothèse défavorable le jour de l'achat
                        sortie_prix, jour, stoppee = stop, execution, True
                    else:
                        # simulation à partir du lendemain : on reproduit la position ouverte à l'achat
                        sortie_prix, jour, stoppee = simuler_depuis(o, h, b, c, atr, execution, prix, stop, sortie)
                    occupe[cle] = jour
                    positions.append({**base, "statut": "prise", "date_entree": g.at[execution, "date"].date(),
                                      "delai_seances": execution - j2, "entree": prix, "stop_niveau": stop,
                                      "risque_pct": risque / prix, "sortie_prix": sortie_prix,
                                      "date_sortie": g.at[jour, "date"].date(), "stoppee": stoppee,
                                      "rendement": sortie_prix / prix - 1 - e5.FRAIS_ALLER_RETOUR,
                                      "R": (sortie_prix - prix) / risque - e5.FRAIS_ALLER_RETOUR * prix / risque,
                                      "annee": g.at[execution, "date"].year})
                    distance_atr = risque / atr[j2]
                    candidats = eligibles[(eligibles >= 1) & (eligibles + e5.SORTIES[sortie][1] < n)]
                    for x in rng.choice(candidats, size=min(e5.N_TIRAGES, len(candidats)), replace=False):
                        st = o[x] - distance_atr * atr[x - 1]
                        if o[x] <= st:
                            continue
                        s, _, _ = e5.simuler(o, h, b, c, atr, x, st, sortie)
                        hasard.append({"filtre": filtre, "stop": nom, "sortie": sortie,
                                       "R": (s - o[x]) / (o[x] - st) - e5.FRAIS_ALLER_RETOUR * o[x] / (o[x] - st)})
    return positions, hasard


def simuler_depuis(o, h, b, c, atr, e: int, prix: float, stop: float, sortie: str):
    """Position achetée en séance e au prix `prix` (déjà vérifié : le stop n'est pas touché en séance e)."""
    mode, duree = e5.SORTIES[sortie]
    plus_haut = h[e]
    if mode == "chandelier":
        stop = max(stop, plus_haut - e5.MULTIPLE_CHANDELIER * atr[e])
    for d in range(e + 1, e + duree + 1):
        if o[d] <= stop:
            return o[d], d, True
        if b[d] <= stop:
            return stop, d, True
        if mode == "chandelier":
            plus_haut = max(plus_haut, h[d])
            stop = max(stop, plus_haut - e5.MULTIPLE_CHANDELIER * atr[d])
    return c[e + duree], e + duree, False


def reference_cassure(donnees: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Entrée à la cassure (règles de l'étape 5) recalculée sur la même moitié."""
    toutes = []
    for symbole, g in donnees.groupby("symbole"):
        p, _ = e5.traiter_action(g, rng)
        toutes += p
    p = pd.DataFrame(toutes)
    p = p[(p["statut"] == "prise") & p["horizon"].isin(SORTIES_TESTEES) & p["stop"].isin(STOPS_TESTES)]
    return p.groupby(["filtre", "stop", "horizon"])["R"].mean().rename("R_moyen_entree_cassure")


def main() -> None:
    colonnes = None  # lecture complète puis filtrage : la moitié B n'est ni conservée ni calculée
    donnees = pd.read_csv(FICHIER_ENTREE, sep=";", parse_dates=["date"], usecols=colonnes)
    retenus = symboles_moitie(donnees["symbole"].unique(), MOITIE)
    donnees = donnees[donnees["symbole"].isin(retenus)]
    print(f"Moitié {MOITIE} : {len(retenus)} actions (la moitié B est réservée et n'est pas calculée)")

    rng = np.random.default_rng(e5.GRAINE)
    toutes, hasard = [], []
    for symbole, g in donnees.groupby("symbole"):
        p, hz = traiter_action(g, rng)
        toutes += p
        hasard += hz
    positions, hasard = pd.DataFrame(toutes), pd.DataFrame(hasard)
    positions.to_csv(FICHIER_POSITIONS, sep=";", index=False, float_format="%.4f")
    cassure = reference_cassure(donnees, np.random.default_rng(e5.GRAINE))

    lignes = []
    for (filtre, stop, sortie), p in positions.groupby(["filtre", "stop", "sortie"]):
        prises = p[p["statut"] == "prise"]
        hz = hasard[(hasard["filtre"] == filtre) & (hasard["stop"] == stop) & (hasard["sortie"] == sortie)]
        demi_ic = 1.96 * prises["R"].std(ddof=1) / np.sqrt(len(prises)) if len(prises) > 1 else np.nan
        signaux = p[~p["statut"].str.startswith("ignorée")]
        lignes.append({"filtre": filtre, "stop": stop, "sortie": sortie, "signaux": len(signaux),
                       "part_executees": (signaux["statut"] == "prise").mean(), "positions": len(prises),
                       "delai_median": prises["delai_seances"].median(), "part_stoppees": prises["stoppee"].mean(),
                       "part_gagnantes": (prises["rendement"] > 0).mean(), "risque_median_pct": prises["risque_pct"].median(),
                       "R_moyen": prises["R"].mean(), "R_ic95_bas": prises["R"].mean() - demi_ic,
                       "R_ic95_haut": prises["R"].mean() + demi_ic, "R_moyen_hasard": hz["R"].mean(),
                       "R_moyen_entree_cassure": cassure.get((filtre, stop, sortie), np.nan),
                       f"R_avant_{e5.ANNEE_COUPURE}": prises.loc[prises["annee"] < e5.ANNEE_COUPURE, "R"].mean(),
                       f"R_depuis_{e5.ANNEE_COUPURE}": prises.loc[prises["annee"] >= e5.ANNEE_COUPURE, "R"].mean()})
    synth = pd.DataFrame(lignes)
    synth["criteres_remplis"] = ((synth[f"R_avant_{e5.ANNEE_COUPURE}"] > 0) & (synth[f"R_depuis_{e5.ANNEE_COUPURE}"] > 0)
                                 & (synth["R_moyen"] > synth["R_moyen_hasard"])
                                 & (synth["R_moyen"] > synth["R_moyen_entree_cassure"]) & (synth["R_ic95_bas"] > 0))
    synth.to_csv(FICHIER_SYNTHESE, sep=";", index=False, float_format="%.4f")
    print(synth.round(3).to_string(index=False))
    print(f"\nVariantes remplissant tous les critères : {int(synth['criteres_remplis'].sum())} / {len(synth)}")


if __name__ == "__main__":
    main()
