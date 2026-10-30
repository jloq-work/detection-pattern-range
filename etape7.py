"""Étape 7 (piste B) : filtre de marché sur les cassures — moitié A de l'univers seulement.

Règle fixée avant tout calcul :
- une cassure confirmée (J2) n'est retenue que si, à la clôture de J2, le CAC 40 (^FCHI) clôture au-dessus de sa
  moyenne mobile à MOYENNE_MOBILE séances (information connue à J2) ;
- positions selon les règles de l'étape 5 (entrée à l'ouverture de J3), stops S1 / S3 / S4,
  sorties fixe_10 et T1_chandelier, avec et sans filtre de volume ;
- comparaisons : entrées au hasard restreintes aux séances où le marché est haussier (sinon l'effet du filtre se
  confondrait avec la hausse générale du marché), et les mêmes cassures sans filtre de marché ;
- critères : R moyen > 0 avant et depuis 2024, R moyen > hasard (marché haussier) et > cassures sans filtre de
  marché, borne basse de l'IC 95 % > 0.
La moitié B est réservée (non chargée). Si l'indice ne peut pas être téléchargé, un indice de remplacement
équipondéré construit sur la moitié A est utilisé, et le résultat n'est alors qu'indicatif.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import etape5 as e5
from etape6 import symboles_moitie

INDICE = "^FCHI"
MOYENNE_MOBILE = 200
DEBUT_INDICE = "2019-01-01"  # assez tôt pour disposer de la moyenne mobile dès avril 2020
FIN_INDICE = "2026-10-01"
STOPS_TESTES = ["S1", "S3", "S4"]
SORTIES_TESTEES = ["fixe_10", "T1_chandelier"]
MOITIE = "A"

DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_ENTREE = DOSSIER_DATA / "etape4.csv"
FICHIER_INDICE = DOSSIER_DATA / "cac40.csv"
FICHIER_SYNTHESE = DOSSIER_DATA / "etape7_synthese.csv"


def charger_indice(donnees: pd.DataFrame) -> tuple[pd.Series, str]:
    """Clôtures de l'indice ; téléchargement puis cache local, sinon indice de remplacement."""
    if not FICHIER_INDICE.exists():
        try:
            import yfinance as yf
            df = yf.download(INDICE, start=DEBUT_INDICE, end=FIN_INDICE, auto_adjust=True, progress=False)
            cloture = df["Close"].squeeze().dropna()
            if len(cloture) > MOYENNE_MOBILE:
                cloture.rename("cloture").to_frame().to_csv(FICHIER_INDICE, sep=";", date_format="%Y-%m-%d")
        except Exception as exc:  # noqa: BLE001 — repli explicite ci-dessous
            print(f"Téléchargement de {INDICE} impossible : {exc}")
    if FICHIER_INDICE.exists():
        s = pd.read_csv(FICHIER_INDICE, sep=";", parse_dates=["date"], index_col="date")["cloture"]
        return s, f"{INDICE} (CAC 40)"
    rendements = (donnees.pivot(index="date", columns="symbole", values="cloture_ajustee")
                  .pct_change().clip(-0.3, 0.3).mean(axis=1).fillna(0))
    return (1 + rendements).cumprod().rename("cloture"), "INDICE DE REMPLACEMENT équipondéré (moitié A) — résultat indicatif"


def lancer(donnees: pd.DataFrame, haussier: pd.Series | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(e5.GRAINE)
    toutes, hasard = [], []
    for _, g in donnees.groupby("symbole"):
        g = g.copy()
        if haussier is not None:
            ok = g["date"].map(haussier).fillna(False).astype(bool)
            g.loc[~ok, "evenement"] = ""          # cassures hors marché haussier ignorées
            g.loc[~ok, "regime"] = "exclu"         # tirages au hasard restreints au marché haussier
        p, hz = e5.traiter_action(g, rng)
        toutes += p
        hasard += hz
    p, hz = pd.DataFrame(toutes), pd.DataFrame(hasard)
    p = p[(p["statut"] == "prise") & p["horizon"].isin(SORTIES_TESTEES) & p["stop"].isin(STOPS_TESTES)]
    hz = hz[hz["horizon"].isin(SORTIES_TESTEES) & hz["stop"].isin(STOPS_TESTES)]
    p["annee"] = pd.to_datetime(p["date_entree"]).dt.year
    return p, hz


def main() -> None:
    donnees = pd.read_csv(FICHIER_ENTREE, sep=";", parse_dates=["date"])
    donnees = donnees[donnees["symbole"].isin(symboles_moitie(donnees["symbole"].unique(), MOITIE))]
    indice, source = charger_indice(donnees)
    haussier = indice > indice.rolling(MOYENNE_MOBILE).mean()
    haussier = haussier[indice.rolling(MOYENNE_MOBILE).mean().notna()]
    print(f"Moitié {MOITIE} : {donnees['symbole'].nunique()} actions ; indice : {source}")
    print(f"Part des séances en marché haussier : {haussier.mean():.0%}")

    sans_marche, _ = lancer(donnees, None)
    avec_marche, hasard = lancer(donnees, haussier)
    ref = sans_marche.groupby(["filtre", "stop", "horizon"])["R"].agg(["mean", "size"])

    lignes = []
    for (filtre, stop, sortie), p in avec_marche.groupby(["filtre", "stop", "horizon"]):
        hz = hasard[(hasard["filtre"] == filtre) & (hasard["stop"] == stop) & (hasard["horizon"] == sortie)]
        demi_ic = 1.96 * p["R"].std(ddof=1) / np.sqrt(len(p))
        lignes.append({"filtre": filtre, "stop": stop, "sortie": sortie, "positions": len(p),
                       "positions_sans_filtre_marche": ref.loc[(filtre, stop, sortie), "size"],
                       "part_gagnantes": (p["rendement"] > 0).mean(), "R_moyen": p["R"].mean(),
                       "R_ic95_bas": p["R"].mean() - demi_ic, "R_ic95_haut": p["R"].mean() + demi_ic,
                       "R_moyen_hasard_marche_haussier": hz["R"].mean(),
                       "R_moyen_sans_filtre_marche": ref.loc[(filtre, stop, sortie), "mean"],
                       f"R_avant_{e5.ANNEE_COUPURE}": p.loc[p["annee"] < e5.ANNEE_COUPURE, "R"].mean(),
                       f"R_depuis_{e5.ANNEE_COUPURE}": p.loc[p["annee"] >= e5.ANNEE_COUPURE, "R"].mean()})
    synth = pd.DataFrame(lignes)
    synth["criteres_remplis"] = ((synth[f"R_avant_{e5.ANNEE_COUPURE}"] > 0) & (synth[f"R_depuis_{e5.ANNEE_COUPURE}"] > 0)
                                 & (synth["R_moyen"] > synth["R_moyen_hasard_marche_haussier"])
                                 & (synth["R_moyen"] > synth["R_moyen_sans_filtre_marche"])
                                 & (synth["R_ic95_bas"] > 0))
    synth.to_csv(FICHIER_SYNTHESE, sep=";", index=False, float_format="%.4f")
    print(synth.round(3).to_string(index=False))
    print(f"\nVariantes remplissant tous les critères : {int(synth['criteres_remplis'].sum())} / {len(synth)}")


if __name__ == "__main__":
    main()
