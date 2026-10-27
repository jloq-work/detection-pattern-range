"""Étape 1 : récupération des cours de clôture, haut et bas journaliers ajustés et du volume de plusieurs actions.

Sortie au format « long » : une ligne par (symbole, date), colonne `symbole` en tête.
Colonne `saut` : vraie si la clôture varie de plus de 30 % (en valeur absolue) par rapport à la veille.
Règle déterministe : ces séances rendent l'action inéligible tant qu'une fenêtre de 44 séances les contient
(scission mal ajustée par Yahoo, offre publique, choc de résultats...). Cf. étapes 2 et 3.
Le volume Yahoo des tickers .PA correspond en principe aux échanges sur Euronext Paris uniquement
(hors autres plateformes) : à utiliser en relatif (comparé à sa moyenne) plutôt qu'en absolu.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

SYMBOLES = ["VIE.PA", "SESG.PA", "VIV.PA", "AIR.PA"]
DATE_DEBUT = "2020-04-01"
DATE_FIN = "2026-10-01"  # borne exclusive pour yfinance -> inclut le 30/09/2026
SEUIL_SAUT = 0.30  # variation journalière absolue de la clôture au-delà de laquelle la séance est un « saut »
DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_CSV = DOSSIER_DATA / "etape1.csv"


def recuperer_cours() -> pd.DataFrame:
    # auto_adjust=True : les colonnes Close, High et Low sont ajustées des dividendes et splits
    # (pas des scissions / spin-offs : voir controler_sauts)
    df = yf.download(SYMBOLES, start=DATE_DEBUT, end=DATE_FIN, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Aucune donnée récupérée pour {SYMBOLES}")

    morceaux = []
    for symbole in SYMBOLES:
        if symbole not in df.columns.get_level_values("Ticker"):
            print(f"Attention : aucune donnée pour {symbole}")
            continue
        cours = df.xs(symbole, axis=1, level="Ticker")[["Close", "High", "Low", "Volume"]].rename(
            columns={"Close": "cloture_ajustee", "High": "haut_ajuste", "Low": "bas_ajuste", "Volume": "volume"}
        ).dropna()
        cours.columns.name = None
        cours.index.name = "date"
        cours.insert(0, "symbole", symbole)
        morceaux.append(cours)
    cours = pd.concat(morceaux).reset_index().sort_values(["symbole", "date"]).reset_index(drop=True)
    cours["volume"] = cours["volume"].astype("int64")
    cours = cours[["symbole", "date", "cloture_ajustee", "haut_ajuste", "bas_ajuste", "volume"]]
    return marquer_sauts(cours)


def marquer_sauts(cours: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne booléenne `saut` (variation de clôture > SEUIL_SAUT en valeur absolue, action par action)."""
    variation = cours.groupby("symbole")["cloture_ajustee"].pct_change()
    return cours.assign(saut=variation.abs() > SEUIL_SAUT)


def controler(cours: pd.DataFrame) -> None:
    for symbole, c in cours.groupby("symbole"):
        print(f"\n{symbole} : {len(c)} séances, {c['date'].min().date()} -> {c['date'].max().date()}")
        nb_volume_nul = int((c["volume"] == 0).sum())
        if nb_volume_nul:
            print(f"  Attention : {nb_volume_nul} séance(s) avec un volume nul")
        variation = c["cloture_ajustee"].pct_change()
        for date, v in zip(c.loc[c["saut"], "date"], variation[c["saut"]]):
            print(f"  Saut : variation de {v:+.0%} le {date.date()} -> action exclue tant qu'une fenêtre "
                  f"de 44 séances contient cette séance")


def main() -> None:
    cours = recuperer_cours()
    DOSSIER_DATA.mkdir(exist_ok=True)
    cours.to_csv(FICHIER_CSV, sep=";", index=False, date_format="%Y-%m-%d")
    print(f"{len(cours)} lignes enregistrées dans {FICHIER_CSV}")
    controler(cours)
    print()
    print(cours.groupby("symbole")[["cloture_ajustee", "volume"]].describe().T.round(2))


if __name__ == "__main__":
    main()
