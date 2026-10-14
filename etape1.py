"""Étape 1 : récupération des cours de clôture, haut et bas journaliers ajustés et du volume de Veolia SA (VIE.PA).

Le volume Yahoo des tickers .PA correspond en principe aux échanges sur Euronext Paris uniquement
(hors autres plateformes) : à utiliser en relatif (comparé à sa moyenne) plutôt qu'en absolu.
"""

from pathlib import Path

import yfinance as yf

TICKER = "VIE.PA"
DATE_DEBUT = "2020-04-01"
DATE_FIN = "2026-10-01"  # borne exclusive pour yfinance -> inclut le 30/09/2026
DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_CSV = DOSSIER_DATA / "veolia_cours.csv"


def recuperer_cours() -> "pd.DataFrame":
    # auto_adjust=True : les colonnes Close, High et Low sont ajustées des dividendes et splits
    df = yf.download(TICKER, start=DATE_DEBUT, end=DATE_FIN, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Aucune donnée récupérée pour {TICKER}")
    cours = df.xs(TICKER, axis=1, level="Ticker")[["Close", "High", "Low", "Volume"]].rename(
        columns={"Close": "cloture_ajustee", "High": "haut_ajuste", "Low": "bas_ajuste", "Volume": "volume"}
    )
    cours.columns.name = None
    cours.index.name = "date"
    return cours.dropna()


def main() -> None:
    cours = recuperer_cours()
    DOSSIER_DATA.mkdir(exist_ok=True)
    cours.to_csv(FICHIER_CSV, sep=";")
    print(f"{len(cours)} séances enregistrées dans {FICHIER_CSV}")
    print(f"Période : {cours.index.min().date()} -> {cours.index.max().date()}")
    print(cours.describe())
    nb_volume_nul = int((cours["volume"] == 0).sum())
    if nb_volume_nul:
        print(f"Attention : {nb_volume_nul} séance(s) avec un volume nul")


if __name__ == "__main__":
    main()
