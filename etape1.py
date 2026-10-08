"""Étape 1 : récupération du cours de clôture journalier ajusté de Veolia SA (VIE.PA)."""

from pathlib import Path

import yfinance as yf

TICKER = "VIE.PA"
DATE_DEBUT = "2020-04-01"
DATE_FIN = "2026-10-01"  # borne exclusive pour yfinance -> inclut le 30/09/2026
DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_CSV = DOSSIER_DATA / "veolia_cours.csv"


def recuperer_cours() -> "pd.DataFrame":
    # auto_adjust=True : la colonne Close est ajustée des dividendes et splits
    df = yf.download(TICKER, start=DATE_DEBUT, end=DATE_FIN, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Aucune donnée récupérée pour {TICKER}")
    cours = df["Close"][TICKER].rename("cloture_ajustee").to_frame()
    cours.index.name = "date"
    return cours.dropna()


def main() -> None:
    cours = recuperer_cours()
    DOSSIER_DATA.mkdir(exist_ok=True)
    cours.to_csv(FICHIER_CSV, sep=";")
    print(f"{len(cours)} séances enregistrées dans {FICHIER_CSV}")
    print(f"Période : {cours.index.min().date()} -> {cours.index.max().date()}")
    print(cours.describe())


if __name__ == "__main__":
    main()
