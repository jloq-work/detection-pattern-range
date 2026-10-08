"""Étape 2 : calcul des 3 features par fenêtre glissante de 44 séances sur le cours de Veolia SA."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TAILLE_FENETRE = 44
DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_COURS = DOSSIER_DATA / "veolia_cours.csv"
FICHIER_FEATURES = DOSSIER_DATA / "veolia_features.csv"


def true_range(cours: pd.DataFrame) -> pd.Series:
    cloture_veille = cours["cloture_ajustee"].shift()
    return pd.concat(
        [
            cours["haut_ajuste"] - cours["bas_ajuste"],
            (cours["haut_ajuste"] - cloture_veille).abs(),
            (cours["bas_ajuste"] - cloture_veille).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)


def pente_log_cours(cloture: pd.Series) -> np.ndarray:
    """Pente de la régression linéaire du log du cours sur chaque fenêtre (en log-points par séance)."""
    fenetres = sliding_window_view(np.log(cloture.to_numpy()), TAILLE_FENETRE)
    x = np.arange(TAILLE_FENETRE) - (TAILLE_FENETRE - 1) / 2
    return (fenetres - fenetres.mean(axis=1, keepdims=True)) @ x / (x @ x)


def calculer_features(cours: pd.DataFrame) -> pd.DataFrame:
    cloture = cours["cloture_ajustee"]
    roulant = cloture.rolling(TAILLE_FENETRE)
    cours_moyen = roulant.mean()

    features = pd.DataFrame(
        {
            "date_debut": cours.index.to_series().shift(TAILLE_FENETRE - 1),
            "date_fin": cours.index,
            "pente": np.r_[np.full(TAILLE_FENETRE - 1, np.nan), pente_log_cours(cloture)],
            "volatilite_atr": true_range(cours).rolling(TAILLE_FENETRE).mean() / cours_moyen,
            "ratio_consolidation": (roulant.max() - roulant.min()) / cours_moyen,
        }
    )
    # la première fenêtre est écartée : son premier true range n'a pas de clôture de la veille
    features = features.dropna().reset_index(drop=True)
    features.insert(0, "id_fenetre", features.index + 1)
    return features


def main() -> None:
    cours = pd.read_csv(FICHIER_COURS, sep=";", parse_dates=["date"], index_col="date")
    features = calculer_features(cours)
    features.to_csv(FICHIER_FEATURES, sep=";", index=False, date_format="%Y-%m-%d")
    print(f"{len(features)} fenêtres de {TAILLE_FENETRE} séances enregistrées dans {FICHIER_FEATURES}")
    print(features.drop(columns="id_fenetre").describe())


if __name__ == "__main__":
    main()
