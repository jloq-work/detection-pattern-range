"""Étape 2 : calcul des 3 features par fenêtre glissante de 22 séances, action par action.

Fenêtre de détection ramenée de 44 à 22 séances (problème 2) : elle correspond mieux à la durée des ranges
de la référence annotée (30 séances en médiane). TAILLE_FENETRE doit rester identique dans etape3.py.

Les fenêtres ne chevauchent jamais deux actions ; id_fenetre repart à 1 pour chaque symbole.
Colonne `fenetre_avec_saut` : vraie si au moins une séance de la fenêtre est un saut (cf. étape 1) ;
ces fenêtres sont exclues du clustering à l'étape 3.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

TAILLE_FENETRE = 22
DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_COURS = DOSSIER_DATA / "etape1.csv"
FICHIER_FEATURES = DOSSIER_DATA / "etape2.csv"


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
            "fenetre_avec_saut": cours["saut"].astype(int).rolling(TAILLE_FENETRE).sum() > 0,
        }
    )
    # la première fenêtre est écartée : son premier true range n'a pas de clôture de la veille
    features = features.dropna().reset_index(drop=True)
    features.insert(0, "id_fenetre", features.index + 1)
    return features


def main() -> None:
    cours = pd.read_csv(FICHIER_COURS, sep=";", parse_dates=["date"])
    morceaux = []
    for symbole, c in cours.groupby("symbole", sort=False):
        f = calculer_features(c.set_index("date"))
        f.insert(0, "symbole", symbole)
        morceaux.append(f)
    features = pd.concat(morceaux, ignore_index=True)
    features.to_csv(FICHIER_FEATURES, sep=";", index=False, date_format="%Y-%m-%d")
    print(f"{len(features)} fenêtres de {TAILLE_FENETRE} séances enregistrées dans {FICHIER_FEATURES}")
    avec_saut = features.groupby("symbole")["fenetre_avec_saut"].sum()
    print(f"Fenêtres contenant un saut (exclues du clustering) : {int(avec_saut.sum())}, dans {int((avec_saut > 0).sum())} action(s)")
    print(features[["pente", "volatilite_atr", "ratio_consolidation"]].describe().T.round(4).to_string())


if __name__ == "__main__":
    main()
