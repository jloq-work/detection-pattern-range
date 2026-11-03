"""Étape 1 : récupération des cours de clôture, haut et bas journaliers ajustés et du volume de plusieurs actions.

Sortie au format « long » : une ligne par (symbole, date), colonne `symbole` en tête.
Colonne `saut` : vraie si la clôture varie de plus de 30 % (en valeur absolue) par rapport à la veille.
Règle déterministe : ces séances rendent l'action inéligible tant qu'une fenêtre de détection (22 séances, cf. étape 2) les contient
(scission mal ajustée par Yahoo, offre publique, choc de résultats...). Cf. étapes 2 et 3.
Colonnes `ouverture_ajustee` et `ouverture_brute` : cours d'ouverture (prix d'entrée en position à l'étape 5).
Univers : les actions de la référence annotée (SYMBOLES_REFERENCE) + la liste du SRD (data_test/symbols.csv),
sans doublon ; seuls les symboles Euronext Paris (.PA) sont gardés (les cotations de Francfort .F ont un volume
sans rapport) ; les actions avec moins de MIN_SEANCES séances sont écartées (historique trop court).
Colonnes `cloture_brute`, `haut_brut`, `bas_brut` : cours non ajustés des dividendes (Yahoo les ajuste
seulement des divisions d'actions), pour comparer des niveaux lus sur un graphique de cours réels.
Le volume Yahoo des tickers .PA correspond en principe aux échanges sur Euronext Paris uniquement
(hors autres plateformes) : à utiliser en relatif (comparé à sa moyenne) plutôt qu'en absolu.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

SYMBOLES_REFERENCE = ["VIE.PA", "SW.PA", "VIV.PA", "AIR.PA", "S30.PA"]
SUFFIXES_ACCEPTES = (".PA",)
MIN_SEANCES = 250  # environ un an de cotations
TAILLE_LOT = 40    # symboles par requête Yahoo
DATE_DEBUT = "2020-04-01"
DATE_FIN = "2026-10-01"  # borne exclusive pour yfinance -> inclut le 30/09/2026
SEUIL_SAUT = 0.30  # variation journalière absolue de la clôture au-delà de laquelle la séance est un « saut »
DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_CSV = DOSSIER_DATA / "etape1.csv"
FICHIER_SYMBOLES = DOSSIER_DATA / "symbols.csv" # Liste de tickers boursiers utilisée pour un projet d'analyse de données


def lire_symboles() -> list[str]:
    symboles = list(SYMBOLES_REFERENCE)
    if FICHIER_SYMBOLES.exists():
        liste = pd.read_csv(FICHIER_SYMBOLES)["symbol"].astype(str).str.strip()
        symboles += liste.tolist()
    symboles = list(dict.fromkeys(symboles))  # sans doublon, ordre conservé
    ecartes = [s for s in symboles if not s.endswith(SUFFIXES_ACCEPTES)]
    if ecartes:
        print(f"Symboles écartés (hors Euronext Paris) : {', '.join(ecartes)}")
    return [s for s in symboles if s.endswith(SUFFIXES_ACCEPTES)]


SYMBOLES = lire_symboles()


def recuperer_cours() -> pd.DataFrame:
    # auto_adjust=True : les colonnes Close, High et Low sont ajustées des dividendes et splits
    # (pas des scissions / spin-offs : voir controler_sauts)
    # téléchargement par lots pour limiter le risque de refus par Yahoo
    morceaux_ajustes, morceaux_bruts = [], []
    for k in range(0, len(SYMBOLES), TAILLE_LOT):
        lot = SYMBOLES[k:k + TAILLE_LOT]
        print(f"Téléchargement des symboles {k + 1} à {k + len(lot)} sur {len(SYMBOLES)}")
        morceaux_ajustes.append(yf.download(lot, start=DATE_DEBUT, end=DATE_FIN, auto_adjust=True,
                                            progress=False, group_by="column"))
        # auto_adjust=False : Open, Close, High et Low bruts (non ajustés des dividendes)
        morceaux_bruts.append(yf.download(lot, start=DATE_DEBUT, end=DATE_FIN, auto_adjust=False,
                                          progress=False, group_by="column"))
    df = pd.concat(morceaux_ajustes, axis=1)
    brut = pd.concat(morceaux_bruts, axis=1)
    if df.empty or brut.empty:
        raise RuntimeError(f"Aucune donnée récupérée pour {SYMBOLES}")

    morceaux = []
    for symbole in SYMBOLES:
        if symbole not in df.columns.get_level_values("Ticker"):
            print(f"Attention : aucune donnée pour {symbole}")
            continue
        cours = df.xs(symbole, axis=1, level="Ticker")[["Open", "Close", "High", "Low", "Volume"]].rename(
            columns={"Open": "ouverture_ajustee", "Close": "cloture_ajustee", "High": "haut_ajuste",
                     "Low": "bas_ajuste", "Volume": "volume"}
        ).dropna()
        if len(cours) < MIN_SEANCES:
            print(f"Attention : {symbole} écarté ({len(cours)} séances, minimum {MIN_SEANCES})")
            continue
        cours = cours.join(brut.xs(symbole, axis=1, level="Ticker")[["Open", "Close", "High", "Low"]].rename(
            columns={"Open": "ouverture_brute", "Close": "cloture_brute", "High": "haut_brut", "Low": "bas_brut"}
        ), how="left")
        cours.columns.name = None
        cours.index.name = "date"
        cours.insert(0, "symbole", symbole)
        morceaux.append(cours)
    cours = pd.concat(morceaux).reset_index().sort_values(["symbole", "date"]).reset_index(drop=True)
    cours["volume"] = cours["volume"].astype("int64")
    cours = cours[["symbole", "date", "ouverture_ajustee", "cloture_ajustee", "haut_ajuste", "bas_ajuste", "volume",
                   "ouverture_brute", "cloture_brute", "haut_brut", "bas_brut"]]
    return marquer_sauts(cours)


def marquer_sauts(cours: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne booléenne `saut` (variation de clôture > SEUIL_SAUT en valeur absolue, action par action)."""
    variation = cours.groupby("symbole")["cloture_ajustee"].pct_change()
    return cours.assign(saut=variation.abs() > SEUIL_SAUT)


def controler(cours: pd.DataFrame) -> None:
    """Affiche seulement les anomalies, action par action, puis une synthèse."""
    print("\nContrôles (seules les actions présentant une anomalie sont listées) :")
    for symbole, c in cours.groupby("symbole"):
        messages = []
        nb_volume_nul = int((c["volume"] == 0).sum())
        if nb_volume_nul:
            messages.append(f"{nb_volume_nul} séance(s) avec un volume nul")
        nb_brut_manquant = int(c["cloture_brute"].isna().sum())
        if nb_brut_manquant:
            messages.append(f"{nb_brut_manquant} séance(s) sans cours non ajusté")
        variation = c["cloture_ajustee"].pct_change()
        for date, v in zip(c.loc[c["saut"], "date"], variation[c["saut"]]):
            messages.append(f"saut de {v:+.0%} le {date.date()} -> exclue tant qu'une fenêtre de détection contient cette séance")
        if messages:
            print(f"  {symbole} ({len(c)} séances, {c['date'].min().date()} -> {c['date'].max().date()}) : " + " ; ".join(messages))
    facteur = (cours["cloture_ajustee"] / cours["cloture_brute"]).groupby(cours["symbole"])
    print(f"\nSynthèse : {cours['symbole'].nunique()} actions, {len(cours)} lignes ; "
          f"{int(cours['saut'].sum())} saut(s) ; facteur d'ajustement médian en début de période "
          f"{facteur.first().median():.3f}, en fin de période {facteur.last().median():.3f}")


def main() -> None:
    cours = recuperer_cours()
    DOSSIER_DATA.mkdir(exist_ok=True)
    cours.to_csv(FICHIER_CSV, sep=";", index=False, date_format="%Y-%m-%d")
    print(f"{len(cours)} lignes enregistrées dans {FICHIER_CSV} ({cours['symbole'].nunique()} actions)")
    controler(cours)


if __name__ == "__main__":
    main()
