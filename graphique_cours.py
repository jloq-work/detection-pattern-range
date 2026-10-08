"""Graphique du cours de clôture ajusté de Veolia SA récupéré à l'étape 1."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DOSSIER_DATA = Path(__file__).parent / "data_test"
FICHIER_CSV = DOSSIER_DATA / "veolia_cours.csv"
FICHIER_PNG = DOSSIER_DATA / "veolia_cours.png"

COULEUR_SERIE = "#2a78d6"
ENCRE_PRINCIPALE = "#0b0b0b"
ENCRE_SECONDAIRE = "#52514e"
ENCRE_DISCRETE = "#898781"
GRILLE = "#e1e0d9"
AXE = "#c3c2b7"


def main() -> None:
    cours = pd.read_csv(FICHIER_CSV, sep=";", parse_dates=["date"], index_col="date")["cloture_ajustee"]

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=120)
    ax.plot(cours.index, cours.values, color=COULEUR_SERIE, linewidth=1.6, solid_joinstyle="round")
    ax.fill_between(cours.index, cours.values, cours.min() * 0.95, color=COULEUR_SERIE, alpha=0.08, linewidth=0)

    dernier_jour, dernier_cours = cours.index[-1], cours.iloc[-1]
    ax.scatter([dernier_jour], [dernier_cours], s=40, color=COULEUR_SERIE, edgecolor="white", linewidth=2, zorder=3)
    ax.annotate(f"{dernier_cours:.2f} €", (dernier_jour, dernier_cours), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=10, color=ENCRE_PRINCIPALE)
    for jour, label in [(cours.idxmin(), "min"), (cours.idxmax(), "max")]:
        valeur = cours[jour]
        ax.annotate(f"{label} {valeur:.2f} € ({jour:%d/%m/%Y})", (jour, valeur),
                    xytext=(10, -2) if label == "min" else (0, 8), textcoords="offset points",
                    ha="left" if label == "min" else "center", va="center" if label == "min" else "bottom",
                    fontsize=9, color=ENCRE_SECONDAIRE)

    ax.set_title("Veolia SA (VIE.PA) : cours de clôture ajusté journalier, avril 2020 - septembre 2026",
                 loc="left", fontsize=13, color=ENCRE_PRINCIPALE, pad=12)
    ax.set_ylabel("€", color=ENCRE_DISCRETE, rotation=0, labelpad=12)
    ax.set_ylim(cours.min() * 0.95, cours.max() * 1.05)
    ax.set_xlim(cours.index[0], cours.index[-1] + pd.Timedelta(days=90))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", color=GRILLE, linewidth=1)
    ax.tick_params(colors=ENCRE_DISCRETE, length=0)
    for cote in ["top", "right", "left"]:
        ax.spines[cote].set_visible(False)
    ax.spines["bottom"].set_color(AXE)

    fig.tight_layout()
    fig.savefig(FICHIER_PNG)
    print(f"Graphique enregistré dans {FICHIER_PNG}")


if __name__ == "__main__":
    main()
