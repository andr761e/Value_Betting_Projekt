import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from lightGBM_functions import LeagueConfig, run_pipeline

cfg = LeagueConfig(
    league_dir="Italian Serie A",
    x_file="X.xlsx",
    y_file="Y.xlsx",
    matches_file="Alle_kampe_med_elo_og_odds.xlsx",
    elo_pkl="elo_dict_serie_a.pkl",
    out_xpred="X_pred.xlsx",
    device="gpu",
    random_state=42
)

# Skriv ugens kampe her (samme format som før: home, away, oddsH, oddsU, oddsA)
fixtures = np.array([
    ["Cagliari", "Parma", 2.20, 3.20, 3.50],
    ["Juventus", "Internazionale", 2.62, 3.00, 2.90],
    ["Fiorentina", "Napoli", 3.60, 3.30, 2.10],
    ["Roma", "Torino", 1.53, 4.00, 6.50],
    ["Atalanta", "Lecce", 1.42, 4.50, 8.00],
    ["Pisa", "Udinese", 2.87, 3.10, 2.62],
    ["Sassuolo", "Lazio", 3.50, 3.60, 2.05],
    ["Milan", "Bologna", 1.95, 3.40, 4.00],
    ["Hellas Verona", "Cremonese", 2.30, 3.25, 3.25],
    ["Como", "Genoa", 1.66, 3.70, 5.50]
], dtype=object)

df, xpred_path = run_pipeline(cfg, fixtures, calibrate=False)
print(df)
print(f"\nX_pred skrevet til: {xpred_path}")