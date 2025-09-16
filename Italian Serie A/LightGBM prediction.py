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
    ["Lecce", "Cagliari", 2.62, 3.10, 2.75],
    ["Bologna", "Genoa", 1.85, 3.30, 4.75],
    ["Hellas Verona", "Juventus", 6.50, 3.60, 1.57],
    ["Udinese", "Milan", 4.33, 3.60, 1.80],
    ["Lazio", "Roma", 2.70, 3.30, 2.55],
    ["Cremonese", "Parma", 2.45, 3.20, 2.87],
    ["Torino", "Atalanta", 3.60, 3.50, 2.05],
    ["Napoli", "Pisa", 1.27, 5.50, 11.00],
    ["Fiorentina", "Como", 2.62, 3.20, 2.70],
    ["Internazionale", "Sassuolo", 1.28, 5.00, 11.00]
], dtype=object)


df, xpred_path = run_pipeline(cfg, fixtures, calibrate=False)
print(df)
print(f"\nX_pred skrevet til: {xpred_path}")