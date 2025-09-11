import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from lightGBM_functions import LeagueConfig, run_pipeline

cfg = LeagueConfig(
    league_dir="Spanish La Liga",
    x_file="X.xlsx",
    y_file="Y.xlsx",
    matches_file="Alle_kampe_med_elo_og_odds.xlsx",
    elo_pkl="elo_dict_la_liga.pkl",
    out_xpred="X_pred.xlsx",
    device="gpu",
    random_state=42
)

# Skriv ugens kampe her (samme format som før: home, away, oddsH, oddsU, oddsA)
fixtures = np.array([
    ["Sevilla", "Elche", 2.00, 3.40, 4.0],
    ["Getafe", "Oviedo", 2.00, 3.10, 4.20],
    ["Real Sociedad", "Real Madrid", 5.25, 4.0, 1.6],
    ["Athletic Club", "Alavés", 1.55, 4.00, 6.25],
    ["Atlético Madrid", "Villarreal", 1.85, 3.50, 4.20],
    ["Celta Vigo", "Girona", 1.66, 3.90, 5.00],
    ["Levante", "Real Betis", 3.30, 3.40, 2.15],
    ["Osasuna", "Rayo Vallecano", 2.35, 3.30, 3.10],
    ["Barcelona", "Valencia", 1.25, 7.00, 9.00],
    ["Espanyol", "Mallorca", 2.2, 3.40, 3.20]
], dtype=object)

df, xpred_path = run_pipeline(cfg, fixtures, calibrate=False)
print(df)
print(f"\nX_pred skrevet til: {xpred_path}")