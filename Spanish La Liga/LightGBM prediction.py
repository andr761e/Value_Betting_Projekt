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
    ["Real Betis", "Real Sociedad", 2.05, 3.10, 4.00],
    ["Girona", "Levante", 1.90, 3.50, 3.90],
    ["Alavés", "Sevilla", 2.37, 3.25, 3.10],
    ["Real Madrid", "Espanyol", 1.22, 7.00, 11.00],
    ["Valencia", "Athletic Club", 3.10, 3.10, 2.45],
    ["Rayo Vallecano", "Celta Vigo", 2.35, 3.30, 3.00],
    ["Mallorca", "Atlético Madrid", 5.00, 3.50, 1.75],
    ["Barcelona", "Getafe", 1.25, 6.25, 9.50],
    ["Elche", "Oviedo", 2.05, 3.00, 4.10],
    ["Villarreal", "Osasuna", 1.53, 4.10, 5.75]
], dtype=object)


df, xpred_path = run_pipeline(cfg, fixtures, calibrate=False)
print(df)
print(f"\nX_pred skrevet til: {xpred_path}")