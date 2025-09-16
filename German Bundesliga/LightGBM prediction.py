import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from lightGBM_functions import LeagueConfig, run_pipeline

cfg = LeagueConfig(
    league_dir="German Bundesliga",
    x_file="X.xlsx",
    y_file="Y.xlsx",
    matches_file="Alle_kampe_med_elo_og_odds.xlsx",
    elo_pkl="elo_dict_bundesliga.pkl",
    out_xpred="X_pred.xlsx",
    device="gpu",
    random_state=42
)

# Skriv ugens kampe her (samme format som før: home, away, oddsH, oddsU, oddsA)
fixtures = np.array([
    ["Stuttgart", "St. Pauli", 1.70, 3.90, 4.50],
    ["Augsburg", "Mainz 05", 2.62, 3.40, 2.60],
    ["Hamburger SV", "Heidenheim", 2.05, 3.60, 3.50],
    ["Hoffenheim", "Bayern Munich", 7.00, 5.25, 1.36],
    ["Werder Bremen", "Freiburg", 2.30, 3.60, 2.80],
    ["RB Leipzig", "Köln", 1.61, 4.20, 4.75],
    ["Eintracht Frankfurt", "Union Berlin", 1.55, 4.10, 5.50],
    ["Bayer Leverkusen", "Mönchengladbach", 1.57, 4.75, 4.75],
    ["Dortmund", "Wolfsburg", 1.48, 5.00, 5.50]
], dtype=object)

df, xpred_path = run_pipeline(cfg, fixtures, calibrate=False)
print(df)
print(f"\nX_pred skrevet til: {xpred_path}")