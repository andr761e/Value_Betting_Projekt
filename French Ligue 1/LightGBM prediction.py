import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from lightGBM_functions import LeagueConfig, run_pipeline

cfg = LeagueConfig(
    league_dir="French Ligue 1",
    x_file="X.xlsx",
    y_file="Y.xlsx",
    matches_file="Alle_kampe_med_elo_og_odds.xlsx",
    elo_pkl="elo_dict_ligue_1.pkl",
    out_xpred="X_pred.xlsx",
    device="gpu",
    random_state=42
)

# Skriv ugens kampe her (samme format som før: home, away, oddsH, oddsU, oddsA)
fixtures = np.array([
    ["Lyon", "Angers", 1.36, 4.75, 8.50],
    ["Nantes", "Rennes", 3.50, 3.30, 2.10],
    ["Brest", "Nice", 2.75, 3.50, 2.45],
    ["Lens", "Lille", 2.45, 3.50, 2.70],
    ["Paris FC", "Strasbourg", 2.45, 3.50, 2.75],
    ["Monaco", "Metz", 1.27, 6.00, 10.00],
    ["Auxerre", "Toulouse", 2.62, 3.30, 2.60],
    ["Le Havre", "Lorient", 2.20, 3.40, 3.10],
    ["Marseille", "Paris Saint-Germain", 3.40, 3.80, 2.00]
], dtype=object)

df, xpred_path = run_pipeline(cfg, fixtures, calibrate=False)
print(df)
print(f"\nX_pred skrevet til: {xpred_path}")