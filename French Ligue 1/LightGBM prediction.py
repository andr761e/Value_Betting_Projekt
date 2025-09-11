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
    ["Marseille", "Lorient", 1.40, 4.50, 7.50],
    ["Nice", "Nantes", 1.65, 3.60, 5.50],
    ["Strasbourg", "Le Havre", 1.55, 3.90, 6.25],
    ["Auxerre", "Monaco", 4.75, 4.10, 1.61],
    ["Lille", "Toulouse", 1.80, 3.60, 4.50],
    ["Brest", "Paris FC", 2.10, 3.40, 3.40],
    ["Metz", "Angers", 2.10, 3.40, 3.40],
    ["Paris Saint-Germain", "Lens", 1.27, 6.50, 8.00],
    ["Rennes", "Lyon", 2.70, 3.60, 2.40]
], dtype=object)

df, xpred_path = run_pipeline(cfg, fixtures, calibrate=False)
print(df)
print(f"\nX_pred skrevet til: {xpred_path}")