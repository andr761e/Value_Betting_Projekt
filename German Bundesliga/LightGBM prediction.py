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
    ["Bayer Leverkusen", "Eintracht Frankfurt", 2.25, 3.75, 2.8],
    ["Union Berlin", "Hoffenheim", 2.4, 3.3, 2.9],
    ["Freiburg","Stuttgart",3.10, 3.40,2.25],
    ["Wolfsburg","Köln",1.95,3.5,3.8],
    ["Mainz 05","RB Leipzig",2.35,3.6,2.88],
    ["Heidenheim","Dortmund",4.75,4.00,1.65],
    ["Bayern Munich","Hamburger SV",1.08,10.0,26.0],
    ["St. Pauli","Augsburg",2.15,3.6,3.10],
    ["Mönchengladbach","Werder Bremen",1.75,4.1,3.9]
], dtype=object)

df, xpred_path = run_pipeline(cfg, fixtures, calibrate=False)
print(df)
print(f"\nX_pred skrevet til: {xpred_path}")