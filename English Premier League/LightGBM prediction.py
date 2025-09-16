import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from lightGBM_functions import LeagueConfig, run_pipeline

cfg = LeagueConfig(
    league_dir="English Premier League",
    x_file="X.xlsx",
    y_file="Y.xlsx",
    matches_file="Alle_kampe_med_elo_og_odds.xlsx",
    elo_pkl="elo_dict_premier_league.pkl",
    out_xpred="X_pred.xlsx",
    device="gpu",
    random_state=42
)

# Skriv ugens kampe her (samme format som før: home, away, oddsH, oddsU, oddsA)
fixtures = np.array([
    ["Liverpool", "Everton", 1.48, 4.50, 6.25],
    ["Brighton & Hove Albion", "Tottenham Hotspur", 2.20, 3.70, 3.00],
    ["Burnley", "Nottingham Forest", 3.20, 3.40, 2.20],
    ["West Ham United", "Crystal Palace", 2.87, 3.50, 2.35],
    ["Wolverhampton Wanderers", "Leeds United", 2.60, 3.20, 2.87],
    ["Manchester United", "Chelsea", 2.60, 3.80, 2.45],
    ["Fulham", "Brentford", 2.00, 3.40, 3.75],
    ["Bournemouth", "Newcastle United", 2.55, 3.60, 2.60],
    ["Sunderland", "Aston Villa", 3.40, 3.50, 2.05],
    ["Arsenal", "Manchester City", 1.90, 3.75, 3.80]
], dtype=object)


df, xpred_path = run_pipeline(cfg, fixtures, calibrate=False)
print(df)
print(f"\nX_pred skrevet til: {xpred_path}")