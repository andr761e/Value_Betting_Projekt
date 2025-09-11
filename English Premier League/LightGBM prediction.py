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
    ["Arsenal", "Nottingham Forest", 1.40, 5.00, 7.50],
    ["Bournemouth", "Brighton & Hove Albion", 2.45, 3.60, 2.75],
    ["Crystal Palace", "Sunderland", 1.61, 3.90, 5.50],
    ["Everton", "Aston Villa", 2.40, 3.30, 3.00],
    ["Fulham", "Leeds United", 1.90, 3.50, 4.10],
    ["Newcastle United", "Wolverhampton Wanderers", 1.44, 4.50, 7.50],
    ["West Ham United", "Tottenham Hotspur", 3.30, 3.60, 2.10],
    ["Brentford", "Chelsea", 4.50, 3.80, 1.75],
    ["Burnley", "Liverpool", 9.50, 5.50, 1.30],
    ["Manchester City", "Manchester United", 1.66, 4.00, 4.75]
], dtype=object)

df, xpred_path = run_pipeline(cfg, fixtures, calibrate=False)
print(df)
print(f"\nX_pred skrevet til: {xpred_path}")