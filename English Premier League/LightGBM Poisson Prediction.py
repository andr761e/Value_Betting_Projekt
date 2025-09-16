import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lightGBM_functions3 import Config, train_full_models, predict_from_fixtures

# ----- træning på hele datasættet -----
cfg = Config(league_dir="English Premier League")
model_GH, model_GA, best_k, feat_names = train_full_models(cfg)

# ----- fixtures som i dit screenshot -----
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

pred_df = predict_from_fixtures(fixtures, cfg, model_GH, model_GA, best_k, include_pgd=True)
print(pred_df.head(10))

fixtures_ah1 = np.array([
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

