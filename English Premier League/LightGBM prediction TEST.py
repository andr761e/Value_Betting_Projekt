import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lightGBM_functions3 import Config, backtest_tail20, tau_sweep

cfg = Config(league_dir="English Premier League", device="gpu",
             gamma_var=1.10, dc_strength=0.05, alpha_power=1.05, beta_max=0.15,
             overround=1.05, stake=100.0, tau_default=0.05)

summary, bets_df, diffs_df, avg_pgd_df = backtest_tail20(cfg, tau_prob=0.05, tau_ev=0.15)
print(summary)
print(summary["diff"])
print(summary["ev"])

sweep_df = tau_sweep(cfg, taus_prob=[0.00,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12], taus_ev=[0.00,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12])
print(sweep_df)

