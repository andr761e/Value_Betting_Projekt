import numpy as np
import pandas as pd

# -----------------------------
# Indstillinger
# -----------------------------
PATH = "English Premier League/test/LightGBMResultsUnfiltered.xlsx"
TAU  = 0.05       # tærskel for (p_model - p_market)
STAKE_PER_BET = 100  # fast indsats pr. bet (kr.)

# -----------------------------
# Hjælpefunktioner
# -----------------------------
LABEL_MAP = {"H":0, "D":1, "A":2}

# -----------------------------
# 1) Indlæs data
# -----------------------------
df = pd.read_excel(PATH)

# Model og market probs
P_mod = df[["YpredH","YpredD","YpredA"]].to_numpy(dtype=float)
P_mkt = df[["YtrueH","YtrueD","YtrueA"]].to_numpy(dtype=float)
ODDS  = df[["OddsH","OddsD","OddsA"]].to_numpy(dtype=float)

# -----------------------------
# 2) Screen efter tau og vælg udfald
# -----------------------------
edge = P_mod - P_mkt
ok_mask = edge >= TAU
any_ok = ok_mask.any(axis=1)

selected_rows = np.where(any_ok)[0]
bets = []
for i in selected_rows:
    # vælg udfaldet med STØRST edge
    j = np.argmax(edge[i])  # 0=H,1=D,2=A
    if edge[i, j] < TAU:
        continue
    side = ["H","D","A"][j]
    p_m  = P_mod[i, j]
    p_k  = P_mkt[i, j]
    odd  = ODDS[i, j]
    bets.append((i, side, j, p_m, p_k, odd, STAKE_PER_BET))

bets_df = pd.DataFrame(bets, columns=["row","side","j","p_model","p_market","odds","stake"])

# læg info fra datasættet på
for col in ["Dato","HjemmeholdNavn","UdeholdNavn","FuldtidsResultat",
            "OddsH","OddsD","OddsA","YpredH","YpredD","YpredA","YtrueH","YtrueD","YtrueA"]:
    if col in df.columns:
        bets_df[col] = df.loc[bets_df["row"], col].values

# -----------------------------
# 3) Evaluér mod facit
# -----------------------------
metrics = {
    "tau": TAU,
    "n_bets": int(len(bets_df)),
    "total_stake": float(bets_df["stake"].sum()) if len(bets_df) else 0.0,
    "hit_rate": np.nan,
    "total_payout": np.nan,
    "pnl_profit": np.nan,
    "roi": np.nan
}

if "FuldtidsResultat" in bets_df.columns and len(bets_df):
    y = bets_df["FuldtidsResultat"].astype(str).str.strip().str.upper().map(LABEL_MAP).to_numpy()
    win = (y == bets_df["j"].to_numpy())
    payout = np.where(win, bets_df["stake"] * bets_df["odds"], 0.0)
    profit = payout - bets_df["stake"]

    bets_df["win"] = win
    bets_df["payout"] = payout
    bets_df["profit"] = profit

    metrics.update({
        "hit_rate": float(win.mean()),
        "total_payout": float(payout.sum()),
        "pnl_profit": float(profit.sum()),
        "roi": float(profit.sum() / bets_df["stake"].sum())
    })

# -----------------------------
# 4) Print statistik
# -----------------------------
print(f"Valgte {metrics['n_bets']} bets ved tau={TAU}.")
print(f"Total stake: {metrics['total_stake']:.2f} kr.")
if not np.isnan(metrics["hit_rate"]):
    print(f"Hit rate: {metrics['hit_rate']:.3f} | Total payout: {metrics['total_payout']:.2f} kr. "
          f"| Profit (PnL): {metrics['pnl_profit']:.2f} kr. | ROI: {metrics['roi']:.3f}")

# -----------------------------
# 5) Gem resultat
# -----------------------------
out_path = f"English Premier League/test/LightGBM_selected_bets_tau_{TAU:.3f}.xlsx"
bets_df.to_excel(out_path, index=False)
print(f"Skrev bets til: {out_path}")