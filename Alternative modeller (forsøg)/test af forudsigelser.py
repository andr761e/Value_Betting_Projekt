import numpy as np
import pandas as pd

# -----------------------------
# Indstillinger
# -----------------------------
PATH = "Alternative modeller (forsøg)/BTD_model_probs.xlsx"
TAU  = 0.1          # tærskel for (p_model - p_market) pr. udfald
STAKE_PER_BET = 100  # fast indsats pr. bet (kr.)
USE_BLEND_IF_AVAILABLE = False  # brug p*_blend hvis tilgængelig, ellers p*_mod

# -----------------------------
# Hjælpefunktioner
# -----------------------------
LABEL_MAP = {"H":0, "D":1,"A":2}

def pick_prob_cols(df, prefer_blend=True):
    """Vælg hvilke model-prob-kolonner vi bruger."""
    if prefer_blend and all(c in df.columns for c in ["pH_blend","pD_blend","pA_blend"]):
        return ("pH_blend","pD_blend","pA_blend","blend")
    if all(c in df.columns for c in ["pH_mod","pD_mod","pA_mod"]):
        return ("pH_mod","pD_mod","pA_mod","mod")
    raise ValueError("Kunne ikke finde model-probabiliteter (hverken *_blend eller *_mod).")

def ensure_cols(df, cols, where="DataFrame"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Mangler kolonner i {where}: {missing}")

# -----------------------------
# 1) Indlæs data
# -----------------------------
df = pd.read_excel(PATH)

# tjek nødvendige kolonner
ensure_cols(df, ["PROB H","PROB D","PROB A","AVG H","AVG D","AVG A"], "BTD_model_probs.xlsx")

pH_col, pD_col, pA_col, probs_used = pick_prob_cols(df, prefer_blend=USE_BLEND_IF_AVAILABLE)

# pak arrays til beregning
P_mod = df[[pH_col, pD_col, pA_col]].to_numpy(dtype=float)              # model/blend
P_mkt = df[["PROB H","PROB D","PROB A"]].to_numpy(dtype=float)          # market
ODDS  = df[["AVG H","AVG D","AVG A"]].to_numpy(dtype=float)             # decimalodds

# -----------------------------
# 2) Screen efter tau og vælg udfald
# -----------------------------
edge = P_mod - P_mkt                    # forskel i sandsynlighed pr. udfald
# mask for udfald hvor edge >= TAU
ok_mask = edge >= TAU
any_ok = ok_mask.any(axis=1)

selected_rows = np.where(any_ok)[0]
bets = []
for i in selected_rows:
    # vælg udfaldet med STØRST edge (hvis flere ≥ tau)
    j = np.argmax(edge[i])             # 0=H,1=D,2=A
    if edge[i, j] < TAU:
        continue                       # sikkerhed, hvis max stadig < tau
    side = ["H","D","A"][j]
    p_m  = P_mod[i, j]
    p_k  = P_mkt[i, j]
    odd  = ODDS[i, j]
    bets.append((i, side, j, p_m, p_k, odd, STAKE_PER_BET))

bets_df = pd.DataFrame(bets, columns=["row","side","j","p_model","p_market","odds","stake"])

# læg læsevenlige kolonner på, hvis de findes
for col in ["Hjemmehold","Udehold","Dato","Liga",
            "AVG H","AVG D","AVG A","PROB H","PROB D","PROB A",
            pH_col, pD_col, pA_col]:
    if col in df.columns:
        bets_df[col] = df.loc[bets_df["row"], col].values

# -----------------------------
# 3) Evaluér mod facit (hvis tilgængelig)
# -----------------------------
metrics = {
    "tau": TAU,
    "probs_used": probs_used,
    "n_bets": int(len(bets_df)),
    "total_stake": float(bets_df["stake"].sum()) if len(bets_df) else 0.0,
    "hit_rate": np.nan,
    "total_payout": np.nan,
    "pnl_profit": np.nan,
    "roi": np.nan
}

if "FuldtidsResultat" in df.columns and len(bets_df):
    y = df.loc[bets_df["row"], "FuldtidsResultat"].astype(str).str.strip().str.upper().map(LABEL_MAP).to_numpy()
    win = (y == bets_df["j"].to_numpy())
    payout = np.where(win, bets_df["stake"] * bets_df["odds"], 0.0)  # udbetaling
    profit = payout - bets_df["stake"]                               # profit (udbetaling minus indsats)

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
# 4) Ekstra statistik (hvis muligt)
# -----------------------------
extras = {}
if len(bets_df):
    # Fordeling pr. udfald
    extras["by_side_counts"] = bets_df["side"].value_counts().to_dict()
    if "win" in bets_df.columns:
        extras["by_side_hit"] = bets_df.groupby("side")["win"].mean().to_dict()
        extras["by_side_roi"] = (bets_df.groupby("side")["profit"].sum() /
                                 bets_df.groupby("side")["stake"].sum()).to_dict()
    # Pr. liga (hvis findes)
    if "Liga" in bets_df.columns:
        grp = bets_df.groupby("Liga")
        extras["by_league_n"] = grp.size().to_dict()
        if "win" in bets_df.columns:
            extras["by_league_roi"] = (grp["profit"].sum() / grp["stake"].sum()).to_dict()

# -----------------------------
# 5) Gem resultat
# -----------------------------
out_path = f"Alternative modeller (forsøg)/BTD_selected_bets_tau_{TAU:.3f}.xlsx"
bets_df.to_excel(out_path, index=False)

print(f"Valgte {metrics['n_bets']} bets ved tau={TAU} (brugte {metrics['probs_used']} probabilities).")
print(f"Total stake: {metrics['total_stake']:.2f} kr.")
if not np.isnan(metrics["hit_rate"]):
    print(f"Hit rate: {metrics['hit_rate']:.3f} | Total payout: {metrics['total_payout']:.2f} kr. | "
          f"Profit (PnL): {metrics['pnl_profit']:.2f} kr. | ROI: {metrics['roi']:.3f}")
else:
    print("Ingen facit-kolonne ('FuldtidsResultat') fundet i filen – statistik på træf/ROI kan ikke beregnes.")
print(f"Skrev bets til: {out_path}")

# Vis evt. et par ekstra nøgletal
if extras:
    if "by_side_counts" in extras:
        print("Antal bets pr. udfald:", extras["by_side_counts"])
    if "by_side_hit" in extras:
        print("Hit rate pr. udfald:", {k: round(v,3) for k,v in extras["by_side_hit"].items()})
    if "by_side_roi" in extras:
        print("ROI pr. udfald:", {k: round(v,3) for k,v in extras["by_side_roi"].items()})
    if "by_league_n" in extras:
        print("Antal bets pr. liga:", extras["by_league_n"])
    if "by_league_roi" in extras:
        print("ROI pr. liga:", {k: round(v,3) for k,v in extras["by_league_roi"].items()})
