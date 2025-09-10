# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize

# =========================
# Konfiguration / filstier
# =========================
BASE = "Alternative modeller (forsøg)/"
PATH_X  = BASE + "X.xlsx"                      # features (samme rækkefølge som Y,M,PM)
PATH_Y  = BASE + "Y.xlsx"                      # markedets sandsynligheder (valgfrit)
PATH_M  = BASE + "M.xlsx"                      # indeholder 'FuldtidsResultat'
PATH_PM = BASE + "predicted_matches.xlsx"      # navne/metadata pr. kamp (samme rækkefølge)
OUTFILE = BASE + "BTD_model_probs.xlsx"

# ==============
# Hjælpefunktioner
# ==============
def safe_logit(p, eps=1e-12):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))

def log_loss(P, y):
    eps = 1e-12
    P = np.clip(P, eps, 1-eps)
    return -np.mean(np.log(P[np.arange(len(y)), y]))

def rps(P, y):
    Yoh = np.zeros_like(P)
    Yoh[np.arange(len(y)), y] = 1.0
    Pcum = np.cumsum(P, axis=1)
    Ycum = np.cumsum(Yoh, axis=1)
    return np.mean(np.sum((Pcum - Ycum)**2, axis=1) / 2.0)

def simplex_project(P):
    s = P.sum(1, keepdims=True)
    s = np.where(s==0, 1.0, s)
    return P / s

def guess_market_cols(df):
    candH = [c for c in df.columns if c.lower() in ["ph","prob_h","p_h","home_prob","h_prob","h_prob_norm"]]
    candD = [c for c in df.columns if c.lower() in ["pd","prob_d","p_d","draw_prob","u_prob","x_prob"]]
    candA = [c for c in df.columns if c.lower() in ["pa","prob_a","p_a","away_prob","b_prob","a_prob"]]
    if candH and candD and candA:
        pH = df[candH[0]].to_numpy()
        pD = df[candD[0]].to_numpy()
        pA = df[candA[0]].to_numpy()
        s = pH + pD + pA
        return pH/s, pD/s, pA/s
    return None

def pick_cols(df):
    """Find (hjemme, ude, dato, liga) kolonner i predicted_matches."""
    cols = {k: None for k in ["home","away","date","league"]}
    cand_home = [c for c in df.columns if any(k in c.lower() for k in ["hjemme","home"]) and "odds" not in c.lower()]
    cand_away = [c for c in df.columns if any(k in c.lower() for k in ["ude","away"]) and "odds" not in c.lower()]
    cand_date = [c for c in df.columns if any(k in c.lower() for k in ["dato","date","kickoff","match_date"])]
    cand_leag = [c for c in df.columns if any(k in c.lower() for k in ["liga","league","competition"])]
    if cand_home: cols["home"]   = cand_home[0]
    if cand_away: cols["away"]   = cand_away[0]
    if cand_date: cols["date"]   = cand_date[0]
    if cand_leag: cols["league"] = cand_leag[0]
    return cols

# =================
# 1) Indlæs data
# =================
X_df  = pd.read_excel(PATH_X)
Y_df  = pd.read_excel(PATH_Y)      # må gerne mangle market-probs
M_df  = pd.read_excel(PATH_M)
PM_df = pd.read_excel(PATH_PM)     # kampnavne/metadata

assert len(X_df)==len(Y_df)==len(M_df)==len(PM_df), "X, Y, M og predicted_matches skal have samme længde/rækkefølge."

# ===========================
# 2) Labels (H/D/A -> 0/1/2)
# ===========================
labels_raw = M_df["FuldtidsResultat"].astype(str).str.strip().str.upper()
label_map = {"H":0, "1":0,
             "D":1, "X":1, "U":1,
             "A":2, "2":2, "B":2}
y_series = labels_raw.map(label_map)

bad_mask = y_series.isna()
if bad_mask.any():
    print("⚠️ Ugyldige/manglende labels i M['FuldtidsResultat']:")
    print(M_df.loc[bad_mask, ["FuldtidsResultat"]])
    print("Rækkenumre:", M_df.index[bad_mask].to_list())

train_mask = ~bad_mask
X_train_df = X_df.loc[train_mask].reset_index(drop=True)
y = y_series.loc[train_mask].astype("int64").to_numpy()
X_pred_df = X_df.loc[~train_mask].reset_index(drop=True)

# ===========================
# 3) Market-probs (valgfrit)
# ===========================
market_full = guess_market_cols(Y_df)
market_train = market_pred = None
if market_full is not None:
    pH_mkt, pD_mkt, pA_mkt = market_full
    market_train = np.vstack([pH_mkt[train_mask], pD_mkt[train_mask], pA_mkt[train_mask]]).T
    if (~train_mask).any():
        market_pred = np.vstack([pH_mkt[~train_mask], pD_mkt[~train_mask], pA_mkt[~train_mask]]).T

# ==========================================
# 4) Feature-rens: numeriske, finite, varians, anti-kollinearitet, standardisering
# ==========================================
def prepare_X(df):
    X = df.select_dtypes(include=[np.number]).to_numpy().astype(float)
    finite_mask = np.isfinite(X).all(axis=0)
    X = X[:, finite_mask]
    var = X.var(axis=0)
    keep_var = var > 1e-8
    X = X[:, keep_var]
    if X.shape[1] > 1:
        C = np.corrcoef(X, rowvar=False)
        drop = set(); thr = 0.999
        for i in range(C.shape[0]):
            if i in drop: continue
            for j in range(i+1, C.shape[0]):
                if abs(C[i,j]) > thr: drop.add(j)
        keep_idx = [k for k in range(C.shape[0]) if k not in drop]
        X = X[:, keep_idx]
    mu = X.mean(0); sd = X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd

Xz_train, mu, sd = prepare_X(X_train_df)

def transform_X(df, mu, sd):
    X = df.select_dtypes(include=[np.number]).to_numpy().astype(float)
    return (X - mu) / sd

Xz_pred = transform_X(X_pred_df, mu, sd) if len(X_pred_df) else None

n, p = Xz_train.shape
print(f"Trænings-matrix: n={n}, p={p} (efter rens)")

# ====================================
# 5) BTD-likelihood + robust optimering
# ====================================
def make_nll(Xz, y, l2=1e-2):
    n, p = Xz.shape
    y = y.astype(int)
    def nll(theta):
        beta = theta[:-2]; b0 = theta[-2]
        log_nu = np.clip(theta[-1], -5.0, 5.0)
        nu = np.exp(log_nu)
        z = b0 + Xz @ beta
        z = np.clip(z, -50, 50)
        r = np.exp(z); sr = np.sqrt(r)
        denom = r + 1.0 + 2.0*nu*sr
        pH = r / denom; pA = 1.0 / denom; pD = (2.0*nu*sr) / denom
        P = np.clip(np.vstack([pH, pD, pA]).T, 1e-12, 1-1e-12)
        ll = np.log(P[np.arange(n), y])
        pen = l2 * np.sum(beta*beta)
        return -(ll.sum() - pen)
    return nll

nll = make_nll(Xz_train, y, l2=1e-2)

best = None
for seed in [0,1,2,3,4]:
    rng = np.random.default_rng(seed)
    theta0 = np.zeros(p + 2)
    theta0[:-2] = rng.normal(0, 0.01, size=p)
    theta0[-2] = 0.0; theta0[-1] = 0.0
    res = minimize(
        nll, theta0, method="L-BFGS-B",
        bounds=[(None,None)]*(p+1) + [(-5.0, 5.0)],
        options={"maxiter": 2000, "maxfun": 200000, "ftol": 1e-9}
    )
    if (best is None) or (res.fun < best.fun): best = res

res = best
if not res.success:
    print("⚠️ Optimering ikke success, fortsætter med bedste fund:", res.message)
theta_hat = res.x
beta_hat = theta_hat[:-2]; b0_hat = theta_hat[-2]; nu_hat = float(np.exp(theta_hat[-1]))
print(f"Konvergerede med nu={nu_hat:.4f}")

# ===============================
# 6) Probabiliteter (train & pred)
# ===============================
def btd_probs(Xz, beta, b0, nu):
    z = b0 + Xz @ beta
    z = np.clip(z, -50, 50)
    r = np.exp(z); sr = np.sqrt(r)
    den = r + 1.0 + 2.0*nu*sr
    P = np.vstack([r/den, (2.0*nu*sr)/den, 1.0/den]).T  # [H,D,A]
    return simplex_project(P)

P_mod_tr = btd_probs(Xz_train, beta_hat, b0_hat, nu_hat)
ll_mod = log_loss(P_mod_tr, y); rps_mod = rps(P_mod_tr, y)
print(f"BTD — LogLoss: {ll_mod:.4f} | RPS: {rps_mod:.4f}")

# Market benchmark på træningsdelen
if market_train is not None:
    ll_mkt = log_loss(market_train, y); rps_mkt = rps(market_train, y)
    print(f"Market — LogLoss: {ll_mkt:.4f} | RPS: {rps_mkt:.4f}")
    print(f"ΔLogLoss (model - market): {ll_mod - ll_mkt:+.4f} (negativt er bedre)")

# Valgfri blending (hold-out: sidste 20% af TRAIN)
best_w = None
if market_train is not None:
    n_tr = len(y); cut = int(n_tr*0.8)
    Pm_tr, Pk_tr, y_tr = P_mod_tr[:cut], market_train[:cut], y[:cut]
    ws = np.linspace(0, 1, 21)
    def blend(Pm, Pk, w):
        out = np.zeros_like(Pm)
        for j in range(3):
            out[:, j] = expit(w*safe_logit(Pm[:, j]) + (1-w)*safe_logit(Pk[:, j]))
        return simplex_project(out)
    best_w, best_ll = 0.0, 1e9
    for w in ws:
        ll = log_loss(blend(Pm_tr, Pk_tr, w), y_tr)
        if ll < best_ll: best_ll, best_w = ll, w
    print(f"Optimal w (val på train): {best_w:.2f}")

# Forecast til rækker uden resultat (hvis nogen)
P_mod_pred = P_blend_pred = None
if len(X_pred_df):
    P_mod_pred = btd_probs(Xz_pred, beta_hat, b0_hat, nu_hat)
    if (market_pred is not None) and (best_w is not None):
        def blend(Pm, Pk, w):
            out = np.zeros_like(Pm)
            for j in range(3):
                out[:, j] = expit(w*safe_logit(Pm[:, j]) + (1-w)*safe_logit(Pk[:, j]))
            return simplex_project(out)
        P_blend_pred = blend(P_mod_pred, market_pred, best_w)

# ========================
# 7) Gem samlet output
# ========================
out = pd.DataFrame(index=range(len(X_df)))

# Tilføj kamp-info (fra predicted_matches)
pm_cols = pick_cols(PM_df)
if pm_cols["home"] is not None:   out["Hjemmehold"] = PM_df[pm_cols["home"]].astype(str)
if pm_cols["away"] is not None:   out["Udehold"]    = PM_df[pm_cols["away"]].astype(str)
if pm_cols["date"] is not None:   out["Dato"]       = PM_df[pm_cols["date"]]
if pm_cols["league"] is not None: out["Liga"]       = PM_df[pm_cols["league"]].astype(str)

# Tilføj odds og probs fra predicted_matches, hvis de findes
extra_cols = ["FuldtidsResultat","AVG H","AVG D","AVG A","PROB H","PROB D","PROB A"]
for c in extra_cols:
    if c in PM_df.columns:
        out[c] = PM_df[c]

# Model/market/blend probabilities for alle rækker
out.loc[train_mask, ["pH_mod","pD_mod","pA_mod"]] = P_mod_tr
if market_full is not None:
    out[["pH_mkt","pD_mkt","pA_mkt"]] = np.vstack([pH_mkt, pD_mkt, pA_mkt]).T
if (market_train is not None) and (best_w is not None):
    out.loc[train_mask, ["pH_blend","pD_blend","pA_blend"]] = blend(P_mod_tr, market_train, best_w)

# For rækker uden resultat
if P_mod_pred is not None:
    out.loc[~train_mask, ["pH_mod","pD_mod","pA_mod"]] = P_mod_pred
if P_blend_pred is not None:
    out.loc[~train_mask, ["pH_blend","pD_blend","pA_blend"]] = P_blend_pred

# Metadata
meta = {"nu_hat": nu_hat, "p_features": p,
        "n_train": int(train_mask.sum()), "n_pred": int((~train_mask).sum())}
for k,v in meta.items(): out[k] = v

out.to_excel(OUTFILE, index=False)
print(f"Skrev: {OUTFILE}")