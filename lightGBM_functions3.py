
# value_betting_poisson.py
# ------------------------------------------------------------
# "Ultimativt" (kompakt) script til value betting med:
# - Poisson LightGBM for GH/GA
# - Skellam-konvolution til P(Δ)
# - DC-lite korrektion (ned med Δ=0, lidt op på ±1)
# - Varians-skalering (gamma) der bevarer margin (favorit-underskud)
# - (Valgfri) power-sharpen på 1X2
# - Backtest på sidste 20% med tau-udtagning
# ------------------------------------------------------------

from dataclasses import dataclass, field
import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from lightgbm import LGBMRegressor, early_stopping

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ------------------------------------------------------------
# Konfiguration
# ------------------------------------------------------------

@dataclass
class Config:
    league_dir: str                  # mappe med X.xlsx, Y.xlsx, Y_odds.xlsx
    x_file: str = "X.xlsx"
    y_file: str = "Y.xlsx"           # Y: to kolonner [GH, GA]
    y_odds_file: str = "Y_odds.xlsx" # Y_odds: [Result ('H','D','A'), pH, pD, pA]
    device: str = "gpu"              # 'gpu' eller 'cpu'
    random_state: int = 42

    # Modelparametre (LightGBM Poisson)
    learning_rate: float = 0.02
    num_leaves: int = 384
    min_child_samples: int = 12
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    reg_alpha: float = 0.0
    reg_lambda: float = 0.05
    max_bin: int = 255
    n_estimators: int = 20000
    es_rounds: int = 800

    # Skellam/kalibrering
    gd_values: np.ndarray = field(default_factory=lambda: np.arange(-5, 6))    # målforskel-støtte
    gamma_var: float = 1.10                     # varians-skalering (>1 => mindre draw)
    dc_strength: float = 0.05                   # flyt lidt masse fra Δ=0 til ±1
    alpha_power: float = 1.05                   # power-sharpen for 1X2 (1.0 = fra)
    # Markeds-blend (valgfrit – slå til/off med beta_max>0)
    beta_max: float = 0.25                      # vægt til p_book når margin er lav (0 = fra)
    use_k_shrink: bool = True
    k_grid: tuple = (0.0,0.1,0.2,0.3,0.4,0.5,0.7,1.0)  # kandidater til k

    # Backtest
    overround: float = 1.05                     # pålæg lille overround til book probs før odds
    stake: float = 100.0                        # flat stake pr. valgt bet
    tau_default: float = 0.05                   # default grænse for udtagning

    # --- ROI-tuning & bet-filtre ---
    auto_tune: bool = True
    tau_prob_grid: tuple = (0.00,0.02,0.04,0.06,0.08,0.10,0.12)
    tau_ev_grid:   tuple = (-0.05,0.00,0.02,0.03,0.04,0.05,0.08)
    min_margin: float = 0.05        # kræv top1-top2 >= min_margin
    max_odds: float = 6.0           # cap odds for at undgå lange haler
    include_draw: bool = False       # hvis False: drop D-bets helt
    draw_extra_ev: float = 0.02     # EV-særkrav oven i tau_ev for D (0 betyder samme)


# ------------------------------------------------------------
# Data-indlæsning
# ------------------------------------------------------------

def _read_excel(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Filen findes ikke: {path}")
    return pd.read_excel(path)

def load_data(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = _read_excel(os.path.join(cfg.league_dir, cfg.x_file)).to_numpy(dtype=float)
    Y_df = _read_excel(os.path.join(cfg.league_dir, cfg.y_file))
    if Y_df.shape[1] < 2:
        raise ValueError("Y.xlsx skal have mindst to kolonner: [GH, GA].")
    GH = Y_df.iloc[:, 0].fillna(0).astype(int).to_numpy()
    GA = Y_df.iloc[:, 1].fillna(0).astype(int).to_numpy()

    Yod = _read_excel(os.path.join(cfg.league_dir, cfg.y_odds_file))
    if Yod.shape[1] < 4:
        raise ValueError("Y_odds.xlsx skal have mindst 4 kolonner: [Result, pH, pD, pA].")
    res = Yod.iloc[:, 0].astype(str).to_numpy()
    book_probs = Yod.iloc[:, 1:4].astype(float).to_numpy()

    return X, GH, GA, res, book_probs

# ------------------------------------------------------------
# Poisson & Skellam hjælpefunktioner
# ------------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0.0:
        return 1.0 if k == 0 else 0.0
    if k < 0:
        return 0.0
    return math.exp(k*math.log(lam) - lam - math.lgamma(k+1))

def skellam_row(mu_h: float, mu_a: float, gd_values: np.ndarray) -> np.ndarray:
    # dynamisk truncation for at fange ~al masse
    max_h = int(max(0, math.ceil(mu_h + 7.0*math.sqrt(max(mu_h, 1e-9)))))
    max_a = int(max(0, math.ceil(mu_a + 7.0*math.sqrt(max(mu_a, 1e-9)))))
    Kmax  = max(20, max(max_h, max_a))

    PH = np.array([poisson_pmf(k, mu_h) for k in range(Kmax+1)], dtype=float)
    PA = np.array([poisson_pmf(k, mu_a) for k in range(Kmax+1)], dtype=float)

    out = np.zeros(len(gd_values), dtype=float)
    for i, k in enumerate(gd_values):
        if k >= 0:
            limit = Kmax - k
            if limit >= 0:
                out[i] = float(np.dot(PH[k:k+limit+1], PA[:limit+1]))
        else:
            kk = -k
            limit = Kmax - kk
            if limit >= 0:
                out[i] = float(np.dot(PH[:limit+1], PA[kk:kk+limit+1]))
    s = out.sum()
    return out / s if s > 0 else out

def widen_keep_margin(muH: np.ndarray, muA: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    # Bevar D, skaler S -> S' = gamma*S
    S = muH + muA
    D = muH - muA
    S_prime = gamma * S
    muH_prime = np.maximum((S_prime + D) * 0.5, 1e-6)
    muA_prime = np.maximum((S_prime - D) * 0.5, 1e-6)
    return muH_prime, muA_prime

def dc_adjust_pgd(pgd: np.ndarray, strength: float) -> np.ndarray:
    # Flyt en smule masse fra 0 til ±1
    if strength <= 0.0: 
        return pgd
    K = len(pgd); mid = K // 2
    delta = min(strength * pgd[mid], pgd[mid])
    if delta <= 1e-15:
        return pgd
    pgd = pgd.copy()
    pgd[mid] -= delta
    if mid-1 >= 0: pgd[mid-1] += 0.5 * delta
    if mid+1 < K: pgd[mid+1] += 0.5 * delta
    return pgd

def one_x_two_from_pgd(pgd: np.ndarray, gd_values: np.ndarray) -> Tuple[float,float,float]:
    # P(H) = P(Δ>0), P(D)=P(Δ=0), P(A)=P(Δ<0)
    mask_pos = gd_values > 0
    mask_zero = gd_values == 0
    pH = float(pgd[mask_pos].sum())
    pD = float(pgd[mask_zero].sum())
    pA = 1.0 - pH - pD
    return pH, pD, pA

def power_sharpen_rows(P: np.ndarray, alpha: float) -> np.ndarray:
    if alpha is None or abs(alpha-1.0) < 1e-9:
        return P
    P = np.clip(P, 1e-12, 1.0)
    P = P ** alpha
    return P / P.sum(axis=1, keepdims=True)

def market_blend(p_model: np.ndarray, p_book: np.ndarray, margin: np.ndarray, beta_max: float=0.25) -> np.ndarray:
    """Log-blanding: p ∝ p_model^1 * p_book^beta(margin). beta er størst når margin er lav."""
    if beta_max <= 0.0:
        return p_model
    # Skaler margin til [0,1] baseret på percentiler for robusthed
    lo = np.percentile(margin, 10); hi = np.percentile(margin, 90)
    m = np.clip((margin - lo) / max(hi - lo, 1e-9), 0, 1)
    beta = beta_max * (1.0 - m)  # høj beta ved lav margin
    # log-blanding = multiplikativ
    out = (p_model ** 1.0) * (p_book ** beta[:, None])
    out /= out.sum(axis=1, keepdims=True)
    return out

def _logit(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return np.log(p/(1-p))

def _inv_logit(z):
    return 1.0/(1.0+np.exp(-z))

def shrink_toward_market_logit(p_model: np.ndarray, p_book: np.ndarray, k: float) -> np.ndarray:
    """
    p' = logistic( logit(p_book) + k*(logit(p_model)-logit(p_book)) ), rækkevis for 3 udfald.
    """
    z_m = _logit(p_model)
    z_b = _logit(p_book)
    z = z_b + k*(z_m - z_b)
    P = _inv_logit(z)
    # normalisér, så H+D+A=1 numerisk stabilt
    P /= P.sum(axis=1, keepdims=True)
    return P

def _goals_to_outcome_labels(GH: np.ndarray, GA: np.ndarray) -> np.ndarray:
    # 'H','D','A' som strenge
    out = np.empty_like(GH, dtype=object)
    out[GH > GA] = 'H'
    out[GH == GA] = 'D'
    out[GH < GA] = 'A'
    return out

def _logloss_1x2(p: np.ndarray, y: np.ndarray) -> float:
    # p: (N,3), y: array af 'H','D','A'
    idx = {'H':0, 'D':1, 'A':2}
    rows = np.arange(len(y))
    col = np.array([idx[a] for a in y])
    return -float(np.mean(np.log(np.clip(p[rows, col], 1e-12, 1.0))))

def _roi_from_bets(take: np.ndarray, won: np.ndarray, picked_odds: np.ndarray, stake: float) -> Tuple[float,float,float]:
    payout = np.where(take & won, stake * picked_odds, 0.0)
    cost   = np.where(take, stake, 0.0)
    total_staked = float(cost.sum())
    total_return = float(payout.sum())
    profit = total_return - total_staked
    roi = (profit / total_staked) if total_staked > 0 else np.nan
    return total_staked, total_return, roi

def _apply_filters(pick_idx: np.ndarray, metric: np.ndarray, onextwo: np.ndarray, odds: np.ndarray,
                   cfg: Config, mode: str, tau: float) -> np.ndarray:
    """
    Returnér boolean mask for bets der tages:
      - metric: diff_max (mode='diff') eller EV_max (mode='ev')
      - tau   : tærskel på metric (for ev = EV)
      - min_margin, max_odds, include_draw, draw_extra_ev anvendes her.
    """
    # grundtærskel
    take = metric >= float(tau)

    # odds cap
    picked_odds = odds[np.arange(len(odds)), pick_idx]
    take &= (picked_odds <= cfg.max_odds)

    # margin-krav
    sortp = np.sort(onextwo, axis=1)[:, ::-1]
    margin = sortp[:,0] - sortp[:,1]
    take &= (margin >= cfg.min_margin)

    # udelad evt. uafgjort eller kræv ekstra EV for D
    if not cfg.include_draw:
        take &= (pick_idx != 1)
    if mode == "ev" and cfg.draw_extra_ev > 0.0:
        take &= ~((pick_idx == 1) & (metric < (tau + cfg.draw_extra_ev)))

    return take

# ------------------------------------------------------------
# Træning af to Poisson-modeller (GH/GA) og 80/20-split
# ------------------------------------------------------------

def train_poisson_models_80(X: np.ndarray, GH: np.ndarray, GA: np.ndarray, cfg: Config):
    n = len(X)
    cut80 = int(n * 0.8)
    cut_val = int(cut80 * 0.9)

    def fit_y(y: np.ndarray) -> LGBMRegressor:
        params = dict(
            objective="poisson",
            metric=["poisson"],
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            feature_fraction=cfg.feature_fraction,
            bagging_fraction=cfg.bagging_fraction,
            bagging_freq=cfg.bagging_freq,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            max_bin=cfg.max_bin,
            device=cfg.device,
            random_state=cfg.random_state,
            verbose=-1,
        )
        reg = LGBMRegressor(**params)
        X_tr, y_tr = X[:cut_val], y[:cut_val]
        X_val, y_val = X[cut_val:cut80], y[cut_val:cut80]

        w = np.ones_like(y_tr, dtype=float)
        w += 0.25 * (y_tr >= 2)  # lidt ekstra vægt på mål >= 2

        cb = [early_stopping(stopping_rounds=cfg.es_rounds, verbose=False)]
        reg.fit(X_tr, y_tr, sample_weight=w, eval_set=[(X_val, y_val)], callbacks=cb)
        return reg

    model_GH = fit_y(GH)
    model_GA = fit_y(GA)

    # ---------- NYT: find k* på validation ----------
    # Byg 1X2 på validation (samme pipeline som test)
    X_val = X[cut_val:cut80]
    muH_val = np.clip(model_GH.predict(X_val), 1e-6, 15.0)
    muA_val = np.clip(model_GA.predict(X_val), 1e-6, 15.0)

    if abs(cfg.gamma_var - 1.0) > 1e-9:
        muH_val, muA_val = widen_keep_margin(muH_val, muA_val, cfg.gamma_var)

    Pgd_val = np.zeros((len(X_val), len(cfg.gd_values)), dtype=float)
    for i in range(len(X_val)):
        pgd = skellam_row(float(muH_val[i]), float(muA_val[i]), cfg.gd_values)
        pgd = dc_adjust_pgd(pgd, cfg.dc_strength)
        Pgd_val[i, :] = pgd

    onextwo_val = np.zeros((len(X_val), 3), dtype=float)
    for i in range(len(X_val)):
        onextwo_val[i, :] = one_x_two_from_pgd(Pgd_val[i, :], cfg.gd_values)

    onextwo_val = power_sharpen_rows(onextwo_val, cfg.alpha_power)

    # Bookmaker 1X2 til validation (samme rækkefølge som X/Y)
    _, _, _, _, book_probs_all = load_data(cfg)
    book_val = book_probs_all[cut_val:cut80]

    # Facit (H/D/A) fra GH/GA
    GH_val = GH[cut_val:cut80]; GA_val = GA[cut_val:cut80]
    y_val_1x2 = _goals_to_outcome_labels(GH_val, GA_val)

    # Grid-søg efter k*
    best_k = 1.0
    best_loss = _logloss_1x2(onextwo_val, y_val_1x2)  # uden shrink
    if cfg.use_k_shrink:
        for k in cfg.k_grid:
            Pk = shrink_toward_market_logit(onextwo_val, book_val, k)
            loss = _logloss_1x2(Pk, y_val_1x2)
            if loss < best_loss:
                best_k, best_loss = k, loss

    # auto-tune τ på validation, hvis ønsket
    if cfg.auto_tune:
        best_k, tau_prob_star, tau_ev_star = _tune_thresholds_on_validation(
            model_GH, model_GA, X, GH, GA, cfg, cut_val, cut80, load_data(cfg)[4]
        )
    else:
        tau_prob_star, tau_ev_star = cfg.tau_default, cfg.tau_default

    return model_GH, model_GA, cut80, best_k, tau_prob_star, tau_ev_star


def _tune_thresholds_on_validation(model_GH, model_GA, X: np.ndarray, GH: np.ndarray, GA: np.ndarray,
                                   cfg: Config, cut_val: int, cut80: int, book_probs_all: np.ndarray):
    """
    Byg 1X2 på val, lav odds, og find tau_prob* og tau_ev* der maksimerer ROI (med dine filtre).
    """
    X_val = X[cut_val:cut80]
    muH = np.clip(model_GH.predict(X_val), 1e-6, 15.0)
    muA = np.clip(model_GA.predict(X_val), 1e-6, 15.0)
    if abs(cfg.gamma_var-1.0) > 1e-9:
        muH, muA = widen_keep_margin(muH, muA, cfg.gamma_var)

    # Skellam -> 1X2
    K = len(cfg.gd_values)
    Pgd = np.zeros((len(X_val), K), dtype=float)
    for i in range(len(X_val)):
        Pgd[i,:] = dc_adjust_pgd(skellam_row(float(muH[i]), float(muA[i]), cfg.gd_values), cfg.dc_strength)
    onextwo = np.zeros((len(X_val), 3), dtype=float)
    for i in range(len(X_val)):
        onextwo[i,:] = one_x_two_from_pgd(Pgd[i,:], cfg.gd_values)
    onextwo = power_sharpen_rows(onextwo, cfg.alpha_power)

    book_val = np.clip(book_probs_all[cut_val:cut80], 1e-12, 1.0)
    s = book_val.sum(axis=1, keepdims=True)
    scaled = book_val * (cfg.overround / np.clip(s, 1e-12, None))
    odds = 1.0 / np.clip(scaled, 1e-12, None)

    # shrink mod marked før tuning
    if cfg.use_k_shrink:
        # brug samme k-søgning som i train; vi kan genbruge onextwo_val ovenfor
        best_k, best_loss = 1.0, _logloss_1x2(onextwo, _goals_to_outcome_labels(GH[cut_val:cut80], GA[cut_val:cut80]))
        for k in cfg.k_grid:
            Pk = shrink_toward_market_logit(onextwo, book_val, k)
            loss = _logloss_1x2(Pk, _goals_to_outcome_labels(GH[cut_val:cut80], GA[cut_val:cut80]))
            if loss < best_loss:
                best_k, best_loss = k, loss
        onextwo = shrink_toward_market_logit(onextwo, book_val, best_k)
    else:
        best_k = 1.0

    # forbered diff, EV, picks
    diffs = onextwo - book_val
    EV_all = onextwo * odds - 1.0
    pick_idx_diff = np.argmax(diffs, axis=1)
    pick_val_diff = diffs[np.arange(len(diffs)), pick_idx_diff]
    pick_idx_ev   = np.argmax(EV_all, axis=1)
    pick_val_ev   = EV_all[np.arange(len(EV_all)), pick_idx_ev]

    actual = _goals_to_outcome_labels(GH[cut_val:cut80], GA[cut_val:cut80])
    idx2lab = {0:"H",1:"D",2:"A"}
    won_diff = (np.array([idx2lab[i] for i in pick_idx_diff]) == actual)
    won_ev   = (np.array([idx2lab[i] for i in pick_idx_ev])   == actual)

    # grid-søg τ’er
    best_tau_prob, best_tau_ev = cfg.tau_default, cfg.tau_default
    best_roi_prob, best_roi_ev = -1e9, -1e9

    for tau in cfg.tau_prob_grid:
        take = _apply_filters(pick_idx_diff, pick_val_diff, onextwo, odds, cfg, mode="diff", tau=tau)
        st, ret, roi = _roi_from_bets(take, won_diff, odds[np.arange(len(odds)), pick_idx_diff], cfg.stake)
        if np.isfinite(roi) and roi > best_roi_prob:
            best_roi_prob, best_tau_prob = roi, tau

    for tau in cfg.tau_ev_grid:
        take = _apply_filters(pick_idx_ev, pick_val_ev, onextwo, odds, cfg, mode="ev", tau=tau)
        st, ret, roi = _roi_from_bets(take, won_ev, odds[np.arange(len(odds)), pick_idx_ev], cfg.stake)
        if np.isfinite(roi) and roi > best_roi_ev:
            best_roi_ev, best_tau_ev = roi, tau

    return float(best_k), float(best_tau_prob), float(best_tau_ev)

# ------------------------------------------------------------
# Scoring af sidste 20% og afledning af P(Δ) -> 1X2
# ------------------------------------------------------------

def score_tail20_and_probs(model_GH, model_GA, X: np.ndarray, cut80: int, cfg: Config,
                           book_tail: np.ndarray=None) -> Tuple[np.ndarray, np.ndarray, Dict[str,float]]:
    X_te = X[cut80:]
    muH = np.clip(model_GH.predict(X_te), 1e-6, 15.0)
    muA = np.clip(model_GA.predict(X_te), 1e-6, 15.0)

    # Varians-skalering (bevarer margin)
    if cfg.gamma_var and abs(cfg.gamma_var-1.0) > 1e-9:
        muH, muA = widen_keep_margin(muH, muA, cfg.gamma_var)

    # Skellam (+ DC-lite)
    K = len(cfg.gd_values)
    Pgd_te = np.zeros((len(X_te), K), dtype=float)
    for i in range(len(X_te)):
        pgd = skellam_row(float(muH[i]), float(muA[i]), cfg.gd_values)
        pgd = dc_adjust_pgd(pgd, cfg.dc_strength)
        Pgd_te[i, :] = pgd

    # 1X2
    onextwo = np.zeros((len(X_te), 3), dtype=float)
    for i in range(len(X_te)):
        onextwo[i, :] = one_x_two_from_pgd(Pgd_te[i, :], cfg.gd_values)

    # Power-sharpen (valgfri)
    onextwo = power_sharpen_rows(onextwo, cfg.alpha_power)

    # (Valgfri) marked-blend (kun hvis book_tail gives)
    if book_tail is not None and cfg.beta_max > 0.0:
        # margin = top1 - top2 i modelens 1X2
        sortp = np.sort(onextwo, axis=1)[:, ::-1]
        margin = sortp[:, 0] - sortp[:, 1]
        onextwo = market_blend(onextwo, book_tail, margin, beta_max=cfg.beta_max)

    # Returns + simple diagnostics
    diag = dict(
        muH_mean=float(np.mean(muH)),
        muA_mean=float(np.mean(muA)),
        draw_mean=float(np.mean(onextwo[:,1])),
    )
    return Pgd_te, onextwo, diag

# ------------------------------------------------------------
# Backtest på tail 20% – BEGGE METODER (diff & ev) + auto-tunede τ
# ------------------------------------------------------------
def backtest_tail20(
    cfg: Config,
    tau_prob: float | None = None,
    tau_ev: float | None = None
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Træner på 80%, tuner k og τ på validation (i 80%), scorer tail 20% og
    backtester to udtagningsmetoder:
      - diff: pick max(model - book)  >= tau_prob
      - ev:   pick max(EV = p*odds-1) >= tau_ev

    Returnerer:
      summary (dict med nøgletal for begge metoder),
      bets_df (rækker for begge metoder),
      diffs_df (klassiske diff-kolonner),
      avg_pgd_df (diagnostik for P(GD)).
    """
    # --- data & træning ---
    X, GH, GA, res, book_probs = load_data(cfg)
    # train_poisson_models_80 forventes at returnere: model_GH, model_GA, cut80, best_k, tau_prob_star, tau_ev_star
    model_GH, model_GA, cut80, best_k, tau_prob_star, tau_ev_star = train_poisson_models_80(X, GH, GA, cfg)

    # odds i tail (med overround)
    book_tail = np.clip(book_probs[cut80:], 1e-12, 1.0)
    s = book_tail.sum(axis=1, keepdims=True)
    scaled_probs = book_tail * (cfg.overround / np.clip(s, 1e-12, None))
    odds_tail = 1.0 / np.clip(scaled_probs, 1e-12, None)

    # model-probs i tail
    Pgd_te, onextwo_te, diag = score_tail20_and_probs(model_GH, model_GA, X, cut80, cfg, book_tail=book_tail)

    # k-shrink mod marked (global) før udtagning
    if cfg.use_k_shrink:
        onextwo_te = shrink_toward_market_logit(onextwo_te, book_tail, best_k)

    # klassiske diffs (til tabel)
    diffs = onextwo_te - book_tail
    diffs_df = pd.DataFrame(diffs, columns=["Diff_H", "Diff_D", "Diff_A"])

    # thresholds (brug auto-tunede hvis ikke specificeret)
    tau_prob = float(tau_prob if tau_prob is not None else tau_prob_star)
    tau_ev   = float(tau_ev   if tau_ev   is not None else tau_ev_star)

    # pick pr. metode
    pick_idx_diff = np.argmax(diffs, axis=1)
    pick_val_diff = diffs[np.arange(len(diffs)), pick_idx_diff]

    EV_all = onextwo_te * odds_tail - 1.0
    pick_idx_ev = np.argmax(EV_all, axis=1)
    pick_val_ev = EV_all[np.arange(len(EV_all)), pick_idx_ev]

    # take-masker med filtre
    take_diff = _apply_filters(pick_idx_diff, pick_val_diff, onextwo_te, odds_tail, cfg, mode="diff", tau=tau_prob)
    take_ev   = _apply_filters(pick_idx_ev,   pick_val_ev,   onextwo_te, odds_tail, cfg, mode="ev",   tau=tau_ev)

    # facit & afregning
    actual_tail = res[cut80:]
    idx2lab = {0: "H", 1: "D", 2: "A"}

    # diff
    label_diff = np.array([idx2lab[i] for i in pick_idx_diff])
    won_diff = (label_diff == actual_tail)
    odds_diff = odds_tail[np.arange(len(odds_tail)), pick_idx_diff]
    st_diff, ret_diff, roi_diff = _roi_from_bets(take_diff, won_diff, odds_diff, cfg.stake)

    # ev
    label_ev = np.array([idx2lab[i] for i in pick_idx_ev])
    won_ev = (label_ev == actual_tail)
    odds_ev = odds_tail[np.arange(len(odds_tail)), pick_idx_ev]
    st_ev, ret_ev, roi_ev = _roi_from_bets(take_ev, won_ev, odds_ev, cfg.stake)

    # summaries
    summary = {
        "diff": dict(
            method="diff",
            n_matches=int(diffs.shape[0]),
            n_bets=int(take_diff.sum()),
            stake=float(cfg.stake),
            overround=float(cfg.overround),
            total_staked=round(st_diff, 2),
            total_return=round(ret_diff, 2),
            profit=round(ret_diff - st_diff, 2),
            roi=round(roi_diff, 4) if np.isfinite(roi_diff) else np.nan,
            tau_prob_used=round(tau_prob, 3),
            min_margin=cfg.min_margin,
            max_odds=cfg.max_odds,
            include_draw=cfg.include_draw,
            draw_extra_ev=cfg.draw_extra_ev,
            best_k=round(best_k, 3) if cfg.use_k_shrink else 1.0,
            draw_mean=round(diag["draw_mean"], 4),
            muH_mean=round(diag["muH_mean"], 3),
            muA_mean=round(diag["muA_mean"], 3),
        ),
        "ev": dict(
            method="ev",
            n_matches=int(diffs.shape[0]),
            n_bets=int(take_ev.sum()),
            stake=float(cfg.stake),
            overround=float(cfg.overround),
            total_staked=round(st_ev, 2),
            total_return=round(ret_ev, 2),
            profit=round(ret_ev - st_ev, 2),
            roi=round(roi_ev, 4) if np.isfinite(roi_ev) else np.nan,
            tau_ev_used=round(tau_ev, 3),
            min_margin=cfg.min_margin,
            max_odds=cfg.max_odds,
            include_draw=cfg.include_draw,
            draw_extra_ev=cfg.draw_extra_ev,
            best_k=round(best_k, 3) if cfg.use_k_shrink else 1.0,
            draw_mean=round(diag["draw_mean"], 4),
            muH_mean=round(diag["muH_mean"], 3),
            muA_mean=round(diag["muA_mean"], 3),
        )
    }

    # detaljeret bet-tabel (begge metoder)
    bets_df = pd.DataFrame({
        "Method": np.concatenate([np.full(diffs.shape[0], "diff"), np.full(diffs.shape[0], "ev")]),
        "Pick?":  np.concatenate([take_diff, take_ev]),
        "Pick":   np.concatenate([label_diff, label_ev]),
        "Actual": np.concatenate([actual_tail, actual_tail]),
        "Odds_used": np.concatenate([odds_diff, odds_ev]),
        "Won":    np.concatenate([won_diff, won_ev]),
        "Stake":  np.concatenate([np.where(take_diff, cfg.stake, 0.0),
                                  np.where(take_ev, cfg.stake, 0.0)]),
        "Payout": np.concatenate([np.where(take_diff & won_diff, cfg.stake * odds_diff, 0.0),
                                  np.where(take_ev & won_ev,     cfg.stake * odds_ev,   0.0)]),
        "Profit": np.concatenate([np.where(take_diff & won_diff, cfg.stake * odds_diff, 0.0) - np.where(take_diff, cfg.stake, 0.0),
                                  np.where(take_ev & won_ev,     cfg.stake * odds_ev,   0.0) - np.where(take_ev,   cfg.stake, 0.0)]),
        "Metric": np.concatenate([pick_val_diff, pick_val_ev]),  # diff_max / EV_max
        "Pick_idx": np.concatenate([pick_idx_diff, pick_idx_ev]),
        "Model_pH": np.concatenate([onextwo_te[:,0], onextwo_te[:,0]]),
        "Model_pD": np.concatenate([onextwo_te[:,1], onextwo_te[:,1]]),
        "Model_pA": np.concatenate([onextwo_te[:,2], onextwo_te[:,2]]),
        "Book_pH":  np.concatenate([book_tail[:,0],   book_tail[:,0]]),
        "Book_pD":  np.concatenate([book_tail[:,1],   book_tail[:,1]]),
        "Book_pA":  np.concatenate([book_tail[:,2],   book_tail[:,2]]),
    })

    # gennemsnitlig P(GD) til diagnose
    avg_probs = Pgd_te.mean(axis=0)
    avg_pgd_df = pd.DataFrame({"GD": cfg.gd_values, "MeanProb": avg_probs})

    return summary, bets_df, diffs_df, avg_pgd_df


# ------------------------------------------------------------
# Tau-sweep – BEGGE METODER (med filtre og auto-tunet k)
# ------------------------------------------------------------
def tau_sweep(
    cfg: Config,
    taus_prob: tuple = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12),
    taus_ev:   tuple = (-0.05, 0.00, 0.02, 0.03, 0.04, 0.05, 0.08)
) -> pd.DataFrame:
    """
    Træn én gang, brug auto-tunet k og scor tail 20%.
    Sweeper:
      - prob-τ for diff-metoden (model - book)
      - EV-τ   for EV-metoden (p*odds - 1)
    Anvender samme bet-filtre (min_margin, max_odds, draw-penalty).
    Returnerer samlet DataFrame med method='diff'/'ev'.
    """
    X, GH, GA, res, book_probs = load_data(cfg)
    model_GH, model_GA, cut80, best_k, tau_prob_star, tau_ev_star = train_poisson_models_80(X, GH, GA, cfg)

    # odds & model-probs i tail
    book_tail = np.clip(book_probs[cut80:], 1e-12, 1.0)
    s = book_tail.sum(axis=1, keepdims=True)
    scaled_probs = book_tail * (cfg.overround / np.clip(s, 1e-12, None))
    odds_tail = 1.0 / np.clip(scaled_probs, 1e-12, None)

    Pgd_te, onextwo_te, _ = score_tail20_and_probs(model_GH, model_GA, X, cut80, cfg, book_tail=book_tail)
    if cfg.use_k_shrink:
        onextwo_te = shrink_toward_market_logit(onextwo_te, book_tail, best_k)

    diffs = onextwo_te - book_tail
    EV_all = onextwo_te * odds_tail - 1.0

    pick_idx_diff = np.argmax(diffs, axis=1)
    pick_val_diff = diffs[np.arange(len(diffs)), pick_idx_diff]
    pick_idx_ev = np.argmax(EV_all, axis=1)
    pick_val_ev = EV_all[np.arange(len(EV_all)), pick_idx_ev]

    actual_tail = res[cut80:]
    idx2lab = {0:"H",1:"D",2:"A"}
    label_diff = np.array([idx2lab[i] for i in pick_idx_diff])
    label_ev   = np.array([idx2lab[i] for i in pick_idx_ev])
    won_diff = (label_diff == actual_tail)
    won_ev   = (label_ev   == actual_tail)
    odds_diff = odds_tail[np.arange(len(odds_tail)), pick_idx_diff]
    odds_ev   = odds_tail[np.arange(len(odds_tail)), pick_idx_ev]

    rows = []

    # sweep for diff
    for tau in taus_prob:
        take = _apply_filters(pick_idx_diff, pick_val_diff, onextwo_te, odds_tail, cfg, mode="diff", tau=float(tau))
        st, ret, roi = _roi_from_bets(take, won_diff, odds_diff, cfg.stake)
        rows.append(dict(
            method="diff",
            tau=round(float(tau),3),
            n_bets=int(take.sum()),
            total_staked=round(st,2),
            total_return=round(ret,2),
            profit=round(ret - st, 2),
            roi=round(roi,4) if np.isfinite(roi) else np.nan,
            best_k=round(best_k,3) if cfg.use_k_shrink else 1.0,
            min_margin=cfg.min_margin,
            max_odds=cfg.max_odds,
            include_draw=cfg.include_draw,
            draw_extra_ev=cfg.draw_extra_ev,
        ))

    # sweep for ev
    for tau in taus_ev:
        take = _apply_filters(pick_idx_ev, pick_val_ev, onextwo_te, odds_tail, cfg, mode="ev", tau=float(tau))
        st, ret, roi = _roi_from_bets(take, won_ev, odds_ev, cfg.stake)
        rows.append(dict(
            method="ev",
            tau=round(float(tau),3),
            n_bets=int(take.sum()),
            total_staked=round(st,2),
            total_return=round(ret,2),
            profit=round(ret - st, 2),
            roi=round(roi,4) if np.isfinite(roi) else np.nan,
            best_k=round(best_k,3) if cfg.use_k_shrink else 1.0,
            min_margin=cfg.min_margin,
            max_odds=cfg.max_odds,
            include_draw=cfg.include_draw,
            draw_extra_ev=cfg.draw_extra_ev,
        ))

    return pd.DataFrame(rows)

# ===================== FULD UDRULNING =====================

def train_full_models(cfg: Config):
    """
    Træn Poisson-modeller på HELE X. Bruger sidste 10% som validering
    til early-stopping og til at kalibrere k* (logit-shrink mod markedet).
    Returnerer (model_GH, model_GA, best_k, feat_names)
    """
    # læs data
    X = _read_excel(os.path.join(cfg.league_dir, cfg.x_file))
    feat_names = list(X.columns)
    X = X.to_numpy(dtype=float)
    Y = _read_excel(os.path.join(cfg.league_dir, cfg.y_file)).to_numpy(dtype=int)
    GH, GA = Y[:,0], Y[:,1]

    n = len(X); cut = int(n*0.9)
    def fit_y(y):
        params = dict(
            objective="poisson",
            metric=["poisson"],
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            feature_fraction=cfg.feature_fraction,
            bagging_fraction=cfg.bagging_fraction,
            bagging_freq=cfg.bagging_freq,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            max_bin=cfg.max_bin,
            device=cfg.device,
            random_state=cfg.random_state,
            verbose=-1,
        )
        reg = LGBMRegressor(**params)
        w = np.ones_like(y[:cut], dtype=float)
        w += 0.25*(y[:cut] >= 2)
        cb = [early_stopping(stopping_rounds=cfg.es_rounds, verbose=False)]
        reg.fit(X[:cut], y[:cut], sample_weight=w, eval_set=[(X[cut:], y[cut:])], callbacks=cb)
        return reg

    model_GH = fit_y(GH)
    model_GA = fit_y(GA)

    # --- kalibrér k* på hold-out (sidste 10%)
    # Byg 1X2 på val
    Xv = X[cut:]
    muH = np.clip(model_GH.predict(Xv), 1e-6, 15.0)
    muA = np.clip(model_GA.predict(Xv), 1e-6, 15.0)
    if abs(cfg.gamma_var-1.0)>1e-9:
        muH, muA = widen_keep_margin(muH, muA, cfg.gamma_var)

    K = len(cfg.gd_values)
    Pgd = np.zeros((len(Xv), K))
    for i in range(len(Xv)):
        Pgd[i,:] = dc_adjust_pgd(skellam_row(float(muH[i]), float(muA[i]), cfg.gd_values), cfg.dc_strength)

    P1x2 = np.array([one_x_two_from_pgd(Pgd[i,:], cfg.gd_values) for i in range(len(Xv))])
    P1x2 = power_sharpen_rows(P1x2, cfg.alpha_power)

    # Bookmaker til samme rækker
    book_probs = _read_excel(os.path.join(cfg.league_dir, cfg.y_odds_file)).iloc[cut:, 1:4].to_numpy(dtype=float)
    y_val = _goals_to_outcome_labels(GH[cut:], GA[cut:])

    best_k = 1.0
    best_loss = _logloss_1x2(P1x2, y_val)
    if cfg.use_k_shrink:
        for k in cfg.k_grid:
            Pk = shrink_toward_market_logit(P1x2, book_probs, k)
            loss = _logloss_1x2(Pk, y_val)
            if loss < best_loss:
                best_k, best_loss = k, loss

    return model_GH, model_GA, best_k, feat_names


def _build_fixture_rows(fixtures: np.ndarray, cfg: Config, windows=(5,)):
    """
    Byg feature-rækker for fixtures ved at genberegne long_df (shift(1)+rolling)
    på Alle_kampe_med_elo_og_odds.xlsx og trække SENESTE rad pr. hold.
    Returnerer (X_new, rows_meta) i samme kolonnerækkefølge som X.xlsx.
    """
    # læs feature-navne fra trænings-X for at holde samme kolonneorden
    feat_names = list(_read_excel(os.path.join(cfg.league_dir, cfg.x_file)).columns)

    # læs rå kampe og rekonstruér long_df med samme logik som build_xy_core
    raw = pd.read_excel(os.path.join(cfg.league_dir, "Alle_kampe_med_elo_og_odds.xlsx"))
    raw["Dato"] = pd.to_datetime(raw["Dato"], errors="coerce").dt.normalize()
    for c in ["HjemmeholdNavn","UdeholdNavn"]:
        raw[c] = raw[c].astype(str).str.strip()
    raw = raw.reset_index(drop=True)
    raw["match_id"] = raw.index

    def _mk_side(prefix_team, prefix_opp, role_flag):
        return pd.DataFrame({
            "match_id":      raw["match_id"],
            "Dato":          raw["Dato"],
            "Team":          raw[f"{prefix_team}Navn"].astype(str).str.strip(),
            "Opp":           raw[f"{prefix_opp}Navn"].astype(str).str.strip(),
            "ELO":           pd.to_numeric(raw[f"{prefix_team}ELO"], errors="coerce"),
            "OppELO":        pd.to_numeric(raw[f"{prefix_opp}ELO"], errors="coerce"),
            "GoalsFor":      pd.to_numeric(raw[f"{prefix_team}Mal"], errors="coerce"),
            "GoalsAg":       pd.to_numeric(raw[f"{prefix_opp}Mal"], errors="coerce"),
            "Skud":          pd.to_numeric(raw[f"{prefix_team}Skud"], errors="coerce"),
            "SkudPaMal":     pd.to_numeric(raw[f"{prefix_team}SkudPaMal"], errors="coerce"),
            "Possession":    pd.to_numeric(raw[f"{prefix_team}Possession"], errors="coerce"),
            "PassAccuracy":  pd.to_numeric(raw[f"{prefix_team}PassAccuracy"], errors="coerce"),
            "Fouls":         pd.to_numeric(raw[f"{prefix_team}Fouls"], errors="coerce"),
            "Corners":       pd.to_numeric(raw[f"{prefix_team}Corners"], errors="coerce"),
            "Role":          role_flag
        })

    long_df = pd.concat([_mk_side("Hjemmehold","Udehold","H"),
                         _mk_side("Udehold","Hjemmehold","A")], ignore_index=True)
    long_df = long_df.sort_values(["Team","Dato","match_id"]).reset_index(drop=True)
    long_df["RestDays"] = long_df.groupby("Team")["Dato"].diff().dt.days.astype("float")
    long_df["prior_games"] = long_df.groupby("Team").cumcount()

    roll_feats = ["GoalsFor","GoalsAg","Skud","SkudPaMal","Possession","PassAccuracy","Fouls","Corners"]
    grp = long_df.groupby("Team", group_keys=False)
    for W in windows:
        for col in roll_feats + ["OppELO","RestDays"]:
            long_df[f"{col}_avg{W}"] = grp[col].apply(lambda s: s.shift(1).rolling(W, min_periods=W).mean()).reset_index(level=0, drop=True)

    # Funktion der henter SENESTE rad for et hold (uanset H/A rolle)
    def _latest_row(team_name: str) -> pd.Series | None:
        s = long_df[long_df["Team"].str.casefold()==str(team_name).strip().casefold()]
        if s.empty:
            return None
        return s.iloc[-1]    # seneste kamp

    # Byg feature-række pr. fixture
    rows = []
    rows_meta = []
    X_cols = feat_names

    bases = ["GoalsFor","GoalsAg","Skud","SkudPaMal","Possession","PassAccuracy","Fouls","Corners","OppELO","RestDays"]
    for home, away, oddsH, oddsD, oddsA in fixtures:
        rH = _latest_row(home); rA = _latest_row(away)
        if rH is None or rA is None:
            # mangler historik → skip
            continue

        row = {c: np.nan for c in X_cols}

        # ELO + diff
        row["ELO_H"] = float(rH["ELO"]); row["ELO_A"] = float(rA["ELO"])
        if "ELO_diff" in row:
            row["ELO_diff"] = row["ELO_H"] - row["ELO_A"]

        # rolling H og A
        for W in windows:
            for b in bases:
                hcol = f"H_{b}_avg{W}"
                acol = f"A_{b}_avg{W}"
                if hcol in row: row[hcol] = float(rH.get(f"{b}_avg{W}", np.nan))
                if acol in row: row[acol] = float(rA.get(f"{b}_avg{W}", np.nan))
                dcol = f"Delta_{b}_avg{W}"
                if dcol in row:
                    vh = float(rH.get(f"{b}_avg{W}", np.nan))
                    va = float(rA.get(f"{b}_avg{W}", np.nan))
                    row[dcol] = vh - va if (np.isfinite(vh) and np.isfinite(va)) else np.nan

        # restdage "latest"
        if f"H_RestDays_latest" in row:
            row["H_RestDays_latest"] = float(rH.get(f"RestDays_avg{windows[0]}", np.nan))
        if f"A_RestDays_latest" in row:
            row["A_RestDays_latest"] = float(rA.get(f"RestDays_avg{windows[0]}", np.nan))
        if f"Delta_RestDays_latest" in row:
            vh = float(rH.get(f"RestDays_avg{windows[0]}", np.nan))
            va = float(rA.get(f"RestDays_avg{windows[0]}", np.nan))
            row["Delta_RestDays_latest"] = vh - va if (np.isfinite(vh) and np.isfinite(va)) else np.nan

        # odds fra fixtures – gem i meta til senere (ikke features)
        rows_meta.append(dict(Home=home, Away=away,
                              Odds_H=float(oddsH), Odds_D=float(oddsD), Odds_A=float(oddsA)))

        rows.append([row[c] if c in row else np.nan for c in X_cols])

    X_new = np.array(rows, dtype=float) if rows else np.zeros((0, len(X_cols)))
    return X_new, pd.DataFrame(rows_meta), feat_names


def predict_from_fixtures(fixtures: np.ndarray, cfg: Config,
                          model_GH, model_GA, best_k: float,
                          include_pgd: bool = True) -> pd.DataFrame:
    """
    Scorer fixtures og returnerer en tabel med Odds/Implied, Model_p, Model_odds (samme overround),
    EV_H/D/A, og (valgfrit) P(Δ=k) kolonner.
    """
    X_new, meta, feat_names = _build_fixture_rows(fixtures, cfg, windows=(5,))  # brug samme vinduer som i X
    if len(X_new)==0:
        return pd.DataFrame()

    # forudsig mål
    muH = np.clip(model_GH.predict(X_new), 1e-6, 15.0)
    muA = np.clip(model_GA.predict(X_new), 1e-6, 15.0)
    if abs(cfg.gamma_var-1.0)>1e-9:
        muH, muA = widen_keep_margin(muH, muA, cfg.gamma_var)

    # Skellam -> Pgd -> 1X2
    K = len(cfg.gd_values)
    Pgd = np.zeros((len(X_new), K))
    for i in range(len(X_new)):
        Pgd[i,:] = dc_adjust_pgd(skellam_row(float(muH[i]), float(muA[i]), cfg.gd_values), cfg.dc_strength)
    P1x2 = np.array([one_x_two_from_pgd(Pgd[i,:], cfg.gd_values) for i in range(len(X_new))])
    P1x2 = power_sharpen_rows(P1x2, cfg.alpha_power)

    # implied fra fixtures + overround
    imp = 1.0 / np.clip(meta[["Odds_H","Odds_D","Odds_A"]].to_numpy(dtype=float), 1e-9, None)
    over = imp.sum(axis=1, keepdims=True)  # faktisk overround i fixtures

    # (valgfri) k-shrink mod markedet
    if cfg.use_k_shrink and best_k is not None:
        P1x2 = shrink_toward_market_logit(P1x2, imp, best_k)

    # model-odds med SAMME overround som fixtures
    P_model_scaled = P1x2 * over
    model_odds = 1.0 / np.clip(P_model_scaled, 1e-9, None)

    # EV
    EV = P1x2 * meta[["Odds_H","Odds_D","Odds_A"]].to_numpy(dtype=float) - 1.0

    # saml output
    out = meta.copy()
    out["Impl_H"] = imp[:,0]; out["Impl_D"] = imp[:,1]; out["Impl_A"] = imp[:,2]
    out["Pred_H"] = P1x2[:,0]; out["Pred_D"] = P1x2[:,1]; out["Pred_A"] = P1x2[:,2]
    out["Predodds_H"] = model_odds[:,0]; out["Predodds_D"] = model_odds[:,1]; out["Predodds_A"] = model_odds[:,2]
    out["EV_H"] = EV[:,0]; out["EV_D"] = EV[:,1]; out["EV_A"] = EV[:,2]

    # (valgfrit) P(Δ=k) kolonner
    if include_pgd:
        for j, gd in enumerate(cfg.gd_values):
            out[f"Pgd_{gd:+d}"] = Pgd[:, j]

    return out

# ===================== ASIAN HANDICAP HELPERS =====================

def _split_quarter_line(line: float) -> list[float]:
    """
    Split en AH-linje til komponenter med .0 eller .5 (half-stakes).
    Eksempler:
      0.25 -> [0.0, 0.5]
      0.75 -> [0.5, 1.0]
      1.25 -> [1.0, 1.5]
    Returnerer en liste på 1 (heltal/.5) eller 2 (kvartlinje) elementer.
    """
    n = np.floor(line)
    frac = round(line - n, 2)
    if frac in (0.0, 0.5):
        return [float(line)]
    elif frac == 0.25:
        return [float(n), float(n + 0.5)]
    elif frac == 0.75:
        return [float(n + 0.5), float(n + 1.0)]
    else:
        # afrund til nærmeste .0/.25/.5/.75 for robusthed
        f = round(frac * 4) / 4
        if f in (0.0, 0.5):
            return [float(n + f)]
        elif f == 0.25:
            return [float(n), float(n + 0.5)]
        elif f == 0.75:
            return [float(n + 0.5), float(n + 1.0)]
        raise ValueError(f"Ugyldig AH-linje: {line}")

def _prob_home_component(Pgd_row: np.ndarray, gd_values: np.ndarray, comp: float) -> tuple[float,float,float]:
    """
    Sandsynlighed for Home -comp (én komponent, comp har .0 eller .5):
      - Win: GD - comp > 0
      - Push: GD - comp = 0 (kun hvis comp heltal)
      - Lose: GD - comp < 0
    Returnerer (p_win, p_push, p_lose).
    """
    n = int(np.floor(comp))
    frac = round(comp - n, 2)

    # map P(GD=k) via mask
    # find indexer for ==, >=, <=
    gd = gd_values

    if frac == 0.0:  # heltalslinje
        p_push = float(Pgd_row[gd == n].sum())
        p_win  = float(Pgd_row[gd >= (n + 1)].sum())
        p_lose = float(Pgd_row[gd <= (n - 1)].sum())
    elif frac == 0.5:  # halvtal -> ingen push
        p_push = 0.0
        p_win  = float(Pgd_row[gd >= (n + 1)].sum())
        p_lose = float(Pgd_row[gd <= n].sum())
    else:
        raise ValueError("Komponent skal ende på .0 eller .5")
    return p_win, p_push, p_lose

def _prob_away_component(Pgd_row: np.ndarray, gd_values: np.ndarray, comp: float) -> tuple[float,float,float]:
    """
    Sandsynlighed for Away +comp (samme comp som for Home -comp):
      - Win  (for Away): GD <  -comp
      - Push (for Away): GD == -comp (kun hvis comp heltal)
      - Lose (for Away): GD >  -comp
    """
    n = int(np.floor(comp))
    frac = round(comp - n, 2)
    gd = gd_values

    if frac == 0.0:
        p_push = float(Pgd_row[gd == -n].sum())
        p_win  = float(Pgd_row[gd <= -(n + 1)].sum())
        p_lose = float(Pgd_row[gd >= (n + 1)].sum())
    elif frac == 0.5:
        p_push = 0.0
        p_win  = float(Pgd_row[gd <= - (n + 1)].sum())  # <= -(n+1)  (da -comp = -(n+0.5))
        p_lose = float(Pgd_row[gd >= (n + 1)].sum())    # >= +(n+1)
    else:
        raise ValueError("Komponent skal ende på .0 eller .5")
    return p_win, p_push, p_lose

def _ev_from_probs(p_win: float, p_push: float, p_lose: float, odds: float) -> float:
    """
    EV per 1 kr. stake: p_win*(odds-1) - p_lose*1  (push -> 0)
    """
    return p_win * (odds - 1.0) - p_lose


def predict_from_fixtures_ah(fixtures: np.ndarray,
                             cfg: Config,
                             model_GH,
                             model_GA,
                             line: float | list[float] | np.ndarray,
                             include_pgd: bool = False) -> pd.DataFrame:
    """
    Asian Handicap scoring for fixtures.
    fixtures: np.array([...,[Home, Away, odds_home, odds_away],...], dtype=object)
    line: én float (samme for alle) ELLER array/list med samme længde som fixtures.
          Standard: "Home -line / Away +line".
    Returnerer DataFrame med:
      Home, Away, Line, Odds_H, Odds_A,
      pH_win/push/lose, pA_win/push/lose, EV_H, EV_A,
      samt (valgfrit) Pgd_* kolonner.
    """
    # 1) byg feature-rækker (samme pipeline som i 1X2-udgaven)
    X_new, meta, feat_names = _build_fixture_rows(fixtures, cfg, windows=(5,))
    if len(X_new) == 0:
        return pd.DataFrame()

    # 2) forudsig mål -> P(GD)
    muH = np.clip(model_GH.predict(X_new), 1e-6, 15.0)
    muA = np.clip(model_GA.predict(X_new), 1e-6, 15.0)
    if abs(cfg.gamma_var - 1.0) > 1e-9:
        muH, muA = widen_keep_margin(muH, muA, cfg.gamma_var)

    K = len(cfg.gd_values)
    Pgd = np.zeros((len(X_new), K))
    for i in range(len(X_new)):
        Pgd[i, :] = dc_adjust_pgd(skellam_row(float(muH[i]), float(muA[i]), cfg.gd_values), cfg.dc_strength)

    # 3) for hver kamp: beregn AH-probabiliteter og EV for Home -L og Away +L
    lines = np.array(line if np.ndim(line) else [line]*len(meta), dtype=float)
    out_rows = []
    for i in range(len(meta)):
        L = float(lines[i])
        comps = _split_quarter_line(abs(L))  # vi bruger absolut linje; Home får -L

        # aggreger over komponenter (half-stakes)
        pH_win = pH_push = pH_lose = 0.0
        pA_win = pA_push = pA_lose = 0.0
        for c in comps:
            # Home -c
            w = 1.0 / len(comps)  # halv/halv ved kvartlinjer
            w_win, w_push, w_lose = _prob_home_component(Pgd[i, :], cfg.gd_values, c)
            pH_win  += w * w_win
            pH_push += w * w_push
            pH_lose += w * w_lose
            # Away +c
            w_winA, w_pushA, w_loseA = _prob_away_component(Pgd[i, :], cfg.gd_values, c)
            pA_win  += w * w_winA
            pA_push += w * w_pushA
            pA_lose += w * w_loseA

        oddsH = float(meta.loc[i, "Odds_H"])
        oddsA = float(meta.loc[i, "Odds_A"])
        EV_H = _ev_from_probs(pH_win, pH_push, pH_lose, oddsH)
        EV_A = _ev_from_probs(pA_win, pA_push, pA_lose, oddsA)

        row = dict(
            Home=meta.loc[i, "Home"],
            Away=meta.loc[i, "Away"],
            Line=L,
            Odds_H=oddsH, Odds_A=oddsA,
            pH_win=pH_win, pH_push=pH_push, pH_lose=pH_lose,
            pA_win=pA_win, pA_push=pA_push, pA_lose=pA_lose,
            EV_H=EV_H, EV_A=EV_A
        )
        if include_pgd:
            for j, gd in enumerate(cfg.gd_values):
                row[f"Pgd_{gd:+d}"] = Pgd[i, j]
        out_rows.append(row)

    return pd.DataFrame(out_rows)
