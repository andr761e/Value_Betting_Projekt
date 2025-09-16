# functions.py
import numpy as np
import pandas as pd
import pickle, hashlib, os
from dataclasses import dataclass
from lightgbm import LGBMRegressor, early_stopping
from numpy.polynomial.polynomial import Polynomial

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

# ---------- Konfiguration ----------
@dataclass
class LeagueConfig:
    league_dir: str                         # fx "English Premier League"
    x_file: str = "X.xlsx"                  # sti relativt til league_dir
    y_file: str = "Y.xlsx"
    matches_file: str = "Alle_kampe_med_elo.xlsx"
    elo_pkl: str = "elo_dict_premier_league.pkl"
    out_xpred: str = "Træning og forudsigelse/X_pred.xlsx"
    device: str = "gpu"                     # "gpu" eller "cpu"
    random_state: int = 42

# ---------- Hjælpere (hash, ELO, afledte features) ----------
def hash64_bits(name: str) -> np.ndarray:
    if pd.isna(name):
        name = ""
    h = hashlib.sha256(str(name).encode("utf-8")).digest()[:8]
    x = int.from_bytes(h, byteorder="big", signed=False)
    return np.array([(x >> (63 - i)) & 1 for i in range(64)], dtype=np.float32)

def _extract_year(val):
    if pd.isna(val):
        return None
    try:
        return pd.to_datetime(val).year
    except Exception:
        s = str(val);  return int(s[:4]) if s[:4].isdigit() else None

def direct_normalization(x):
    return x / np.sum(x, axis=1, keepdims=True)

def polynomial_calibration(Y_pred_proba, Y_test, degree=3):
    calibrated_probs = []
    for i in range(Y_pred_proba.shape[1]):
        poly = Polynomial.fit(Y_pred_proba[:, i], Y_test[:, i], deg=degree)
        calibrated_probs.append(poly(Y_pred_proba[:, i]))
    return np.column_stack(calibrated_probs)

def _elo_lookup(elo_dict, team, ref_year=None):
    d = elo_dict.get(team, None)
    if d is None: return np.nan
    try: keys = sorted(int(k) for k in d.keys())
    except Exception: keys = sorted(d.keys())
    if not keys: return np.nan
    if ref_year is None: return float(d[keys[-1]])
    le_keys = [k for k in keys if k <= ref_year]
    if le_keys: return float(d[max(le_keys)])
    closest = min(keys, key=lambda k: abs(k - ref_year))
    return float(d[closest])

def _safe_div(n, d):
    n = float(n) if pd.notna(n) else np.nan
    d = float(d) if pd.notna(d) else np.nan
    if d == 0 or np.isnan(d): return np.nan
    return n / d

def _row_derived_for_team(r, team_is_home: bool):
    if team_is_home:
        g,s,sot = r.get("HjemmeholdMal",np.nan), r.get("HjemmeholdSkud",np.nan), r.get("HjemmeholdSkudPaMal",np.nan)
        g_opp,s_opp,sot_opp = r.get("UdeholdMal",np.nan), r.get("UdeholdSkud",np.nan), r.get("UdeholdSkudPaMal",np.nan)
        passes,passes_opp = r.get("HjemmeholdPasses",np.nan), r.get("UdeholdPasses",np.nan)
        succ_pass,passacc = r.get("HjemmeholdSuccesfulPasses",np.nan), r.get("HjemmeholdPassAccuracy",np.nan)
        crosses,long_balls = r.get("HjemmeholdCrosses",np.nan), r.get("HjemmeholdLong_balls",np.nan)
        aerials,aerials_opp = r.get("HjemmeholdAerials_won",np.nan), r.get("UdeholdAerials_won",np.nan)
        tackles,inter = r.get("HjemmeholdTackles",np.nan), r.get("HjemmeholdInterceptions",np.nan)
        touches,touches_opp = r.get("HjemmeholdTouches",np.nan), r.get("UdeholdTouches",np.nan)
        corners = r.get("HjemmeholdCorners",np.nan)
    else:
        g,s,sot = r.get("UdeholdMal",np.nan), r.get("UdeholdSkud",np.nan), r.get("UdeholdSkudPaMal",np.nan)
        g_opp,s_opp,sot_opp = r.get("HjemmeholdMal",np.nan), r.get("HjemmeholdSkud",np.nan), r.get("HjemmeholdSkudPaMal",np.nan)
        passes,passes_opp = r.get("UdeholdPasses",np.nan), r.get("HjemmeholdPasses",np.nan)
        succ_pass,passacc = r.get("UdeholdSuccesfulPasses",np.nan), r.get("UdeholdPassAccuracy",np.nan)
        crosses,long_balls = r.get("UdeholdCrosses",np.nan), r.get("UdeholdLong_balls",np.nan)
        aerials,aerials_opp = r.get("UdeholdAerials_won",np.nan), r.get("HjemmeholdAerials_won",np.nan)
        tackles,inter = r.get("UdeholdTackles",np.nan), r.get("UdeholdInterceptions",np.nan)
        touches,touches_opp = r.get("UdeholdTouches",np.nan), r.get("HjemmeholdTouches",np.nan)
        corners = r.get("UdeholdCorners",np.nan)

    conv = _safe_div(g, s)
    sot_rate = _safe_div(sot, s)
    def_conv = _safe_div(g_opp, s_opp)
    def_sot_rate = _safe_div(sot_opp, s_opp)
    cross_rate = _safe_div(crosses, passes)
    long_ball_share = _safe_div(long_balls, passes)
    pass_succ_rate = (float(passacc)/100.0) if pd.notna(passacc) else _safe_div(succ_pass, passes)
    aerial_den = (float(aerials)+float(aerials_opp)) if (pd.notna(aerials) and pd.notna(aerials_opp)) else np.nan
    aerial_win_rate = _safe_div(aerials, aerial_den)
    defensive_intensity = _safe_div((float(tackles) if pd.notna(tackles) else 0.0) +
                                    (float(inter) if pd.notna(inter) else 0.0), passes_opp)
    field_tilt = _safe_div(passes, (float(passes) if pd.notna(passes) else 0.0) +
                                  (float(passes_opp) if pd.notna(passes_opp) else 0.0))
    touch_tilt = _safe_div(touches, (float(touches) if pd.notna(touches) else 0.0) +
                                  (float(touches_opp) if pd.notna(touches_opp) else 0.0))
    set_piece_pressure = _safe_div((float(corners) if pd.notna(corners) else 0.0) +
                                   (float(crosses) if pd.notna(crosses) else 0.0), passes)
    return np.array([
        conv, sot_rate, def_conv, def_sot_rate,
        cross_rate, long_ball_share, pass_succ_rate,
        aerial_win_rate, defensive_intensity, field_tilt,
        touch_tilt, set_piece_pressure
    ], dtype=float)

# ---------- Indlæsning af data + navne ----------
def load_league_resources(cfg: LeagueConfig):
    league = cfg.league_dir
    X = pd.read_excel(os.path.join(league, cfg.x_file)).to_numpy()
    Y = pd.read_excel(os.path.join(league, cfg.y_file)).to_numpy()

    # matches
    columns_to_use = ["Dato","HjemmeholdNavn","HjemmeholdELO","HjemmeholdMal","HjemmeholdSkud","HjemmeholdSkudPaMal",
                      "HjemmeholdPossession","HjemmeholdFouls","HjemmeholdCorners","HjemmeholdCrosses","HjemmeholdTouches",
                      "HjemmeholdTackles","HjemmeholdInterceptions","HjemmeholdAerials_won","HjemmeholdClearances",
                      "HjemmeholdOffsides","HjemmeholdGoal_kicks","HjemmeholdThrow_ins","HjemmeholdLong_balls",
                      "HjemmeholdPasses","HjemmeholdSuccesfulPasses","HjemmeholdPassAccuracy",
                      "UdeholdNavn","UdeholdELO","UdeholdMal","UdeholdSkud","UdeholdSkudPaMal",
                      "UdeholdPossession","UdeholdFouls","UdeholdCorners","UdeholdCrosses","UdeholdTouches",
                      "UdeholdTackles","UdeholdInterceptions","UdeholdAerials_won","UdeholdClearances",
                      "UdeholdOffsides","UdeholdGoal_kicks","UdeholdThrow_ins","UdeholdLong_balls",
                      "UdeholdPasses","UdeholdSuccesfulPasses","UdeholdPassAccuracy"]
    matches = pd.read_excel(os.path.join(league, cfg.matches_file), usecols=columns_to_use)

    # elo_dict
    with open(os.path.join(league, cfg.elo_pkl), "rb") as f:
        elo_dict = pickle.load(f)

    # stat-navne (bevarer rækkefølge)
    _home_exclude = {"HjemmeholdNavn","HjemmeholdELO","HjemmeholdDato","Dato"}
    home_stats_cols = [c for c in matches.columns if c.startswith("Hjemmehold") and c not in _home_exclude]
    def _strip_prefix(col, prefix): return col[len(prefix):]
    stat_names = [_strip_prefix(c, "Hjemmehold") for c in home_stats_cols]
    stat_names = [s for s in stat_names if ("Udehold"+s) in matches.columns]

    derived_names = [
        "conv","sot_rate","def_conv","def_sot_rate",
        "cross_rate","long_ball_share","pass_succ_rate",
        "aerial_win_rate","defensive_intensity","field_tilt",
        "touch_tilt","set_piece_pressure"
    ]

    # skaleringsparametre til elo_dict -> [0,5]
    all_elos = []
    for _t, _years in elo_dict.items():
        for _y, _v in _years.items():
            try: all_elos.append(float(_v))
            except: pass
    if len(all_elos) == 0:
        elo_min, elo_max = 0.0, 1.0
    else:
        elo_min, elo_max = float(np.nanmin(all_elos)), float(np.nanmax(all_elos))

    def scale_dict_elo(v):
        if pd.isna(v): return np.nan
        if elo_max == elo_min: return 2.5
        return float(np.clip(5.0 * (float(v)-elo_min)/(elo_max-elo_min), 0.0, 5.0))

    return X, Y, matches, elo_dict, stat_names, derived_names, scale_dict_elo

# ---------- Feature-bygning til én kamp ----------
def _last5_weighted_stats(team: str, matches: pd.DataFrame, stat_names):
    idx = matches.index[(matches["HjemmeholdNavn"]==team) | (matches["UdeholdNavn"]==team)]
    if len(idx) < 5: return None, None

    last5 = idx[-5:]
    base_rows, derived_rows, opp_elos, years = [], [], [], []

    for i in last5:
        r = matches.loc[i]
        y = _extract_year(r["Dato"]) if "Dato" in matches.columns else None
        if y is not None: years.append(y)
        if r["HjemmeholdNavn"] == team:
            base_rows.append([r.get("Hjemmehold"+s, np.nan) for s in stat_names])
            derived_rows.append(_row_derived_for_team(r, True))
            opp_elos.append(r.get("UdeholdELO", np.nan))
        else:
            base_rows.append([r.get("Udehold"+s, np.nan) for s in stat_names])
            derived_rows.append(_row_derived_for_team(r, False))
            opp_elos.append(r.get("HjemmeholdELO", np.nan))

    base_arr = np.array(base_rows, dtype=float)
    der_arr  = np.array(derived_rows, dtype=float)
    base_mean, der_mean = np.nanmean(base_arr, axis=0), np.nanmean(der_arr, axis=0)

    opp_elo_avg = float(np.nanmean(np.array(opp_elos, dtype=float)))
    base_weighted = base_mean * opp_elo_avg
    der_weighted  = der_mean  * opp_elo_avg
    ref_year = max(years) if years else None
    return np.concatenate([base_weighted, der_weighted]).astype(float), ref_year

def build_feature_row(home, away, matches, elo_dict, stat_names, derived_names, scale_dict_elo):
    home_vec, y_home = _last5_weighted_stats(home, matches, stat_names)
    away_vec, y_away = _last5_weighted_stats(away, matches, stat_names)
    if home_vec is None or away_vec is None: return None

    home_elo_now = scale_dict_elo(_elo_lookup(elo_dict, home, y_home))
    away_elo_now = scale_dict_elo(_elo_lookup(elo_dict, away, y_away))
    delta_vec = home_vec - away_vec

    return np.concatenate([
        hash64_bits(home), np.array([home_elo_now], float), home_vec,
        hash64_bits(away), np.array([away_elo_now], float), away_vec,
        delta_vec
    ], dtype=float)

# ---------- Træning ----------
def train_lightgbm_models(X, Y, device="gpu", random_state=42):
    y_h, y_d, y_a = Y[:,0], Y[:,1], Y[:,2]
    n = len(X);  cut = int(n*0.9)
    X_tr, X_val = X[:cut], X[cut:]
    y_h_tr, y_h_val = y_h[:cut], y_h[cut:]
    y_d_tr, y_d_val = y_d[:cut], y_d[cut:]
    y_a_tr, y_a_val = y_a[:cut], y_a[cut:]

    params = dict(
        objective="l2",
        n_estimators=10000, learning_rate=0.015,
        max_depth=-1, num_leaves=127, min_child_samples=40,
        feature_fraction=0.85, bagging_fraction=0.8, bagging_freq=1,
        reg_alpha=0.0, reg_lambda=0.5, max_bin=255,
        device=device, random_state=random_state, verbose=-1
    )
    mH, mD, mA = LGBMRegressor(**params), LGBMRegressor(**params), LGBMRegressor(**params)
    cb = [early_stopping(stopping_rounds=300, verbose=False)]
    mH.fit(X_tr, y_h_tr, eval_set=[(X_val, y_h_val)], callbacks=cb)
    mD.fit(X_tr, y_d_tr, eval_set=[(X_val, y_d_val)], callbacks=cb)
    mA.fit(X_tr, y_a_tr, eval_set=[(X_val, y_a_val)], callbacks=cb)
    return mH, mD, mA

# ---------- Forudsigelser på fixtures ----------
def predict_from_fixtures(fixtures_array, models, matches, elo_dict, stat_names, derived_names, scale_dict_elo,
                          calibrate=True, degree=3):
    X_rows, meta = [], []
    for row in fixtures_array:
        home, away, oh, od, oa = str(row[0]), str(row[1]), float(row[2]), float(row[3]), float(row[4])
        x = build_feature_row(home, away, matches, elo_dict, stat_names, derived_names, scale_dict_elo)
        if x is None:
            num_base, num_der = len(stat_names), len(derived_names)
            feat_len = (64+1+(num_base+num_der) + 64+1+(num_base+num_der) + (num_base+num_der))
            x = np.full(feat_len, np.nan, dtype=float)
        X_rows.append(x); meta.append((home, away, oh, od, oa))

    X_pred = np.vstack(X_rows)

    odds = np.array([[m[2], m[3], m[4]] for m in meta], dtype=float)
    inv_odds = 1.0/odds
    implied = inv_odds / inv_odds.sum(axis=1, keepdims=True)
    K = inv_odds.sum(axis=1)

    valid = ~np.isnan(X_pred).any(axis=1)
    preds = np.full((len(X_pred),3), np.nan, dtype=float)
    if valid.any():
        p_h = models[0].predict(X_pred[valid])
        p_d = models[1].predict(X_pred[valid])
        p_a = models[2].predict(X_pred[valid])
        preds_valid = np.column_stack([p_h, p_d, p_a])
        preds_valid = direct_normalization(preds_valid)
        if calibrate and (np.sum(valid) > degree):
            try:
                preds_valid = polynomial_calibration(preds_valid, implied[valid], degree=degree)
                preds_valid = direct_normalization(preds_valid)
            except Exception as e:
                print(f"[Kalibrering sprang over] {e}")
        preds[valid] = preds_valid

    with np.errstate(divide='ignore', invalid='ignore'):
        pred_odds = 1.0 / (preds * K[:, None])
        pred_odds[~np.isfinite(pred_odds)] = np.nan

    out = pd.DataFrame({
        "Home":[m[0] for m in meta], "Away":[m[1] for m in meta],
        "Odds_H":odds[:,0], "Odds_U":odds[:,1], "Odds_A":odds[:,2],
        "Impl_H":implied[:,0], "Impl_U":implied[:,1], "Impl_A":implied[:,2],
        "Pred_H":preds[:,0], "Pred_U":preds[:,1], "Pred_A":preds[:,2],
        "PredOdds_H":pred_odds[:,0], "PredOdds_U":pred_odds[:,1], "PredOdds_A":pred_odds[:,2],
        "(>=5)?": valid
    })
    out["Diff_H"] = out["Pred_H"] - out["Impl_H"]
    out["Diff_U"] = out["Pred_U"] - out["Impl_U"]
    out["Diff_A"] = out["Pred_A"] - out["Impl_A"]
    out["Diff_H_%"] = 100.0*out["Diff_H"]
    out["Diff_D_%"] = 100.0*out["Diff_U"]
    out["Diff_A_%"] = 100.0*out["Diff_A"]
    return out, X_pred

# ---------- Kolonnenavne + eksport ----------
def build_xpred_columns(stat_names, derived_names):
    home_bits = [f"home_team_bit_{i:02d}" for i in range(64)]
    away_bits = [f"away_team_bit_{i:02d}" for i in range(64)]
    home_base = [f"home_{s}_avg5_weighted" for s in stat_names]
    home_der  = [f"home_{f}_avg5_weighted" for f in derived_names]
    away_base = [f"away_{s}_avg5_weighted" for s in stat_names]
    away_der  = [f"away_{f}_avg5_weighted" for f in derived_names]
    delta     = [f"delta_{s}" for s in stat_names] + [f"delta_{f}" for f in derived_names]
    return home_bits + ["HjemmeholdELO"] + home_base + home_der + \
           away_bits + ["UdeholdELO"]   + away_base + away_der + delta

def save_xpred_excel(X_pred, cfg: LeagueConfig, stat_names, derived_names):
    cols = build_xpred_columns(stat_names, derived_names)
    df = pd.DataFrame(X_pred, columns=cols)
    out_path = os.path.join(cfg.league_dir, cfg.out_xpred)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_excel(out_path, index=False)
    return out_path

# ---------- Convenience: fuld pipeline ----------
def run_pipeline(cfg: LeagueConfig, fixtures_array, calibrate=False, degree=3):
    X, Y, matches, elo_dict, stat_names, derived_names, scale_dict_elo = load_league_resources(cfg)
    models = train_lightgbm_models(X, Y, device=cfg.device, random_state=cfg.random_state)
    result_df, X_pred = predict_from_fixtures(
        fixtures_array, models, matches, elo_dict, stat_names, derived_names, scale_dict_elo,
        calibrate=calibrate, degree=degree
    )
    out_path = save_xpred_excel(X_pred, cfg, stat_names, derived_names)
    return result_df, out_path
