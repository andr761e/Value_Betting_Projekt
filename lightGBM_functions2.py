# functions.py
import numpy as np
import pandas as pd
import pickle, hashlib, os
from dataclasses import dataclass
from lightgbm import LGBMClassifier, early_stopping
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

def _power_sharpen(p: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """alpha>1 skærper, alpha<1 blødgør. Bevarer sum=1 pr. række."""
    p = np.clip(p, 1e-12, 1.0)
    q = p**alpha
    return q / q.sum(axis=1, keepdims=True)

def smooth_pgd(P, w=(0.20, 0.60, 0.20)):
    """Glatter P(GD) lidt på tværs af naboklasser. P er (n, K)."""
    w = np.asarray(w, float)
    w /= w.sum()
    K = P.shape[1]
    P_pad = np.pad(P, ((0,0),(1,1)), mode='edge')           # [x0, x0, x1, .., xK-2, xK-1, xK-1]
    P_sm  = (w[0]*P_pad[:, :-2] + w[1]*P_pad[:, 1:-1] + w[2]*P_pad[:, 2:])
    P_sm /= P_sm.sum(axis=1, keepdims=True)
    return P_sm

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

# Vi arbejder på GD ∈ [-6..6]  =>  13 klasser (0..12)
GD_VALUES = np.arange(-4, 5)  # [-6..6]
GD_MIN, GD_MAX = GD_VALUES[0], GD_VALUES[-1]

# -------------------- Træning (kun multiclass) --------------------
def train_lightgbm_models(X, Y, device="gpu", random_state=42):
    """
    Ren multiclass-træning på GD_class.
    Forvent Y har form (N, 1) med heltal i [0..12], mappet fra GD + 6.
    Returnerer (clf,) så eksisterende pipeline fortsat kan 'unpacke' en tuple.
    """
    if not (Y.ndim == 2 and Y.shape[1] == 1):
        raise ValueError("Forvent Y som (N,1) med GD_class (0..12).")

    y_cls = Y[:, 0].astype(int)

    # NB: I praksis bør du bruge time-based split + purge/embargo.
    n = len(X)
    cut = int(n * 0.9)
    X_tr, X_val = X[:cut], X[cut:]
    y_tr, y_val = y_cls[:cut], y_cls[cut:]

    params = dict(
        objective="multiclass",
        num_class=len(GD_VALUES),
        metric=["multi_logloss"],
        n_estimators=20000,
        learning_rate=0.015,
        num_leaves=255,
        min_child_samples=20,
        feature_fraction=0.85,
        bagging_fraction=0.8,
        bagging_freq=1,
        reg_alpha=0.0,
        reg_lambda=0.2,
        max_bin=255,
        device=device,
        random_state=random_state,
        verbose=-1,
    )
    clf = LGBMClassifier(**params)
    # efter y_tr, y_val er defineret
    # --- class weights (inverse frekvens) ---
    uni, cnt = np.unique(y_tr, return_counts=True)
    inv = {c: 1.0/float(n) for c, n in zip(uni, cnt)}
    gd_tr = y_tr + GD_MIN

    # boost midten (GD=0) og de tætte (±1)
    w_tr = np.ones_like(y_tr, dtype=float)
    w_tr += 0.5 * (gd_tr == 0)        # +50% vægt på draws
    w_tr += 0.25 * (np.abs(gd_tr) == 1)

    # (valgfrit) dæmp halerne lidt
    w_tr += 0.075 * (np.abs(gd_tr) >= 3)  # sæt fx 0.15 hvis haler er ekstreme

    cb = [early_stopping(stopping_rounds=800, verbose=False)]
    clf.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], callbacks=cb)


    # Lille markør så vi kan genkende modellen nedstrøms (hvis du vil)
    setattr(clf, "_gd_multiclass", True)
    return (clf,)

# -------------------- 1X2- & AH-hjælpere baseret på P(GD) --------------------
def _one_x_two_from_Pgd(Pgd_row: np.ndarray):
    """Konverter én P(GD=-6..6) række til (pH, pD, pA)."""
    pH = float(Pgd_row[GD_VALUES >=  1].sum())
    pD = float(Pgd_row[GD_VALUES ==  0].sum())
    pA = float(Pgd_row[GD_VALUES <= -1].sum())
    s = pH + pD + pA
    if s > 0:
        pH, pD, pA = pH/s, pD/s, pA/s
    return pH, pD, pA


def _ah_cover_push_lose_from_Pgd(Pgd_row: np.ndarray, a: float):
    """
    Asian Handicap (home-siden) på linje a.
    - Win hvis (GD - a) > 0
    - Push hvis (GD - a) = 0 (kun hvis a er heltal)
    - Lose ellers
    """
    if abs(a - round(a)) < 1e-12:  # heltalslinje
        a_int = int(round(a))
        idx = a_int - GD_MIN
        p_push = float(Pgd_row[idx]) if GD_MIN <= a_int <= GD_MAX else 0.0
        p_win  = float(Pgd_row[(GD_VALUES > a_int)].sum())
        p_lose = 1.0 - p_win - p_push
        return p_win, p_push, p_lose
    # halvlinjer (±0.5): ingen push
    p_win  = float(Pgd_row[(GD_VALUES > a)].sum())
    p_push = 0.0
    p_lose = 1.0 - p_win
    return p_win, p_push, p_lose


def _ah_quarter_from_Pgd(Pgd_row: np.ndarray, a: float):
    """
    Kvartlinjer (x.25/x.75) = ½ på (a-0.25) + ½ på (a+0.25).
    """
    is_quarter = (abs(a*2 - round(a*2)) < 1e-12) and (int(abs(round(a*2))) % 2 == 1)
    if is_quarter:
        p1 = _ah_cover_push_lose_from_Pgd(Pgd_row, a - 0.25)
        p2 = _ah_cover_push_lose_from_Pgd(Pgd_row, a + 0.25)
        return tuple(0.5*(p1[i] + p2[i]) for i in range(3))
    return _ah_cover_push_lose_from_Pgd(Pgd_row, a)


def _row_Pgd_from_df(df_row: pd.Series) -> np.ndarray:
    """Læs P_GD_-6..6 kolonner fra en resultatrække og renormalisér let."""
    vals = [df_row.get(f"P_GD_{gd}", np.nan) for gd in GD_VALUES]
    arr = np.array(vals, dtype=float)
    if np.isnan(arr).any():
        raise ValueError("P_GD_* kolonner mangler (kørte du multiclass?).")
    s = arr.sum()
    if s > 0:
        arr = arr / s
    return arr


def add_ah_columns(df: pd.DataFrame, line_a: float, odds_home: float | None = None,
                   prefix: str | None = None) -> pd.DataFrame:
    """
    Tilføj kolonner for AH (home på linje 'a') baseret på P_GD_* i df.
    - line_a: fx -1.25, +0.5, 0, -1, ...
    - odds_home: hvis sat, beregnes EV for home-siden (decimalodds).
    - prefix: præfiks til kolonnenavne (default: 'AH_{a}')
    Returnerer en *kopi* af df med ekstra kolonner:
      {prefix}_pWin, {prefix}_pPush, {prefix}_pLose, og evtl. {prefix}_EV_home
    """
    if prefix is None:
        prefix = f"AH_{line_a}"

    out = df.copy()
    pW, pP, pL, EV = [], [], [], []
    for _, r in out.iterrows():
        Pgd = _row_Pgd_from_df(r)
        w, p, l = _ah_quarter_from_Pgd(Pgd, line_a)
        pW.append(w); pP.append(p); pL.append(l)
        if odds_home is not None:
            EV.append(odds_home * w - (1.0 - w - p))
        else:
            EV.append(np.nan)
    out[f"{prefix}_pWin"]  = np.array(pW)
    out[f"{prefix}_pPush"] = np.array(pP)
    out[f"{prefix}_pLose"] = np.array(pL)
    if odds_home is not None:
        out[f"{prefix}_EV_home"] = np.array(EV)
    return out

# -------------------- Prediction fra fixtures --------------------
def predict_from_fixtures(fixtures_array, models, matches, elo_dict, stat_names, derived_names, scale_dict_elo,
                          calibrate=True, degree=3):
    """
    Forbliver API-kompatibel med din eksisterende pipeline.
    Nu henter vi P(GD) via én multiclass-model og afleder 1X2.
    Lægger kolonner:
      Pred_H, Pred_U, Pred_A
      PredOdds_H, PredOdds_U, PredOdds_A   (modelodds skaleret med market overround K)
      P_GD_-6 ... P_GD_6                   (til AH/EV efterfølgende)
    """
    # Fixtures → features (samme mekanik som før i din kodebase)
    X_rows, meta = [], []
    for row in fixtures_array:
        home, away, oh, od, oa = str(row[0]), str(row[1]), float(row[2]), float(row[3]), float(row[4])
        x = build_feature_row(home, away, matches, elo_dict, stat_names, derived_names, scale_dict_elo)
        if x is None:
            # Fallback: bevar længde; du har allerede en build-feature-længdelogik i dit projekt
            num_base, num_der = len(stat_names), len(derived_names)
            feat_len = (64+1+(num_base+num_der) + 64+1+(num_base+num_der) + (num_base+num_der))
            x = np.full(feat_len, np.nan, dtype=float)
        X_rows.append(x); meta.append((home, away, oh, od, oa))

    X_pred = np.vstack(X_rows)

    # Bookmaker odds → implied & overround K (kun til at skalere modelodds; ikke nødvendig for P(GD))
    odds = np.array([[m[2], m[3], m[4]] for m in meta], dtype=float)
    inv_odds = 1.0 / odds
    implied = inv_odds / inv_odds.sum(axis=1, keepdims=True)
    K = inv_odds.sum(axis=1)

    valid = ~np.isnan(X_pred).any(axis=1)

    # Output ramme
    out = pd.DataFrame({
        "Home":[m[0] for m in meta], "Away":[m[1] for m in meta],
        "Odds_H":odds[:,0], "Odds_U":odds[:,1], "Odds_A":odds[:,2],
        "Impl_H":implied[:,0], "Impl_U":implied[:,1], "Impl_A":implied[:,2],
        "Valid?(>=5)": valid
    })

    # ----- Multiclass for P(GD) -----
    clf = models[0]
    preds_1x2 = np.full((len(X_pred), 3), np.nan, dtype=float)
    Pgd_full  = np.full((len(X_pred), len(GD_VALUES)), np.nan, dtype=float)

    if valid.any():
        Pgd_valid = clf.predict_proba(X_pred[valid])  # (n_valid, 13) summerer til 1
        #Pgd_valid = _power_sharpen(Pgd_valid, alpha=0.9)
        #Pgd_valid = smooth_pgd(Pgd_valid, w=(0.2, 0.6, 0.2))
        Pgd_full[valid] = Pgd_valid
        onextwo = np.array([_one_x_two_from_Pgd(row) for row in Pgd_valid], dtype=float)
        preds_1x2[valid] = onextwo

    # Model-odds (valgfrit: skaler med K hvis du vil “matche” market overround)
    with np.errstate(divide='ignore', invalid='ignore'):
        with np.errstate(divide='ignore', invalid='ignore'):
            pred_odds = 1.0 / np.clip(preds_1x2, 1e-12, 1.0)
            pred_odds[~np.isfinite(pred_odds)] = np.nan
        pred_odds[~np.isfinite(pred_odds)] = np.nan

    out["Pred_H"] = preds_1x2[:,0]
    out["Pred_U"] = preds_1x2[:,1]
    out["Pred_A"] = preds_1x2[:,2]
    out["PredOdds_H"] = pred_odds[:,0]
    out["PredOdds_U"] = pred_odds[:,1]
    out["PredOdds_A"] = pred_odds[:,2]
    out["Diff_H"] = out["Pred_H"] - out["Impl_H"]
    out["Diff_U"] = out["Pred_U"] - out["Impl_U"]
    out["Diff_A"] = out["Pred_A"] - out["Impl_A"]
    out["Diff_H_%"] = 100.0*out["Diff_H"]
    out["Diff_D_%"] = 100.0*out["Diff_U"]
    out["Diff_A_%"] = 100.0*out["Diff_A"]

    # Læg P(GD=-6..6) i output til videre brug (AH/EV)
    for i, gd in enumerate(GD_VALUES):
        out[f"P_GD_{gd}"] = Pgd_full[:, i]

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