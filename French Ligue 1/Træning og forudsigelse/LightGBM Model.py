import numpy as np
import pandas as pd
import pickle
import hashlib
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from Methods import methods
met = methods()
pd.set_option('display.max_columns', None)  # vis alle kolonner
pd.set_option('display.width', None)        # brug fuld terminalbredde
pd.set_option('display.max_colwidth', None) # afkort ikke celleindhold

# Indlæs data
X = pd.read_excel("French Ligue 1/X.xlsx").to_numpy()
Y = pd.read_excel("French Ligue 1/Y.xlsx").to_numpy()

# Opdel Y_train i separate arrays for hver klasse
Y_train_home = Y[:, 0]   # Sandsynlighed for hjemmebanesejr
Y_train_draw = Y[:, 1]   # Sandsynlighed for uafgjort
Y_train_away = Y[:, 2]   # Sandsynlighed for udebanesejr

# Initialiser modeller
model_home = LGBMRegressor(boosting_type='gbdt', device='gpu', n_estimators=1000, learning_rate=0.01, max_depth=3, random_state=42,verbose =-1)
model_draw = LGBMRegressor(boosting_type='gbdt', device='gpu', n_estimators=1000, learning_rate=0.01, max_depth=3, random_state=42,verbose =-1)
model_away = LGBMRegressor(boosting_type='gbdt', device='gpu', n_estimators=1000, learning_rate=0.01, max_depth=3, random_state=42,verbose =-1)

# Træn modellerne
model_home.fit(X, Y_train_home)
model_draw.fit(X, Y_train_draw)
model_away.fit(X, Y_train_away)


#Læs de originale statistikker
# Læs kun relevante kolonner fra Excel-arket
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
matches = pd.read_excel("French Ligue 1/Alle_kampe_med_elo_og_odds.xlsx", usecols=columns_to_use)

# load elo_dict
with open("French Ligue 1/elo_dict_ligue_1.pkl", "rb") as f:
    elo_dict = pickle.load(f)


# =========================
# Hjælpefunktioner (samme 64-bit encoding som tidligere)
# =========================

# --- Skaler elo_dict-værdier til [0,5] ---
_all_dict_elos = []
for _team, _years in elo_dict.items():
    for _y, _v in _years.items():
        try:
            _all_dict_elos.append(float(_v))
        except Exception:
            pass

if len(_all_dict_elos) > 0:
    DICT_ELO_MIN = float(np.nanmin(_all_dict_elos))
    DICT_ELO_MAX = float(np.nanmax(_all_dict_elos))
else:
    DICT_ELO_MIN, DICT_ELO_MAX = 0.0, 1.0  # fallback, bruges aldrig i praksis

def scale_dict_elo(v):
    if pd.isna(v):
        return np.nan
    if DICT_ELO_MAX == DICT_ELO_MIN:
        return 2.5  # neutral fallback
    return float(np.clip(5.0 * (float(v) - DICT_ELO_MIN) / (DICT_ELO_MAX - DICT_ELO_MIN), 0.0, 5.0))

def hash64_bits(name: str) -> np.ndarray:
    if pd.isna(name):
        name = ""
    h = hashlib.sha256(str(name).encode("utf-8")).digest()[:8]
    x = int.from_bytes(h, byteorder="big", signed=False)
    return np.array([(x >> (63 - i)) & 1 for i in range(64)], dtype=np.float32)

def _extract_year(val):
    """Returnér årstal som int fra Dato-kolonnen (uanset dtype)."""
    if pd.isna(val):
        return None
    try:
        # Timestamp eller datetime
        return pd.to_datetime(val).year
    except Exception:
        # String "YYYY-..." eller "YYYY"
        s = str(val)
        return int(s[:4]) if s[:4].isdigit() else None

def _elo_lookup(elo_dict, team, ref_year=None):
    """Find ELO for 'team' i 'elo_dict' for ref_year.
       Fald tilbage til nærmeste år <= ref_year, ellers seneste år."""
    d = elo_dict.get(team, None)
    if d is None:
        return np.nan
    # Tving nøgler til int
    try:
        keys = sorted(int(k) for k in d.keys())
    except Exception:
        keys = sorted(d.keys())
    if not keys:
        return np.nan
    if ref_year is None:
        return float(d[keys[-1]])
    le_keys = [k for k in keys if k <= ref_year]
    if le_keys:
        return float(d[max(le_keys)])
    # ellers nærmeste år i det hele taget
    closest = min(keys, key=lambda k: abs(k - ref_year))
    return float(d[closest])

# Identificér stats-basenavne (samme rækkefølge som i matches)
_home_exclude = {"HjemmeholdNavn", "HjemmeholdELO", "HjemmeholdDato", "Dato"}
_home_stats_cols = [c for c in matches.columns if c.startswith("Hjemmehold") and c not in _home_exclude]
def _strip_prefix(col, prefix): return col[len(prefix):]
_stat_names = [_strip_prefix(c, "Hjemmehold") for c in _home_stats_cols]
# behold kun dem, der også findes for Udehold
_stat_names = [s for s in _stat_names if ("Udehold" + s) in matches.columns]

# --- Nye afledte feature-navne (SAMME rækkefølge som i træningen) ---
_derived_names = [
    "conv","sot_rate","def_conv","def_sot_rate",
    "cross_rate","long_ball_share","pass_succ_rate",
    "aerial_win_rate","defensive_intensity","field_tilt",
    "touch_tilt","set_piece_pressure"
]

def _safe_div(n, d):
    n = float(n) if pd.notna(n) else np.nan
    d = float(d) if pd.notna(d) else np.nan
    if d == 0 or np.isnan(d):
        return np.nan
    return n / d

def _row_derived_for_team(r, team_is_home: bool):
    """Afledte pr.-kamp features for et hold givet en kamp-række r."""
    if team_is_home:
        g, s, sot = r.get("HjemmeholdMal", np.nan), r.get("HjemmeholdSkud", np.nan), r.get("HjemmeholdSkudPaMal", np.nan)
        g_opp, s_opp, sot_opp = r.get("UdeholdMal", np.nan), r.get("UdeholdSkud", np.nan), r.get("UdeholdSkudPaMal", np.nan)
        passes, passes_opp = r.get("HjemmeholdPasses", np.nan), r.get("UdeholdPasses", np.nan)
        succ_pass, passacc = r.get("HjemmeholdSuccesfulPasses", np.nan), r.get("HjemmeholdPassAccuracy", np.nan)
        crosses, long_balls = r.get("HjemmeholdCrosses", np.nan), r.get("HjemmeholdLong_balls", np.nan)
        aerials, aerials_opp = r.get("HjemmeholdAerials_won", np.nan), r.get("UdeholdAerials_won", np.nan)
        tackles, inter = r.get("HjemmeholdTackles", np.nan), r.get("HjemmeholdInterceptions", np.nan)
        touches, touches_opp = r.get("HjemmeholdTouches", np.nan), r.get("UdeholdTouches", np.nan)
        corners = r.get("HjemmeholdCorners", np.nan)
    else:
        g, s, sot = r.get("UdeholdMal", np.nan), r.get("UdeholdSkud", np.nan), r.get("UdeholdSkudPaMal", np.nan)
        g_opp, s_opp, sot_opp = r.get("HjemmeholdMal", np.nan), r.get("HjemmeholdSkud", np.nan), r.get("HjemmeholdSkudPaMal", np.nan)
        passes, passes_opp = r.get("UdeholdPasses", np.nan), r.get("HjemmeholdPasses", np.nan)
        succ_pass, passacc = r.get("UdeholdSuccesfulPasses", np.nan), r.get("UdeholdPassAccuracy", np.nan)
        crosses, long_balls = r.get("UdeholdCrosses", np.nan), r.get("UdeholdLong_balls", np.nan)
        aerials, aerials_opp = r.get("UdeholdAerials_won", np.nan), r.get("HjemmeholdAerials_won", np.nan)
        tackles, inter = r.get("UdeholdTackles", np.nan), r.get("UdeholdInterceptions", np.nan)
        touches, touches_opp = r.get("UdeholdTouches", np.nan), r.get("HjemmeholdTouches", np.nan)
        corners = r.get("UdeholdCorners", np.nan)

    # Afledte mål (match træningsscriptet)
    conv         = _safe_div(g, s)
    sot_rate     = _safe_div(sot, s)
    def_conv     = _safe_div(g_opp, s_opp)
    def_sot_rate = _safe_div(sot_opp, s_opp)
    cross_rate      = _safe_div(crosses, passes)
    long_ball_share = _safe_div(long_balls, passes)
    pass_succ_rate  = (float(passacc)/100.0) if pd.notna(passacc) else _safe_div(succ_pass, passes)
    aerial_den      = (float(aerials) + float(aerials_opp)) if (pd.notna(aerials) and pd.notna(aerials_opp)) else np.nan
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


def _last5_weighted_stats(team: str, matches: pd.DataFrame):
    """
    Returnér (weighted_vec, ref_year):
      weighted_vec = [base_avg5_weighted..., derived_avg5_weighted...]  (samme rækkefølge som træning)
      ref_year = reference-år til elo_dict-lookup (seneste år i de 5 kampe)
    """
    idx = matches.index[(matches["HjemmeholdNavn"] == team) | (matches["UdeholdNavn"] == team)]
    if len(idx) < 5:
        return None, None

    last5 = idx[-5:]
    base_rows = []
    derived_rows = []
    opp_elos = []
    years = []

    for i in last5:
        r = matches.loc[i]
        # år fra Dato (til elo_dict-lookup senere)
        y = _extract_year(r["Dato"]) if "Dato" in matches.columns else None
        if y is not None: years.append(y)

        if r["HjemmeholdNavn"] == team:
            # base-stats i præcis _stat_names rækkefølge
            base_rows.append([r.get("Hjemmehold" + s, np.nan) for s in _stat_names])
            derived_rows.append(_row_derived_for_team(r, team_is_home=True))
            opp_elos.append(r.get("UdeholdELO", np.nan))   # allerede skaleret i matches
        else:
            base_rows.append([r.get("Udehold" + s, np.nan) for s in _stat_names])
            derived_rows.append(_row_derived_for_team(r, team_is_home=False))
            opp_elos.append(r.get("HjemmeholdELO", np.nan))

    base_arr = np.array(base_rows, dtype=float)
    der_arr  = np.array(derived_rows, dtype=float)

    base_mean = np.nanmean(base_arr, axis=0)
    der_mean  = np.nanmean(der_arr,  axis=0)

    opp_elo_avg = float(np.nanmean(np.array(opp_elos, dtype=float)))  # skaleret gennemsnit
    base_weighted = base_mean * opp_elo_avg
    der_weighted  = der_mean  * opp_elo_avg

    ref_year = max(years) if years else None
    return np.concatenate([base_weighted, der_weighted]).astype(float), ref_year


def _build_feature_row(home: str, away: str, elo_dict, matches: pd.DataFrame):
    """Byg feature-række i præcis samme format som træningens X (inkl. delta-blok)."""
    home_vec, home_ref_year = _last5_weighted_stats(home, matches)
    away_vec, away_ref_year = _last5_weighted_stats(away, matches)
    if home_vec is None or away_vec is None:
        return None  # utilstrækkelig historik

    # Skaler ELO fra elo_dict (matches-ELO er allerede skaleret)
    home_elo_now = scale_dict_elo(_elo_lookup(elo_dict, home, home_ref_year))
    away_elo_now = scale_dict_elo(_elo_lookup(elo_dict, away, away_ref_year))

    # Delta-blok (home - away) på HELE det vægtede feature-sæt (base + derived)
    delta_vec = home_vec - away_vec

    x = np.concatenate([
        hash64_bits(home),                    # 64 hjemme-bits
        np.array([home_elo_now], float),      # hjemme-ELO (skaleret)
        home_vec,                             # hjemme vægtede base+derived
        hash64_bits(away),                    # 64 ude-bits
        np.array([away_elo_now], float),      # ude-ELO (skaleret)
        away_vec,                             # ude vægtede base+derived
        delta_vec,                            # delta-blokken (til sidst)
    ], dtype=float)
    return x

# ==========================================
# Hovedfunktion: forudsig fra kampliste + odds
# ==========================================
def predict_from_fixtures(fixtures_array: np.ndarray, calibrate: bool = True, degree: int = 3):
    """
    fixtures_array: shape (n, 5)
      [:,0]=Hjemmehold, [:,1]=Udehold, [:,2]=oddsH, [:,3]=oddsU, [:,4]=oddsA

    Returnerer:
      result_df (med odds, implied, kalibrerede pred, differencer, model-odds, procent-diff)
      og X_pred (feature-matrixen i samme format som træning).
    """
    # Byg feature-matrix
    X_rows, meta, ok_flags = [], [], []
    for row in fixtures_array:
        home, away, oh, od, oa = row
        home, away = str(home), str(away)
        oh, od, oa = float(oh), float(od), float(oa)
        x = _build_feature_row(home, away, elo_dict, matches)
        if x is None:
            num_base = len(_stat_names)
            num_der  = len(_derived_names)
            feat_len = (64 + 1 + (num_base + num_der)    # home-blok
                        + 64 + 1 + (num_base + num_der)  # away-blok
                        + (num_base + num_der))          # delta-blok
            x = np.full(feat_len, np.nan, dtype=float)
            ok_flags.append(False)
        else:
            ok_flags.append(True)
        X_rows.append(x)
        meta.append((home, away, oh, od, oa))

    X_pred = np.vstack(X_rows)

    # --- Markeds-implied fra odds (før vi kører modellen) ---
    odds = np.array([[m[2], m[3], m[4]] for m in meta], dtype=float)
    inv_odds = 1.0 / odds
    implied = inv_odds / inv_odds.sum(axis=1, keepdims=True)   # Y_test i din notation
    K = inv_odds.sum(axis=1)  # overround til senere model-odds

    # Forudsig kun for gyldige rækker
    valid = ~np.isnan(X_pred).any(axis=1)
    preds = np.full((len(X_pred), 3), np.nan, dtype=float)
    if valid.any():
        p_h = model_home.predict(X_pred[valid])
        p_d = model_draw.predict(X_pred[valid])
        p_a = model_away.predict(X_pred[valid])
        preds_valid = np.column_stack([p_h, p_d, p_a])

        # Direct normalization
        preds_valid = met.direct_normalization(preds_valid)

        # --- Kalibrering mod implied (markedsfordeling) ---
        if calibrate and (np.sum(valid) > degree):
            try:
                preds_valid = met.polynomial_calibration(preds_valid, implied[valid], degree=degree)
                # normalisér igen for sikkerheds skyld
                preds_valid = met.direct_normalization(preds_valid)
            except Exception as e:
                print(f"[Kalibrering sprang over] {e}")

        preds[valid] = preds_valid

    # Model-odds med SAMME overround som input
    with np.errstate(divide='ignore', invalid='ignore'):
        pred_odds = 1.0 / (preds * K[:, None])   # 1 / (p * K)
        pred_odds[~np.isfinite(pred_odds)] = np.nan

    # Resultattabel
    out = pd.DataFrame({
        "Hjemme": [m[0] for m in meta],
        "Ude":    [m[1] for m in meta],
        "Odds_H": odds[:, 0],
        "Odds_U": odds[:, 1],
        "Odds_A": odds[:, 2],
        "Impl_H": implied[:, 0],
        "Impl_U": implied[:, 1],
        "Impl_A": implied[:, 2],
        "Pred_H": preds[:, 0],
        "Pred_U": preds[:, 1],
        "Pred_A": preds[:, 2],
        "PredOdds_H": pred_odds[:, 0],
        "PredOdds_U": pred_odds[:, 1],
        "PredOdds_A": pred_odds[:, 2],
        "Valid?(>=5)": valid
    })

    # Differencer (model - implied)
    out["Diff_H"] = out["Pred_H"] - out["Impl_H"]
    out["Diff_U"] = out["Pred_U"] - out["Impl_U"]
    out["Diff_A"] = out["Pred_A"] - out["Impl_A"]

    # Procent-kolonner (D = Draw)
    out["Diff_H_%"] = 100.0 * out["Diff_H"]
    out["Diff_D_%"] = 100.0 * out["Diff_U"]
    out["Diff_A_%"] = 100.0 * out["Diff_A"]

    return out, X_pred

#SKRIV UGENS KAMPE IND HER
fixtures = np.array([
    ["Marseille", "Lorient", 1.40, 4.50, 7.50],
    ["Nice", "Nantes", 1.65, 3.60, 5.50],
    ["Strasbourg", "Le Havre", 1.55, 3.90, 6.25],
    ["Auxerre", "Monaco", 4.75, 4.10, 1.61],
    ["Lille", "Toulouse", 1.80, 3.60, 4.50],
    ["Brest", "Paris FC", 2.10, 3.40, 3.40],
    ["Metz", "Angers", 2.10, 3.40, 3.40],
    ["Paris Saint-Germain", "Lens", 1.27, 6.50, 8.00],
    ["Rennes", "Lyon", 2.70, 3.60, 2.40]
], dtype=object)

result_df, X_pred = predict_from_fixtures(fixtures,calibrate=False)
print(result_df)

# Byg kolonnenavne (samme ordre som i vores træning):
home_bits_cols = [f"home_team_bit_{i:02d}" for i in range(64)]
away_bits_cols = [f"away_team_bit_{i:02d}" for i in range(64)]
home_base_cols = [f"home_{s}_avg5_weighted" for s in _stat_names]
home_der_cols  = [f"home_{f}_avg5_weighted" for f in _derived_names]
away_base_cols = [f"away_{s}_avg5_weighted" for s in _stat_names]
away_der_cols  = [f"away_{f}_avg5_weighted" for f in _derived_names]
delta_cols     = [f"delta_{s}" for s in _stat_names] + [f"delta_{f}" for f in _derived_names]

cols = (home_bits_cols + ["HjemmeholdELO"] + home_base_cols + home_der_cols
        + away_bits_cols + ["UdeholdELO"] + away_base_cols + away_der_cols
        + delta_cols)

df_Xpred = pd.DataFrame(X_pred, columns=cols)

df_Xpred.to_excel("French Ligue 1/Træning og forudsigelse/X_pred.xlsx", index=False)