import pandas as pd
import numpy as np
import warnings  # Import Warnings to suppress unnecessary warnings
import shap
import matplotlib.pyplot as plt
import hashlib

# -------------------------------------------------------------------
# 1) Hjælpefunktioner
# -------------------------------------------------------------------
def hash64_bits(name: str) -> np.ndarray:
    """
    Stabil 64-bit binær encoding af et holdnavn.
    Vi tager de første 8 bytes af SHA-256 og laver dem til 64 bit.
    Returnerer en (64,) numpy array af 0/1.
    """
    if pd.isna(name):
        name = ""
    h = hashlib.sha256(str(name).encode("utf-8")).digest()[:8]
    x = int.from_bytes(h, byteorder="big", signed=False)
    bits = np.array([(x >> (63 - i)) & 1 for i in range(64)], dtype=np.uint8)
    return bits

def make_binary_cols(series, prefix):
    """
    Laver 64 binære kolonner for hver streng i en pandas Series.
    """
    bit_mat = np.vstack(series.apply(hash64_bits).to_numpy())
    cols = [f"{prefix}_bit_{i:02d}" for i in range(64)]
    return pd.DataFrame(bit_mat, columns=cols, index=series.index)


#Y DELEN
#Indlæs sandsynligheder
HomeWinProb = pd.read_excel("French Ligue 1/Alle_kampe_med_elo_og_odds.xlsx", usecols="AV", skiprows=0, nrows=7981)
DrawProb = pd.read_excel("French Ligue 1/Alle_kampe_med_elo_og_odds.xlsx", usecols="AW", skiprows=0, nrows=7981)
AwayWinProb = pd.read_excel("French Ligue 1/Alle_kampe_med_elo_og_odds.xlsx", usecols="AX", skiprows=0, nrows=7981)

#Concatenate data to matrix
y = pd.concat([HomeWinProb, DrawProb, AwayWinProb],axis=1)


#X DELEN
# Læs kun relevante kolonner fra Excel-arket
columns_to_use = ["HjemmeholdNavn","HjemmeholdELO","HjemmeholdMal","HjemmeholdSkud","HjemmeholdSkudPaMal",
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

# 0) Byg long-form datasæt med hjemme/ude + modstander-statistik pr. kamp
matches = matches.copy()
matches.reset_index(drop=True, inplace=True)
matches["match_id"] = np.arange(len(matches))

# Identificér base-statnavne (fælles for hjemme/ude)
home_exclude = {"HjemmeholdNavn", "HjemmeholdELO", "Dato"}
home_stats_cols = [c for c in matches.columns if c.startswith("Hjemmehold") and c not in home_exclude]
def _strip_prefix(col, prefix): return col[len(prefix):]
stat_names = [_strip_prefix(c, "Hjemmehold") for c in home_stats_cols]
# behold kun baser, der også findes for udeholdet
stat_names = [s for s in stat_names if ("Udehold" + s) in matches.columns]

# long-form: én række pr. hold pr. kamp (home + away)
home_part = pd.DataFrame({
    "match_id": matches["match_id"],
    "team": matches["HjemmeholdNavn"],
    "opp":  matches["UdeholdNavn"],
    "team_elo": matches["HjemmeholdELO"],   # allerede skaleret i matches
    "opp_elo":  matches["UdeholdELO"],      # allerede skaleret i matches
    "side": "home",
}, index=matches.index)

away_part = pd.DataFrame({
    "match_id": matches["match_id"],
    "team": matches["UdeholdNavn"],
    "opp":  matches["HjemmeholdNavn"],
    "team_elo": matches["UdeholdELO"],      # allerede skaleret i matches
    "opp_elo":  matches["HjemmeholdELO"],   # allerede skaleret i matches
    "side": "away",
}, index=matches.index)

# Tilføj både egne stats og modstanderens stats pr. kamp
for s in stat_names:
    home_part[s] = matches["Hjemmehold" + s]
    home_part["opp_" + s] = matches["Udehold" + s]
    away_part[s] = matches["Udehold" + s]
    away_part["opp_" + s] = matches["Hjemmehold" + s]

long_df = pd.concat([home_part, away_part], ignore_index=True)
long_df.sort_values("match_id", inplace=True)
long_df.reset_index(drop=True, inplace=True)

# 1) Afledte pr.-kamp features (uden læk; vi ruller/shift’er senere)
g = long_df  # alias for korthed

# Hjælpefunktion til sikkert at slå kolonner op (NaN hvis mangler)
def _safe(name):
    return g[name] if name in g.columns else pd.Series(np.nan, index=g.index)

den_shots      = _safe("Skud").replace(0, np.nan)
den_opp_shots  = _safe("opp_Skud").replace(0, np.nan)
den_passes     = _safe("Passes").replace(0, np.nan)
den_opp_passes = _safe("opp_Passes").replace(0, np.nan)

# Angreb/effektivitet
g["conv"]        = _safe("Mal") / den_shots
g["sot_rate"]    = _safe("SkudPaMal") / den_shots  # justér til dit kolonnenavn hvis nødvendigt

# Forsvar/effektivitet (imod)
g["def_conv"]     = _safe("opp_Mal") / den_opp_shots
g["def_sot_rate"] = _safe("opp_SkudPaMal") / den_opp_shots

# Stil
g["cross_rate"]      = _safe("Crosses") / den_passes
g["long_ball_share"] = _safe("Long_balls") / den_passes

# Pasningskvalitet (du har typisk PassAccuracy i %)
if "PassAccuracy" in g.columns:
    g["pass_succ_rate"] = g["PassAccuracy"] / 100.0
else:
    g["pass_succ_rate"] = _safe("SuccesfulPasses") / den_passes

# Dueller/pres
g["aerial_win_rate"]     = _safe("Aerials_won") / (_safe("Aerials_won") + _safe("opp_Aerials_won")).replace(0, np.nan)
g["defensive_intensity"] = (_safe("Tackles") + _safe("Interceptions")) / den_opp_passes

# Tilt/territorium
g["field_tilt"] = _safe("Passes")  / (_safe("Passes")  + _safe("opp_Passes")).replace(0, np.nan)
g["touch_tilt"] = _safe("Touches") / (_safe("Touches") + _safe("opp_Touches")).replace(0, np.nan)

# Dødbolds-tryk
g["set_piece_pressure"] = (_safe("Corners") + _safe("Crosses")) / den_passes

derived_feats = [
    "conv","sot_rate","def_conv","def_sot_rate",
    "cross_rate","long_ball_share","pass_succ_rate",
    "aerial_win_rate","defensive_intensity","field_tilt",
    "touch_tilt","set_piece_pressure"
]

# 2) Rullende 5-kamps gennemsnit (kun historik) og vægtning med modstanderes ELO
rolling_window = 5
grouped = long_df.groupby("team", sort=False)

# gennemsnit af modstanderes ELO over de sidste 5 kampe (historik)
long_df["opp_elo_avg5"] = (
    grouped["opp_elo"]
    .apply(lambda x: x.shift(1).rolling(rolling_window, min_periods=5).mean())
    .reset_index(level=0, drop=True)
)

# Vægt de oprindelige stats-baser
for s in stat_names:
    avg5 = (
        grouped[s]
        .apply(lambda x: x.shift(1).rolling(rolling_window, min_periods=5).mean())
        .reset_index(level=0, drop=True)
    )
    long_df[f"{s}_avg5_weighted"] = avg5 * long_df["opp_elo_avg5"]

# Vægt de afledte features
for f in derived_feats:
    avg5 = (
        grouped[f]
        .apply(lambda x: x.shift(1).rolling(rolling_window, min_periods=5).mean())
        .reset_index(level=0, drop=True)
    )
    long_df[f"{f}_avg5_weighted"] = avg5 * long_df["opp_elo_avg5"]

# 3) Split tilbage i hjemme/ude og behold KUN *_avg5_weighted
#    (vi fastholder din ønskede kolonnerækkefølge)
home_w_stats = (
    long_df[long_df["side"] == "home"]
    .sort_values("match_id").set_index("match_id")
    [[f"{s}_avg5_weighted" for s in stat_names]]
).copy()
home_w_stats.columns = [f"home_{c}" for c in home_w_stats.columns]

away_w_stats = (
    long_df[long_df["side"] == "away"]
    .sort_values("match_id").set_index("match_id")
    [[f"{s}_avg5_weighted" for s in stat_names]]
).copy()
away_w_stats.columns = [f"away_{c}" for c in away_w_stats.columns]

home_w_extra = (
    long_df[long_df["side"] == "home"]
    .sort_values("match_id").set_index("match_id")
    [[f"{f}_avg5_weighted" for f in derived_feats]]
).copy()
home_w_extra.columns = [f"home_{c}" for c in home_w_extra.columns]

away_w_extra = (
    long_df[long_df["side"] == "away"]
    .sort_values("match_id").set_index("match_id")
    [[f"{f}_avg5_weighted" for f in derived_feats]]
).copy()
away_w_extra.columns = [f"away_{c}" for c in away_w_extra.columns]

# Saml hjemme/ude blokke (først de "gamle" stats, dernæst de nye afledte)
home_w = pd.concat([home_w_stats, home_w_extra], axis=1)
away_w = pd.concat([away_w_stats, away_w_extra], axis=1)

# 4) 64-bit hash af holdnavne (forudsætter hash64_bits/make_binary_cols er defineret)
home_bits = make_binary_cols(matches["HjemmeholdNavn"], "home_team").set_index(matches["match_id"])
away_bits = make_binary_cols(matches["UdeholdNavn"], "away_team").set_index(matches["match_id"])

# 5) Byg X i præcis ønsket rækkefølge
X = pd.concat(
    [
        home_bits,                                                      # 1) 64 hjemme-bits
        matches.set_index("match_id")[["HjemmeholdELO"]],               # 2) hjemme-ELO (allerede skaleret)
        home_w,                                                         # 3) hjemme vægtede stats (gamle + afledte)
        away_bits,                                                      # 4) 64 ude-bits
        matches.set_index("match_id")[["UdeholdELO"]],                  # 5) ude-ELO (allerede skaleret)
        away_w,                                                         # 6) ude vægtede stats (gamle + afledte)
    ],
    axis=1
).reset_index(drop=True)

# (Valgfrit) 6) Delta-blok (home - away) for alle vægtede features – lægges TIL SIDST
#       Kommentér ud, hvis du ikke ønsker delta-features.
home_base = home_w.copy(); home_base.columns = [c.replace("home_", "") for c in home_base.columns]
away_base = away_w.copy(); away_base.columns = [c.replace("away_", "") for c in away_base.columns]
delta_block = (home_base - away_base); delta_block.columns = [f"delta_{c}" for c in delta_block.columns]
X = pd.concat([X, delta_block], axis=1)

# 7) Fjern rækker uden fuld historik (NaN) og filtrér y tilsvarende
mask_complete = ~X.isna().any(axis=1)
X_clean = X.loc[mask_complete].reset_index(drop=True)
y_aligned = y.loc[mask_complete].reset_index(drop=True)

print(f"Form før filter: X={X.shape}, y={getattr(y, 'shape', (len(y),))}")
print(f"Form efter filter: X={X_clean.shape}, y={y_aligned.shape}")

# 8) Gem til Excel
X_clean.to_excel("French Ligue 1/X.xlsx", index=False)
y_aligned.to_excel("French Ligue 1/Y.xlsx", index=False)
print("Gemte 'X_features.xlsx' og 'y_aligned.xlsx'.")
