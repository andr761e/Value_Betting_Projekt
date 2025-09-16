# xy_builder.py
# Bygger X/Y til træning ud fra "Alle_kampe_med_elo_og_odds.xlsx".
# Generaliseret ud fra dine Ligue 1 og Bundesliga scripts.

import os
import hashlib
import numpy as np
import pandas as pd
from dataclasses import dataclass

# --------------------- Hjælpere ---------------------

def hash64_bits(name: str) -> np.ndarray:
    if pd.isna(name):
        name = ""
    h = hashlib.sha256(str(name).encode("utf-8")).digest()[:8]
    x = int.from_bytes(h, byteorder="big", signed=False)
    return np.array([(x >> (63 - i)) & 1 for i in range(64)], dtype=np.uint8)

def make_binary_cols(series: pd.Series, prefix: str) -> pd.DataFrame:
    bit_mat = np.vstack(series.apply(hash64_bits).to_numpy())
    cols = [f"{prefix}_bit_{i:02d}" for i in range(64)]
    return pd.DataFrame(bit_mat, columns=cols, index=series.index)

def _strip_prefix(col: str, prefix: str) -> str:
    return col[len(prefix):]

# --------------------- Konfiguration ---------------------

@dataclass
class XYConfig:
    league_dir: str = "French Ligue 1"
    matches_filename: str = "Alle_kampe_med_elo_og_odds.xlsx"
    # Y-kolonner (Excel kolonnebogstaver) – dine filer brugte AV, AW, AX
    y_cols: tuple = ("AV", "AW", "AX")
    y_nrows: int | None = None               # sæt None for at læse alle rækker
    out_X: str = "X.xlsx"
    out_Y: str = "Y.xlsx"
    rolling_window: int = 5
    include_delta: bool = True               # svarer til din delta-blok til sidst

    # Navne som i dine filer
    home_prefix: str = "Hjemmehold"
    away_prefix: str = "Udehold"

# --------------------- Kernefunktion ---------------------

def build_xy(cfg: XYConfig):
    # --- Y: læs tre sandsynlighedskolonner (samme som dine scripts) ---
    y_path = os.path.join(cfg.league_dir, cfg.matches_filename)
    # Læs hver kolonne separat for at matche din oprindelige tilgang
    #MatchResult = pd.read_excel(y_path, usecols="B", nrows=cfg.y_nrows)
    HomeWinProb = pd.read_excel(y_path, usecols=cfg.y_cols[0], nrows=cfg.y_nrows)
    DrawProb    = pd.read_excel(y_path, usecols=cfg.y_cols[1], nrows=cfg.y_nrows)
    AwayWinProb = pd.read_excel(y_path, usecols=cfg.y_cols[2], nrows=cfg.y_nrows)
    y = pd.concat([HomeWinProb, DrawProb, AwayWinProb], axis=1)
    #y = pd.concat([MatchResult,HomeWinProb, DrawProb, AwayWinProb], axis=1)

    # --- X: læs nødvendige kolonner ---
    columns_to_use = [
        f"{cfg.home_prefix}Navn", f"{cfg.home_prefix}ELO", f"{cfg.home_prefix}Mal",
        f"{cfg.home_prefix}Skud", f"{cfg.home_prefix}SkudPaMal", f"{cfg.home_prefix}Possession",
        f"{cfg.home_prefix}Fouls", f"{cfg.home_prefix}Corners", f"{cfg.home_prefix}Crosses",
        f"{cfg.home_prefix}Touches", f"{cfg.home_prefix}Tackles", f"{cfg.home_prefix}Interceptions",
        f"{cfg.home_prefix}Aerials_won", f"{cfg.home_prefix}Clearances", f"{cfg.home_prefix}Offsides",
        f"{cfg.home_prefix}Goal_kicks", f"{cfg.home_prefix}Throw_ins", f"{cfg.home_prefix}Long_balls",
        f"{cfg.home_prefix}Passes", f"{cfg.home_prefix}SuccesfulPasses", f"{cfg.home_prefix}PassAccuracy",
        f"{cfg.away_prefix}Navn", f"{cfg.away_prefix}ELO", f"{cfg.away_prefix}Mal",
        f"{cfg.away_prefix}Skud", f"{cfg.away_prefix}SkudPaMal", f"{cfg.away_prefix}Possession",
        f"{cfg.away_prefix}Fouls", f"{cfg.away_prefix}Corners", f"{cfg.away_prefix}Crosses",
        f"{cfg.away_prefix}Touches", f"{cfg.away_prefix}Tackles", f"{cfg.away_prefix}Interceptions",
        f"{cfg.away_prefix}Aerials_won", f"{cfg.away_prefix}Clearances", f"{cfg.away_prefix}Offsides",
        f"{cfg.away_prefix}Goal_kicks", f"{cfg.away_prefix}Throw_ins", f"{cfg.away_prefix}Long_balls",
        f"{cfg.away_prefix}Passes", f"{cfg.away_prefix}SuccesfulPasses", f"{cfg.away_prefix}PassAccuracy"
    ]
    matches = pd.read_excel(y_path, usecols=columns_to_use).copy()
    matches.reset_index(drop=True, inplace=True)
    matches["match_id"] = np.arange(len(matches))

    # --- Identificér fælles base-statnavne (som i dine scripts) ---
    home_exclude = {f"{cfg.home_prefix}Navn", f"{cfg.home_prefix}ELO", "Dato"}
    home_stats_cols = [c for c in matches.columns if c.startswith(cfg.home_prefix) and c not in home_exclude]
    stat_names = [_strip_prefix(c, cfg.home_prefix) for c in home_stats_cols]
    stat_names = [s for s in stat_names if (cfg.away_prefix + s) in matches.columns]

    # --- Long-form: én række pr. hold pr. kamp ---
    home_part = pd.DataFrame({
        "match_id": matches["match_id"],
        "team": matches[f"{cfg.home_prefix}Navn"],
        "opp":  matches[f"{cfg.away_prefix}Navn"],
        "team_elo": matches[f"{cfg.home_prefix}ELO"],   # allerede skaleret
        "opp_elo":  matches[f"{cfg.away_prefix}ELO"],   # allerede skaleret
        "side": "home",
    }, index=matches.index)

    away_part = pd.DataFrame({
        "match_id": matches["match_id"],
        "team": matches[f"{cfg.away_prefix}Navn"],
        "opp":  matches[f"{cfg.home_prefix}Navn"],
        "team_elo": matches[f"{cfg.away_prefix}ELO"],
        "opp_elo":  matches[f"{cfg.home_prefix}ELO"],
        "side": "away",
    }, index=matches.index)

    # Tilføj egne + modstanderens stats
    for s in stat_names:
        home_part[s] = matches[f"{cfg.home_prefix}{s}"]
        home_part[f"opp_{s}"] = matches[f"{cfg.away_prefix}{s}"]
        away_part[s] = matches[f"{cfg.away_prefix}{s}"]
        away_part[f"opp_{s}"] = matches[f"{cfg.home_prefix}{s}"]

    long_df = pd.concat([home_part, away_part], ignore_index=True)
    long_df.sort_values("match_id", inplace=True)
    long_df.reset_index(drop=True, inplace=True)

    # --- Afledte pr.-kamp features (samme som dine) ---
    g = long_df

    def _safe(name):
        return g[name] if name in g.columns else pd.Series(np.nan, index=g.index)

    den_shots      = _safe("Skud").replace(0, np.nan)
    den_opp_shots  = _safe("opp_Skud").replace(0, np.nan)
    den_passes     = _safe("Passes").replace(0, np.nan)
    den_opp_passes = _safe("opp_Passes").replace(0, np.nan)

    g["conv"]        = _safe("Mal") / den_shots
    g["sot_rate"]    = _safe("SkudPaMal") / den_shots

    g["def_conv"]     = _safe("opp_Mal") / den_opp_shots
    g["def_sot_rate"] = _safe("opp_SkudPaMal") / den_opp_shots

    g["cross_rate"]      = _safe("Crosses") / den_passes
    g["long_ball_share"] = _safe("Long_balls") / den_passes

    if "PassAccuracy" in g.columns:
        g["pass_succ_rate"] = g["PassAccuracy"] / 100.0
    else:
        g["pass_succ_rate"] = _safe("SuccesfulPasses") / den_passes

    g["aerial_win_rate"]     = _safe("Aerials_won") / (_safe("Aerials_won") + _safe("opp_Aerials_won")).replace(0, np.nan)
    g["defensive_intensity"] = (_safe("Tackles") + _safe("Interceptions")) / den_opp_passes

    g["field_tilt"] = _safe("Passes")  / (_safe("Passes")  + _safe("opp_Passes")).replace(0, np.nan)
    g["touch_tilt"] = _safe("Touches") / (_safe("Touches") + _safe("opp_Touches")).replace(0, np.nan)

    g["set_piece_pressure"] = (_safe("Corners") + _safe("Crosses")) / den_passes

    derived_feats = [
        "conv","sot_rate","def_conv","def_sot_rate",
        "cross_rate","long_ball_share","pass_succ_rate",
        "aerial_win_rate","defensive_intensity","field_tilt",
        "touch_tilt","set_piece_pressure"
    ]

    # --- Rullende 5 kampe (kun historik) + vægtning med opp_elo ---
    rw = cfg.rolling_window
    grouped = long_df.groupby("team", sort=False)

    long_df["opp_elo_avg5"] = (
        grouped["opp_elo"].apply(lambda x: x.shift(1).rolling(rw, min_periods=rw).mean())
        .reset_index(level=0, drop=True)
    )

    for s in stat_names:
        avg5 = (
            grouped[s].apply(lambda x: x.shift(1).rolling(rw, min_periods=rw).mean())
            .reset_index(level=0, drop=True)
        )
        long_df[f"{s}_avg5_weighted"] = avg5 * long_df["opp_elo_avg5"]

    for f in derived_feats:
        avg5 = (
            grouped[f].apply(lambda x: x.shift(1).rolling(rw, min_periods=rw).mean())
            .reset_index(level=0, drop=True)
        )
        long_df[f"{f}_avg5_weighted"] = avg5 * long_df["opp_elo_avg5"]

    # --- Split tilbage i hjemme/ude og behold *_avg5_weighted ---
    home_w_stats = (
        long_df[long_df["side"] == "home"].sort_values("match_id").set_index("match_id")
        [[f"{s}_avg5_weighted" for s in stat_names]]
    ).copy()
    home_w_stats.columns = [f"home_{c}" for c in home_w_stats.columns]

    away_w_stats = (
        long_df[long_df["side"] == "away"].sort_values("match_id").set_index("match_id")
        [[f"{s}_avg5_weighted" for s in stat_names]]
    ).copy()
    away_w_stats.columns = [f"away_{c}" for c in away_w_stats.columns]

    home_w_extra = (
        long_df[long_df["side"] == "home"].sort_values("match_id").set_index("match_id")
        [[f"{f}_avg5_weighted" for f in derived_feats]]
    ).copy()
    home_w_extra.columns = [f"home_{c}" for c in home_w_extra.columns]

    away_w_extra = (
        long_df[long_df["side"] == "away"].sort_values("match_id").set_index("match_id")
        [[f"{f}_avg5_weighted" for f in derived_feats]]
    ).copy()
    away_w_extra.columns = [f"away_{c}" for c in away_w_extra.columns]

    home_w = pd.concat([home_w_stats, home_w_extra], axis=1)
    away_w = pd.concat([away_w_stats, away_w_extra], axis=1)

    # --- 64-bit hash af holdnavne ---
    home_bits = make_binary_cols(matches[f"{cfg.home_prefix}Navn"], "home_team").set_index(matches["match_id"])
    away_bits = make_binary_cols(matches[f"{cfg.away_prefix}Navn"], "away_team").set_index(matches["match_id"])

    # --- Byg X i ønsket rækkefølge ---
    X = pd.concat(
        [
            home_bits,
            matches.set_index("match_id")[[f"{cfg.home_prefix}ELO"]],
            home_w,
            away_bits,
            matches.set_index("match_id")[[f"{cfg.away_prefix}ELO"]],
            away_w,
        ],
        axis=1
    ).reset_index(drop=True)

    # (valgfri) delta-blok til sidst
    if cfg.include_delta:
        home_base = home_w.copy(); home_base.columns = [c.replace("home_", "") for c in home_base.columns]
        away_base = away_w.copy(); away_base.columns = [c.replace("away_", "") for c in away_base.columns]
        delta_block = (home_base - away_base); delta_block.columns = [f"delta_{c}" for c in delta_block.columns]
        X = pd.concat([X, delta_block], axis=1)

    # --- Filtrér rækker uden fuld historik og align y ---
    mask_complete = ~X.isna().any(axis=1)
    X_clean = X.loc[mask_complete].reset_index(drop=True)
    y_aligned = y.loc[mask_complete].reset_index(drop=True)

    # --- Gem ---
    out_X_path = os.path.join(cfg.league_dir, cfg.out_X)
    out_Y_path = os.path.join(cfg.league_dir, cfg.out_Y)
    X_clean.to_excel(out_X_path, index=False)
    y_aligned.to_excel(out_Y_path, index=False)

    info = {
        "X_shape_raw": X.shape,
        "X_shape": X_clean.shape,
        "Y_shape": y_aligned.shape,
        "saved_X": out_X_path,
        "saved_Y": out_Y_path
    }
    return info
