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

def _logit(p):
    p = np.clip(p.astype(float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))

# --------------------- Konfiguration ---------------------

@dataclass
class XYConfig:
    league_dir: str
    matches_filename: str
    y_cols: tuple = ("E","Z")       # ikke brugt i denne version, Y bygges fra GH/GA
    y_nrows: int | None = None
    out_X: str = "X.xlsx"
    out_Y: str = "Y.xlsx"
    rolling_window: int = 5
    include_delta: bool = True
    use_bookmaker_features: bool = True
    windows: tuple = (5,)

# --------------------- Kernefunktion ---------------------

def build_xy(cfg: XYConfig):
    matches_path = os.path.join(cfg.league_dir, cfg.matches_filename)
    X, Y, feat_names, meta, Y_odds, raw_len = build_xy_core(
        matches_path=matches_path,
        use_bookmaker_features=cfg.use_bookmaker_features,
        min_games_rolling=cfg.rolling_window,
        windows=cfg.windows
    )

    # gem X, Y
    outX = os.path.join(cfg.league_dir, cfg.out_X)
    outY = os.path.join(cfg.league_dir, cfg.out_Y)
    outYodds = os.path.join(cfg.league_dir, "Y_odds.xlsx")

    pd.DataFrame(X, columns=feat_names).to_excel(outX, index=False)
    pd.DataFrame(Y, columns=["GH","GA"]).to_excel(outY, index=False)
    Y_odds.to_excel(outYodds, index=False)

    dropped = int(raw_len - X.shape[0])
    info = dict(
        X_shape_raw=(int(raw_len), len(feat_names)),
        X_shape=X.shape,
        Y_shape=Y.shape,
        Yodds_shape=Y_odds.shape,
        dropped_rows=dropped,
        dropped_pct=round(100.0 * dropped / max(raw_len, 1), 2),
        saved_X=outX,
        saved_Y=outY,
        saved_Yodds=outYodds
    )
    return info

def build_xy_core(
    matches_path: str,
    use_bookmaker_features: bool = True,
    min_games_rolling: int = 5,
    windows: tuple = (5, 10),
):
    """
    Byg X og Y til Poisson/Skellam-modellen ud fra 'Alle_kampe_med_elo_og_odds.xlsx'.

    - Ingen lækage: rolling beregnes på tidligere kampe via shift(1).
    - Smid rækker hvis et hold ikke har mindst 'min_games_rolling' tidligere kampe.
    - Merge via match_id (stabil nøgle), ikke dato+team.
    - Kolonne-rækkefølge: H... , A... , Delta..., (odds-logits), ELO_diff.

    Returnerer:
      X : np.ndarray
      Y : np.ndarray  (N,2) = [GH, GA]
      feature_names : list[str]
      meta_out : pd.DataFrame (Dato, HjemmeholdNavn, UdeholdNavn, FuldtidsResultat, PROB H/D/A)
    """

    # ---------- Hjælpere ----------
    def _logit(p):
        p = np.clip(pd.Series(p, dtype="float").to_numpy(), 1e-12, 1-1e-12)
        return np.log(p/(1-p))

    # ---------- Læs og klargør ----------
    df = pd.read_excel(matches_path)
    req_cols = [
        "Dato", "FuldtidsResultat",
        "HjemmeholdNavn", "UdeholdNavn",
        "HjemmeholdELO", "UdeholdELO",
        "HjemmeholdMal", "UdeholdMal",
        "HjemmeholdSkud", "UdeholdSkud",
        "HjemmeholdSkudPaMal", "UdeholdSkudPaMal",
        "HjemmeholdPossession", "UdeholdPossession",
        "HjemmeholdPassAccuracy", "UdeholdPassAccuracy",
        "HjemmeholdFouls", "UdeholdFouls",
        "HjemmeholdCorners", "UdeholdCorners",
        "PROB H", "PROB D", "PROB A",
    ]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Mangler kolonne i data: {c}")

    # datatyper + rens
    df["Dato"] = pd.to_datetime(df["Dato"], errors="coerce").dt.normalize()
    for col in ["HjemmeholdNavn", "UdeholdNavn"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.reset_index(drop=True)
    df["match_id"] = df.index  # stabil nøgle pr. kamp

    # targets
    GH = df["HjemmeholdMal"].astype(int).to_numpy()
    GA = df["UdeholdMal"].astype(int).to_numpy()
    Y = np.column_stack([GH, GA])

    # meta til debug/trace
    meta_out = df[["Dato", "HjemmeholdNavn", "UdeholdNavn", "FuldtidsResultat", "PROB H", "PROB D", "PROB A"]].copy()

    # ---------- Long tabel (én række pr. hold pr. kamp) ----------
    def _mk_side(prefix_team, prefix_opp, role_flag):
        return pd.DataFrame({
            "match_id":      df["match_id"],
            "Dato":          df["Dato"],
            "Team":          df[f"{prefix_team}Navn"].astype(str).str.strip(),
            "Opp":           df[f"{prefix_opp}Navn"].astype(str).str.strip(),
            "ELO":           pd.to_numeric(df[f"{prefix_team}ELO"], errors="coerce"),
            "OppELO":        pd.to_numeric(df[f"{prefix_opp}ELO"], errors="coerce"),
            "GoalsFor":      pd.to_numeric(df[f"{prefix_team}Mal"], errors="coerce"),
            "GoalsAg":       pd.to_numeric(df[f"{prefix_opp}Mal"], errors="coerce"),
            "Skud":          pd.to_numeric(df[f"{prefix_team}Skud"], errors="coerce"),
            "SkudPaMal":     pd.to_numeric(df[f"{prefix_team}SkudPaMal"], errors="coerce"),
            "Possession":    pd.to_numeric(df[f"{prefix_team}Possession"], errors="coerce"),
            "PassAccuracy":  pd.to_numeric(df[f"{prefix_team}PassAccuracy"], errors="coerce"),
            "Fouls":         pd.to_numeric(df[f"{prefix_team}Fouls"], errors="coerce"),
            "Corners":       pd.to_numeric(df[f"{prefix_team}Corners"], errors="coerce"),
            "Role":          role_flag
        })

    long_home = _mk_side("Hjemmehold", "Udehold", "H")
    long_away = _mk_side("Udehold", "Hjemmehold", "A")
    long_df = pd.concat([long_home, long_away], axis=0, ignore_index=True)

    # sortér pr. hold og beregn restdage/“tidligere kampe”
    long_df["Dato"] = pd.to_datetime(long_df["Dato"], errors="coerce").dt.normalize()
    long_df = long_df.sort_values(["Team", "Dato", "match_id"]).reset_index(drop=True)

    # restdage (dage mellem kampe)
    long_df["RestDays"] = (
        long_df.groupby("Team")["Dato"].diff().dt.days.astype("float")
    )

    # antal tidligere kampe for holdet
    long_df["prior_games"] = long_df.groupby("Team").cumcount()

    # ---------- Rolling uden lækage (shift(1) før rolling) ----------
    roll_feats = ["GoalsFor", "GoalsAg", "Skud", "SkudPaMal", "Possession", "PassAccuracy", "Fouls", "Corners"]
    grp = long_df.groupby("Team", group_keys=False)

    for W in windows:
        for col in roll_feats + ["OppELO", "RestDays"]:
            name = f"{col}_avg{W}"
            long_df[name] = (
                grp[col]
                .apply(lambda s: s.shift(1).rolling(W, min_periods=W).mean())
                .reset_index(level=0, drop=True)
            )

    # ---------- Udtræk side-blokke pr. kamp via match_id ----------
    def _extract_side_by_match(long_df, role_prefix: str, team_prefix: str):
        sel = long_df[long_df["Role"] == role_prefix].copy()
        keep = ["match_id", "prior_games", "ELO", "OppELO", "GoalsFor", "GoalsAg", "RestDays"] + \
               [c for c in sel.columns if any(c.endswith(f"_avg{W}") for W in windows)]
        sel = sel[keep]
        # omdøb prior_games og præfiksér øvrige features
        rename_map = {"prior_games": f"{team_prefix}_prior_games"}
        for c in sel.columns:
            if c not in ["match_id", "prior_games"]:
                rename_map[c] = f"{team_prefix}_{c}"
        sel = sel.rename(columns=rename_map)
        return sel

    H_block = _extract_side_by_match(long_df, "H", "H")
    A_block = _extract_side_by_match(long_df, "A", "A")

    # kamp-ramme (én række pr. match_id)
    m = df[["match_id", "Dato", "HjemmeholdNavn", "UdeholdNavn",
            "HjemmeholdELO", "UdeholdELO", "PROB H", "PROB D", "PROB A"]].copy()

    # merge via match_id (robust)
    m = m.merge(H_block, on="match_id", how="left")
    m = m.merge(A_block, on="match_id", how="left")

    # RestDays "latest" (brug den korteste rullede hvis tilgængelig)
    if len(windows) > 0:
        W0 = windows[0]
        if f"H_RestDays_avg{W0}" in m.columns:
            m["H_RestDays_latest"] = m[f"H_RestDays_avg{W0}"]
        else:
            m["H_RestDays_latest"] = m.get("H_RestDays", np.nan)
        if f"A_RestDays_avg{W0}" in m.columns:
            m["A_RestDays_latest"] = m[f"A_RestDays_avg{W0}"]
        else:
            m["A_RestDays_latest"] = m.get("A_RestDays", np.nan)
        m["Delta_RestDays_latest"] = m["H_RestDays_latest"] - m["A_RestDays_latest"]

    # ELO-features og diff
    m["ELO_H"] = pd.to_numeric(m["HjemmeholdELO"], errors="coerce")
    m["ELO_A"] = pd.to_numeric(m["UdeholdELO"], errors="coerce")
    m["ELO_diff"] = m["ELO_H"] - m["ELO_A"]

    # Odds-baseline (svage features)
    if use_bookmaker_features:
        m["logit_PROB_H"] = _logit(m["PROB H"])
        m["logit_PROB_D"] = _logit(m["PROB D"])
        m["logit_PROB_A"] = _logit(m["PROB A"])

    # ---------- Delta-features ----------
    bases = ["GoalsFor", "GoalsAg", "Skud", "SkudPaMal", "Possession", "PassAccuracy", "Fouls", "Corners", "OppELO", "RestDays"]
    for W in windows:
        for b in bases:
            h_col = f"H_{b}_avg{W}"
            a_col = f"A_{b}_avg{W}"
            if h_col in m.columns and a_col in m.columns:
                m[f"Delta_{b}_avg{W}"] = m[h_col] - m[a_col]

    # ---------- Kolonne-rækkefølge ----------
    feat_cols_H, feat_cols_A, feat_cols_delta = [], [], []

    # H først, så A
    feat_cols_H += ["ELO_H"]
    feat_cols_A += ["ELO_A"]

    for W in windows:
        feat_cols_H += [f"H_{b}_avg{W}" for b in bases if f"H_{b}_avg{W}" in m.columns]
        feat_cols_A += [f"A_{b}_avg{W}" for b in bases if f"A_{b}_avg{W}" in m.columns]

    feat_cols_H += [c for c in ["H_RestDays_latest"] if c in m.columns]
    feat_cols_A += [c for c in ["A_RestDays_latest"] if c in m.columns]

    # deltas
    for W in windows:
        for b in bases:
            d = f"Delta_{b}_avg{W}"
            if d in m.columns:
                feat_cols_delta.append(d)
    feat_cols_delta += [c for c in ["Delta_RestDays_latest"] if c in m.columns]

    # odds (til sidst)
    feat_cols_odds = []
    if use_bookmaker_features:
        feat_cols_odds = [c for c in ["logit_PROB_H", "logit_PROB_D", "logit_PROB_A"] if c in m.columns]

    # ELO_diff allersidst
    feat_cols_tail = ["ELO_diff"]

    feature_names = [c for c in (feat_cols_H + feat_cols_A + feat_cols_delta + feat_cols_odds + feat_cols_tail) if c in m.columns]

    raw_len = len(m)  # antal kampe før vi filtrerer for historik/NaN

    # ---------- Maskering: kræv historik + ingen NaN ----------
    # min. historik pr. hold (forudgående kampe)
    if "H_prior_games" in m.columns and "A_prior_games" in m.columns:
        mask_hist = (m["H_prior_games"] >= min_games_rolling) & (m["A_prior_games"] >= min_games_rolling)
    else:
        # fallback hvis prior_games ikke findes (burde ikke ske)
        mask_hist = pd.Series(True, index=m.index)

    Xtmp = m[feature_names].copy()
    mask_nonan = ~Xtmp.isna().any(axis=1)

    mask = (mask_hist & mask_nonan)

    # ---------- Udtræk X/Y + meta ----------
    X = Xtmp.loc[mask].to_numpy(dtype=float)
    Y = np.column_stack([df["HjemmeholdMal"].astype(int), df["UdeholdMal"].astype(int)])[mask.values, :]
    meta_out = meta_out.loc[mask].reset_index(drop=True)

    # ----- Y_odds (resultat + odds) -----
    # Vi antager at "B" = kampresultat, "AV"/"AW"/"AX" = bookie probs
    y_odds_raw = df[["FuldtidsResultat", "PROB H", "PROB D", "PROB A"]].copy()
    y_odds_raw = y_odds_raw.rename(columns={
        "B": "Result",
        "AV": "PROB_H",
        "AW": "PROB_D",
        "AX": "PROB_A"
    })

    Y_odds = y_odds_raw.loc[mask].reset_index(drop=True)

    return X, Y, feature_names, meta_out, Y_odds, raw_len
