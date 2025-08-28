import pickle
import pandas as pd
import numpy as np

# --- Funktioner ---

def flatten_elo_values(elo_dict):
    vals = []
    for year_map in elo_dict.values():
        for v in year_map.values():
            if v is not None and pd.notna(v):
                vals.append(float(v))
    return np.array(vals, dtype=float)

def make_scaler_0_5(elo_dict):
    vals = flatten_elo_values(elo_dict)
    vmin, vmax = np.min(vals), np.max(vals)
    return lambda x: 5 * (x - vmin) / (vmax - vmin)

def parse_year(date_val):
    s = str(date_val)
    return int(s[:4]) if len(s) >= 4 else None

def add_elo_columns(df, elo_dict):
    scaler = make_scaler_0_5(elo_dict)

    home_elo = []
    away_elo = []

    for _, row in df.iterrows():
        year = parse_year(row["Dato"])
        home = row["HjemmeholdNavn"]
        away = row["UdeholdNavn"]

        h_val = elo_dict.get(home, {}).get(year, np.nan)
        a_val = elo_dict.get(away, {}).get(year, np.nan)

        home_elo.append(np.nan if pd.isna(h_val) else scaler(h_val))
        away_elo.append(np.nan if pd.isna(a_val) else scaler(a_val))

    # indsæt lige efter holdnavn-kolonnerne
    idx_home = df.columns.get_loc("HjemmeholdNavn")
    df.insert(idx_home + 1, "HjemmeholdELO", home_elo)

    idx_away = df.columns.get_loc("UdeholdNavn")
    df.insert(idx_away + 1, "UdeholdELO", away_elo)

    return df


# --- Kørsel ---

# load elo_dict
with open("English Premier League/elo_dict_premier_league.pkl", "rb") as f:
    elo_dict = pickle.load(f)

# load excel
df = pd.read_excel("English Premier League/season_14_15_to_24_25.xlsx")

# tilføj ELO kolonner
df = add_elo_columns(df, elo_dict)

# gem som ny fil
df.to_excel("English Premier League/Alle_tidligere_seasons_med_elo.xlsx", index=False)
print("Færdig – fil gemt som kampe_med_elo.xlsx")
