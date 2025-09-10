import pandas as pd

#English Premier League Matches
English = pd.read_excel("English Premier League/Alle_kampe_med_elo_og_odds.xlsx").to_numpy()

#French Ligue 1 Matches
French = pd.read_excel("French Ligue 1/Alle_kampe_med_elo_og_odds.xlsx").to_numpy()

#German Bundesliga Matches
German = pd.read_excel("German Bundesliga/Alle_kampe_med_elo_og_odds.xlsx").to_numpy()

#italian Serie A Matches
Italian = pd.read_excel("Italian Serie A/Alle_kampe_med_elo_og_odds.xlsx").to_numpy()

#Spanish La Liga Matches
Spanish = pd.read_excel("Spanish La Liga/Alle_kampe_med_elo_og_odds.xlsx").to_numpy()

from pathlib import Path

# Hjælp: ensret kolonnenavne og parse dato
def load_league(path, liga_navn):
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip()                # fjern evt. whitespace i kolonnenavne
    # Sørg for at 'Dato' er datetime. Format som "2014-08-16" parses fint.
    df["Dato"] = pd.to_datetime(df["Dato"], errors="coerce", utc=False)
    df["Liga"] = liga_navn                             # nyttigt at bevare oprindelse
    return df

# Indsæt dine stier her (de kan også findes med glob, hvis du vil)
dfs = [
    load_league("English Premier League/Alle_kampe_med_elo_og_odds.xlsx", "Premier League"),
    load_league("French Ligue 1/Alle_kampe_med_elo_og_odds.xlsx", "Ligue 1"),
    load_league("German Bundesliga/Alle_kampe_med_elo_og_odds.xlsx", "Bundesliga"),
    load_league("Italian Serie A/Alle_kampe_med_elo_og_odds.xlsx", "Serie A"),
    load_league("Spanish La Liga/Alle_kampe_med_elo_og_odds.xlsx", "La Liga"),
]

# Saml alt og sorter efter dato
alle = pd.concat(dfs, ignore_index=True)
alle = alle.sort_values("Dato").reset_index(drop=True)

# (Valgfrit) tjek for rækker hvor dato ikke kunne parses
# print(alle[alle["Dato"].isna()].head())

# Skriv til ét excel-ark
alle.to_excel("Alternative modeller (forsøg)/alle_ligaer_sorteret_efter_dato.xlsx", index=False)  # bruger openpyxl under motorhjelmen
