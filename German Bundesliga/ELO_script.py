# -*- coding: utf-8 -*-
import io, unicodedata, re, pickle
from difflib import get_close_matches

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ----------------- Netværk + normalisering -----------------
def _session():
    retry = Retry(total=4, connect=4, read=4, backoff_factor=0.8,
                  status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "Mozilla/5.0 (elo-scraper/sep-snapshot/1.0)"})
    return s

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip().replace("&", " and ")
    s = re.sub(r"\b(f\.?c\.?|a\.?f\.?c\.?)\b", " ", s)   # fjern FC/AFC
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

ALIASES = {
    "brighton and hove albion": ["brighton"],
    "manchester city": ["man city"],
    "manchester united": ["man united"],
    "tottenham hotspur": ["tottenham", "spurs"],
    "wolverhampton wanderers": ["wolverhampton", "wolves"],
    "west ham united": ["west ham"],
    "queens park rangers": ["qpr"],
    "sheffield united": ["sheff utd", "sheff united", "sheffield utd"],
    "west bromwich albion": ["west brom"],
}

def _candidates(name: str):
    n = _norm(name)
    return [n] + ALIASES.get(n, [])

# ----------------- Snapshot-cache (kun 1. september) -----------------
_session_singleton = _session()
_snapshot_cache: dict[int, pd.DataFrame] = {}  # year -> df(["club","elo","club_norm"])

def _fetch_snapshot_september(year: int) -> pd.DataFrame:
    if year in _snapshot_cache:
        return _snapshot_cache[year]
    # prøv https -> http
    for base in ("https://api.clubelo.com/", "http://api.clubelo.com/"):
        try:
            url = f"{base}{year}-09-01"
            r = _session_singleton.get(url, timeout=15)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
            # kolonne-robusthed
            cols = {c.lower(): c for c in df.columns}
            name_col = (cols.get("club") or cols.get("team")
                        or [c for c in df.columns if "club" in c.lower() or "team" in c.lower()][0])
            elo_col = [c for c in df.columns if "elo" in c.lower()][0]
            out = df.rename(columns={name_col: "club", elo_col: "elo"})[["club", "elo"]].copy()
            out["club_norm"] = out["club"].map(_norm)
            _snapshot_cache[year] = out
            return out
        except Exception:
            continue
    raise RuntimeError(f"Kunne ikke hente ClubElo snapshot for {year}-09-01")

# ----------------- Din offentlige API -----------------
elo_dict: dict[str, dict[int, int | None]] = {}  # {"Arsenal": {2014: 2140, ...}, ...}

def add_team_elo(dict_ref: dict, key_name: str, clubelo_name: str, start_year: int, end_year: int):
    """
    Tilføj/overskriv dict_ref[key_name] med {år: Elo} baseret på 1. september-snapshot pr. år.
    - key_name: din egen nøgle (fx "Arsenal")
    - clubelo_name: navnet i ClubElo (fx "Arsenal", "Tottenham Hotspur", "Wolverhampton Wanderers")
    - år: inklusivt interval [start_year, end_year]
    """
    cand = _candidates(clubelo_name)
    yearly = {}
    for y in range(start_year, end_year + 1):
        dfy = _fetch_snapshot_september(y)
        # 1) exact kandidatnavne
        elo = None
        for c in cand:
            hit = dfy.loc[dfy["club_norm"].eq(c)]
            if not hit.empty:
                elo = int(round(hit["elo"].iloc[0])); break
        # 2) fuzzy fallback
        if elo is None:
            close = get_close_matches(_norm(clubelo_name), dfy["club_norm"].tolist(), n=1, cutoff=0.92)
            if close:
                elo = int(round(dfy.loc[dfy["club_norm"].eq(close[0]), "elo"].iloc[0]))
        yearly[y] = elo
    dict_ref[key_name] = yearly

# ----------------- Gem/indlæs via Pickle -----------------
def save_elo_dict_pickle(dict_obj: dict, path: str = "elo_dict.pkl"):
    with open(path, "wb") as f:
        pickle.dump(dict_obj, f)

def load_elo_dict_pickle(path: str = "elo_dict.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)

# Byg dit dictionary (kun 12 netkald i alt for 2014–2025 pga. cache pr. år)
add_team_elo(elo_dict, "Arminia", "Bielefeld", 2016, 2025)
print(elo_dict["Arminia"])
add_team_elo(elo_dict, "Augsburg", "Augsburg", 2016, 2025)
add_team_elo(elo_dict, "Bayer Leverkusen", "Leverkusen", 2016, 2025)
add_team_elo(elo_dict, "Bayern Munich", "Bayern", 2016, 2025)
add_team_elo(elo_dict, "Bochum", "Bochum", 2016, 2025)
add_team_elo(elo_dict, "Darmstadt 98", "Darmstadt", 2016, 2025)
add_team_elo(elo_dict, "Dortmund", "Dortmund", 2016, 2025)
add_team_elo(elo_dict, "Düsseldorf", "Düsseldorf", 2016, 2025)
add_team_elo(elo_dict, "Eintracht Frankfurt", "Frankfurt", 2016, 2025)
add_team_elo(elo_dict, "Freiburg", "Freiburg", 2016, 2025)
add_team_elo(elo_dict, "Greuther Fürth", "Fuerth", 2016, 2025)
add_team_elo(elo_dict, "Hamburger SV", "Hamburg", 2016, 2025)
add_team_elo(elo_dict, "Hannover 96", "Hannover", 2016, 2025)
add_team_elo(elo_dict, "Heidenheim", "Heidenheim", 2016, 2025)
add_team_elo(elo_dict, "Hertha BSC", "Hertha", 2016, 2025)
add_team_elo(elo_dict, "Hoffenheim", "Hoffenheim", 2016, 2025)
add_team_elo(elo_dict, "Holstein Kiel", "Holstein", 2016, 2025)
add_team_elo(elo_dict, "Ingolstadt 04", "Ingolstadt", 2016, 2025)
add_team_elo(elo_dict, "Köln", "Koeln", 2016, 2025)
add_team_elo(elo_dict, "Mainz 05", "Mainz", 2016, 2025)
add_team_elo(elo_dict, "Mönchengladbach", "Gladbach", 2016, 2025)
add_team_elo(elo_dict, "Nürnberg", "Nürnberg", 2016, 2025)
add_team_elo(elo_dict, "Paderborn 07", "Paderborn", 2016, 2025)
add_team_elo(elo_dict, "RB Leipzig", "RB Leipzig", 2016, 2025)
add_team_elo(elo_dict, "Schalke 04", "Schalke", 2016, 2025)
add_team_elo(elo_dict, "St. Pauli", "St. Pauli", 2016, 2025)
add_team_elo(elo_dict, "Stuttgart", "Stuttgart", 2016, 2025)
add_team_elo(elo_dict, "Union Berlin", "Union Berlin", 2016, 2025)
add_team_elo(elo_dict, "Werder Bremen", "Werder", 2016, 2025)
add_team_elo(elo_dict, "Wolfsburg", "Wolfsburg", 2016, 2025)



'''cardiff_elo = {
    2015: 1400,
    2016: 1400,
    2017: 1400,
    2018: 1550,
    2019: 1550,
    2020: 1450,
    2021: 1450,
    2022: 1450,
    2023: 1400,
    2024: 1400,
    2025: 1400
}
huddersfield_elo = {
    2015: 1500,
    2016: 1500,
    2017: 1550,
    2018: 1600,
    2019: 1550,
    2020: 1400,
    2021: 1400,
    2022: 1400,
    2023: 1400,
    2024: 1400,
    2025: 1400
}
luton_elo = {
    2015: 1400,
    2016: 1400,
    2017: 1400,
    2018: 1450,
    2019: 1430,
    2020: 1400,
    2021: 1360,
    2022: 1550,
    2023: 1550,
    2024: 1400,
    2025: 1400
}
elo_dict["Cardiff City"] = cardiff_elo
elo_dict["Huddersfield Town"] = huddersfield_elo
elo_dict["Luton Town"] = luton_elo'''

# Gem én gang
save_elo_dict_pickle(elo_dict, "German Bundesliga/elo_dict_bundesliga.pkl")

# …senere / i et nyt script:
elo_dict = load_elo_dict_pickle("German Bundesliga/elo_dict_bundesliga.pkl")
print(elo_dict["Greuther Fürth"])
