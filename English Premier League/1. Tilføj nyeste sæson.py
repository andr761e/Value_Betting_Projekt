# ---------- Pakker ----------
import time, random, re, os
from datetime import datetime
from urllib.parse import urljoin
import pandas as pd
from lxml import html
import cloudscraper  # Cloudflare-bypass
import pickle
import numpy as np

# ---------- Session med Cloudflare-bypass ----------
def make_session():
    s = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/122.0 Safari/537.36"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://fbref.com/",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    })
    # varm cookies
    s.get("https://fbref.com/")
    return s

def _from_local_html(path):
    with open(path, "r", encoding="utf-8") as f:
        return html.fromstring(f.read())

def fetch_doc(session, url, tries=6, backoff=(0.8, 1.8), timeout=30):
    # Tillad lokal HTML for offline test
    if os.path.exists(url) and url.lower().endswith(".html"):
        return _from_local_html(url)
    last_exc = None
    for _ in range(tries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 403:
                session.get("https://fbref.com/")  # opdatér cookies
                time.sleep(random.uniform(*backoff))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}")
            return html.fromstring(r.text)
        except Exception as e:
            last_exc = e
            time.sleep(random.uniform(*backoff))
    raise last_exc

# ---------- Hjælpere ----------
def as_num(x):
    if x is None:
        return None
    x = re.sub(r"[,%]", "", str(x)).strip()
    try:
        return float(x)
    except ValueError:
        return None

def get_first(lst, i, default=None):
    return lst[i] if len(lst) > i else default

def normalize_label(lbl: str) -> str:
    """Normaliserer label fra 'Team Stats Extra'."""
    lbl = (lbl or "").strip().lower()
    lbl = lbl.replace("-", " ")
    lbl = re.sub(r"\s+", " ", lbl)
    lbl = re.sub(r"[^a-z ]", "", lbl)
    return lbl.replace(" ", "")

# Map fra normaliseret label -> kolonnenavn-suffix (ASCII)
EXTRA_MAP = {
    "fouls": "Fouls",
    "corners": "Corners",
    "crosses": "Crosses",
    "touches": "Touches",
    "tackles": "Tackles",
    "interceptions": "Interceptions",
    "aerialswon": "Aerials_won",
    "clearances": "Clearances",
    "offsides": "Offsides",
    "goalkicks": "Goal_kicks",
    "throwins": "Throw_ins",
    "longballs": "Long_balls",
}

# ---------- Parser for én kamp ----------
def scrape_match(url, session=None):
    s = session or make_session()
    doc = fetch_doc(s, url)

    # Scorebox
    team_nodes  = doc.cssselect(".scorebox strong a")
    score_nodes = doc.cssselect(".scorebox .scores .score")

    # Dato
    date_txt_el = doc.cssselect(".scorebox_meta")
    date_iso = None
    if date_txt_el:
        m = re.search(r"[A-Za-z]+\s+\d{1,2},\s*\d{4}", date_txt_el[0].text_content())
        if m:
            date_iso = datetime.strptime(m.group(0), "%B %d, %Y").strftime("%Y-%m-%d")

    hjem_navn = get_first(team_nodes, 0).text_content().strip() if team_nodes else None
    ude_navn  = get_first(team_nodes, 1).text_content().strip() if team_nodes else None
    hjem_mal  = as_num(get_first(score_nodes, 0).text_content() if score_nodes else None)
    ude_mal   = as_num(get_first(score_nodes, 1).text_content() if score_nodes else None)

    if hjem_mal is not None and ude_mal is not None:
        if hjem_mal > ude_mal: ftr = "H"
        elif hjem_mal < ude_mal: ftr = "A"
        else: ftr = "D"
    else:
        ftr = None

    # Possession
    poss_nodes = doc.xpath("//div[@id='team_stats']//tr[th[contains(.,'Possession')]]"
                           "/following-sibling::tr[1]//strong")
    if len(poss_nodes) >= 2:
        hjem_poss = as_num(poss_nodes[0].text_content().replace("%",""))
        ude_poss  = as_num(poss_nodes[1].text_content().replace("%",""))
    else:
        hjem_poss = ude_poss = None

    # Shots + SOT
    tf_shots = doc.xpath("//table[starts-with(@id,'stats_') and contains(@id,'_summary')]/tfoot")
    def get_tot(tfs, data_stat):
        if not tfs: return None
        cell = tfs[0].xpath(f".//td[@data-stat='{data_stat}']")
        return as_num(cell[0].text_content()) if cell else None
    hjem_skud = get_tot(tf_shots[0:1], "shots")
    ude_skud  = get_tot(tf_shots[1:2], "shots")
    hjem_sot  = get_tot(tf_shots[0:1], "shots_on_target")
    ude_sot   = get_tot(tf_shots[1:2], "shots_on_target")

    # Team stats extra (robust triads)
    extra_vals = {f"Hjemmehold{v}": None for v in EXTRA_MAP.values()}
    extra_vals.update({f"Udehold{v}": None for v in EXTRA_MAP.values()})

    groups = doc.xpath("//div[@id='team_stats_extra']/div")
    for g in groups:
        # skip header cells: div.th
        subs = g.xpath("./div[not(@class='th')]")
        txt  = [d.text_content().strip() for d in subs]
        # chunk 3‑og‑3: [home, label, away]
        for i in range(0, len(txt), 3):
            tri = txt[i:i+3]
            if len(tri) != 3:
                continue
            home, label, away = tri
            key_norm = normalize_label(label)
            if key_norm in EXTRA_MAP:
                suf = EXTRA_MAP[key_norm]
                extra_vals[f"Hjemmehold{suf}"] = as_num(home)
                extra_vals[f"Udehold{suf}"]    = as_num(away)

    # Passing totals (ny + fallback for ældre kampe)
    tf_pass = doc.xpath("//table[starts-with(@id,'stats_') and contains(@id,'_passing') "
                        "and not(contains(@id,'_passing_types'))]/tfoot")

    def parse_passing_accuracy_from_team_stats(doc):
        # Finder rækken lige under "Passing Accuracy"
        rows = doc.xpath("//div[@id='team_stats']//tr[th[contains(., 'Passing Accuracy')]]/following-sibling::tr[1]")
        if len(rows) < 1:
            return None
        cells = rows[0].xpath(".//td/div/div[1]")  # den tekst, hvor der står "482 of 558 — 86%"
        # Forventet to celler: [home, away]
        out = []
        for c in cells[:2]:
            txt = "".join(c.itertext()).strip()
            # Ekstraher "completed of attempted — pct%"
            # Eksempler: "482 of 558 — 86%" eller "80% — 307 of 383" (rækkefølgen kan variere)
            import re
            # Fang begge rækkefølger robust
            m1 = re.search(r"(\d+)\s+of\s+(\d+).+?(\d+)%", txt)  # "482 of 558 — 86%"
            m2 = re.search(r"(\d+)%.+?(\d+)\s+of\s+(\d+)", txt)  # "80% — 307 of 383"
            if m1:
                comp, att, pct = m1.group(1), m1.group(2), m1.group(3)
            elif m2:
                pct, comp, att = m2.group(1), m2.group(2), m2.group(3)
            else:
                comp = att = pct = None
            out.append( (comp, att, pct) )
        return out if len(out) == 2 else None

    def get_pass_from_tfoot(tfs, stat):
        if not tfs: return None
        cell = tfs[0].xpath(f".//td[@data-stat='{stat}']")
        return as_num(cell[0].text_content()) if cell else None

    if tf_pass:
        hjem_pass     = get_pass_from_tfoot(tf_pass[0:1], "passes")
        ude_pass      = get_pass_from_tfoot(tf_pass[1:2], "passes")
        hjem_pass_ok  = get_pass_from_tfoot(tf_pass[0:1], "passes_completed")
        ude_pass_ok   = get_pass_from_tfoot(tf_pass[1:2], "passes_completed")
        hjem_pass_pct = get_pass_from_tfoot(tf_pass[0:1], "passes_pct")
        ude_pass_pct  = get_pass_from_tfoot(tf_pass[1:2], "passes_pct")
    else:
        acc = parse_passing_accuracy_from_team_stats(doc)
        if acc:
            (h_comp, h_att, h_pct), (a_comp, a_att, a_pct) = acc
            hjem_pass_ok  = as_num(h_comp)
            hjem_pass     = as_num(h_att)
            hjem_pass_pct = as_num(h_pct)
            ude_pass_ok   = as_num(a_comp)
            ude_pass      = as_num(a_att)
            ude_pass_pct  = as_num(a_pct)
        else:
            hjem_pass = ude_pass = hjem_pass_ok = ude_pass_ok = hjem_pass_pct = ude_pass_pct = None


    # Række (ASCII kolonnenavne – ingen æ/ø/å)
    row = {
        "Dato": date_iso, "FuldtidsResultat": ftr,
        "HjemmeholdNavn": hjem_navn, "HjemmeholdMal": hjem_mal,
        "HjemmeholdSkud": hjem_skud, "HjemmeholdSkudPaMal": hjem_sot, "HjemmeholdPossession": hjem_poss,
        "UdeholdNavn": ude_navn, "UdeholdMal": ude_mal,
        "UdeholdSkud": ude_skud, "UdeholdSkudPaMal": ude_sot, "UdeholdPossession": ude_poss,
        "HjemmeholdPasses": hjem_pass, "HjemmeholdSuccesfulPasses": hjem_pass_ok, "HjemmeholdPassAccuracy": hjem_pass_pct,
        "UdeholdPasses": ude_pass, "UdeholdSuccesfulPasses": ude_pass_ok, "UdeholdPassAccuracy": ude_pass_pct,
    }
    row.update(extra_vals)

    cols = [
        "Dato","FuldtidsResultat",
        "HjemmeholdNavn","HjemmeholdMal",
        "HjemmeholdSkud","HjemmeholdSkudPaMal","HjemmeholdPossession",
        "HjemmeholdFouls","HjemmeholdCorners","HjemmeholdCrosses","HjemmeholdTouches",
        "HjemmeholdTackles","HjemmeholdInterceptions","HjemmeholdAerials_won",
        "HjemmeholdClearances","HjemmeholdOffsides","HjemmeholdGoal_kicks",
        "HjemmeholdThrow_ins","HjemmeholdLong_balls",
        "HjemmeholdPasses","HjemmeholdSuccesfulPasses","HjemmeholdPassAccuracy",
        "UdeholdNavn","UdeholdMal",
        "UdeholdSkud","UdeholdSkudPaMal","UdeholdPossession",
        "UdeholdFouls","UdeholdCorners","UdeholdCrosses","UdeholdTouches",
        "UdeholdTackles","UdeholdInterceptions","UdeholdAerials_won",
        "UdeholdClearances","UdeholdOffsides","UdeholdGoal_kicks",
        "UdeholdThrow_ins","UdeholdLong_balls",
        "UdeholdPasses","UdeholdSuccesfulPasses","UdeholdPassAccuracy"
    ]
    for c in cols:
        row.setdefault(c, None)

    return pd.DataFrame([row], columns=cols)

# ---------- Finder alle Match Reports fra en sæsonside ----------
def find_match_report_links(fixtures_url, session=None):
    s = session or make_session()
    doc = fetch_doc(s, fixtures_url)
    links = [urljoin("https://fbref.com", a) for a in doc.xpath("//a[contains(., 'Match Report')]/@href")]
    return list(dict.fromkeys(links))  # unikke i rækkefølge

# ---------- Scrape en hel sæson ----------
def scrape_season(fixtures_url, sleep_range=(0.9, 1.8)):
    s = make_session()
    urls = find_match_report_links(fixtures_url, session=s)
    rows = []
    for i, u in enumerate(urls, 1):
        try:
            df = scrape_match(u, session=s)
            rows.append(df)
        except Exception as e:
            print(f"[{i}/{len(urls)}] Fejl på {u}: {e}")
        time.sleep(random.uniform(*sleep_range))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


#Hent alle 25/26 kampe
fixtures = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
df_season = scrape_season(fixtures)
df_season.to_excel("English Premier League/Raw/25_26.xlsx", index=False)


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
df1 = pd.read_excel("English Premier League/Raw/25_26.xlsx")

# tilføj ELO kolonner
df1 = add_elo_columns(df1, elo_dict)

# load alle tidligere sæsoner 
df2 = pd.read_excel("English Premier League/Alle_tidligere_seasons_med_elo.xlsx")

# Sæt den nye sæson efter de gamle (samme kolonner forudsættes)
df_combined = pd.concat([df2, df1], ignore_index=True)

# gem filen
df_combined.to_excel("English Premier League/Alle_kampe_med_elo.xlsx", index=False)


