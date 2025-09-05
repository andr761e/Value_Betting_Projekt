import pandas as pd
import sys

# --- Indlæs dine to filer (tilpas stier) ---
df_master  = pd.read_excel("German Bundesliga/Alle_kampe_med_elo.xlsx")                 # korrekt rækkefølge
df_shuffle = pd.read_excel("German Bundesliga/Odds-rækkefølge/Odds_forkert_rækkefølge.xlsx")  # skal sorteres

assert len(df_master) == len(df_shuffle), "Datasættene har forskelligt antal kampe!"

# --- Normalisering af holdnavne / alias-håndtering ---
ALIASES = {
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Köln" : "FC Koln",
    "Hamburger SV" : "Hamburg",
    "Ingolstadt 04" : "Ingolstadt",
    "Darmstadt 98" : "Darmstadt",
    "Mainz 05" : "Mainz",
    "Mönchengladbach" : "M'gladbach",
    "Bayer Leverkusen" : "Leverkusen",
    "Hertha BSC" : "Hertha",
    "Hannover 96" : "Hannover",
    "Nürnberg" : "Nurnberg",
    "Düsseldorf" : "Fortuna Dusseldorf",
    "Paderborn 07" : "Paderborn",
    "Arminia" : "Bielefeld",
    "Greuther Fürth" : "Greuther Furth",
    "St. Pauli" : "St Pauli"
}

# 1) Brug en liste af blokstørrelser (i rækkefølge) i stedet for SEASON_SIZE
BLOCK_SIZES = [306, 306, 306, 306, 306, 306, 306, 306, 306, 18]

def clean(s):
    if pd.isna(s):
        return ""
    return str(s).replace("\u00A0", " ").replace("\u200b", "").strip()

def as_shuffle_name(master_raw):
    return clean(ALIASES.get(master_raw, master_raw))

# Pre-clean (rører ikke originalkolonnerne)
m = df_master.copy()
s = df_shuffle.copy()
m["__home__"] = m["HjemmeholdNavn"].map(clean)
m["__away__"] = m["UdeholdNavn"].map(clean)
s["__home__"] = s["HomeTeam"].map(clean)
s["__away__"] = s["AwayTeam"].map(clean)

sorted_blocks = []
start = 0

for bsize in BLOCK_SIZES:
    if start >= len(m):
        break

    stop = min(start + bsize, len(m))

    block_master  = m.iloc[start:stop].copy()
    block_shuffle = s.iloc[start:stop].copy()

    # pool = resterende rækker i shuffle-blokken, der ikke er brugt endnu (behold original index)
    pool = block_shuffle.reset_index(drop=False)

    rows = []
    for idx in range(len(block_master)):
        hm_raw = block_master.iloc[idx]["HjemmeholdNavn"]
        am_raw = block_master.iloc[idx]["UdeholdNavn"]

        h1, a1 = clean(hm_raw), clean(am_raw)
        h2, a2 = as_shuffle_name(hm_raw), as_shuffle_name(am_raw)

        candidates = [
            (h1, a1),
            (h2, a1) if h2 != h1 else None,
            (h1, a2) if a2 != a1 else None,
            (h2, a2) if (h2 != h1 or a2 != a1) else None,
        ]
        candidates = [c for c in candidates if c is not None]

        hit_row = None
        for (hh, aa) in candidates:
            hit = pool[(pool["__home__"] == hh) & (pool["__away__"] == aa)]
            if not hit.empty:
                hit_row = hit.iloc[0]
                break

        if hit_row is None:
            print(f"\n❌ Ingen match i blok {start}:{stop-1} ved master-række {start+idx}")
            print("Master Hjemme/Ude (raw):", hm_raw, "/", am_raw)
            print("Prøvede kombinationer (clean/alias):", candidates)
            remaining = list(zip(pool["__home__"].tolist(), pool["__away__"].tolist()))
            print("Eksempler på resterende Home/Away (første 20):", remaining[:20])
            sys.exit("Tilføj/ret ALIASES (master->shuffle) og kør igen.")

        # Tilføj fundet række i korrekt rækkefølge og fjern den fra pool
        rows.append(block_shuffle.loc[hit_row["index"]])
        pool = pool[pool["index"] != hit_row["index"]].reset_index(drop=True)

    block_sorted = pd.DataFrame(rows).reset_index(drop=True)
    sorted_blocks.append(block_sorted)
    start = stop  # flyt start til næste blok

# Sæt alle blokke sammen i korrekt odds-rækkefølge
df_aligned = pd.concat(sorted_blocks, ignore_index=True)

# Gem til Excel (den sorterede odds-rækkefølge)
sorted_odds_path = "German Bundesliga/Odds-rækkefølge/odds_kampe_sorteret.xlsx"
df_aligned.to_excel(sorted_odds_path, index=False)
print(f"Færdig! Ny fil gemt som {sorted_odds_path}")

# 2) Genindlæs odds-filen og læs kolonnerne V:AG
odds_V_AO = pd.read_excel(sorted_odds_path, usecols="V:AD")

# Gør ALT numerisk (tomme felter/tekst -> NaN) og fjern evt. 0'ere
odds_V_AO = odds_V_AO.apply(pd.to_numeric, errors="coerce").replace(0, pd.NA)

# 3) Kolonnelister til H/D/A (fx B365H, B365D, B365A, ..., VCH, VCD, VCA)
h_cols = [c for c in odds_V_AO.columns if str(c).endswith("H")]
d_cols = [c for c in odds_V_AO.columns if str(c).endswith("D")]
a_cols = [c for c in odds_V_AO.columns if str(c).endswith("A")]

# Gennemsnit (ignorerer NaN automatisk)
avg_h = odds_V_AO[h_cols].mean(axis=1, skipna=True)
avg_d = odds_V_AO[d_cols].mean(axis=1, skipna=True)
avg_a = odds_V_AO[a_cols].mean(axis=1, skipna=True)

# Implicitte sandsynligheder (rå) – beskyt mod division med 0/NaN
prob_h_raw = 1.0 / avg_h
prob_d_raw = 1.0 / avg_d
prob_a_raw = 1.0 / avg_a

probs_raw = pd.concat([prob_h_raw, prob_d_raw, prob_a_raw], axis=1)
probs_raw.columns = ["PROB H", "PROB D", "PROB A"]

# 3b) Normalisér så radsum = 1 (rækker hvor alt er NaN forbliver NaN)
row_sum = probs_raw.sum(axis=1, min_count=1)
probs_norm = probs_raw.div(row_sum, axis=0)

# 4) Tilføj kolonner til master
df_master = df_master.copy()
df_master["AVG H"]  = avg_h
df_master["AVG D"]  = avg_d
df_master["AVG A"]  = avg_a
df_master["PROB H"] = probs_norm["PROB H"]
df_master["PROB D"] = probs_norm["PROB D"]
df_master["PROB A"] = probs_norm["PROB A"]

# Gem som før
master_out_path = "German Bundesliga/Alle_kampe_med_elo_og_odds.xlsx"
df_master.to_excel(master_out_path, index=False)
print(f"Master opdateret og gemt som {master_out_path}")