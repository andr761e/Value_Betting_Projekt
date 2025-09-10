import pandas as pd

M_df = pd.read_excel("Alternative modeller (forsøg)/alle_ligaer_sorteret_efter_dato.xlsx",skiprows=0)  # indeholder 'FuldtidsResultat'

labels_raw = M_df["FuldtidsResultat"].astype(str).str.strip().str.upper()

# robust map (H/D/A + evt. 1/X/2)
label_map = {"H":0,"D":1, "A":2}

y_series = labels_raw.map(label_map)

# find ugyldige / NaN
bad_mask = y_series.isna()
if bad_mask.any():
    print("⚠️ Der er ugyldige eller manglende labels:\n")
    print(M_df.loc[bad_mask, ["FuldtidsResultat"]])   # viser kolonnen for de rækker
    print("\nRækkenumre i datasættet:", M_df.index[bad_mask].to_list())
    raise ValueError("Stopper her så du kan tjekke de pågældende rækker ovenfor.")