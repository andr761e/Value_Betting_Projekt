import pandas as pd

# Indlæs Excel-arket
df = pd.read_excel("French Ligue 1/season_16_17_to_24_25.xlsx")  # <- skift til dit filnavn

# Hent kolonnen "HjemmeholdNavn"
hjemmehold = df["HjemmeholdNavn"]

# Find unikke værdier
unikke_hold = sorted(hjemmehold.unique())

# Print dem
print("Unikke hjemmehold:")
for hold in unikke_hold:
    print(hold)
