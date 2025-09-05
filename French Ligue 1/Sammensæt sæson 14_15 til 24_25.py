import pandas as pd

# Liste over dine filer
filnavne = [
    "French Ligue 1/Raw/16_17.xlsx", 
    "French Ligue 1/Raw/17_18.xlsx", "French Ligue 1/Raw/18_19.xlsx", "French Ligue 1/Raw/19_20.xlsx", 
    "French Ligue 1/Raw/20_21.xlsx", "French Ligue 1/Raw/21_22.xlsx", "French Ligue 1/Raw/22_23.xlsx", 
    "French Ligue 1/Raw/23_24.xlsx", "French Ligue 1/Raw/24_25.xlsx"
]

# Læs alle filer og sæt dem sammen
alle = pd.concat([pd.read_excel(f, engine="openpyxl") for f in filnavne],
                 ignore_index=True)

# Gem samlet fil
alle.to_excel("French Ligue 1/season_16_17_to_24_25.xlsx", index=False)

print("Færdig – samlet fil gemt")